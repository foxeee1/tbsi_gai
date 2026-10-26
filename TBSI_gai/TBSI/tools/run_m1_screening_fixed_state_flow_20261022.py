#!/usr/bin/env python3
"""Capture target/hard-background evidence on calibration onset frame pairs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLOW_MODULE = ROOT / 'tools/diagnose_m1_lowauc_information_flow_20261020.py'
SPEC = importlib.util.spec_from_file_location('m1_flow_capture_base', str(FLOW_MODULE))
flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow)
base = flow.base


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_events(path):
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    out = {}
    for row in events:
        seq = row['sequence']
        pair = []
        if row.get('pre_frame') is not None:
            pair.append({'phase': 'pre_onset', 'frame': row['pre_frame'],
                         'trace_iou': row['pre_iou'], 'state_before': row['pre_state_before'],
                         'gt_box': row['pre_gt_box'], 'crop_xywh': row['pre_crop_xywh']})
        pair.append({'phase': 'severe_onset', 'frame': row['frame'],
                     'trace_iou': row['m1_iou'], 'state_before': row['state_before'],
                     'gt_box': row['gt_box'], 'crop_xywh': row['crop_xywh']})
        out[seq] = pair
    return out


def feature_space_rank(tensor, gt_mask, hard_bg_y, hard_bg_x):
    """Measure self-consistency of target features against the strongest background."""
    values = tensor.detach().float().cpu().numpy()[0]
    feature = values.transpose(1, 2, 0)
    mask = gt_mask.astype(bool)
    if mask.shape != feature.shape[:2]:
        raise ValueError('GT mask and fused feature map have different resolutions')
    prototype = feature[mask].mean(axis=0)
    prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
    norms = np.maximum(np.linalg.norm(feature, axis=-1, keepdims=True), 1e-12)
    score = (feature / norms * prototype[None, None, :]).sum(axis=-1)
    target_peak = float(score[mask].max())
    background = score[~mask]
    bg_peak = float(background.max())
    top = np.unravel_index(int(np.argmax(score)), score.shape)
    return {
        'target_peak': target_peak,
        'hard_bg_peak': bg_peak,
        'target_hard_bg_margin': target_peak - bg_peak,
        'target_rank': 1 + int(np.sum(background > target_peak)),
        'top1_in_gt': bool(mask[top]),
        'fixed_center_hard_bg_score': float(score[int(hard_bg_y), int(hard_bg_x)]),
    }


def foreground_background_query_stats(template_tokens, search_tokens, template_mask,
                                      gt_mask, hard_bg_y, hard_bg_x):
    """Use the same normalized foreground-minus-background query as query_map()."""
    template = template_tokens.detach().float().cpu().numpy()
    search = search_tokens.detach().float().cpu().numpy()
    if template.ndim == 4:
        template = template[0].transpose(1, 2, 0).reshape(-1, template.shape[1])
    if search.ndim == 4:
        search = search[0].transpose(1, 2, 0).reshape(-1, search.shape[1])
        search = search.reshape(16, 16, -1)
    elif search.ndim == 3:
        if search.shape[0] != 1:
            raise ValueError('batched token search input must have batch size 1')
        search = search[0].reshape(16, 16, -1)
    elif search.ndim == 2:
        if search.shape[0] != 256:
            raise ValueError('flat search token input must contain 256 tokens')
        search = search.reshape(16, 16, -1)
    weights = np.asarray(template_mask, dtype=np.float32).reshape(-1)
    if template.shape[0] != weights.size or search.shape[:2] != gt_mask.shape:
        raise ValueError('template/search/query-mask shape contract mismatch')
    if not np.isfinite(weights).all() or weights.min() < 0 or weights.max() > 1:
        raise ValueError('template foreground mask must be finite and in [0, 1]')
    template = template / np.maximum(np.linalg.norm(template, axis=-1, keepdims=True), 1e-6)
    search = search / np.maximum(np.linalg.norm(search, axis=-1, keepdims=True), 1e-6)
    fg_weight = weights[:, None]
    bg_weight = (1.0 - weights)[:, None]
    fg_proto = (template * fg_weight).sum(axis=0) / max(float(fg_weight.sum()), 1e-6)
    bg_proto = (template * bg_weight).sum(axis=0) / max(float(bg_weight.sum()), 1e-6)
    fg_proto /= max(float(np.linalg.norm(fg_proto)), 1e-6)
    bg_proto /= max(float(np.linalg.norm(bg_proto)), 1e-6)
    score = (search * fg_proto[None, None, :]).sum(axis=-1)
    score -= (search * bg_proto[None, None, :]).sum(axis=-1)
    mask = gt_mask.astype(bool)
    target_peak = float(score[mask].max())
    background = score[~mask]
    strongest_bg_peak = float(background.max())
    fixed_bg_score = float(score[int(hard_bg_y), int(hard_bg_x)])
    top = np.unravel_index(int(np.argmax(score)), score.shape)
    return {
        'query_gt_peak': target_peak,
        'query_hard_bg': fixed_bg_score,
        'query_gt_hard_bg_margin': target_peak - fixed_bg_score,
        'query_strongest_bg_peak': strongest_bg_peak,
        'query_gt_strongest_bg_margin': target_peak - strongest_bg_peak,
        'query_gt_rank': 1 + int(np.sum(background > target_peak)),
        'query_peak_in_gt': bool(mask[top]),
    }


def sparse_candidate_coverage(context, template_mask, gt_mask, control, topk):
    if context is None:
        return {'sparse_gt_coverage': 'NA', 'sparse_gt_cell_fraction': 'NA'}
    rgb_t, tir_t, rgb_s, tir_s = context
    masks = []
    for template, search in ((rgb_t, rgb_s), (tir_t, tir_s)):
        weight = template_mask.to(device=template.device, dtype=template.dtype).clamp(0, 1)
        proto = (template * weight.unsqueeze(-1)).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-6)
        proto = F.normalize(proto.float(), dim=-1, eps=1e-6)
        search_n = F.normalize(search.float(), dim=-1, eps=1e-6)
        score = (search_n * proto.unsqueeze(1)).sum(-1)
        idx = score.topk(min(max(1, topk), score.shape[1]), dim=-1).indices
        mask = torch.zeros_like(score).scatter(1, idx, 1.0)
        mask = F.max_pool2d(mask.reshape(1, 1, 16, 16), 3, stride=1, padding=1).reshape(1, -1)
        masks.append(mask)
    candidate = torch.maximum(masks[0], masks[1]).reshape(16, 16)
    if control == 'shifted':
        candidate = torch.roll(candidate, shifts=(8, 8), dims=(0, 1))
    candidate = candidate.detach().cpu().bool()
    gt = torch.as_tensor(gt_mask, dtype=torch.bool)
    gt_count = int(gt.sum())
    return {
        'sparse_gt_coverage': float((candidate & gt).sum()) / max(gt_count, 1),
        'sparse_gt_cell_fraction': float((candidate & gt).sum()) / max(int(candidate.sum()), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', required=True)
    parser.add_argument('--sequence-list', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-states', type=int, default=0,
                        help='optional state cap for measurement-contract smoke checks')
    parser.add_argument('--bypass-block', type=int, choices=(2, 5, 11), default=0,
                        help='reversible diagnostic: replace one block output with its input')
    parser.add_argument('--bypass-tbsi-loc', type=int, choices=(3, 6, 9), default=0,
                        help='reversible diagnostic: replace one TBSI output by its RGB/TIR inputs')
    parser.add_argument('--search-fusion-arm', choices=('rgb', 'tir'), default='',
                        help='replace fused search output with a single-modality fuser output')
    parser.add_argument('--sparse-mask-control', choices=('off', 'none', 'shifted'), default='off',
                        help='disable sparse fusion, use target-conditioned union, or use an equal-density shifted mask')
    parser.add_argument('--config', default='',
                        help='explicit tracker config path for post-training evaluation')
    parser.add_argument('--checkpoint', default='',
                        help='explicit tracker checkpoint path for post-training evaluation')
    parser.add_argument('--head-map-archive', default='',
                        help='optional NPZ path for CENTER, size, offset, Hann, GT and fixed-state arrays')
    parser.add_argument('--spatial-token-bridge-mode', choices=('off', 'target', 'shifted'),
                        default='off', help='explicit bridge arm; requires a bridge-enabled config')
    parser.add_argument('--spatial-token-bridge-scale', type=float, default=1.0,
                        help='diagnostic post-bridge message multiplier; default preserves behavior')
    parser.add_argument('--split', choices=('trainingset', 'testingset'), default='trainingset',
                        help='LasHeR split containing the supplied event sequences')
    args = parser.parse_args()
    if sum(bool(value) for value in
           (args.bypass_block, args.bypass_tbsi_loc, args.search_fusion_arm,
            args.sparse_mask_control != 'off')) > 1:
        raise ValueError('select exactly one intervention per run')
    if bool(args.config) != bool(args.checkpoint):
        raise ValueError('--config and --checkpoint must be supplied together')
    if args.config:
        config_path = Path(args.config).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError('explicit config or checkpoint does not exist')
        base.CONFIG = config_path
        base.CHECKPOINT = checkpoint_path
        base.CONFIG_NAME = config_path.stem
    elif sha256(base.CONFIG) != base.CONFIG_SHA256 or sha256(base.CHECKPOINT) != base.CHECKPOINT_SHA256:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')

    event_path = Path(args.events).resolve()
    sequence_list = Path(args.sequence_list).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not event_path.is_file() or not sequence_list.is_file():
        raise FileNotFoundError('events or cohort sequence list not found')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output directory')
    output_dir.mkdir(parents=True, exist_ok=True)

    events = read_events(event_path)
    names = [line.strip() for line in sequence_list.read_text().splitlines() if line.strip()]
    if set(events) - set(names):
        raise RuntimeError('event sequence outside the supplied calibration cohort')
    torch.set_num_threads(1)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '0'
    os.environ['TBSI_TEMPLATE_UNCERTAINTY_MODE'] = 'off'
    os.environ['TBSI_SPATIAL_TOKEN_BRIDGE'] = args.spatial_token_bridge_mode
    os.environ['TBSI_SPARSE_CROSS_MODE'] = 'off' if args.sparse_mask_control == 'off' else 'neck_union'
    os.environ['TBSI_SPARSE_MASK_CONTROL'] = 'shifted' if args.sparse_mask_control == 'shifted' else 'none'
    os.environ.setdefault('TBSI_SPARSE_TOPK', '8')
    tracker = base.new_tracker('off', 'm1_screening_fixed_state_flow')
    sequence_map = {seq.name: seq for seq in base.LasHeRDataset(
        args.split, str(sequence_list)).get_sequence_list()}

    per_event, stages, head_maps = [], [], []
    for seq_index, name in enumerate(sorted(events), 1):
        sequence = sequence_map[name]
        tracker.initialize(base.read_pair(sequence, 0), {
            'init_bbox': np.asarray(sequence.ground_truth_rect[0]), 'sequence_name': name})
        for event in events[name]:
            if args.max_states and len(per_event) >= args.max_states:
                break
            fusion_tensors = {}
            def capture_fusion_input(module, inputs):
                fusion_tensors['input'] = inputs[0].detach().float().cpu()
            def capture_fusion_output(module, inputs, output):
                fusion_tensors['output'] = output.detach().float().cpu()
            def capture_norm(module, inputs, output):
                fusion_tensors['norm_tokens'] = output.detach().float().cpu()
            fusion_handles = [tracker.network.tbsi_fuse_search.register_forward_pre_hook(
                capture_fusion_input)]
            if args.spatial_token_bridge_mode != 'off' and args.spatial_token_bridge_scale != 1.0:
                def scale_bridge_message(module, inputs, output):
                    rgb_in, tir_in = inputs[:2]
                    rgb_out, tir_out, stats = output
                    scale = float(args.spatial_token_bridge_scale)
                    return (rgb_in + scale * (rgb_out - rgb_in),
                            tir_in + scale * (tir_out - tir_in), stats)
                fusion_handles.append(tracker.network.spatial_token_bridge.register_forward_hook(
                    scale_bridge_message))
            if args.search_fusion_arm:
                def replace_search_fusion(module, inputs, output):
                    feature = inputs[0]
                    channels = feature.shape[1] // 2
                    zeros = torch.zeros_like(feature[:, :channels])
                    if args.search_fusion_arm == 'rgb':
                        replacement = torch.cat((feature[:, :channels], zeros), dim=1)
                    else:
                        replacement = torch.cat((zeros, feature[:, channels:]), dim=1)
                    return module.forward(replacement)
                fusion_handles.append(tracker.network.tbsi_fuse_search.register_forward_hook(
                    replace_search_fusion))
            fusion_handles.extend([
                tracker.network.tbsi_fuse_search.register_forward_hook(capture_fusion_output),
                tracker.network.backbone.norm.register_forward_hook(capture_norm),
            ])
            if args.bypass_block:
                block = tracker.network.backbone.blocks[args.bypass_block]
                def bypass_block_output(module, inputs, output):
                    return inputs[0]
                fusion_handles.append(block.register_forward_hook(bypass_block_output))
            if args.bypass_tbsi_loc:
                locations = tracker.network.backbone.tbsi_loc
                if args.bypass_tbsi_loc not in locations:
                    raise RuntimeError('requested TBSI location is not present: {}'.format(
                        args.bypass_tbsi_loc))
                layer = tracker.network.backbone.tbsi_layers[locations.index(
                    args.bypass_tbsi_loc)]
                def bypass_tbsi_output(module, inputs, output):
                    return inputs[0], inputs[1]
                fusion_handles.append(layer.register_forward_hook(bypass_tbsi_output))
            fusion_tensors.clear()
            try:
                row, stage_rows = flow.run_event(tracker, event, sequence)
                diag = tracker._last_fixed_state_diag
                head_maps.append({
                    'event_id': '{}:{}:{}'.format(name, event['phase'], event['frame']),
                    'score_map': diag['score_map'][0, 0].numpy().astype(np.float32),
                    'size_map': diag['size_map'][0].numpy().astype(np.float32),
                    'offset_map': diag['offset_map'][0].numpy().astype(np.float32),
                    'hann': tracker.output_window[0, 0].detach().float().cpu().numpy().astype(np.float32),
                    'gt_box': np.asarray(event['gt_box'], dtype=np.float32),
                    'crop_xywh': np.asarray(event['crop_xywh'], dtype=np.float32),
                    'state_before': np.asarray(event['state_before'], dtype=np.float32),
                    'resize_factor': np.float32(diag['resize_factor']),
                    'image_shape': np.asarray(diag['image_shape'], dtype=np.int32),
                    'gt_mask': base.grid_mask(event['gt_box'], event['crop_xywh']).astype(np.uint8),
                })
            finally:
                for handle in fusion_handles:
                    handle.remove()
            sparse_stats = getattr(tracker.network, '_last_sparse_stats', None)
            if sparse_stats:
                for key, value in sparse_stats.items():
                    row['sparse_' + key] = value
            bridge_stats = getattr(
                tracker.network, '_last_spatial_token_bridge_stats', None)
            if bridge_stats:
                for key, value in bridge_stats.items():
                    row['bridge_' + key] = value
            row['bridge_output_scale'] = float(args.spatial_token_bridge_scale)
            coverage = sparse_candidate_coverage(
                getattr(tracker.network.backbone, '_last_early_sparse_context', None),
                tracker.template_fg_mask, base.grid_mask(event['gt_box'], event['crop_xywh']),
                args.sparse_mask_control, int(os.environ['TBSI_SPARSE_TOPK']))
            row.update(coverage)
            gt_mask = base.grid_mask(event['gt_box'], event['crop_xywh'])
            fixed_bg_y, fixed_bg_x = int(row['hard_bg_y']), int(row['hard_bg_x'])
            fused_input = fusion_tensors.get('input')
            tokens = fusion_tensors.get('norm_tokens')
            if fused_input is None or 'output' not in fusion_tensors or tokens is None:
                raise RuntimeError('search fusion feature capture is incomplete')
            channels = fused_input.shape[1] // 2
            for modality, feature in (('rgb', fused_input[:, :channels]),
                                      ('tir', fused_input[:, channels:])):
                stats = feature_space_rank(feature, gt_mask, fixed_bg_y, fixed_bg_x)
                for key, value in stats.items():
                    row['prefusion_{}_{}'.format(modality, key)] = value
            token_dim = tokens.shape[-1]
            rgb_template = tokens[0, :64]
            rgb_search = tokens[0, 64:320]
            tir_template = tokens[0, 320:384]
            tir_search = tokens[0, 384:640]
            template_mask = tracker.c1_template_mask.detach().float().cpu().numpy().reshape(-1)
            for modality, template, search in (('rgb', rgb_template, rgb_search),
                                               ('tir', tir_template, tir_search)):
                stats = foreground_background_query_stats(
                    template, search, template_mask, gt_mask, fixed_bg_y, fixed_bg_x)
                for key, value in stats.items():
                    row['prefusion_{}_{}'.format(modality, key)] = value
            rgb_template_map = rgb_template.transpose(0, 1).reshape(1, token_dim, 8, 8)
            tir_template_map = tir_template.transpose(0, 1).reshape(1, token_dim, 8, 8)
            if args.search_fusion_arm == 'rgb':
                template_map = torch.cat((rgb_template_map, torch.zeros_like(tir_template_map)), dim=1)
            elif args.search_fusion_arm == 'tir':
                template_map = torch.cat((torch.zeros_like(rgb_template_map), tir_template_map), dim=1)
            else:
                template_map = torch.cat((rgb_template_map, tir_template_map), dim=1)
            search_map = fusion_tensors['output']
            fusion_device = next(tracker.network.tbsi_fuse_search.parameters()).device
            with torch.no_grad():
                fused_template = tracker.network.tbsi_fuse_search(
                    template_map.to(device=fusion_device)).detach().float().cpu()
            for key, value in foreground_background_query_stats(
                    fused_template, search_map, template_mask, gt_mask,
                    fixed_bg_y, fixed_bg_x).items():
                row['postfusion_{}'.format(key)] = value
            row['event_id'] = '{}:{}:{}'.format(name, event['phase'], event['frame'])
            row['event_pair_frame'] = event['phase']
            per_event.append(row)
            for stage in stage_rows:
                stage['event_id'] = row['event_id']
                stages.append(stage)
            print('[event] {}/{} {} {} frame={} trace_iou={:.4f} replay_iou={:.4f} rank={} margin={:+.5f}'.format(
                len(per_event), sum(len(v) for v in events.values()), name, event['phase'],
                event['frame'], event['trace_iou'], row['fixed_replay_iou'],
                row['center_gt_rank'], row['center_gt_hard_bg_margin']), flush=True)
        if args.max_states and len(per_event) >= args.max_states:
            break

    for filename, rows in (('per_event.csv', per_event), ('stage_metrics.csv', stages)):
        columns = sorted({key for row in rows for key in row})
        with (output_dir / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    if args.head_map_archive:
        archive_path = Path(args.head_map_archive).resolve()
        if archive_path.exists():
            raise RuntimeError('refusing to overwrite head-map archive: {}'.format(archive_path))
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(archive_path),
            event_id=np.asarray([item['event_id'] for item in head_maps]),
            score_map=np.stack([item['score_map'] for item in head_maps]),
            size_map=np.stack([item['size_map'] for item in head_maps]),
            offset_map=np.stack([item['offset_map'] for item in head_maps]),
            hann=head_maps[0]['hann'],
            gt_box=np.stack([item['gt_box'] for item in head_maps]),
            crop_xywh=np.stack([item['crop_xywh'] for item in head_maps]),
            state_before=np.stack([item['state_before'] for item in head_maps]),
            resize_factor=np.asarray([item['resize_factor'] for item in head_maps], dtype=np.float32),
            image_shape=np.stack([item['image_shape'] for item in head_maps]),
            gt_mask=np.stack([item['gt_mask'] for item in head_maps]),
        )
    if args.bypass_block:
        intervention_name = 'BLOCK{}_BYPASS'.format(args.bypass_block)
    elif args.bypass_tbsi_loc:
        intervention_name = 'TBSI_LOC{}_BYPASS'.format(args.bypass_tbsi_loc)
    elif args.search_fusion_arm:
        intervention_name = 'SEARCH_FUSION_{}_ONLY'.format(args.search_fusion_arm.upper())
    elif args.sparse_mask_control == 'shifted':
        intervention_name = 'SPARSE_NECK_SHIFTED_CONTROL'
    elif args.sparse_mask_control == 'none':
        intervention_name = 'SPARSE_NECK_UNION'
    else:
        intervention_name = 'PARENT'
    manifest = {
        'experiment': 'M1-SCREENING-FIXED-STATE-FLOW-{}_20261022'.format(intervention_name),
        'parent': base.CONFIG_NAME,
        'config_sha256': sha256(base.CONFIG), 'checkpoint_sha256': sha256(base.CHECKPOINT),
        'runner_sha256': sha256(Path(__file__)), 'flow_capture_sha256': sha256(FLOW_MODULE),
        'events_sha256': sha256(event_path), 'sequence_list_sha256': sha256(sequence_list),
        'split': 'LasHeR official trainingset calibration cohort',
        'event_count': len(events), 'fixed_state_count': len(per_event),
        'stages': 'block outputs, TBSI loc3/6/9, backbone-normalized pre-fusion query, tbsi_fuse_search post-fusion query, CENTER*Hann readout',
        'query_metric': 'L2-normalize every template/search token; weighted foreground/background template prototypes; normalized cosine foreground minus background score map; fixed CENTER-selected background margin plus rank against all non-GT cells and query-own peak margin',
        'hard_background': 'highest M1 CENTER*Hann cell outside GT, fixed across stages for each state',
        'arms': ['M1 only; sparse fusion disabled'] if args.sparse_mask_control == 'off'
            and not any((args.bypass_block, args.bypass_tbsi_loc, args.search_fusion_arm)) else [
            'reversible block{} bypass: replace output by input'.format(args.bypass_block)
            if args.bypass_block else
            'reversible TBSI loc{} bypass: return RGB/TIR inputs'.format(args.bypass_tbsi_loc)
            if args.bypass_tbsi_loc else
            'search fusion single-modality replacement: {}'.format(args.search_fusion_arm)
            if args.search_fusion_arm else
            'equal-density circularly shifted sparse candidate mask control'
            if args.sparse_mask_control == 'shifted' else
            'sparse candidate-conditioned neck fusion: RGB/TIR top-8 union with 3x3 dilation'],
        'bypass_block_index': int(args.bypass_block),
        'bypass_tbsi_loc': int(args.bypass_tbsi_loc),
        'search_fusion_arm': args.search_fusion_arm or 'off',
        'sparse_cross_mode': os.environ['TBSI_SPARSE_CROSS_MODE'],
        'spatial_token_bridge_mode': args.spatial_token_bridge_mode,
        'spatial_token_bridge_output_scale': float(args.spatial_token_bridge_scale),
        'sparse_mask_control': args.sparse_mask_control,
        'sparse_topk_per_modality': int(os.environ.get('TBSI_SPARSE_TOPK', '8')),
        'explicit_model_override': bool(args.config),
        'precision': 'FP32', 'torch_threads': 1,
        'interpretation': 'fixed-state reversible location intervention; diagnostic only, not a tracking-gain claim',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'outputs': ['per_event.csv', 'stage_metrics.csv'],
    }
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'events': len(events), 'fixed_states': len(per_event),
                      'stage_rows': len(stages), 'output_dir': str(output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
