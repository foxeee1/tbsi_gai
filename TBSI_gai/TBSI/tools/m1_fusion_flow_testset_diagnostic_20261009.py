#!/usr/bin/env python3
"""Measure M1 fusion activations and localization-loss gradients on frozen train traces."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ['TBSI_INFER_FP16'] = '0'
os.environ.pop('TBSI_TEMPLATE_CROSS_SCALE', None)
os.environ.pop('TBSI_TEMPLATE_ROUTE_R2T', None)
os.environ.pop('TBSI_TEMPLATE_ROUTE_T2R', None)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss, clip_box
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap


TRACE_DEFAULT = ROOT / 'output/experiments/m1_temporal_trace_v1_20261006'
CHECKPOINT_DEFAULT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_DEFAULT = ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pair(sequence, frame_index):
    images = []
    for modality in range(2):
        bgr = cv2.imread(sequence.frames[modality][frame_index], cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError('failed to read {} frame {}'.format(sequence.name, frame_index))
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.concatenate(images, axis=2)


def iou_xywh(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def scalar(value):
    return value.detach().float().reshape(()).clamp(-1e30, 1e30)


def tensor_stats(prefix, value, record):
    value = value.float()
    record[prefix + '_norm'] = scalar(value.norm())
    record[prefix + '_rms'] = scalar(value.square().mean().sqrt())
    record[prefix + '_mean_abs'] = scalar(value.abs().mean())


def feature_pair_stats(prefix, rgb, tir, record):
    rgb32, tir32 = rgb.float(), tir.float()
    record[prefix + '_rgb_norm'] = scalar(rgb32.norm())
    record[prefix + '_tir_norm'] = scalar(tir32.norm())
    record[prefix + '_cosine'] = scalar(F.cosine_similarity(
        rgb32.reshape(1, -1), tir32.reshape(1, -1), dim=1, eps=1e-8)[0])


def hook_capture(store, name):
    def capture(module, inputs, output):
        def tensors(value):
            if torch.is_tensor(value):
                return [value]
            if isinstance(value, (tuple, list)):
                result = []
                for item in value:
                    result.extend(tensors(item))
                return result
            return []
        store[name] = {'inputs': tensors(inputs), 'outputs': tensors(output)}
    return capture


def register_target(name, tensor, targets, labels):
    if tensor is None or not torch.is_tensor(tensor) or not tensor.requires_grad:
        return
    targets.append(tensor)
    labels.append(name)


def rank_values(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def corr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def correlations(rows):
    outcomes = {'tracking_iou': 'trace_tracking_iou', 'tracking_loss': 'loss',
                'center_error_tokens': 'center_error_tokens',
                'crop_gt_area_fraction': 'crop_gt_area_fraction'}
    candidates = [key for key in rows[0] if key.startswith((
        'mhsa_', 'search_', 'backbone_', 'response_')) or key.endswith(('_grad_norm', '_grad_rms'))]
    sequence_indices = {}
    for index, row in enumerate(rows):
        sequence_indices.setdefault(row['sequence'], []).append(index)
    sequence_indices = [np.asarray(indices, dtype=np.int64)
                        for _, indices in sorted(sequence_indices.items())]
    arrays = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
              for key in set(candidates) | set(outcomes.values())}
    result = []
    for metric in candidates:
        for outcome, outcome_column in outcomes.items():
            x, y = arrays[metric], arrays[outcome_column]
            result.append({'metric': metric, 'outcome': outcome, 'frames': len(rows),
                           'pearson': corr(x, y),
                           'spearman': corr(rank_values(x), rank_values(y))})
            centered_x, centered_y = x.copy(), y.copy()
            for indices in sequence_indices:
                centered_x[indices] -= centered_x[indices].mean()
                centered_y[indices] -= centered_y[indices].mean()
            result[-1]['within_sequence_pearson'] = corr(centered_x, centered_y)
            result[-1]['within_sequence_spearman'] = corr(
                rank_values(centered_x), rank_values(centered_y))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', default=str(TRACE_DEFAULT))
    parser.add_argument('--checkpoint', default=str(CHECKPOINT_DEFAULT))
    parser.add_argument('--config', default=str(CONFIG_DEFAULT))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit-frames', type=int, default=0)
    parser.add_argument('--frames-per-sequence', type=int, default=0)
    parser.add_argument('--split', choices=['trainingset', 'testingset'], default='trainingset')
    parser.add_argument('--parity-tolerance', type=float, default=1e-2)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('refusing non-empty output directory: {}'.format(output_dir))
    if not trace_dir.is_dir() or not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError('trace, checkpoint, or config does not exist')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the bounded M1 diagnostic')

    seq_names = [line.strip() for line in (trace_dir / 'trace_sequences.txt').read_text().splitlines()
                 if line.strip()]
    trace_manifest = trace_dir / 'manifest.json'
    trace_manifest_data = json.loads(trace_manifest.read_text())
    experiment_id = trace_manifest_data.get(
        'experiment_id', 'M1-FUSION-FLOW-TESTSET-DIAG-20261009')
    dataset = LasHeRDataset(split=args.split,
                            sequence_list_file=str(trace_dir / 'trace_sequences.txt')).get_sequence_list()
    sequence_map = {sequence.name: sequence for sequence in dataset}
    missing = sorted(set(seq_names) - set(sequence_map))
    if missing:
        raise RuntimeError('trace sequences absent from LasHeR {}: {}'.format(args.split, missing))

    params = parameters(config_path.stem)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_test' if args.split == 'testingset' else 'lasher_train')
    network = tracker.network
    network.eval()
    network.zero_grad(set_to_none=True)
    for parameter in network.parameters():
        parameter.requires_grad_(True)

    captures = {}
    handles = []
    fusion_modules = []
    locations = list(getattr(tracker.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9]))
    for location, layer in zip(locations, network.backbone.tbsi_layers):
        module = getattr(layer, 'v2_template_fusion', None)
        if module is None:
            continue
        fusion_modules.append((int(location), module))
        handles.append(module.register_forward_hook(hook_capture(captures, 'mhsa_{}'.format(location))))
        module.analysis_enabled = True
        def make_attention_collector(layer_location):
            def collect(**kwargs):
                attention = kwargs['attn_prob'].float()
                masks = kwargs.get('fg_masks')
                stats = {
                    'cross_mass_rgb_to_tir': attention[:, :, :64, 64:128].mean(),
                    'cross_mass_tir_to_rgb': attention[:, :, 64:128, :64].mean(),
                    'rgb_cross_message_norm': kwargs['rgb_cross_norm'],
                    'rgb_self_message_norm': kwargs['rgb_self_norm'],
                    'tir_cross_message_norm': kwargs['tir_cross_norm'],
                    'tir_self_message_norm': kwargs['tir_self_norm'],
                    'message_to_template_ratio': kwargs['output_change_norm'],
                    'entropy': -(attention.clamp_min(1e-8) *
                                 attention.clamp_min(1e-8).log()).sum(-1).mean(),
                }
                if masks is not None and len(masks) == 2:
                    rgb_mask = masks[0].float().reshape(-1) > 0.5
                    tir_mask = masks[1].float().reshape(-1) > 0.5
                    rt = attention[:, :, :64, 64:128].mean(1)[0]
                    tr = attention[:, :, 64:128, :64].mean(1)[0]
                    if rgb_mask.any() and tir_mask.any():
                        stats['rgb_target_query_to_tir_target_key'] = rt[rgb_mask][:, tir_mask].sum(-1).mean()
                        stats['rgb_target_query_to_tir_background_key'] = rt[rgb_mask][:, ~tir_mask].sum(-1).mean()
                        stats['tir_target_query_to_rgb_target_key'] = tr[tir_mask][:, rgb_mask].sum(-1).mean()
                        stats['tir_target_query_to_rgb_background_key'] = tr[tir_mask][:, ~rgb_mask].sum(-1).mean()
                captures['mhsa_attention_{}'.format(layer_location)] = stats
            return collect
        module.analysis_collector = make_attention_collector(int(location))
    if len(fusion_modules) != 3:
        raise RuntimeError('expected M1 joint template MHSA at three configured locations')
    handles.append(network.tbsi_fuse_search.register_forward_hook(
        hook_capture(captures, 'search_fusion')))
    handles.append(network.backbone.register_forward_hook(
        hook_capture(captures, 'backbone')))

    focal = FocalLoss()
    rows = []
    excluded = []
    parity_diffs = []
    parity_iou_diffs = []
    seen_frames = 0
    sampled_trace_frames = {}
    for seq_name in seq_names:
        sequence = sequence_map[seq_name]
        trace_file = trace_dir / seq_name / 'frames.jsonl'
        trace_rows = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
        if not trace_rows:
            raise RuntimeError('empty frozen trace for {}'.format(seq_name))
        if args.frames_per_sequence:
            valid_rows = []
            for trace_row in trace_rows:
                box = np.asarray(trace_row.get('gt_box'), dtype=np.float64)
                if (box.shape == (4,) and np.isfinite(box).all() and
                        float(box[2]) > 0.0 and float(box[3]) > 0.0):
                    valid_rows.append(trace_row)
            count = min(args.frames_per_sequence, len(valid_rows))
            if count:
                positions = np.linspace(0, len(valid_rows) - 1, count, dtype=np.int64)
                trace_rows = [valid_rows[int(position)] for position in positions]
            else:
                trace_rows = []
            sampled_trace_frames[seq_name] = len(trace_rows)
        first_image = read_pair(sequence, 0)
        tracker.initialize(first_image, {'init_bbox': sequence.ground_truth_rect[0],
                                         'sequence_name': seq_name})
        for trace_row in trace_rows:
            if args.limit_frames and seen_frames >= args.limit_frames:
                break
            frame_index = int(trace_row['frame_id'])
            if frame_index >= len(sequence.frames[0]) or frame_index >= len(sequence.ground_truth_rect):
                raise RuntimeError('trace frame out of {} sequence bounds: {} {}'.format(
                    args.split, seq_name, frame_index))
            state = [float(v) for v in trace_row['tracker_state_before']]
            gt = np.asarray(sequence.ground_truth_rect[frame_index], dtype=np.float32)
            if (gt.shape != (4,) or not np.isfinite(gt).all() or
                    float(gt[2]) <= 0.0 or float(gt[3]) <= 0.0):
                excluded.append({'sequence': seq_name, 'frame': frame_index,
                                 'gt_box': gt.tolist(),
                                 'reason': 'non-finite or non-positive GT width/height'})
                continue
            image = read_pair(sequence, frame_index)
            search_crop, resize_factor, search_mask = sample_target(
                image, state, params.search_factor, output_sz=params.search_size)
            search_nested = tracker.preprocessor.process(search_crop, search_mask, half=False)
            captures.clear()
            network.zero_grad(set_to_none=True)
            with torch.enable_grad():
                output = network(
                    template=[tracker.z_dict1.tensors[:, :3], tracker.z_dict1.tensors[:, 3:]],
                    search=[search_nested.tensors[:, :3], search_nested.tensors[:, 3:]],
                    ce_template_mask=tracker.box_mask_z,
                    template_fg_masks=(tracker.template_fg_mask, tracker.template_fg_mask))
                response = tracker.output_window * output['score_map']
                pred_grid = network.box_head.cal_bbox(
                    response, output['size_map'], output['offset_map']).view(-1, 4)
                pred_cxcywh = pred_grid.mean(dim=0)
                pred_box_crop = pred_cxcywh * (params.search_size / resize_factor)
                state_cx = state[0] + 0.5 * state[2]
                state_cy = state[1] + 0.5 * state[3]
                half_side = 0.5 * params.search_size / resize_factor
                box_image = torch.stack((
                    pred_box_crop[0] + state_cx - half_side - 0.5 * pred_box_crop[2],
                    pred_box_crop[1] + state_cy - half_side - 0.5 * pred_box_crop[3],
                    pred_box_crop[2], pred_box_crop[3]))
                box_image = torch.tensor(clip_box(box_image.detach().cpu().tolist(),
                                                  image.shape[0], image.shape[1], margin=10))
                gt_crop = transform_image_to_crop(
                    torch.tensor(gt), torch.tensor(state), resize_factor,
                    torch.tensor([params.search_size, params.search_size]), normalize=True
                ).to(output['score_map'].device).view(1, 4)
                pred_xyxy = box_cxcywh_to_xyxy(pred_cxcywh.view(1, 4))
                gt_xyxy = box_xywh_to_xyxy(gt_crop)
                loss_giou, _ = giou_loss(pred_xyxy, gt_xyxy)
                loss_l1 = F.l1_loss(pred_xyxy, gt_xyxy)
                gt_heatmap = generate_heatmap(gt_crop.unsqueeze(0),
                                              patch_size=params.search_size,
                                              stride=network.backbone.patch_size)[-1].unsqueeze(1)
                loss_focal = focal(output['score_map'], gt_heatmap)
                tracking_loss = (float(tracker.cfg.TRAIN.GIOU_WEIGHT) * loss_giou +
                                 float(tracker.cfg.TRAIN.L1_WEIGHT) * loss_l1 + loss_focal)

                record = {
                    'sequence': seq_name,
                    'frame': frame_index,
                    'state_before': state,
                    'gt_valid': bool(trace_row.get('gt_valid', True)),
                    'loss': tracking_loss,
                    'loss_giou': loss_giou,
                    'loss_l1': loss_l1,
                    'loss_focal': loss_focal,
                    'pred_box': box_image,
                    'gt_box': gt.tolist(),
                }
                pred_iou = iou_xywh(box_image.detach().cpu().tolist(), gt)
                record['tracking_iou'] = torch.tensor(pred_iou, device='cuda')
                recorded_pred = np.asarray(trace_row['pred_box'], dtype=np.float64)
                parity = float(np.max(np.abs(box_image.detach().cpu().numpy() - recorded_pred)))
                trace_iou = iou_xywh(recorded_pred.tolist(), gt)
                parity_iou = abs(pred_iou - trace_iou)
                parity_diffs.append(parity)
                parity_iou_diffs.append(parity_iou)
                record['parity_max_abs'] = torch.tensor(parity, device='cuda')
                record['trace_tracking_iou'] = torch.tensor(trace_iou, device='cuda')
                record['parity_iou_abs_diff'] = torch.tensor(parity_iou, device='cuda')

                gt_area = max(float(gt[2] * gt[3]), 1e-8)
                crop_xywh = trace_row['current_crop_xywh']
                ix = max(0.0, min(float(crop_xywh[0] + crop_xywh[2]), float(gt[0] + gt[2])) -
                         max(float(crop_xywh[0]), float(gt[0])))
                iy = max(0.0, min(float(crop_xywh[1] + crop_xywh[3]), float(gt[1] + gt[3])) -
                         max(float(crop_xywh[1]), float(gt[1])))
                record['crop_gt_area_fraction'] = torch.tensor(ix * iy / gt_area, device='cuda')

                score = output['score_map'].detach().float()[0, 0]
                peak = response.detach().float()[0, 0]
                peak_index = int(peak.argmax().item())
                peak_y, peak_x = divmod(peak_index, peak.shape[-1])
                gt_center = gt_crop[0, :2] + 0.5 * gt_crop[0, 2:]
                gt_token = gt_center * (params.search_size / network.backbone.patch_size)
                record['center_error_tokens'] = torch.tensor(
                    ((peak_x + 0.5 - gt_token[0]).square() +
                     (peak_y + 0.5 - gt_token[1]).square()).sqrt(), device='cuda')
                record['response_peak'] = peak.max()
                gx = int(torch.clamp(gt_token[0], 0, peak.shape[-1] - 1).item())
                gy = int(torch.clamp(gt_token[1], 0, peak.shape[-2] - 1).item())
                record['response_gt_cell'] = score[gy, gx]
                target_cell = torch.zeros_like(score, dtype=torch.bool)
                target_cell[gy, gx] = True
                record['response_target_bg_gap'] = score[gy, gx] - score[~target_cell].max()

                targets, labels = [], []
                for location, _module in fusion_modules:
                    cap = captures['mhsa_{}'.format(location)]
                    for modality_index, modality in enumerate(('rgb', 'tir')):
                        register_target('mhsa_{}_input_{}'.format(location, modality),
                                        cap['inputs'][modality_index], targets, labels)
                        register_target('mhsa_{}_output_{}'.format(location, modality),
                                        cap['outputs'][modality_index], targets, labels)
                    register_target('mhsa_{}_qkv_weight'.format(location),
                                    _module.qkv.weight, targets, labels)
                    register_target('mhsa_{}_proj_weight'.format(location),
                                    _module.proj.weight, targets, labels)
                search_cap = captures['search_fusion']
                register_target('search_input', search_cap['inputs'][0], targets, labels)
                register_target('search_output', search_cap['outputs'][0], targets, labels)
                for param_index, param in enumerate(network.tbsi_fuse_search.parameters()):
                    register_target('search_param_{}'.format(param_index), param, targets, labels)
                backbone_out = captures['backbone']['outputs'][0]
                register_target('backbone_output', backbone_out, targets, labels)
                grads = torch.autograd.grad(tracking_loss, targets, allow_unused=True)
                grad_map = dict(zip(labels, grads))

                for location, _module in fusion_modules:
                    cap = captures['mhsa_{}'.format(location)]
                    for modality_index, modality in enumerate(('rgb', 'tir')):
                        x_in = cap['inputs'][modality_index]
                        x_out = cap['outputs'][modality_index]
                        tensor_stats('mhsa_{}_input_{}'.format(location, modality), x_in, record)
                        tensor_stats('mhsa_{}_output_{}'.format(location, modality), x_out, record)
                        delta = x_out.float() - x_in.float()
                        record['mhsa_{}_delta_norm_{}'.format(location, modality)] = scalar(delta.norm())
                        record['mhsa_{}_delta_ratio_{}'.format(location, modality)] = scalar(
                            delta.norm() / x_in.float().norm().clamp_min(1e-8))
                        record['mhsa_{}_cosine_{}'.format(location, modality)] = scalar(
                            F.cosine_similarity(x_in.float().reshape(1, -1),
                                                x_out.float().reshape(1, -1), dim=1, eps=1e-8)[0])
                        for boundary in ('input', 'output'):
                            name = 'mhsa_{}_{}_{}'.format(location, boundary, modality)
                            grad = grad_map.get(name)
                            if grad is not None:
                                tensor_stats(name + '_grad', grad, record)
                                record[name + '_grad_act_abs'] = scalar((grad.float() *
                                                                          (x_in if boundary == 'input' else x_out).float()).abs().sum())
                    for param_name in ('qkv_weight', 'proj_weight'):
                        grad = grad_map.get('mhsa_{}_{}'.format(location, param_name))
                        if grad is not None:
                            tensor_stats('mhsa_{}_{}_grad'.format(location, param_name), grad, record)

                    attention_stats = captures.get('mhsa_attention_{}'.format(location), {})
                    for key, value in attention_stats.items():
                        record['mhsa_{}_{}'.format(location, key)] = scalar(value)
                    stats = getattr(_module, 'last_route_stats', {})
                    for key, value in stats.items():
                        if key.endswith('_mass'):
                            record['mhsa_{}_{}'.format(location, key)] = torch.tensor(value, device='cuda')

                search_in = search_cap['inputs'][0]
                search_out = search_cap['outputs'][0]
                channels = search_in.shape[1] // 2
                rgb_in, tir_in = search_in[:, :channels], search_in[:, channels:]
                feature_pair_stats('search_input', rgb_in, tir_in, record)
                tensor_stats('search_output', search_out, record)
                for modality, value, start, end in (
                        ('rgb', rgb_in, 0, channels), ('tir', tir_in, channels, 2 * channels)):
                    grad = grad_map.get('search_input')
                    if grad is not None:
                        grad_part = grad[:, start:end]
                        tensor_stats('search_input_grad_' + modality, grad_part, record)
                        record['search_input_grad_act_abs_' + modality] = scalar(
                            (grad_part.float() * value.float()).abs().sum())
                search_grad = grad_map.get('search_output')
                if search_grad is not None:
                    tensor_stats('search_output_grad', search_grad, record)
                    record['search_output_grad_act_abs'] = scalar(
                        (search_grad.float() * search_out.float()).abs().sum())
                for param_index in range(len(list(network.tbsi_fuse_search.parameters()))):
                    grad = grad_map.get('search_param_{}'.format(param_index))
                    if grad is not None:
                        tensor_stats('search_param_{}_grad'.format(param_index), grad, record)

                bb_grad = grad_map.get('backbone_output')
                if bb_grad is not None:
                    tensor_stats('backbone_output_grad', bb_grad, record)
                bb = backbone_out.float()
                if bb.ndim == 3 and bb.shape[1] == 640:
                    feature_pair_stats('backbone_search', bb[:, 64:320], bb[:, 384:640], record)

            record['parity_with_frozen_trace'] = parity
            numeric = {key: value for key, value in record.items()
                       if torch.is_tensor(value) and value.numel() == 1}
            values = torch.stack([scalar(value) for value in numeric.values()]).detach().cpu().tolist()
            row = {key: value for key, value in record.items()
                   if not torch.is_tensor(value)}
            row.update(dict(zip(numeric.keys(), values)))
            rows.append(row)
            seen_frames += 1
            if seen_frames % 100 == 0:
                print('processed {} trace frames'.format(seen_frames), flush=True)
        if args.limit_frames and seen_frames >= args.limit_frames:
            break

    for handle in handles:
        handle.remove()
    if not rows:
        raise RuntimeError('no diagnostic frames were processed')
    max_parity = max(parity_diffs)
    fields = list(rows[0].keys())
    with (output_dir / 'per_frame.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    correlations_result = correlations(rows)
    with (output_dir / 'metric_correlations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(correlations_result[0]))
        writer.writeheader()
        writer.writerows(correlations_result)
    with (output_dir / 'excluded_frames.csv').open('w', newline='') as stream:
        fields = ['sequence', 'frame', 'gt_box', 'reason']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(excluded)
    summary = {
        'experiment_id': experiment_id,
        'parent': 'TBSI_E_MHSA_FULL_4090D_seed42',
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'config': str(config_path), 'config_sha256': sha256(config_path),
        'trace_dir': str(trace_dir), 'trace_manifest_sha256': sha256(trace_manifest),
        'sequences': seq_names, 'frame_count': len(rows),
        'excluded_frame_count': len(excluded),
        'excluded_frames': excluded,
        'sequence_list_sha256': hashlib.sha256(('\n'.join(seq_names) + '\n').encode()).hexdigest(),
        'script_sha256': sha256(Path(__file__)),
        'command': ' '.join(sys.argv),
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'inference_precision': 'FP32', 'model_mode': 'eval; gradients enabled; no parameter updates',
        'gradient_objective': '2*GIoU + 5*L1 on Hann-decoded localization plus CenterNet focal score loss',
        'diagnostic_hooks': 'V2TemplateFusion at configured TBSI locations 3/6/9; tbsi_fuse_search; backbone output',
        'max_abs_prediction_difference_vs_frozen_m1_trace': max_parity,
        'parity_tolerance': args.parity_tolerance,
        'parity_pass': max_parity <= args.parity_tolerance,
        'max_iou_difference_vs_frozen_m1_trace': max(parity_iou_diffs),
        'mean_tracking_iou_on_replayed_trace': float(np.mean([row['tracking_iou'] for row in rows])),
        'mean_loss': float(np.mean([row['loss'] for row in rows])),
        'outputs': ['per_frame.csv', 'metric_correlations.csv', 'excluded_frames.csv'],
        'sampled_trace_frames_per_sequence': sampled_trace_frames,
        'diagnostic_split': args.split,
        'limitations': [
            'LasHeR test-set sequence subset selected using official attributes only; exploratory mechanism diagnostic.',
            'Gradient values are frozen-checkpoint localization-loss sensitivities, not optimizer updates.',
            'Correlation is associational and cannot establish module causality or choose an insertion position.',
            'Closed-loop tracker states were generated using the fixed M1 checkpoint; gradient rows are evenly sampled from each selected sequence.',
            'No training, checkpoint writes, or model parameter updates.'
        ],
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary['parity_pass']:
        raise RuntimeError('re-forward predictions diverge from frozen M1 trace')


if __name__ == '__main__':
    main()
