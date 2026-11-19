#!/usr/bin/env python3
"""Fixed-state, case-study capture of M1 target/context information flow."""
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / 'tools/run_m1_template_query_stage_diagnostic_20261020.py'
SPEC = importlib.util.spec_from_file_location('m1_query_stage_base', str(BASE_PATH))
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
OUT = ROOT / 'output/experiments/M1_LOW_AUC_INFORMATION_FLOW_CASES_20261020'
TRACE_ROOT = ROOT / 'output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/closed_loop_trace/trace'
SEQUENCES = [
    'ab_bikeoccluded',
    '3rdfatboy',
    'leftfarboycomingpicktheball',
    'rightbottlecomes',
]
CODE_FILES = [
    'tools/run_m1_template_query_stage_diagnostic_20261020.py',
    'tools/diagnose_m1_lowauc_information_flow_20261020.py',
    'lib/test/tracker/tbsi_track.py',
    'lib/test/parameter/tbsi_track.py',
    'lib/test/evaluation/lasherdataset.py',
    'lib/models/tbsi_track/tbsi_track.py',
    'lib/models/tbsi_track/vit_tbsi_care.py',
    'lib/models/tbsi_track/base_backbone.py',
    'lib/models/layers/tbsi_layer.py',
    'lib/models/layers/head.py',
    'lib/models/layers/attn.py',
    'lib/models/layers/attn_blocks.py',
]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def iou(a, b):
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if a[2] * a[3] + b[2] * b[3] > inter else 0.


def crop_contains(gt, crop):
    return (gt[0] >= crop[0] and gt[1] >= crop[1] and
            gt[0] + gt[2] <= crop[0] + crop[2] and
            gt[1] + gt[3] <= crop[1] + crop[3])


def discover_event(sequence):
    path = TRACE_ROOT / sequence / 'frames.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scored = []
    for row in rows:
        if row['frame_id'] <= 1 or not row['gt_valid']:
            continue
        scored.append((row, iou(row['pred_box'], row['gt_box'])))
    for idx, (row, value) in enumerate(scored):
        window = scored[idx:idx + 30]
        if (value < 0.5 and len(window) == 30 and
                all(item[1] < 0.5 for item in window) and
                crop_contains(row['gt_box'], row['current_crop_xywh'])):
            prior = scored[idx - 1][0]
            return {
                'sequence': sequence,
                'trace_sha256': sha256(path),
                'onset_frame': int(row['frame_id']),
                'onset_trace_iou': float(value),
                'pre_frame': int(prior['frame_id']),
                'pre_trace_iou': float(scored[idx - 1][1]),
                'events': [
                    {'phase': 'pre_onset', 'frame': int(prior['frame_id']),
                     'trace_iou': float(scored[idx - 1][1]),
                     'state_before': prior['tracker_state_before'],
                     'gt_box': prior['gt_box'], 'crop_xywh': prior['current_crop_xywh']},
                    {'phase': 'severe_onset', 'frame': int(row['frame_id']),
                     'trace_iou': float(value), 'state_before': row['tracker_state_before'],
                     'gt_box': row['gt_box'], 'crop_xywh': row['current_crop_xywh']},
                ],
            }
    raise RuntimeError('no eligible crop-contained 30-frame failure onset: {}'.format(sequence))


def query_map(rgb_tokens, tir_tokens, fg_mask):
    maps = {}
    for name, tokens in (('rgb', rgb_tokens), ('tir', tir_tokens)):
        template = F.normalize(tokens[:, :64].float(), dim=-1, eps=1e-6)
        search = F.normalize(tokens[:, 64:].float(), dim=-1, eps=1e-6)
        fg = fg_mask.to(device=template.device, dtype=template.dtype).view(1, 64)
        bg = 1.0 - fg
        fg_proto = F.normalize((template * fg.unsqueeze(-1)).sum(1) /
                               fg.sum(1, keepdim=True).clamp_min(1e-6), dim=-1, eps=1e-6)
        bg_proto = F.normalize((template * bg.unsqueeze(-1)).sum(1) /
                               bg.sum(1, keepdim=True).clamp_min(1e-6), dim=-1, eps=1e-6)
        fg_match = (search * fg_proto.unsqueeze(1)).sum(-1)
        bg_match = (search * bg_proto.unsqueeze(1)).sum(-1)
        maps[name] = (fg_match - bg_match).reshape(16, 16).detach().float().cpu().numpy()
    return maps


def feature_geometry(feature, mask, bg_yx):
    z = feature.detach().float()[0].permute(1, 2, 0).cpu().numpy()
    target = z[mask].mean(0)
    background = z[bg_yx[0], bg_yx[1]]
    return {
        'target_bg_l2': float(np.linalg.norm(target - background)),
        'target_bg_cosine': float(np.dot(target, background) /
                                  (np.linalg.norm(target) * np.linalg.norm(background) + 1e-12)),
        'target_norm': float(np.linalg.norm(target)),
        'background_norm': float(np.linalg.norm(background)),
    }


def crop_from_state(state, search_factor):
    cx, cy = state[0] + state[2] / 2., state[1] + state[3] / 2.
    side = int(math.ceil(math.sqrt(state[2] * state[3]) * search_factor))
    return [int(round(cx - side / 2.)), int(round(cy - side / 2.)), side, side]


def attach_stage_hooks(tracker, capture):
    net = tracker.network
    backbone = net.backbone
    handles = []
    block_calls = [0 for _ in backbone.blocks]
    for index, block in enumerate(backbone.blocks):
        def block_hook(module, inputs, output, i=index):
            stream = 'rgb' if block_calls[i] % 2 == 0 else 'tir'
            block_calls[i] += 1
            entry = capture['feature_maps'].setdefault('block{:02d}'.format(i), {})
            entry[stream] = output.detach()
            if 'rgb' in entry and 'tir' in entry:
                capture['feature_maps']['block{:02d}'.format(i)] = query_map(
                    entry.pop('rgb'), entry.pop('tir'), tracker.c1_template_mask)
        handles.append(block.register_forward_hook(block_hook))

    for index, layer in enumerate(getattr(backbone, 'tbsi_layers', [])):
        def layer_hook(module, inputs, output, i=index):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                return
            capture['feature_maps']['tbsi_loc{}'.format((3, 6, 9)[i] if i < 3 else i)] = \
                query_map(output[0], output[1], tracker.c1_template_mask)
        handles.append(layer.register_forward_hook(layer_hook))

    def norm_hook(module, inputs, output):
        if output.ndim == 3 and output.shape[1] == 640:
            capture['feature_maps']['backbone_norm'] = query_map(
                output[:, :320], output[:, 320:], tracker.c1_template_mask)
    handles.append(backbone.norm.register_forward_hook(norm_hook))

    def fusion_pre(module, inputs):
        capture['fusion_input'] = inputs[0].detach().float().cpu()

    def fusion_post(module, inputs, output):
        capture['fusion_output'] = output.detach().float().cpu()

    handles.extend([net.tbsi_fuse_search.register_forward_pre_hook(fusion_pre),
                    net.tbsi_fuse_search.register_forward_hook(fusion_post)])
    return handles


def run_event(tracker, event, sequence):
    capture = {'feature_maps': {}}
    handles = attach_stage_hooks(tracker, capture)
    frame = int(event['frame'])
    tracker.state = list(map(float, event['state_before']))
    tracker.prev_search = None
    tracker.prev_temporal_tokens = None
    tracker.prev_head_feature = None
    image = base.read_pair(sequence, frame)
    pred = tracker.track(image, {
        'gt_bbox': event['gt_box'], 'sequence_name': sequence.name, 'frame_num': frame})
    diag = tracker._last_fixed_state_diag
    gt_mask = base.grid_mask(event['gt_box'], event['crop_xywh'])
    hann = tracker.output_window[0, 0].detach().float().cpu().numpy()
    raw_score = diag['score_map'][0, 0].numpy()
    response = raw_score * hann
    background_score = np.where(~gt_mask, response, -np.inf)
    bg_yx = np.unravel_index(int(np.argmax(background_score)), response.shape)
    gt_peak = float(response[gt_mask].max())
    bg_peak = float(response[bg_yx])
    top = np.unravel_index(int(np.argmax(response)), response.shape)
    crop_x, crop_y, crop_w, crop_h = event['crop_xywh']
    gt_x, gt_y, gt_w, gt_h = event['gt_box']
    gt_center_x = (gt_x + 0.5 * gt_w - crop_x) * response.shape[1] / crop_w - 0.5
    gt_center_y = (gt_y + 0.5 * gt_h - crop_y) * response.shape[0] / crop_h - 0.5
    gt_cell_x = int(np.clip(round(gt_center_x), 0, response.shape[1] - 1))
    gt_cell_y = int(np.clip(round(gt_center_y), 0, response.shape[0] - 1))
    row = {
        'sequence': sequence.name, 'phase': event['phase'], 'frame': frame,
        'trace_iou': event['trace_iou'], 'fixed_replay_iou': iou(pred['target_bbox'], event['gt_box']),
        'trace_state_before': json.dumps(event['state_before']),
        'trace_crop_xywh': json.dumps(event['crop_xywh']),
        'reconstructed_crop_xywh': json.dumps(crop_from_state(
            event['state_before'], tracker.params.search_factor)),
        'gt_box': json.dumps(event['gt_box']),
        'hard_bg_y': int(bg_yx[0]), 'hard_bg_x': int(bg_yx[1]),
        'center_gt_rank': 1 + int(np.sum(response[~gt_mask] > gt_peak)),
        'center_top1_is_gt': bool(gt_mask[top]),
        'center_gt_hard_bg_margin': gt_peak - bg_peak,
        'center_raw_score_gt_hard_bg_margin': float(
            raw_score[gt_mask].max() - raw_score[bg_yx]),
        'center_top1_y': int(top[0]), 'center_top1_x': int(top[1]),
        'gt_center_grid_y': float(gt_center_y), 'gt_center_grid_x': float(gt_center_x),
        'gt_center_cell_y': gt_cell_y, 'gt_center_cell_x': gt_cell_x,
        'center_top1_to_gt_center_cells': float(math.hypot(
            top[0] - gt_center_y, top[1] - gt_center_x)),
        'center_response_gt_center_cell': float(response[gt_cell_y, gt_cell_x]),
        'center_response_top1': float(response[top]),
        'center_score_gt_center_cell': float(raw_score[gt_cell_y, gt_cell_x]),
        'center_score_top1': float(raw_score[top]),
        'predicted_bbox': json.dumps(list(map(float, pred['target_bbox']))),
    }
    stage_rows = []
    for stage, maps in capture['feature_maps'].items():
        for modality, score in maps.items():
            outside = np.where(~gt_mask, score, -np.inf)
            bg_peak_q = float(score[bg_yx])
            target_peak_q = float(score[gt_mask].max())
            rank = 1 + int(np.sum(score[~gt_mask] > target_peak_q))
            stage_rows.append({
                'sequence': sequence.name, 'phase': event['phase'], 'frame': frame,
                'stage': stage, 'modality': modality,
                'query_gt_peak': target_peak_q,
                'query_hard_bg': bg_peak_q,
                'query_gt_hard_bg_margin': target_peak_q - bg_peak_q,
                'query_gt_rank': rank,
                'query_peak_in_gt': bool(gt_mask[np.unravel_index(int(np.argmax(score)), score.shape)]),
                'query_strongest_bg_peak': float(np.max(outside)),
            })
    for key in ('fusion_input', 'fusion_output'):
        tensor = capture.get(key)
        if tensor is None:
            continue
        if key == 'fusion_input':
            channels = tensor.shape[1] // 2
            for modality, part in (('rgb', tensor[:, :channels]), ('tir', tensor[:, channels:])):
                for stat, value in feature_geometry(part, gt_mask, bg_yx).items():
                    row['{}_{}_{}'.format(key, modality, stat)] = value
        else:
            for stat, value in feature_geometry(tensor, gt_mask, bg_yx).items():
                row['{}_{}'.format(key, stat)] = value
    for handle in handles:
        handle.remove()
    return row, stage_rows


def main():
    if sha256(base.CONFIG) != base.CONFIG_SHA256 or sha256(base.CHECKPOINT) != base.CHECKPOINT_SHA256:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    os.environ['TBSI_TEMPLATE_UNCERTAINTY_MODE'] = 'off'
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '0'
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    events = [discover_event(name) for name in SEQUENCES]
    (OUT / 'selected_events.json').write_text(json.dumps(events, indent=2) + '\n')
    list_path = OUT / 'sequence_list.txt'
    list_path.write_text('\n'.join(SEQUENCES) + '\n')
    sequences = {seq.name: seq for seq in base.LasHeRDataset(
        'testingset', str(list_path)).get_sequence_list()}
    rows, stage_rows = [], []

    for seq_index, seq_event in enumerate(events, 1):
        sequence = sequences[seq_event['sequence']]
        tracker = base.new_tracker('off', 'm1_lowauc_information_flow_case')
        tracker.initialize(base.read_pair(sequence, 0), {
            'init_bbox': np.asarray(sequence.ground_truth_rect[0]),
            'sequence_name': sequence.name})
        for event in seq_event['events']:
            row, captured_stages = run_event(tracker, event, sequence)
            rows.append(row)
            stage_rows.extend(captured_stages)
            print('[case] {}/{} {} {} frame={} trace_iou={:.3f} replay_iou={:.3f} rank={} margin={:+.4f}'.format(
                seq_index, len(events), sequence.name, event['phase'], event['frame'],
                event['trace_iou'], row['fixed_replay_iou'], row['center_gt_rank'],
                row['center_gt_hard_bg_margin']), flush=True)

    for filename, records in (('per_event.csv', rows), ('stage_metrics.csv', stage_rows)):
        columns = sorted({key for record in records for key in record})
        with (OUT / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    summary = {
        'parent': 'protected TBSI_E_MHSA_FULL_4090D_seed42 epoch15',
        'config_sha256': sha256(base.CONFIG), 'checkpoint_sha256': sha256(base.CHECKPOINT),
        'git_head': os.popen('git -C "{}" rev-parse HEAD'.format(ROOT)).read().strip(),
        'execution_code_sha256': {
            relative: sha256(ROOT / relative) for relative in CODE_FILES
        },
        'runner_sha256': sha256(Path(__file__).resolve()),
        'trace_root': str(TRACE_ROOT),
        'trace_summary': 'current-working-tree M1 replay trace, not archive per-frame parity',
        'selection': 'retrospective four-case study: three Cdiag full-244 large-gain cases and one low-AUC Cdiag regression; first crop-contained 30-frame IoU<0.5 onset and preceding frame',
        'inference': 'frozen M1 checkpoint, FP32, fixed state/crop from trace, no training or intervention',
        'n_sequences': len(SEQUENCES), 'n_fixed_frames': len(rows), 'n_stage_rows': len(stage_rows),
        'dataset_frame_roots': sorted({
            str(Path(sequences[name].frames[0][0]).parents[1]) for name in SEQUENCES
        }),
        'validation_split': 'LasHeR testingset case studies only; no population claim or candidate tuning',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
