#!/usr/bin/env python3
"""Read-only G0 audit for residual RGB/TIR search complement after TBSI-0.

The model is frozen. GT is used only after forward to form token labels.
The primary location is post TBSI index 0, after block 5 and before TBSI index 1.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset  # noqa: E402
from lib.test.parameter.tbsi_track import parameters  # noqa: E402
from lib.test.tracker.tbsi_track import TBSITrack  # noqa: E402
from lib.train.data.processing_utils import sample_target  # noqa: E402
from tools.template_route_probe_v1 import event_frames, read_frame, select_sequences  # noqa: E402


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(v) for v in value)
    return copy.deepcopy(value)


def token_mask(box, state, resize_factor, search_size=256, grid=16):
    cx = float(state[0]) + float(state[2]) / 2.0
    cy = float(state[1]) + float(state[3]) / 2.0
    side = float(search_size) / float(resize_factor)
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    x = (float(box[0]) - x0) * resize_factor
    y = (float(box[1]) - y0) * resize_factor
    w = float(box[2]) * resize_factor
    h = float(box[3]) * resize_factor
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
    px0, py0 = xx * (search_size / grid), yy * (search_size / grid)
    px1, py1 = px0 + search_size / grid, py0 + search_size / grid
    return ((px1 > x) & (px0 < x + w) & (py1 > y) & (py0 < y + h)).reshape(-1)


def crop_center_token(state, resize_factor, search_size=256, grid=16):
    cx = float(state[0]) + float(state[2]) / 2.0
    cy = float(state[1]) + float(state[3]) / 2.0
    side = float(search_size) / float(resize_factor)
    x = (cx - (cx - side / 2.0)) * resize_factor
    y = (cy - (cy - side / 2.0)) * resize_factor
    col = np.clip(int(x / (search_size / grid)), 0, grid - 1)
    row = np.clip(int(y / (search_size / grid)), 0, grid - 1)
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
    return -((xx.reshape(-1) - col) ** 2 + (yy.reshape(-1) - row) ** 2).astype(float)


def local_scores(sim, grid=16):
    sim = sim.reshape(grid, grid, grid, grid)
    values = [sim]
    for dh, dw in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        shifted = torch.full_like(sim, -1e9)
        src_h = slice(max(0, -dh), min(grid, grid - dh))
        src_w = slice(max(0, -dw), min(grid, grid - dw))
        dst_h = slice(max(0, dh), min(grid, grid + dh))
        dst_w = slice(max(0, dw), min(grid, grid + dw))
        shifted[:, :, dst_h, dst_w] = sim[:, :, src_h, src_w]
        values.append(shifted)
    return torch.stack(values, dim=0).max(dim=0).values.reshape(grid * grid, grid * grid)


def average_precision(scores, labels):
    order = np.argsort(-np.asarray(scores))
    labels = np.asarray(labels, dtype=bool)[order]
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / positives)


def feature_metrics(rgb, tir, mask, xy_score):
    rgb = F.normalize(torch.from_numpy(rgb).float(), dim=-1)
    tir = F.normalize(torch.from_numpy(tir).float(), dim=-1)
    sim = rgb @ tir.t()
    diag = torch.diagonal(sim).numpy()
    local = torch.diagonal(local_scores(sim)).numpy()
    mask = np.asarray(mask, dtype=bool)
    bg = ~mask
    return {
        'target_tokens': int(mask.sum()),
        'diag_target_mean': float(diag[mask].mean()),
        'diag_background_mean': float(diag[bg].mean()) if bg.any() else 0.0,
        'diag_target_background_margin': float(diag[mask].mean() - diag[bg].mean()) if bg.any() else 0.0,
        'local_target_mean': float(local[mask].mean()),
        'local_background_mean': float(local[bg].mean()) if bg.any() else 0.0,
        'local_target_background_margin': float(local[mask].mean() - local[bg].mean()) if bg.any() else 0.0,
        'local_rescue_rate_target': float((local[mask] > diag[mask]).mean()),
        'token_ap': average_precision(diag, mask),
        'xy_ap': average_precision(xy_score, mask),
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_tracker(args, checkpoint, sequences):
    params = parameters(args.config)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_train')
    tracker.diagnostic = True
    tracker.diagnostic_tbsi_index = 0
    tracker.diagnostic_post_block = 5
    rows = []
    parity = []
    baseline = None
    if args.parity_sequences > 0:
        base_params = parameters(args.config)
        base_params.debug = 0
        base_params.save_all_boxes = False
        base_params.checkpoint = str(checkpoint)
        baseline = TBSITrack(base_params, 'lasher_train')
    for seq_index, seq in enumerate(sequences):
        print('[{}/{}] {}'.format(seq_index + 1, len(sequences), seq.name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        if baseline is not None and seq_index < args.parity_sequences:
            baseline.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        events = set(event_frames(seq))
        for frame_num in range(1, len(seq.frames[0])):
            state_before = list(tracker.state)
            image = read_frame(seq, frame_num)
            _, resize_factor, _ = sample_target(image, state_before, params.search_factor,
                                                output_sz=params.search_size)
            info = {'sequence_name': seq.name, 'frame_num': frame_num,
                    'gt_bbox': seq.ground_truth_rect[frame_num]}
            result = tracker.track(image, info)
            if baseline is not None and seq_index < args.parity_sequences:
                base_result = baseline.track(image, info)
                if frame_num in events:
                    parity.append({'sequence': seq.name, 'frame': frame_num,
                                   'max_abs_box_diff': float(np.max(np.abs(
                                       np.asarray(result['target_bbox']) - np.asarray(base_result['target_bbox']))))})
            if frame_num not in events:
                continue
            features = tracker._last_diagnostic_features
            required = ('post_tbsi_rgb', 'post_tbsi_tir', 'post_block_rgb', 'post_block_tir')
            if any(key not in features for key in required):
                raise RuntimeError('missing diagnostic features: {}'.format(required))
            arrays = {key: features[key][0].float().cpu().numpy() for key in required}
            mask = token_mask(seq.ground_truth_rect[frame_num], state_before, resize_factor,
                              search_size=params.search_size)
            np.savez_compressed(Path(args.output) / ('event_%04d.npz' % len(rows)),
                                mask=mask.astype(np.uint8), **arrays)
            rows.append({
                'event_id': 'event_%04d' % (len(rows)), 'sequence': seq.name,
                'frame': int(frame_num), 'state_before': [float(x) for x in state_before],
                'pred_box': [float(x) for x in result['target_bbox']],
                'gt_box': [float(x) for x in seq.ground_truth_rect[frame_num]],
                'resize_factor': float(resize_factor), 'target_tokens': int(mask.sum()),
                'target_in_crop': bool(mask.any()), 'token_grid': [16, 16],
            })
    return rows, parity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sequence-list', default=str(ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'))
    ap.add_argument('--output', required=True)
    ap.add_argument('--sequences', type=int, default=24)
    ap.add_argument('--parity-sequences', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    checkpoint = Path(args.checkpoint).resolve()
    sequence_list = Path(args.sequence_list).resolve()
    out = Path(args.output).resolve()
    if not checkpoint.is_file() or not sequence_list.is_file():
        raise FileNotFoundError('checkpoint or sequence list missing')
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(sequence_list)).get_sequence_list()
    sequences = select_sequences(dataset, args.sequences)
    rows, parity = run_tracker(args, checkpoint, sequences)
    (out / 'selected_sequences.txt').write_text('\n'.join(s.name for s in sequences) + '\n')
    with (out / 'events.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    parity_result = {'events': parity, 'max_abs_box_diff': max([r['max_abs_box_diff'] for r in parity] or [0.0]),
                     'passed': all(r['max_abs_box_diff'] == 0.0 for r in parity)}
    (out / 'parity.json').write_text(json.dumps(parity_result, indent=2) + '\n')
    if not parity_result['passed']:
        raise RuntimeError('diagnostic parity failed')

    arrays = [np.load(out / ('event_%04d.npz' % i)) for i in range(len(rows))]
    usable = [i for i, row in enumerate(rows) if row['target_tokens'] > 0]
    if len(usable) < int(0.8 * len(rows)):
        raise RuntimeError('target coverage below fixed G0 minimum')
    metrics = []
    for i in usable:
        data = arrays[i]
        mask = data['mask'].astype(bool)
        xy = crop_center_token(rows[i]['state_before'], rows[i]['resize_factor'])
        row = {'event_id': rows[i]['event_id'], 'sequence': rows[i]['sequence'], 'frame': rows[i]['frame']}
        for stage in ('post_tbsi', 'post_block'):
            row[stage] = feature_metrics(data[stage + '_rgb'], data[stage + '_tir'], mask, xy)
        metrics.append(row)
    # Fixed cyclic event mismatch and fixed one-token spatial shift controls.
    for j, row in enumerate(metrics):
        i = usable[j]
        shuffled = usable[(j + 1) % len(usable)]
        own = arrays[i]; other = arrays[shuffled]
        shifted = np.roll(own['post_tbsi_tir'].reshape(16, 16, -1), 1, axis=1).reshape(256, -1)
        mask = own['mask'].astype(bool)
        xy = crop_center_token(rows[i]['state_before'], rows[i]['resize_factor'])
        row['post_tbsi_event_shuffle'] = feature_metrics(own['post_tbsi_rgb'], other['post_tbsi_tir'], mask, xy)
        row['post_tbsi_spatial_shift'] = feature_metrics(own['post_tbsi_rgb'], shifted, mask, xy)
    (out / 'event_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

    def mean(path):
        values = []
        for row in metrics:
            value = row
            for key in path.split('.'):
                value = value[key]
            values.append(value)
        return float(np.mean(values))

    summary = {'events_total': len(rows), 'events_usable': len(usable),
               'sequences': len(sequences), 'target_coverage': len(usable) / max(len(rows), 1),
               'parity': parity_result, 'checkpoint_sha256': sha256(checkpoint),
               'stages': {}}
    for stage in ('post_tbsi', 'post_block'):
        summary['stages'][stage] = {
            'paired_diag_margin': mean(stage + '.diag_target_background_margin'),
            'paired_local_margin': mean(stage + '.local_target_background_margin'),
            'paired_token_ap': mean(stage + '.token_ap'),
            'xy_only_ap': mean(stage + '.xy_ap'),
        }
    summary['post_tbsi_controls'] = {
        'event_shuffle_diag_margin': mean('post_tbsi_event_shuffle.diag_target_background_margin'),
        'spatial_shift_diag_margin': mean('post_tbsi_spatial_shift.diag_target_background_margin'),
        'event_shuffle_token_ap': mean('post_tbsi_event_shuffle.token_ap'),
        'spatial_shift_token_ap': mean('post_tbsi_spatial_shift.token_ap'),
        'correct_minus_event_margin': summary['stages']['post_tbsi']['paired_diag_margin'] -
                                      mean('post_tbsi_event_shuffle.diag_target_background_margin'),
        'correct_minus_spatial_margin': summary['stages']['post_tbsi']['paired_diag_margin'] -
                                        mean('post_tbsi_spatial_shift.diag_target_background_margin'),
        'correct_minus_event_ap': summary['stages']['post_tbsi']['paired_token_ap'] -
                                  mean('post_tbsi_event_shuffle.token_ap'),
        'correct_minus_spatial_ap': summary['stages']['post_tbsi']['paired_token_ap'] -
                                    mean('post_tbsi_spatial_shift.token_ap'),
    }
    summary['decision_rule'] = {
        'coverage_gt': 0.80, 'correct_minus_event_margin_gt': 0.01,
        'correct_minus_spatial_margin_gt': 0.01, 'correct_minus_event_ap_gt': 0.03,
        'correct_minus_spatial_ap_gt': 0.03,
        'diagnostic_only': True,
    }
    c = summary['post_tbsi_controls']
    summary['decision'] = 'G0_PASS_FOR_P0' if (
        summary['target_coverage'] > 0.80 and
        c['correct_minus_event_margin'] > 0.01 and
        c['correct_minus_spatial_margin'] > 0.01 and
        c['correct_minus_event_ap'] > 0.03 and
        c['correct_minus_spatial_ap'] > 0.03) else 'G0_STOP_NO_RELIABLE_COMPLEMENT'
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
