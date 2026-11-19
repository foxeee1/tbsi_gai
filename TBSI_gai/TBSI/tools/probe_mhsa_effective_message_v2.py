#!/usr/bin/env python3
"""Causal cross-modal message probe on matched regression/gain events."""
import argparse
import copy
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from lib.test.evaluation.lasherdataset import LasHeRDataset  # noqa: E402
from lib.test.parameter.tbsi_track import parameters  # noqa: E402
from lib.test.tracker.tbsi_track import TBSITrack  # noqa: E402

CONDITIONS = {'F': 1.0, 'C_OFF': 0.0, 'C_HALF': 0.5, 'C_1.5': 1.5,
              'FG_CROSS': 1.0}


def clone_value(value):
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value, list): return [clone_value(item) for item in value]
    if isinstance(value, tuple): return tuple(clone_value(item) for item in value)
    if isinstance(value, dict): return {key: clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def snapshot_tracker(tracker):
    return {'state': clone_value(tracker.state), 'prev_search': clone_value(tracker.prev_search),
            'prev_temporal_tokens': clone_value(tracker.prev_temporal_tokens), 'frame_id': tracker.frame_id}


def restore_tracker(tracker, snapshot):
    tracker.state = clone_value(snapshot['state']); tracker.prev_search = clone_value(snapshot['prev_search'])
    tracker.prev_temporal_tokens = clone_value(snapshot['prev_temporal_tokens']); tracker.frame_id = snapshot['frame_id']


def snapshot_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(snapshot):
    random.setstate(snapshot['python']); np.random.set_state(snapshot['numpy'])
    torch.set_rng_state(snapshot['torch']); torch.cuda.set_rng_state_all(snapshot['cuda'])


def hash_value(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item): digest.update(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item: visit(child)
        else: digest.update(repr(item).encode())
    visit(value); return digest.hexdigest()[:16]


def read_frame(seq, frame_num):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][frame_num]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][frame_num]), cv2.COLOR_BGR2RGB)
    return np.concatenate([rgb, tir], axis=2)


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = [float(x) for x in b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah); bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    inter = max(0.0, min(ax2, bx2) - max(ax, bx)) * max(0.0, min(ay2, by2) - max(ay, by))
    union = max(0.0, ax2 - ax) * max(0.0, ay2 - ay) + max(0.0, bx2 - bx) * max(0.0, by2 - by) - inter
    return inter / union if union > 0 else 0.0


def probe_stats(attn_prob, rgb_template, tir_template, **kwargs):
    bsz, heads, length, _ = attn_prob.shape; n = rgb_template.shape[1]
    if bsz != 1 or length != 2 * n: raise ValueError('expected batch-1 equal template streams')
    rgb_self = (attn_prob[:, :, :n, :n] @ kwargs['v_rgb']) if 'v_rgb' in kwargs else None
    # Values are supplied by the model collector; this function only receives summaries.
    del rgb_self
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    parser.add_argument('--checkpoint', default=str(ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'))
    parser.add_argument('--events-dir', default=str(ROOT / 'output/experiments/mhsa_causal_probe_v2/manifest'))
    parser.add_argument('--output', default=str(ROOT / 'output/experiments/mhsa_causal_probe_v2/probe'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--conditions', default='F,C_OFF,C_HALF,C_1.5')
    parser.add_argument('--only-sequence', default='')
    args = parser.parse_args()
    condition_names = [name for name in args.conditions.split(',') if name]
    if not condition_names or any(name not in CONDITIONS and name != 'F_POST' for name in condition_names):
        raise ValueError('conditions must be a subset of {}'.format(','.join(CONDITIONS)))
    os.environ['TBSI_INFER_FP16'] = '0'; os.environ.pop('TBSI_TEMPLATE_CROSS_SCALE', None)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    # Do not override cuDNN algorithm selection here. The stored M1 full-test
    # predictions were produced through the regular test entrypoint; forcing
    # deterministic kernels changes the closed-loop trajectory on this stack.
    checkpoint = Path(args.checkpoint).resolve(); output = Path(args.output).resolve()
    if not checkpoint.is_file(): raise FileNotFoundError(str(checkpoint))
    if output.exists() and any(output.iterdir()): raise RuntimeError('refusing non-empty output: {}'.format(output))
    output.mkdir(parents=True, exist_ok=True)
    event_rows = []
    for filename in ('regression_events.csv', 'gain_events.csv'):
        with open(Path(args.events_dir) / filename, newline='') as handle: event_rows.extend(csv.DictReader(handle))
    events = {}
    for row in event_rows: events.setdefault(row['seq'], {}).setdefault(int(row['frame']), []).append(row)
    if args.only_sequence:
        events = {args.only_sequence: events[args.only_sequence]}
    dataset = LasHeRDataset(split='testingset').get_sequence_list()
    sequence_map = {seq.name: seq for seq in dataset}
    params = parameters(args.config); params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_test')
    rows = []
    for seq_name in sorted(events):
        seq = sequence_map[seq_name]; print('[SEQ] {}'.format(seq_name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        # Match lib.test.evaluation.tracker._track_sequence: initialize() of
        # TBSITrack returns None, so the first track call receives {}.
        previous_output = {}
        for frame_num in range(1, max(events[seq_name]) + 1):
            if frame_num not in events[seq_name]:
                os.environ.pop('TBSI_TEMPLATE_CROSS_SCALE', None)
                info = {'previous_output': previous_output, 'sequence_name': seq.name,
                        'frame_num': frame_num, 'gt_bbox': seq.ground_truth_rect[frame_num]}
                previous_output = tracker.track(read_frame(seq, frame_num), info)
                continue
            base_snapshot, base_rng = snapshot_tracker(tracker), snapshot_rng()
            for event in events[seq_name][frame_num]:
                state_hash = hash_value(base_snapshot['state'])
                for condition in condition_names:
                    scale = CONDITIONS['F'] if condition == 'F_POST' else CONDITIONS[condition]
                    restore_tracker(tracker, base_snapshot); restore_rng(base_rng)
                    os.environ['TBSI_TEMPLATE_CROSS_SCALE'] = str(scale)
                    if condition == 'FG_CROSS':
                        os.environ['TBSI_MSHA_FG_CROSS_MODE'] = 'foreground_queries'
                    else:
                        os.environ.pop('TBSI_MSHA_FG_CROSS_MODE', None)
                    layer_stats = {}
                    for layer_index, layer in enumerate(tracker.network.backbone.tbsi_layers):
                        fusion = getattr(layer, 'v2_template_fusion', None)
                        if fusion is None: continue
                        fusion.analysis_enabled = True
                        def collector(layer_index=layer_index, **kwargs):
                            layer_stats[layer_index] = {
                                'rgb_self_norm': float(kwargs['rgb_self_norm'].item()),
                                'rgb_cross_norm': float(kwargs['rgb_cross_norm'].item()),
                                'tir_self_norm': float(kwargs['tir_self_norm'].item()),
                                'tir_cross_norm': float(kwargs['tir_cross_norm'].item()),
                                'output_change_norm': float(kwargs['output_change_norm'].item()),
                            }
                        fusion.analysis_collector = collector
                    out = tracker.track(read_frame(seq, frame_num), {
                        'previous_output': previous_output, 'sequence_name': seq.name,
                        'frame_num': frame_num, 'gt_bbox': seq.ground_truth_rect[frame_num]})
                    box = [float(x) for x in out['target_bbox']]
                    gt = seq.ground_truth_rect[frame_num]
                    pred_center = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
                    gt_center = (float(gt[0]) + float(gt[2]) / 2.0,
                                 float(gt[1]) + float(gt[3]) / 2.0)
                    for layer, stats in sorted(layer_stats.items()):
                        rgb_ratio = stats['rgb_cross_norm'] / max(stats['rgb_self_norm'], 1e-8)
                        tir_ratio = stats['tir_cross_norm'] / max(stats['tir_self_norm'], 1e-8)
                        rows.append({'event_id': event['event_id'], 'seq': seq.name, 'frame': frame_num,
                                     'group': event['group'], 'condition': condition, 'offset': 0,
                                     'layer': layer, 'iou': iou_xywh(box, gt), 'pred_x': box[0], 'pred_y': box[1],
                                     'pred_w': box[2], 'pred_h': box[3], 'rgb_self_norm': stats['rgb_self_norm'],
                                     'rgb_cross_norm': stats['rgb_cross_norm'], 'tir_self_norm': stats['tir_self_norm'],
                                     'tir_cross_norm': stats['tir_cross_norm'], 'rgb_cross_self_ratio': rgb_ratio,
                                     'tir_cross_self_ratio': tir_ratio, 'output_change_norm': stats['output_change_norm'],
                                     'center_error_px': float(np.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1])),
                                     'cross_scale': scale, 'state_hash': state_hash})
            # Advance the real tracker from the shared state using the original F path.
            restore_tracker(tracker, base_snapshot); restore_rng(base_rng); os.environ.pop('TBSI_TEMPLATE_CROSS_SCALE', None)
            os.environ.pop('TBSI_MSHA_FG_CROSS_MODE', None)
            previous_output = tracker.track(read_frame(seq, frame_num), {
                'previous_output': previous_output, 'sequence_name': seq.name,
                'frame_num': frame_num, 'gt_bbox': seq.ground_truth_rect[frame_num]})
    with open(output / 'event_results.csv', 'w', newline='') as handle:
        fields = list(rows[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with open(output / 'probe_manifest.json', 'w') as handle:
        json.dump({'experiment': 'MHSA-CAUSAL-PROBE-V2', 'config': args.config, 'checkpoint': str(checkpoint),
                   'seed': args.seed, 'events': len(event_rows), 'conditions': condition_names,
                   'layers': 'all TBSI-E layers', 'offsets': [0], 'fp16': False}, handle, indent=2)
    os.environ.pop('TBSI_TEMPLATE_CROSS_SCALE', None)
    print(json.dumps({'events': len(event_rows), 'rows': len(rows)}))


if __name__ == '__main__': main()
