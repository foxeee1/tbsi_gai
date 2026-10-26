#!/usr/bin/env python3
"""Frozen-weight functional probe for M1 template cross-modal routing.

Runs five logit-bias conditions on 24 deterministic LasHeR train sequences,
two uniformly selected events per sequence, and compares event-frame IoU.
No parameters are added or updated. The model is loaded from the trained M1
checkpoint and each condition branches from the same tracker/RNG snapshot.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.evaluation.lasherdataset import LasHeRDataset  # noqa: E402
from lib.test.parameter.tbsi_track import parameters  # noqa: E402
from lib.test.tracker.tbsi_track import TBSITrack  # noqa: E402


CONDITIONS = {
    'F': (0.0, 0.0),
    'RT_down': (-math.log(2.0), 0.0),
    'TR_down': (0.0, -math.log(2.0)),
    'Both_down': (-math.log(2.0), -math.log(2.0)),
    'Both_up': (math.log(2.0), math.log(2.0)),
}


def clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_value(v) for v in value)
    if isinstance(value, dict):
        return {k: clone_value(v) for k, v in value.items()}
    return copy.deepcopy(value)


def snapshot_tracker(tracker):
    return {
        'state': clone_value(tracker.state),
        'prev_search': clone_value(tracker.prev_search),
        'prev_temporal_tokens': clone_value(tracker.prev_temporal_tokens),
        'frame_id': tracker.frame_id,
    }


def restore_tracker(tracker, snapshot):
    tracker.state = clone_value(snapshot['state'])
    tracker.prev_search = clone_value(snapshot['prev_search'])
    tracker.prev_temporal_tokens = clone_value(snapshot['prev_temporal_tokens'])
    tracker.frame_id = snapshot['frame_id']


def snapshot_rng():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
    }


def restore_rng(snapshot):
    random.setstate(snapshot['python'])
    np.random.set_state(snapshot['numpy'])
    torch.set_rng_state(snapshot['torch'])
    torch.cuda.set_rng_state_all(snapshot['cuda'])


def read_frame(seq, frame_num):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][frame_num]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][frame_num]), cv2.COLOR_BGR2RGB)
    return np.concatenate([rgb, tir], axis=2)


def iou_xywh(a, b):
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1, ix2, iy2 = max(ax, bx), max(ay, by), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax) * max(0.0, ay2 - ay) + max(0.0, bx2 - bx) * max(0.0, by2 - by) - inter
    return inter / union if union > 0 else 0.0


def set_route(delta_r_to_t, delta_t_to_r):
    os.environ['TBSI_TEMPLATE_ROUTE_R2T'] = str(delta_r_to_t)
    os.environ['TBSI_TEMPLATE_ROUTE_T2R'] = str(delta_t_to_r)


def clear_route():
    os.environ.pop('TBSI_TEMPLATE_ROUTE_R2T', None)
    os.environ.pop('TBSI_TEMPLATE_ROUTE_T2R', None)


def route_stats(tracker):
    result = {}
    for index, layer in enumerate(tracker.network.backbone.tbsi_layers):
        stats = getattr(layer.v2_template_fusion, 'last_route_stats', None)
        if stats is None:
            raise RuntimeError('missing route stats for TBSI layer {}'.format(index))
        result[index] = dict(stats)
    return result


def select_sequences(dataset, count):
    if count > len(dataset):
        raise ValueError('requested {} sequences, dataset has {}'.format(count, len(dataset)))
    indices = np.linspace(0, len(dataset) - 1, count).round().astype(int).tolist()
    return [dataset[i] for i in indices]


def event_frames(seq):
    n = min(len(seq.frames[0]), len(seq.frames[1]), len(seq.ground_truth_rect))
    if n < 3:
        return []
    return sorted(set(np.linspace(1, n - 1, 2).round().astype(int).tolist()))


def track_condition(tracker, seq, frame_num, snapshot, rng, condition):
    restore_tracker(tracker, snapshot)
    restore_rng(rng)
    set_route(*CONDITIONS[condition])
    out = tracker.track(read_frame(seq, frame_num), {
        'sequence_name': seq.name, 'frame_num': frame_num,
        'gt_bbox': seq.ground_truth_rect[frame_num],
    })
    return [float(x) for x in out['target_bbox']], route_stats(tracker)


def write_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    parser.add_argument('--checkpoint', default=str(ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'))
    parser.add_argument('--output', default=str(ROOT / 'output/experiments/template_route_probe_v1'))
    parser.add_argument('--sequence-list', default=str(ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'))
    parser.add_argument('--sequences', type=int, default=24)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='full')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.environ['TBSI_INFER_FP16'] = '0'
    clear_route()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    out_dir = Path(args.output).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError('refusing to reuse non-empty output directory: {}'.format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    params = parameters(args.config)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    sequence_list = Path(args.sequence_list).resolve()
    if not sequence_list.is_file():
        raise FileNotFoundError(str(sequence_list))
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(sequence_list)).get_sequence_list()
    sequences = select_sequences(dataset, args.sequences)
    if args.mode == 'smoke':
        sequences = sequences[:2]
    with open(out_dir / 'selected_sequences.txt', 'w') as handle:
        handle.write('\n'.join(seq.name for seq in sequences) + '\n')

    tracker = TBSITrack(params, 'lasher_train')
    rows = []
    mass_rows = []
    parity_rows = []
    first_event = True
    for seq_index, seq in enumerate(sequences):
        print('[{}/{}] {}'.format(seq_index + 1, len(sequences), seq.name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        events = set(event_frames(seq))
        for frame_num in range(1, len(seq.frames[0])):
            if frame_num not in events:
                clear_route()
                tracker.track(read_frame(seq, frame_num), {'sequence_name': seq.name, 'frame_num': frame_num})
                continue

            state = snapshot_tracker(tracker)
            rng = snapshot_rng()

            # Default environment versus explicit zero-bias F condition.
            clear_route()
            restore_tracker(tracker, state)
            restore_rng(rng)
            default_box = [float(x) for x in tracker.track(read_frame(seq, frame_num), {
                'sequence_name': seq.name, 'frame_num': frame_num,
            })['target_bbox']]
            explicit_box, explicit_stats = track_condition(tracker, seq, frame_num, state, rng, 'F')
            parity = max(abs(a - b) for a, b in zip(default_box, explicit_box))
            parity_rows.append({'sequence': seq.name, 'frame': frame_num, 'max_abs_box_diff': parity})
            if first_event and parity != 0.0:
                raise RuntimeError('F parity failed: max_abs_box_diff={}'.format(parity))
            first_event = False

            condition_outputs = {'F': (explicit_box, explicit_stats)}
            for condition in ('RT_down', 'TR_down', 'Both_down', 'Both_up'):
                condition_outputs[condition] = track_condition(tracker, seq, frame_num, state, rng, condition)

            gt = seq.ground_truth_rect[frame_num]
            f_iou = iou_xywh(condition_outputs['F'][0], gt)
            for condition, (box, stats) in condition_outputs.items():
                rows.append({
                    'sequence': seq.name, 'frame': frame_num, 'condition': condition,
                    'pred_x': box[0], 'pred_y': box[1], 'pred_w': box[2], 'pred_h': box[3],
                    'gt_x': float(gt[0]), 'gt_y': float(gt[1]), 'gt_w': float(gt[2]), 'gt_h': float(gt[3]),
                    'iou': iou_xywh(box, gt), 'delta_iou_vs_F': iou_xywh(box, gt) - f_iou,
                })
                for layer_index, layer_stats in stats.items():
                    mass_rows.append({
                        'sequence': seq.name, 'frame': frame_num, 'condition': condition,
                        'layer_index': layer_index,
                        'delta_r_to_t': layer_stats['delta_r_to_t'],
                        'delta_t_to_r': layer_stats['delta_t_to_r'],
                        'r_to_t_mass': layer_stats['r_to_t_mass'],
                        't_to_r_mass': layer_stats['t_to_r_mass'],
                    })

            restore_tracker(tracker, state)
            restore_rng(rng)
            clear_route()
            tracker.track(read_frame(seq, frame_num), {'sequence_name': seq.name, 'frame_num': frame_num})

    clear_route()
    write_csv(out_dir / 'event_results.csv', rows)
    write_csv(out_dir / 'attention_mass.csv', mass_rows)
    write_csv(out_dir / 'parity.csv', parity_rows)

    summary = []
    for condition in CONDITIONS:
        vals = [float(row['delta_iou_vs_F']) for row in rows if row['condition'] == condition]
        summary.append({
            'condition': condition, 'n_events': len(vals),
            'mean_delta_iou': float(np.mean(vals)) if vals else None,
            'median_delta_iou': float(np.median(vals)) if vals else None,
            'positive_events': int(sum(value > 0 for value in vals)),
            'negative_events': int(sum(value < 0 for value in vals)),
        })
    write_csv(out_dir / 'condition_summary.csv', summary)

    metadata = {
        'experiment': 'template_route_probe_v1', 'config': args.config, 'seed': args.seed,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'sequence_count': len(sequences), 'events_per_sequence': 2,
        'sequence_list': str(sequence_list),
        'conditions': CONDITIONS, 'mode': args.mode,
        'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'torch': torch.__version__, 'python': sys.version,
        'gpu': torch.cuda.get_device_name(0), 'fp16': False,
        'parity_max_abs_box_diff': max((float(x['max_abs_box_diff']) for x in parity_rows), default=None),
        'rows': len(rows), 'status': 'PASS' if args.mode == 'full' else 'SMOKE_PASS',
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=True))
    print(json.dumps({'output': str(out_dir), 'summary': summary, 'parity': metadata['parity_max_abs_box_diff']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
