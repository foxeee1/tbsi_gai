#!/usr/bin/env python3
"""Frozen-weight, same-state path-effect diagnostic for the Innovation-1 idea."""
import argparse
import copy
import csv
import hashlib
import json
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


ACTION_NAMES = ('F', 'GR', 'GT')


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


def read_frame(seq, frame_num):
    visible = cv2.cvtColor(cv2.imread(seq.frames[0][frame_num]), cv2.COLOR_BGR2RGB)
    infrared = cv2.cvtColor(cv2.imread(seq.frames[1][frame_num]), cv2.COLOR_BGR2RGB)
    return np.concatenate([visible, infrared], axis=2)


def iou_xywh(a, b):
    ax1, ay1, aw, ah = [float(x) for x in a]
    bx1, by1, bw, bh = [float(x) for x in b]
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def select_sequences(dataset, count):
    # Deterministic uniform coverage of the sorted training list; no score-based selection.
    if count > len(dataset):
        raise ValueError('requested %d sequences, but train split has %d' % (count, len(dataset)))
    indices = np.linspace(0, len(dataset) - 1, count).round().astype(int).tolist()
    return [dataset[i] for i in indices]


def point_frames(seq, points, horizon):
    last = len(seq.frames[0]) - horizon
    if last < 1:
        return []
    return sorted(set(np.linspace(1, last, points).round().astype(int).tolist()))


def set_action(action, layer):
    if action is None:
        os.environ.pop('TBSI_PATH_EFFECT_ACTION', None)
    else:
        os.environ['TBSI_PATH_EFFECT_ACTION'] = action
    os.environ['TBSI_PATH_EFFECT_LAYER'] = str(layer)


def track_window(tracker, seq, frame_num, snapshot, action, layer, horizon):
    restore_tracker(tracker, snapshot)
    boxes = []
    set_action(action, layer)
    for offset in range(horizon):
        current = frame_num + offset
        info = {'sequence_name': seq.name, 'frame_num': current}
        out = tracker.track(read_frame(seq, current), info)
        boxes.append([float(x) for x in out['target_bbox']])
        set_action('F', layer)
    return boxes


def run(args):
    os.environ['TBSI_INFER_FP16'] = '0'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_action('F', args.layer)

    params = parameters(args.config)
    params.debug = 0
    params.save_all_boxes = False
    if args.checkpoint:
        params.checkpoint = str(Path(args.checkpoint).resolve())
    if not Path(params.checkpoint).is_file():
        raise FileNotFoundError('checkpoint not found: %s' % params.checkpoint)
    train_list = ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(train_list)).get_sequence_list()
    sequences = select_sequences(dataset, args.sequences)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'selected_sequences.txt', 'w') as f:
        f.write('\n'.join(seq.name for seq in sequences) + '\n')

    meta = {
        'experiment': 'innovation1_path_effect_minimal',
        'config': args.config,
        'checkpoint': str(Path(params.checkpoint).resolve()),
        'checkpoint_sha256': hashlib.sha256(Path(params.checkpoint).read_bytes()).hexdigest(),
        'seed': args.seed,
        'actions': {'F': 'full', 'GR': 'suppress fused-template-to-RGB-search attention message',
                    'GT': 'suppress fused-template-to-TIR-search attention message'},
        'layer_index': args.layer,
        'horizon': args.horizon,
        'sequences': [seq.name for seq in sequences],
        'points_per_sequence': args.points,
        'repeats': args.repeats,
        'fp16': False,
        'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=True))

    tracker = TBSITrack(params, 'lasher_train')
    rows = []
    parity_max_abs = None
    for seq_index, seq in enumerate(sequences):
        print('[%d/%d] %s (%d frames)' % (seq_index + 1, len(sequences), seq.name, len(seq.frames[0])), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0].tolist()})
        frames = point_frames(seq, args.points, args.horizon)
        frame_set = set(frames)
        for frame_num in range(1, len(seq.frames[0])):
            if frame_num in frame_set:
                state_before = snapshot_tracker(tracker)
                if parity_max_abs is None:
                    # Default gate-off path versus explicit F must be identical
                    # before any intervention result is accepted.
                    restore_tracker(tracker, state_before)
                    set_action(None, args.layer)
                    default_boxes = track_window(tracker, seq, frame_num, state_before, 'F', args.layer, args.horizon)
                    restore_tracker(tracker, state_before)
                    set_action('F', args.layer)
                    explicit_boxes = track_window(tracker, seq, frame_num, state_before, 'F', args.layer, args.horizon)
                    parity_max_abs = max(abs(a - b) for aa, bb in zip(default_boxes, explicit_boxes)
                                         for a, b in zip(aa, bb))
                    meta['f_gate_off_vs_explicit_max_abs'] = parity_max_abs
                    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=True))
                reference = {}
                for repeat in range(args.repeats):
                    for action in ACTION_NAMES:
                        boxes = track_window(tracker, seq, frame_num, state_before, action, args.layer, args.horizon)
                        reference[(repeat, action)] = boxes
                        for offset, box in enumerate(boxes):
                            gt = seq.ground_truth_rect[frame_num + offset]
                            rows.append({
                                'sequence': seq.name, 'frame': frame_num, 'offset': offset,
                                'repeat': repeat, 'action': action,
                                'pred_x': box[0], 'pred_y': box[1], 'pred_w': box[2], 'pred_h': box[3],
                                'gt_x': float(gt[0]), 'gt_y': float(gt[1]), 'gt_w': float(gt[2]), 'gt_h': float(gt[3]),
                                'iou': iou_xywh(box, gt),
                            })
                restore_tracker(tracker, state_before)
                set_action('F', args.layer)
                tracker.track(read_frame(seq, frame_num), {'sequence_name': seq.name, 'frame_num': frame_num})
            else:
                set_action('F', args.layer)
                tracker.track(read_frame(seq, frame_num), {'sequence_name': seq.name, 'frame_num': frame_num})

    set_action('F', args.layer)
    fields = list(rows[0].keys()) if rows else []
    with open(out_dir / 'raw_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows, out_dir, args.horizon)
    meta['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    meta['rows'] = len(rows)
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=True))


def summarize(rows, out_dir, horizon):
    by_key = {}
    for row in rows:
        by_key.setdefault((row['sequence'], row['frame'], row['repeat'], row['action']), {})[int(row['offset'])] = float(row['iou'])
    summary = []
    for (seq, frame, repeat, action), vals in sorted(by_key.items()):
        full = by_key[(seq, frame, repeat, 'F')]
        deltas = [vals[k] - full[k] for k in range(horizon)]
        summary.append({'sequence': seq, 'frame': frame, 'repeat': repeat, 'action': action,
                        'delta_k0': deltas[0], 'delta_k1': deltas[1] if horizon > 1 else '',
                        'delta_k2': deltas[2] if horizon > 2 else '',
                        'delta_mean': float(np.mean(deltas)), 'large_gain': int(max(deltas) >= 0.05)})
    with open(out_dir / 'branch_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        writer.writeheader()
        writer.writerows(summary)
    grouped = {}
    for row in summary:
        grouped.setdefault((row['sequence'], row['action']), []).append(row)
    with open(out_dir / 'per_sequence_statistics.csv', 'w', newline='') as f:
        fields = ['sequence', 'action', 'n_points', 'mean_delta', 'median_delta', 'large_gain_rate', 'repeat_max_abs_diff']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (seq, action), vals in sorted(grouped.items()):
            means = [v['delta_mean'] for v in vals]
            reps = [v for v in vals if v['repeat'] == 0]
            rep2 = [v for v in vals if v['repeat'] == 1]
            diffs = []
            if rep2:
                a = {(v['frame']): v['delta_mean'] for v in reps}
                b = {(v['frame']): v['delta_mean'] for v in rep2}
                diffs = [abs(a[k] - b[k]) for k in a if k in b]
            writer.writerow({'sequence': seq, 'action': action, 'n_points': len(vals),
                             'mean_delta': float(np.mean(means)), 'median_delta': float(np.median(means)),
                             'large_gain_rate': float(np.mean([v['large_gain'] for v in vals])),
                             'repeat_max_abs_diff': max(diffs) if diffs else ''})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for action in ('GR', 'GT'):
            vals = [v['delta_mean'] for v in summary if v['action'] == action]
            plt.figure(figsize=(6, 4))
            plt.hist(vals, bins=20)
            plt.axvline(0.0, color='black', linewidth=1)
            plt.xlabel('D_t(%s), mean IoU delta over H=%d' % (action, horizon))
            plt.ylabel('branch points')
            plt.tight_layout()
            plt.savefig(out_dir / ('distribution_%s.png' % action), dpi=160)
            plt.close()
    except ImportError:
        (out_dir / 'plot_status.txt').write_text('matplotlib unavailable; CSV summaries are authoritative\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='v1.0.3-单卡4090D-24G-等效全局128-seed3407')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output', default='output/experiments/innovation1_path_effect')
    parser.add_argument('--sequences', type=int, default=12)
    parser.add_argument('--points', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    if args.repeats < 2 or args.horizon != 3:
        raise ValueError('minimum validation requires repeats >= 2 and horizon == 3')
    run(args)


if __name__ == '__main__':
    main()
