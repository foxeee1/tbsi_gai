#!/usr/bin/env python3
"""Three-arm closed-loop holdout screen for MSHA query-role intervention."""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

CHECKPOINT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
SEQUENCE_LIST = ROOT / 'experiments/protocols/m1_screening_v1_20261022/locked_holdout_sequences.txt'
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
ARMS = ('M1', 'FG_QUERY', 'SHIFTED_QUERY')


def read_frame(seq, index):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][index]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][index]), cv2.COLOR_BGR2RGB)
    return np.concatenate((rgb, tir), axis=2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-frames', type=int, default=128)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output directory: {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_DISABLE_TRACKER_CACHE'] = '1'
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    sequences = LasHeRDataset('trainingset', str(SEQUENCE_LIST)).get_sequence_list()
    if len(sequences) != 24:
        raise ValueError('expected 24 locked holdout sequences, got {}'.format(len(sequences)))
    rows = []
    for arm in ARMS:
        if arm == 'FG_QUERY':
            os.environ['TBSI_MSHA_FG_CROSS_MODE'] = 'foreground_queries'
        elif arm == 'SHIFTED_QUERY':
            os.environ['TBSI_MSHA_FG_CROSS_MODE'] = 'shifted_queries'
        else:
            os.environ.pop('TBSI_MSHA_FG_CROSS_MODE', None)
        for seq in sequences:
            params = parameters(CONFIG)
            params.debug = 0
            params.save_all_boxes = False
            params.checkpoint = str(CHECKPOINT)
            tracker = TBSITrack(params, 'lasher_test')
            tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
            previous_output = {}
            count = min(len(seq.frames[0]), args.max_frames + 1)
            values = []
            for frame in range(1, count):
                result = tracker.track(read_frame(seq, frame), {
                    'previous_output': previous_output,
                    'sequence_name': seq.name,
                    'frame_num': frame,
                    'gt_bbox': seq.ground_truth_rect[frame],
                })
                previous_output = result
                box = [float(x) for x in result['target_bbox']]
                gt = [float(x) for x in seq.ground_truth_rect[frame]]
                value = iou(box, gt)
                values.append(value)
                rows.append({'arm': arm, 'sequence': seq.name, 'frame': frame,
                             'iou': value, 'pred_x': box[0], 'pred_y': box[1],
                             'pred_w': box[2], 'pred_h': box[3]})
            print('[{}] {} frames={} mean_iou={:.5f} failure_lt05={:.4f}'.format(
                arm, seq.name, len(values), float(np.mean(values)),
                float(np.mean(np.asarray(values) < .5))), flush=True)
    with (out / 'per_frame.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for arm in ARMS:
        for seq in sequences:
            values = [r['iou'] for r in rows if r['arm'] == arm and r['sequence'] == seq.name]
            summary.append({'arm': arm, 'sequence': seq.name, 'frames': len(values),
                            'mean_iou': float(np.mean(values)),
                            'failure_rate_iou_lt_05': float(np.mean(np.asarray(values) < .5))})
    with (out / 'per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    for arm in ARMS:
        values = [r['mean_iou'] for r in summary if r['arm'] == arm]
        print('[SUMMARY] {} seq_macro_mean_iou={:.6f}'.format(arm, float(np.mean(values))), flush=True)


if __name__ == '__main__':
    main()
