#!/usr/bin/env python3
"""128-frame paired closed-loop screen for foreground-limited MSHA messages."""
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEQUENCE_LIST = ROOT / 'experiments/manifests/m1_lowauc20_msha_20261022.txt'
CHECKPOINT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
OUT = ROOT / 'output/experiments/M1_LOWAUC20_MSHA_CLOSED_LOOP_20261022'

from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack


def read_frame(seq, index):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][index]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][index]), cv2.COLOR_BGR2RGB)
    return np.concatenate((rgb, tir), axis=2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ['TBSI_INFER_FP16'] = '0'
    torch.set_num_threads(1)
    sequences = LasHeRDataset('testingset', str(SEQUENCE_LIST)).get_sequence_list()
    rows = []
    for condition in ('F', 'FG_CROSS'):
        if condition == 'FG_CROSS':
            os.environ['TBSI_MSHA_FG_CROSS_MODE'] = 'foreground_queries'
        else:
            os.environ.pop('TBSI_MSHA_FG_CROSS_MODE', None)
        for seq in sequences:
            params = parameters('TBSI_E_MHSA_FULL_4090D_seed42')
            params.debug = 0
            params.save_all_boxes = False
            params.checkpoint = str(CHECKPOINT)
            tracker = TBSITrack(params, 'lasher_test')
            tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
            previous_output = {}
            count = min(len(seq.frames[0]), 129)
            overlaps = []
            for frame in range(1, count):
                out = tracker.track(read_frame(seq, frame), {
                    'previous_output': previous_output,
                    'sequence_name': seq.name,
                    'frame_num': frame,
                    'gt_bbox': seq.ground_truth_rect[frame],
                })
                previous_output = out
                box = [float(x) for x in out['target_bbox']]
                gt = [float(x) for x in seq.ground_truth_rect[frame]]
                overlap = iou(box, gt)
                overlaps.append(overlap)
                rows.append({'condition': condition, 'sequence': seq.name,
                             'frame': frame, 'iou': overlap,
                             'pred_x': box[0], 'pred_y': box[1],
                             'pred_w': box[2], 'pred_h': box[3]})
            print('[{}] {} frames={} mean_iou={:.4f} failure_lt05={:.3f}'.format(
                condition, seq.name, len(overlaps), float(np.mean(overlaps)),
                float(np.mean(np.asarray(overlaps) < .5))), flush=True)
    with (OUT / 'per_frame.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for condition in ('F', 'FG_CROSS'):
        for seq in sequences:
            values = [r['iou'] for r in rows if r['condition'] == condition and r['sequence'] == seq.name]
            summary.append({'condition': condition, 'sequence': seq.name,
                            'frames': len(values), 'mean_iou': float(np.mean(values)),
                            'failure_rate_iou_lt_05': float(np.mean(np.asarray(values) < .5))})
    with (OUT / 'per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    for condition in ('F', 'FG_CROSS'):
        values = [r['mean_iou'] for r in summary if r['condition'] == condition]
        print('[SUMMARY] {} seq_macro_mean_iou={:.5f}'.format(condition, float(np.mean(values))), flush=True)


if __name__ == '__main__':
    main()
