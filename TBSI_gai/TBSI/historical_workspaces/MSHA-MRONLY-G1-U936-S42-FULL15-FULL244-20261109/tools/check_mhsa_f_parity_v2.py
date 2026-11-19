#!/usr/bin/env python3
"""Check current frozen-M1 inference against the stored M1 boxes for one sequence."""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from lib.test.evaluation.lasherdataset import LasHeRDataset  # noqa: E402
from lib.test.parameter.tbsi_track import parameters  # noqa: E402
from lib.test.tracker.tbsi_track import TBSITrack  # noqa: E402


def read_frame(seq, frame_num):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][frame_num]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][frame_num]), cv2.COLOR_BGR2RGB)
    return np.concatenate([rgb, tir], axis=2)


def iou_xywh(a, b):
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    inter = max(0.0, min(ax2, bx2) - max(ax, bx)) * max(0.0, min(ay2, by2) - max(ay, by))
    union = max(0.0, ax2 - ax) * max(0.0, ay2 - ay) + max(0.0, bx2 - bx) * max(0.0, by2 - by) - inter
    return inter / union if union else 0.0


def stored_boxes(result_dir, sequence):
    path = Path(result_dir) / (sequence + '.txt')
    rows = []
    with path.open() as handle:
        for line in handle:
            values = [float(x) for x in line.split()[:4]]
            rows.append(values)
    return rows


def run_mode(config, checkpoint, seq, max_frame, mode):
    os.environ['TBSI_INFER_FP16'] = '0'
    if mode == 'cross_scale_1':
        os.environ['TBSI_TEMPLATE_CROSS_SCALE'] = '1.0'
    else:
        os.environ.pop('TBSI_TEMPLATE_CROSS_SCALE', None)
    os.environ.pop('TBSI_FUSION_DIAGNOSTICS', None)
    params = parameters(config)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_test')
    if mode == 'analysis_collector':
        for layer in tracker.network.backbone.tbsi_layers:
            fusion = getattr(layer, 'v2_template_fusion', None)
            if fusion is not None:
                fusion.analysis_enabled = True
                fusion.analysis_collector = lambda **kwargs: None
    tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
    predictions = {}
    for frame_num in range(1, max_frame + 1):
        out = tracker.track(read_frame(seq, frame_num), {
            'sequence_name': seq.name,
            'frame_num': frame_num,
            'gt_bbox': seq.ground_truth_rect[frame_num],
        })
        predictions[frame_num] = [float(x) for x in out['target_bbox']]
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sequence', default='besom3')
    parser.add_argument('--frame', type=int, default=635)
    parser.add_argument('--stored-results', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--modes', default='default,cross_scale_1,analysis_collector')
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    dataset = LasHeRDataset(split='testingset').get_sequence_list()
    sequence = next((item for item in dataset if item.name == args.sequence), None)
    if sequence is None:
        raise KeyError('sequence not found: {}'.format(args.sequence))
    if args.frame >= len(sequence.frames[0]) or args.frame >= len(sequence.ground_truth_rect):
        raise IndexError('frame {} exceeds {}'.format(args.frame, args.sequence))

    stored = stored_boxes(args.stored_results, args.sequence)
    if args.frame >= len(stored):
        raise IndexError('stored result has no 0-based frame {}'.format(args.frame))
    expected = stored[args.frame]
    rows = []
    for mode in tuple(args.modes.split(',')):
        predictions = run_mode(args.config, checkpoint, sequence, args.frame, mode)
        actual = predictions[args.frame]
        rows.append({
            'sequence': args.sequence,
            'frame': args.frame,
            'mode': mode,
            'stored_x': expected[0], 'stored_y': expected[1],
            'stored_w': expected[2], 'stored_h': expected[3],
            'current_x': actual[0], 'current_y': actual[1],
            'current_w': actual[2], 'current_h': actual[3],
            'max_abs_box_diff': float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))),
            'stored_iou_gt': iou_xywh(expected, sequence.ground_truth_rect[args.frame]),
            'current_iou_gt': iou_xywh(actual, sequence.ground_truth_rect[args.frame]),
            'iou_gt_diff': iou_xywh(actual, sequence.ground_truth_rect[args.frame]) - iou_xywh(expected, sequence.ground_truth_rect[args.frame]),
        })
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with output.with_suffix('.json').open('w') as handle:
        json.dump({'experiment': 'MHSA-CAUSAL-PROBE-V2-PARITY', 'config': args.config,
                   'checkpoint': str(checkpoint), 'sequence': args.sequence,
                   'frame': args.frame, 'fp16': False, 'rows': rows}, handle, indent=2)
    os.environ.pop('TBSI_TEMPLATE_CROSS_SCALE', None)
    print(json.dumps({'sequence': args.sequence, 'frame': args.frame, 'rows': rows}, ensure_ascii=False))


if __name__ == '__main__':
    main()
