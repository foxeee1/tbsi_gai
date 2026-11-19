#!/usr/bin/env python3
"""Replay one frozen M1 sequence and locate the first stored-trajectory divergence."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path

# These must be set before importing project modules: tracker construction reads
# TBSI_INFER_FP16 and builds the network during import-driven setup.
os.environ['TBSI_INFER_FP16'] = '0'
os.environ.pop('TBSI_TEMPLATE_CROSS_SCALE', None)
os.environ.pop('TBSI_TEMPLATE_ROUTE_R2T', None)
os.environ.pop('TBSI_TEMPLATE_ROUTE_T2R', None)

import cv2
import numpy as np
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset  # noqa: E402
from lib.test.parameter.tbsi_track import parameters  # noqa: E402
from lib.test.tracker.tbsi_track import TBSITrack  # noqa: E402
from lib.train.data.processing_utils import sample_target  # noqa: E402


def digest(value):
    h = hashlib.sha256()
    if torch.is_tensor(value):
        a = value.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    elif isinstance(value, np.ndarray):
        h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    elif isinstance(value, (list, tuple)):
        for x in value: h.update(digest(x).encode())
    elif isinstance(value, dict):
        for k in sorted(value): h.update(str(k).encode()); h.update(digest(value[k]).encode())
    else:
        h.update(repr(value).encode())
    return h.hexdigest()


def read_modalities(seq, frame_num):
    rgb_path, tir_path = seq.frames[0][frame_num], seq.frames[1][frame_num]
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    tir_bgr = cv2.imread(tir_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None or tir_bgr is None:
        raise IOError('failed to read frame {}'.format(frame_num))
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(tir_bgr, cv2.COLOR_BGR2RGB)
    return (np.concatenate([rgb, tir], axis=2), rgb_path, tir_path, rgb, tir)


def load_stored(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            values = [float(x) for x in line.split()[:4]]
            if len(values) != 4:
                raise ValueError('invalid stored box line: {}'.format(line.strip()))
            rows.append(values)
    return rows


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = [float(x) for x in b]
    ax2, ay2 = ax + max(0., aw), ay + max(0., ah)
    bx2, by2 = bx + max(0., bw), by + max(0., bh)
    inter = max(0., min(ax2, bx2) - max(ax, bx)) * max(0., min(ay2, by2) - max(ay, by))
    union = max(0., ax2 - ax) * max(0., ay2 - ay) + max(0., bx2 - bx) * max(0., by2 - by) - inter
    return inter / union if union else 0.


def rng_snapshot():
    return {'python': repr(random.getstate()), 'numpy': repr(np.random.get_state()),
            'torch': digest(torch.get_rng_state()),
            'cuda': [digest(x) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--stored-results', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--target-frame', type=int, default=-1)
    ap.add_argument('--collector', action='store_true')
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    stored_path = Path(args.stored_results).resolve() / (args.sequence + '.txt')
    stored = load_stored(stored_path)
    dataset = LasHeRDataset(split='testingset').get_sequence_list()
    seq = next((x for x in dataset if x.name == args.sequence), None)
    if seq is None:
        raise KeyError(args.sequence)
    target = len(stored) - 1 if args.target_frame < 0 else args.target_frame
    if target >= len(seq.frames[0]) or target >= len(seq.ground_truth_rect) or target >= len(stored):
        raise IndexError('target frame {} out of bounds'.format(target))

    params = parameters(args.config)
    params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_test')
    if args.collector:
        for layer in tracker.network.backbone.tbsi_layers:
            fusion = getattr(layer, 'v2_template_fusion', None)
            if fusion is not None:
                fusion.analysis_enabled = True
                fusion.analysis_collector = lambda **kwargs: None
    image0, rgb0_path, tir0_path, rgb0, tir0 = read_modalities(seq, 0)
    init_info = {'init_bbox': seq.ground_truth_rect[0]}
    tracker.initialize(image0, init_info)

    rows = []
    init_actual = np.asarray(seq.ground_truth_rect[0], dtype=np.float64)
    init_stored = np.asarray(stored[0], dtype=np.float64)
    first = None
    rows.append({'seq': seq.name, 'frame': 0, 'stored_row': 0,
                 'stored_x': stored[0][0], 'stored_y': stored[0][1], 'stored_w': stored[0][2], 'stored_h': stored[0][3],
                 'replay_x': init_actual[0], 'replay_y': init_actual[1], 'replay_w': init_actual[2], 'replay_h': init_actual[3],
                 'max_abs_diff': float(np.max(np.abs(init_actual - init_stored))),
                 'rounded_box_equal': bool(np.array_equal(np.rint(init_actual).astype(np.int64), np.rint(init_stored).astype(np.int64))),
                 'stored_iou_gt': iou_xywh(init_stored, seq.ground_truth_rect[0]),
                 'replay_iou_gt': iou_xywh(init_actual, seq.ground_truth_rect[0]),
                 'image_rgb_sha256': digest(rgb0), 'image_tir_sha256': digest(tir0),
                 'tracker_state_hash': digest(tracker.state), 'template_hash': digest(tracker.z_dict1.tensors),
                 'search_crop_hash': '', 'input_dtype': str(image0.dtype), 'use_fp16': str(tracker.use_fp16),
                 'rng_torch': rng_snapshot()['torch'], 'info_keys': 'init_bbox'})

    # Mirrors Tracker._track_sequence: initialize() returns None for this
    # tracker, so the first frame's previous_output is an empty dict.
    previous_output = {}
    for frame_num in range(1, target + 1):
        image, rgb_path, tir_path, rgb, tir = read_modalities(seq, frame_num)
        state_before = list(tracker.state)
        search_crop, search_resize, _ = sample_target(image, state_before, tracker.params.search_factor,
                                                       output_sz=tracker.params.search_size)
        info = {'previous_output': previous_output, 'sequence_name': seq.name,
                'frame_num': frame_num, 'gt_bbox': seq.ground_truth_rect[frame_num]}
        out = tracker.track(image, info)
        actual = np.asarray(out['target_bbox'], dtype=np.float64)
        expected = np.asarray(stored[frame_num], dtype=np.float64)
        err = float(np.max(np.abs(actual - expected)))
        rounded_equal = bool(np.array_equal(np.rint(actual).astype(np.int64), np.rint(expected).astype(np.int64)))
        if first is None and err > 1.0:
            first = frame_num
        rows.append({'seq': seq.name, 'frame': frame_num, 'stored_row': frame_num,
                     'stored_x': expected[0], 'stored_y': expected[1], 'stored_w': expected[2], 'stored_h': expected[3],
                     'replay_x': actual[0], 'replay_y': actual[1], 'replay_w': actual[2], 'replay_h': actual[3],
                     'max_abs_diff': err, 'rounded_box_equal': rounded_equal,
                     'stored_iou_gt': iou_xywh(expected, seq.ground_truth_rect[frame_num]),
                     'replay_iou_gt': iou_xywh(actual, seq.ground_truth_rect[frame_num]),
                     'image_rgb_sha256': digest(rgb), 'image_tir_sha256': digest(tir),
                     'tracker_state_hash': digest(tracker.state), 'template_hash': digest(tracker.z_dict1.tensors),
                     'search_crop_hash': digest(search_crop), 'input_dtype': str(image.dtype),
                     'use_fp16': str(tracker.use_fp16), 'rng_torch': rng_snapshot()['torch'],
                     'info_keys': ','.join(sorted(info.keys()))})
        previous_output = out

    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with output.with_suffix('.json').open('w') as handle:
        json.dump({'experiment': 'MHSA-REPLAY-FIRST-DIVERGENCE', 'sequence': seq.name,
                   'target_frame': target, 'first_divergence_gt_1px': first,
                   'config': args.config, 'checkpoint': str(checkpoint), 'fp16_env_before_import': '0',
                   'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'stored_rows': len(stored),
                   'initial_row_equals_gt0': bool(np.array_equal(np.rint(init_actual).astype(np.int64), np.rint(init_stored).astype(np.int64))),
                   'rows': len(rows)}, handle, indent=2)
    print(json.dumps({'sequence': seq.name, 'target_frame': target,
                      'first_divergence_gt_1px': first, 'rows': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
