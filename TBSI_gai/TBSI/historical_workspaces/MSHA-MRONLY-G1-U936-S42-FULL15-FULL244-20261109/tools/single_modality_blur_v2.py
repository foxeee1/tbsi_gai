#!/usr/bin/env python
"""New-sequence RGB Gaussian-blur revalidation (00 vs 10 only)."""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.tracker.tbsi_track import TBSITrack
from lib.test.utils.params import TrackerParams
from synthetic_joint_interference_v1 import image_loader, load_train_sequence, rng_snapshot, restore_rng, sha256_bytes


EXP = 'single_modality_blur_v2'


def box_iou(a, b):
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ix0, iy0, ix1, iy1 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def size_bin(gt, width, height):
    fraction = float(gt[2] * gt[3]) / float(max(width * height, 1))
    if fraction < 0.01:
        return 'small'
    if fraction < 0.05:
        return 'medium'
    return 'large'


def build_params(config_name, checkpoint):
    config_path = os.path.join(ROOT, 'experiments/tbsi_track', config_name + '.yaml')
    update_config_from_file(config_path)
    params = TrackerParams()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = checkpoint
    params.save_all_boxes = False
    params.debug = 0
    return params, config_path


def degrade_rgb(image):
    return np.clip(cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0),
                   0, 255).astype(np.uint8)


def replay(tracker, rgb_frames, tir_frames, gt_boxes, event, rgb_event, rng, horizon):
    restore_rng(rng)
    tracker.initialize(np.concatenate([rgb_frames[0], tir_frames[0]], axis=2), {'init_bbox': gt_boxes[0]})
    outputs = []
    for frame_num in range(1, event['frame'] + horizon + 1):
        rgb = rgb_frames[frame_num]
        if frame_num == event['frame'] and rgb_event is not None:
            rgb = rgb_event
        out = tracker.track(np.concatenate([rgb, tir_frames[frame_num]], axis=2),
                            {'gt_bbox': gt_boxes[frame_num], 'sequence_name': event['seq'],
                             'frame_num': frame_num})
        if frame_num >= event['frame']:
            pred = [float(x) for x in out['target_bbox']]
            outputs.append({'offset': frame_num - event['frame'], 'iou': box_iou(pred, gt_boxes[frame_num]),
                            'pred_sha256': sha256_bytes(json.dumps(pred).encode())})
    return outputs


def write_csv(path, rows):
    fields = ['event_id', 'seq', 'frame', 'condition', 'offset', 'iou', 'error', 'gt_area_frac', 'size_bin',
              'pred_sha256', 'rng_sha256', 'status']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    manifest = json.load(open(args.manifest))
    events = manifest['events'][:8] if args.mode == 'smoke' else manifest['events']
    settings = env_settings()
    sequences = {name: load_train_sequence(settings.lasher_path, name)
                 for name in sorted(set(event['seq'] for event in events))}
    params, config_path = build_params(args.config, args.checkpoint)
    tracker = TBSITrack(params, EXP)
    rng = rng_snapshot()
    rows = []
    skipped = []
    for index, event in enumerate(events):
        sequence = sequences[event['seq']]
        rgb_frames = [image_loader(path) for path in sequence.frames[0]]
        tir_frames = [image_loader(path) for path in sequence.frames[1]]
        n = min(len(rgb_frames), len(tir_frames), len(sequence.ground_truth_rect))
        rgb_frames, tir_frames = rgb_frames[:n], tir_frames[:n]
        gt_boxes = [[float(x) for x in box] for box in sequence.ground_truth_rect[:n]]
        if event['frame'] + args.horizon >= n:
            skipped.append({'event_id': event['event_id'], 'seq': event['seq'], 'reason': 'insufficient_future_frames'})
            continue
        h, w = rgb_frames[event['frame']].shape[:2]
        gt = gt_boxes[event['frame']]
        frac = float(gt[2] * gt[3]) / float(max(w * h, 1))
        bad = degrade_rgb(rgb_frames[event['frame']])
        for condition, rgb_event in (('00', None), ('10', bad)):
            for point in replay(tracker, rgb_frames, tir_frames, gt_boxes, event, rgb_event, rng, args.horizon):
                rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': event['frame'],
                             'condition': condition, 'offset': point['offset'], 'iou': point['iou'],
                             'error': 1.0 - point['iou'], 'gt_area_frac': frac,
                             'size_bin': size_bin(gt, w, h), 'pred_sha256': point['pred_sha256'],
                             'rng_sha256': sha256_bytes(repr(rng).encode()), 'status': 'OK'})
        if index % 4 == 0:
            print('processed {}/{} events'.format(index + 1, len(events)), flush=True)
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, 'blur_event_results.csv'), rows)
    with open(os.path.join(out, 'skipped_events.json'), 'w') as handle:
        json.dump(skipped, handle, indent=2)
    with open(os.path.join(out, 'metadata.json'), 'w') as handle:
        json.dump({'experiment': EXP, 'mode': args.mode, 'manifest': os.path.abspath(args.manifest),
                   'n_events_requested': len(events), 'n_rows': len(rows), 'horizon': args.horizon,
                   'conditions': ['00', '10'], 'blur_sigma': 3.0, 'config': args.config,
                   'config_sha256': hashlib.sha256(open(config_path, 'rb').read()).hexdigest(),
                   'checkpoint': args.checkpoint,
                   'checkpoint_sha256': hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest(),
                   'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'status': 'COMPLETED'},
                  handle, indent=2)
    if skipped:
        raise RuntimeError('skipped events: {}'.format(skipped))
    print('COMPLETED {}'.format(args.mode), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('smoke', 'batch'), default='smoke')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--config', default='v1.0.0-单卡3090-48G-等效全局128基线')
    ap.add_argument('--checkpoint', default=os.path.join(ROOT, 'output/experiments/v1.0.0-单卡3090-48G-等效全局128基线/checkpoints/TBSITrack_ep0015.pth.tar'))
    ap.add_argument('--horizon', type=int, default=5)
    ap.add_argument('--min-free-gb', type=float, default=8.0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if shutil.disk_usage(ROOT).free / 2 ** 30 < args.min_free_gb:
        raise RuntimeError('insufficient free disk')
    lock = os.path.join(os.path.abspath(args.output_dir), '.lock')
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    if os.path.exists(lock):
        raise RuntimeError('lock exists: {}'.format(lock))
    open(lock, 'w').write(str(os.getpid()))
    try:
        run(args)
    finally:
        if os.path.exists(lock):
            os.unlink(lock)


if __name__ == '__main__':
    main()
