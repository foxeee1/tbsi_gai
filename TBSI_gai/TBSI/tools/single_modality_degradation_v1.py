#!/usr/bin/env python
"""GT-evaluable single-modality degradation four-cell diagnostic.

No blind labels, no training, and no test-set-driven severity selection.
The fixed event manifest is inherited from SJI-v1 only as a pre-registered
24-train-sequence/72-frame development manifest; its outcomes are not used.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from lib.test.evaluation.environment import env_settings
from lib.test.tracker.tbsi_track import TBSITrack
from lib.test.utils.params import TrackerParams
from lib.config.tbsi_track.config import cfg, update_config_from_file
from synthetic_joint_interference_v1 import image_loader, load_train_sequence, rng_snapshot, restore_rng, sha256_bytes, write_csv

EXP = 'single_modality_degradation_v1'
V1 = os.path.join(ROOT, 'output/experiments/synthetic_joint_interference_v1/batch')
SUITES = ('blur', 'lowlight', 'contrast')
CONDITIONS = ('00', '10', '01', '11')


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_events():
    selected = {x['event_id']: x for x in json.load(open(os.path.join(V1, 'selected_events.json')))}
    summary = read_csv(os.path.join(V1, 'event_summary.csv'))
    return [{'event_id': r['event_id'], 'seq': r['seq'], 'frame': int(r['frame']),
             'gt': selected[r['event_id']]['gt']} for r in summary]


def build_params(checkpoint, config_name):
    path = os.path.join(ROOT, 'experiments/tbsi_track', config_name + '.yaml')
    update_config_from_file(path)
    params = TrackerParams(); params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR; params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR; params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = checkpoint; params.save_all_boxes = False; params.debug = 0
    return params, path


def degrade(image, suite):
    image = image.astype(np.float32)
    if suite == 'blur':
        out = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0, sigmaY=3.0)
    elif suite == 'lowlight':
        out = image * 0.35
    elif suite == 'contrast':
        out = 127.5 + (image - 127.5) * 0.40
    else:
        raise ValueError(suite)
    return np.clip(out, 0, 255).astype(np.uint8)


def replay(tracker, rgb_frames, tir_frames, gt_boxes, event, rgb_event, tir_event, rng, horizon):
    restore_rng(rng)
    tracker.initialize(np.concatenate([rgb_frames[0], tir_frames[0]], axis=2), {'init_bbox': gt_boxes[0]})
    outputs = []
    for frame_num in range(1, event['frame'] + horizon + 1):
        rgb, tir = rgb_frames[frame_num], tir_frames[frame_num]
        if frame_num == event['frame']:
            if rgb_event is not None: rgb = rgb_event
            if tir_event is not None: tir = tir_event
        out = tracker.track(np.concatenate([rgb, tir], axis=2),
                            {'gt_bbox': gt_boxes[frame_num], 'sequence_name': event['seq'],
                             'frame_num': frame_num})
        if frame_num >= event['frame']:
            pred = [float(x) for x in out['target_bbox']]
            outputs.append({'offset': frame_num - event['frame'], 'pred': pred,
                            'iou': box_iou(pred, gt_boxes[frame_num]),
                            'pred_sha256': sha256_bytes(json.dumps(pred).encode())})
    return outputs


def box_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by); ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def run_stage(args):
    events = load_events()
    if args.mode == 'smoke': events = events[:8]
    settings = env_settings()
    names = sorted(set(x['seq'] for x in events))
    seq_map = {name: load_train_sequence(settings.lasher_path, name) for name in names}
    params, config_path = build_params(args.checkpoint, args.config)
    tracker = TBSITrack(params, EXP)
    rng = rng_snapshot()
    rows = []
    skipped = []
    for index, event in enumerate(events):
        seq = seq_map[event['seq']]
        rgb_frames = [image_loader(p) for p in seq.frames[0]]
        tir_frames = [image_loader(p) for p in seq.frames[1]]
        n = min(len(rgb_frames), len(tir_frames), len(seq.ground_truth_rect))
        rgb_frames, tir_frames = rgb_frames[:n], tir_frames[:n]
        gt_boxes = [list(map(float, x)) for x in seq.ground_truth_rect[:n]]
        if event['frame'] + args.horizon >= n:
            skipped.append({'event_id': event['event_id'], 'seq': event['seq'], 'reason': 'insufficient_future_frames'})
            continue
        rgb = rgb_frames[event['frame']]; tir = tir_frames[event['frame']]
        for suite in SUITES:
            rgb_bad = degrade(rgb, suite); tir_bad = degrade(tir, suite)
            condition_inputs = {'00': (None, None), '10': (rgb_bad, None),
                                '01': (None, tir_bad), '11': (rgb_bad, tir_bad)}
            for condition in CONDITIONS:
                result = replay(tracker, rgb_frames, tir_frames, gt_boxes, event,
                                condition_inputs[condition][0], condition_inputs[condition][1], rng, args.horizon)
                for point in result:
                    rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': event['frame'],
                                 'suite': suite, 'condition': condition, 'offset': point['offset'],
                                 'iou': point['iou'], 'error': 1.0 - point['iou'],
                                 'pred_sha256': point['pred_sha256'], 'rng_sha256': sha256_bytes(repr(rng).encode()),
                                 'status': 'OK'})
            # 00 must be replayed identically to the unmodified baseline path by construction.
            if index % 4 == 0:
                print('processed {}/{} events'.format(index + 1, len(events)), flush=True)
    out = os.path.abspath(args.output_dir); os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, 'degradation_event_results.csv'), rows,
              ['event_id','seq','frame','suite','condition','offset','iou','error','pred_sha256','rng_sha256','status'])
    with open(os.path.join(out, 'skipped_events.json'), 'w') as f: json.dump(skipped, f, indent=2)
    with open(os.path.join(out, 'metadata.json'), 'w') as f:
        json.dump({'experiment': EXP, 'mode': args.mode, 'source_manifest': os.path.join(V1, 'selected_events.json'),
                   'n_events_requested': len(events), 'n_rows': len(rows), 'horizon': args.horizon,
                   'suites': SUITES, 'conditions': CONDITIONS, 'severity': {'blur_sigma': 3.0, 'lowlight_factor': 0.35, 'contrast_factor': 0.40},
                   'config': args.config, 'config_sha256': sha256_bytes(open(config_path,'rb').read()),
                   'checkpoint': args.checkpoint, 'checkpoint_sha256': sha256_bytes(open(args.checkpoint,'rb').read()),
                   'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'status': 'COMPLETED'}, f, indent=2)
    if args.mode == 'smoke' and skipped:
        raise RuntimeError('smoke skipped events: {}'.format(skipped))
    print('COMPLETED {}'.format(args.mode), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('smoke','batch'), default='smoke')
    ap.add_argument('--output-dir', default=os.path.join(ROOT, 'output/experiments', EXP))
    ap.add_argument('--config', default='v1.0.0-单卡3090-48G-等效全局128基线')
    ap.add_argument('--checkpoint', default=os.path.join(ROOT, 'output/experiments/v1.0.0-单卡3090-48G-等效全局128基线/checkpoints/TBSITrack_ep0015.pth.tar'))
    ap.add_argument('--horizon', type=int, default=5)
    ap.add_argument('--min-free-gb', type=float, default=8.0)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    if shutil.disk_usage(ROOT).free / 2**30 < args.min_free_gb: raise RuntimeError('insufficient free disk')
    lock = os.path.join(os.path.abspath(args.output_dir), '.lock'); os.makedirs(os.path.dirname(lock), exist_ok=True)
    if os.path.exists(lock): raise RuntimeError('lock exists: {}'.format(lock))
    open(lock, 'w').write(str(os.getpid()))
    try: run_stage(args)
    finally:
        if os.path.exists(lock): os.unlink(lock)


if __name__ == '__main__': main()
