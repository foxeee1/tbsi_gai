#!/usr/bin/env python
"""Synthetic joint RGB/TIR interference diagnostic (RPP-independent).

This diagnostic never reads blind-review labels. It uses only LasHeR image files
and init.txt boxes, and writes to a dedicated experiment directory.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
import re
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.tracker.tbsi_track import TBSITrack
from lib.test.utils.params import TrackerParams


EXPERIMENT = 'synthetic_joint_interference_v1'
CONDITIONS = ('00', '10', '01', '11', 'bg10', 'bg01', 'bg11')
CORE_CONDITIONS = ('00', '10', '01', '11')


def image_loader(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError('failed to read {}'.format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class SimpleSequence(object):
    def __init__(self, name, frames_rgb, frames_tir, ground_truth_rect):
        self.name = name
        self.frames = [frames_rgb, frames_tir]
        self.ground_truth_rect = ground_truth_rect


class SimpleSequenceList(object):
    def __init__(self, names):
        self.sequence_list = names


def frame_sort_key(path):
    stem = os.path.basename(path)
    numbers = re.findall(r'\d+', stem)
    return int(numbers[-1]) if numbers else -1


def load_train_sequence(data_root, name):
    base = os.path.join(data_root, 'trainingset', name)
    rgb = sorted([os.path.join(base, 'visible', x) for x in os.listdir(os.path.join(base, 'visible'))
                  if x.lower().endswith('.jpg')], key=frame_sort_key)
    tir = sorted([os.path.join(base, 'infrared', x) for x in os.listdir(os.path.join(base, 'infrared'))
                  if x.lower().endswith('.jpg')], key=frame_sort_key)
    gt = np.loadtxt(os.path.join(base, 'init.txt'), delimiter=',', dtype=np.float64).reshape(-1, 4)
    return SimpleSequence(name, rgb, tir, gt)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def array_hash(array):
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def json_hash(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, default=str).encode('utf-8'))


def rng_snapshot():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(snapshot):
    random.setstate(snapshot['python'])
    np.random.set_state(snapshot['numpy'])
    torch.set_rng_state(snapshot['torch_cpu'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot['torch_cuda'])


def rng_hash(snapshot):
    parts = [repr(snapshot['python']).encode('utf-8'),
             snapshot['numpy'][1].tobytes(), snapshot['torch_cpu'].numpy().tobytes()]
    parts.extend(x.cpu().numpy().tobytes() for x in snapshot['torch_cuda'])
    return sha256_bytes(b''.join(parts))


def box_iou(a, b):
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center_distance(a, b):
    return float(np.linalg.norm(np.array([a[0] + a[2] / 2, a[1] + a[3] / 2]) -
                                np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])))


def clip_box(box, width, height):
    x, y, w, h = [float(v) for v in box]
    x = max(0.0, min(x, width - 1.0))
    y = max(0.0, min(y, height - 1.0))
    w = min(w, width - x)
    h = min(h, height - y)
    return [x, y, w, h]


def int_box(box, width, height):
    x, y, w, h = clip_box(box, width, height)
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + w)), int(round(y + h))
    x1, y1 = max(x0 + 1, min(width, x1)), max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def choose_destination(gt, width, height, search_factor=4.0):
    x, y, w, h = [float(v) for v in gt]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(16.0, (w * h) ** 0.5 * search_factor)
    gap = max(8.0, 0.35 * min(w, h))
    candidates = [(x + w + gap, cy - h / 2.0), (x - w - gap, cy - h / 2.0),
                  (cx - w / 2.0, y + h + gap), (cx - w / 2.0, y - h - gap),
                  (x + w + gap, y + h + gap), (x - w - gap, y - h - gap)]
    for dx, dy in candidates:
        box = [dx, dy, w, h]
        if dx >= 0 and dy >= 0 and dx + w <= width and dy + h <= height:
            dcx, dcy = dx + w / 2.0, dy + h / 2.0
            if np.hypot(dcx - cx, dcy - cy) <= side / 2.0 - max(w, h) / 2.0:
                return box
    return None


def patch_image(image, source_box, dest_box, alpha=0.88):
    """Copy source content with a narrow feathered edge; return image and mask."""
    h, w = image.shape[:2]
    sx0, sy0, sx1, sy1 = int_box(source_box, w, h)
    dx0, dy0, dx1, dy1 = int_box(dest_box, w, h)
    patch = image[sy0:sy1, sx0:sx1]
    if patch.size == 0:
        raise ValueError('empty source patch')
    patch = cv2.resize(patch, (dx1 - dx0, dy1 - dy0), interpolation=cv2.INTER_LINEAR)
    ph, pw = patch.shape[:2]
    feather = np.ones((ph, pw), np.float32) * float(alpha)
    edge = max(1, min(8, ph // 8, pw // 8))
    ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
    feather[:edge, :] *= ramp[:, None]
    feather[-edge:, :] *= ramp[::-1, None]
    feather[:, :edge] *= ramp[None, :]
    feather[:, -edge:] *= ramp[::-1][None, :]
    out = image.copy()
    base = out[dy0:dy1, dx0:dx1].astype(np.float32)
    out[dy0:dy1, dx0:dx1] = np.clip(base * (1.0 - feather[..., None]) +
                                     patch.astype(np.float32) * feather[..., None], 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy0:dy1, dx0:dx1] = 1
    return out, mask


def make_background_control(image, gt, dest_box):
    h, w = image.shape[:2]
    candidates = [[0, 0, dest_box[2], dest_box[3]],
                  [w - dest_box[2] - 1, 0, dest_box[2], dest_box[3]],
                  [0, h - dest_box[3] - 1, dest_box[2], dest_box[3]],
                  [w - dest_box[2] - 1, h - dest_box[3] - 1, dest_box[2], dest_box[3]]]
    for source in candidates:
        source = clip_box(source, w, h)
        if box_iou(source, gt) == 0 and box_iou(source, dest_box) == 0:
            return patch_image(image, source, dest_box, alpha=0.88), source
    raise ValueError('no background source patch')


def load_cfg(config_name):
    config_path = os.path.join(ROOT, 'experiments', 'tbsi_track', config_name + '.yaml')
    update_config_from_file(config_path)
    return config_path


def build_params(config_name, checkpoint):
    params = TrackerParams()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = checkpoint
    params.save_all_boxes = False
    params.debug = 0
    return params


def make_conditions(rgb, tir, gt, dest_rgb, dest_tir):
    rgb_copy, rgb_mask = patch_image(rgb, gt, dest_rgb)
    tir_copy, tir_mask = patch_image(tir, gt, dest_tir)
    rgb_bg, rgb_source = make_background_control(rgb, gt, dest_rgb)
    tir_bg, tir_source = make_background_control(tir, gt, dest_tir)
    return {
        '00': (rgb, tir, np.zeros(rgb.shape[:2], np.uint8), np.zeros(tir.shape[:2], np.uint8)),
        '10': (rgb_copy, tir, rgb_mask, np.zeros(tir.shape[:2], np.uint8)),
        '01': (rgb, tir_copy, np.zeros(rgb.shape[:2], np.uint8), tir_mask),
        '11': (rgb_copy, tir_copy, rgb_mask, tir_mask),
        'bg10': (rgb_bg[0], tir, rgb_bg[1], np.zeros(tir.shape[:2], np.uint8)),
        'bg01': (rgb, tir_bg[0], np.zeros(rgb.shape[:2], np.uint8), tir_bg[1]),
        'bg11': (rgb_bg[0], tir_bg[0], rgb_bg[1], tir_bg[1]),
    }, {'rgb_source': rgb_source, 'tir_source': tir_source}


def run_to_frame(tracker, frames_rgb, frames_tir, gt_boxes, target_frame, conditions, event, rng):
    restore_rng(rng)
    tracker.initialize(np.concatenate([frames_rgb[0], frames_tir[0]], axis=2),
                       {'init_bbox': gt_boxes[0]})
    prediction = list(gt_boxes[0])
    input_hash = hashlib.sha256()
    for frame_num in range(1, target_frame + 1):
        rgb, tir, rgb_mask, tir_mask = conditions.get(frame_num, (frames_rgb[frame_num],
                                                                    frames_tir[frame_num],
                                                                    None, None))
        if rgb_mask is None:
            rgb, tir = frames_rgb[frame_num], frames_tir[frame_num]
        input_hash.update(np.ascontiguousarray(rgb).tobytes())
        input_hash.update(np.ascontiguousarray(tir).tobytes())
        info = {'gt_bbox': gt_boxes[frame_num], 'sequence_name': event['seq'],
                'frame_num': frame_num}
        out = tracker.track(np.concatenate([rgb, tir], axis=2), info)
        prediction = [float(x) for x in out['target_bbox']]
    return prediction, input_hash.hexdigest()


def draw_failure(out_path, rgb, gt, dest, preds):
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    def rect(box, color, label):
        x, y, w, h = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, label, (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    rect(gt, (0, 255, 0), 'GT')
    rect(dest, (255, 0, 255), 'D')
    colors = [(255, 255, 0), (0, 128, 255), (255, 128, 0), (0, 0, 255)]
    for (condition, pred), color in zip(preds.items(), colors):
        rect(pred, color, condition)
    cv2.imwrite(out_path, canvas)


def write_csv(path, rows, fields):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_sequences(dataset, blind_sequences, count, seed):
    names = [s for s in dataset.sequence_list if s not in blind_sequences]
    rng = random.Random(seed)
    rng.shuffle(names)
    return sorted(names[:count])


def event_frames(length, count=3, max_frame=60):
    upper = min(length - 1, max_frame)
    if upper < 3:
        return []
    values = np.linspace(max(2, upper // 5), upper, count, dtype=int).tolist()
    return sorted(set(int(x) for x in values if x > 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'batch'], default='smoke')
    parser.add_argument('--config', default='v1.0.0-单卡3090-48G-等效全局128基线')
    parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'output/experiments',
                      'v1.0.0-单卡3090-48G-等效全局128基线/checkpoints/TBSITrack_ep0015.pth.tar'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sequence-count', type=int, default=24)
    parser.add_argument('--max-frame', type=int, default=60)
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'output/experiments', EXPERIMENT))
    parser.add_argument('--min-free-gb', type=float, default=8.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    free_gb = shutil.disk_usage(args.output_dir if os.path.exists(args.output_dir) else ROOT).free / 2**30
    if free_gb < args.min_free_gb:
        raise RuntimeError('refusing to start: only {:.2f} GiB free, threshold {:.2f}'.format(free_gb, args.min_free_gb))
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError('missing checkpoint: {}'.format(args.checkpoint))
    if not os.path.isfile(os.path.join(ROOT, 'experiments/tbsi_track', args.config + '.yaml')):
        raise RuntimeError('missing config')

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    lock = os.path.join(out_dir, '.lock')
    if os.path.exists(lock):
        raise RuntimeError('experiment lock exists: {}'.format(lock))
    with open(lock, 'w') as f:
        f.write('{}\n'.format(os.getpid()))
    try:
        config_path = load_cfg(args.config)
        settings = env_settings()
        data_root = settings.lasher_path
        with open(os.path.join(data_root, 'trainingsetList.txt')) as f:
            train_names = [x.strip() for x in f if x.strip()]
        blind_meta = os.path.join(ROOT, 'output/experiments/false_consensus_week1/metadata.json')
        blind = set(json.load(open(blind_meta)).get('sequences', [])) if os.path.isfile(blind_meta) else set()
        seq_names = select_sequences(SimpleSequenceList(train_names), blind,
                                     8 if args.mode == 'smoke' else args.sequence_count, args.seed)
        seq_map = {name: load_train_sequence(data_root, name) for name in seq_names}
        rng = rng_snapshot()
        params = build_params(args.config, args.checkpoint)
        tracker = TBSITrack(params, 'synthetic_joint_interference_v1')
        rows = []
        sequence_rows = []
        failures = []
        event_counter = 0
        selected_events = []
        for seq_name in seq_names:
            seq = seq_map[seq_name]
            frames_rgb = [image_loader(p) for p in seq.frames[0]]
            frames_tir = [image_loader(p) for p in seq.frames[1]]
            n = min(len(frames_rgb), len(frames_tir), len(seq.ground_truth_rect))
            frames_rgb, frames_tir = frames_rgb[:n], frames_tir[:n]
            gt_boxes = [list(map(float, x)) for x in seq.ground_truth_rect[:n]]
            for frame_num in event_frames(n, count=3, max_frame=args.max_frame):
                gt = gt_boxes[frame_num]
                dest = choose_destination(gt, frames_rgb[frame_num].shape[1], frames_rgb[frame_num].shape[0], cfg.TEST.SEARCH_FACTOR)
                if dest is None:
                    failures.append({'seq': seq_name, 'frame': frame_num, 'error': 'no_destination'})
                    continue
                dest_tir = [dest[0] * frames_tir[frame_num].shape[1] / frames_rgb[frame_num].shape[1],
                            dest[1] * frames_tir[frame_num].shape[0] / frames_rgb[frame_num].shape[0],
                            dest[2] * frames_tir[frame_num].shape[1] / frames_rgb[frame_num].shape[1],
                            dest[3] * frames_tir[frame_num].shape[0] / frames_rgb[frame_num].shape[0]]
                if box_iou(dest, gt) != 0 or box_iou(dest_tir, gt) != 0:
                    failures.append({'seq': seq_name, 'frame': frame_num, 'error': 'destination_overlaps_gt'})
                    continue
                selected_events.append({'event_id': 'SJI-{:05d}'.format(event_counter), 'seq': seq_name,
                                        'frame': frame_num, 'gt': gt, 'dest_rgb': dest, 'dest_tir': dest_tir})
                event_counter += 1
        with open(os.path.join(out_dir, 'selected_events.json'), 'w') as f:
            json.dump(selected_events, f, indent=2)
        if args.mode == 'smoke':
            selected_events = selected_events[:8]
        # Grouping keeps the test deterministic while avoiding repeated disk I/O.
        for event in selected_events:
            seq = seq_map[event['seq']]
            frames_rgb = [image_loader(p) for p in seq.frames[0]]
            frames_tir = [image_loader(p) for p in seq.frames[1]]
            gt_boxes = [list(map(float, x)) for x in seq.ground_truth_rect]
            frame = event['frame']
            conds, control_meta = make_conditions(frames_rgb[frame], frames_tir[frame], event['gt'],
                                                   event['dest_rgb'], event['dest_tir'])
            # Only the event frame is modified; all prior frames are byte-identical.
            cond_frames = {name: {frame: (vals[0], vals[1], vals[2], vals[3])}
                           for name, vals in conds.items()}
            event_outputs = OrderedDict()
            baseline_pred, baseline_input_hash = run_to_frame(
                tracker, frames_rgb, frames_tir, gt_boxes, frame, {}, event, rng)
            for condition in CONDITIONS:
                pred, inp_hash = run_to_frame(tracker, frames_rgb, frames_tir, gt_boxes, frame,
                                               cond_frames[condition], event, rng)
                event_outputs[condition] = pred
                iou = box_iou(pred, event['gt'])
                distractor_hit = center_distance(pred, event['dest_rgb']) <= max(event['dest_rgb'][2], event['dest_rgb'][3])
                rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': frame,
                             'condition': condition, 'gt_x': event['gt'][0], 'gt_y': event['gt'][1],
                             'gt_w': event['gt'][2], 'gt_h': event['gt'][3], 'pred_x': pred[0],
                             'pred_y': pred[1], 'pred_w': pred[2], 'pred_h': pred[3], 'iou': iou,
                             'error': 1.0 - iou, 'distractor_hit': int(distractor_hit),
                             'input_sha256': inp_hash, 'prediction_sha256': json_hash(pred),
                             'rng_sha256': rng_hash(rng), 'status': 'OK'})
            parity_ok = (np.array_equal(np.asarray(baseline_pred), np.asarray(event_outputs['00'])) and
                         baseline_input_hash == next(r['input_sha256'] for r in rows[-len(CONDITIONS):]
                                                     if r['condition'] == '00'))
            if args.mode == 'smoke' and not parity_ok:
                raise RuntimeError('smoke failed: 00 differs from unmodified baseline for {}'.format(event['event_id']))
            e = {c: 1.0 - box_iou(event_outputs[c], event['gt']) for c in CORE_CONDITIONS}
            interaction = e['11'] - e['10'] - e['01'] + e['00']
            for row in rows[-len(CONDITIONS):]:
                row['interaction_I'] = interaction
                row['baseline_00_parity'] = int(parity_ok)
            if args.mode == 'smoke':
                # Order invariance: rerun core conditions in reverse order from identical RNG.
                reverse = {}
                for condition in reversed(CORE_CONDITIONS):
                    pred, _ = run_to_frame(tracker, frames_rgb, frames_tir, gt_boxes, frame,
                                           cond_frames[condition], event, rng)
                    reverse[condition] = pred
                order_ok = all(np.array_equal(np.asarray(reverse[c]), np.asarray(event_outputs[c]))
                               for c in CORE_CONDITIONS)
                for row in rows[-len(CONDITIONS):]:
                    row['order_invariant'] = int(order_ok)
                if not order_ok:
                    raise RuntimeError('smoke failed: condition order changed predictions for {}'.format(event['event_id']))
            bad = interaction > 0.05 or box_iou(event_outputs['11'], event['gt']) + 0.20 < box_iou(event_outputs['00'], event['gt'])
            if bad:
                path = os.path.join(out_dir, 'failure_examples', event['event_id'] + '.jpg')
                os.makedirs(os.path.dirname(path), exist_ok=True)
                draw_failure(path, frames_rgb[frame], event['gt'], event['dest_rgb'],
                             OrderedDict((c, event_outputs[c]) for c in CORE_CONDITIONS))
            print('{} {} frame={} I={:.6f}'.format(event['event_id'], event['seq'], frame, interaction), flush=True)
        fields = ['event_id', 'seq', 'frame', 'condition', 'gt_x', 'gt_y', 'gt_w', 'gt_h',
                  'pred_x', 'pred_y', 'pred_w', 'pred_h', 'iou', 'error', 'distractor_hit',
                  'input_sha256', 'prediction_sha256', 'rng_sha256', 'interaction_I',
                  'baseline_00_parity', 'order_invariant', 'status']
        write_csv(os.path.join(out_dir, 'event_results.csv'), rows, fields)
        for event_id in sorted(set(r['event_id'] for r in rows)):
            er = [r for r in rows if r['event_id'] == event_id]
            core = {r['condition']: r for r in er if r['condition'] in CORE_CONDITIONS}
            sequence_rows.append({'event_id': event_id, 'seq': er[0]['seq'], 'frame': er[0]['frame'],
                                  'n_conditions': len(er), 'iou_00': core['00']['iou'],
                                  'iou_10': core['10']['iou'], 'iou_01': core['01']['iou'],
                                  'iou_11': core['11']['iou'], 'I': core['11']['interaction_I'],
                                  'distractor_hit_10': core['10']['distractor_hit'],
                                  'distractor_hit_01': core['01']['distractor_hit'],
                                  'distractor_hit_11': core['11']['distractor_hit'],
                                  'status': 'OK'})
        write_csv(os.path.join(out_dir, 'event_summary.csv'), sequence_rows,
                  ['event_id', 'seq', 'frame', 'n_conditions', 'iou_00', 'iou_10', 'iou_01',
                   'iou_11', 'I', 'distractor_hit_10', 'distractor_hit_01', 'distractor_hit_11', 'status'])
        seq_groups = defaultdict(list)
        for item in sequence_rows:
            seq_groups[item['seq']].append(item)
        seq_rows = []
        for seq_name in sorted(seq_groups):
            group = seq_groups[seq_name]
            vals = [float(x['I']) for x in group]
            seq_rows.append({
                'seq': seq_name, 'n_events': len(group),
                'mean_iou_00': float(np.mean([float(x['iou_00']) for x in group])),
                'mean_iou_10': float(np.mean([float(x['iou_10']) for x in group])),
                'mean_iou_01': float(np.mean([float(x['iou_01']) for x in group])),
                'mean_iou_11': float(np.mean([float(x['iou_11']) for x in group])),
                'mean_I': float(np.mean(vals)), 'median_I': float(np.median(vals)),
                'positive_I_events': int(sum(v > 0 for v in vals)),
                'large_positive_I_events': int(sum(v > 0.05 for v in vals)),
                'status': 'OK'})
        write_csv(os.path.join(out_dir, 'sequence_summary.csv'), seq_rows,
                  ['seq', 'n_events', 'mean_iou_00', 'mean_iou_10', 'mean_iou_01',
                   'mean_iou_11', 'mean_I', 'median_I', 'positive_I_events',
                   'large_positive_I_events', 'status'])
        with open(os.path.join(out_dir, 'failures.json'), 'w') as f:
            json.dump(failures, f, indent=2)
        meta = {'experiment': EXPERIMENT, 'mode': args.mode, 'config': args.config,
                'checkpoint': os.path.abspath(args.checkpoint), 'seed': args.seed,
                'sequence_count': len(seq_names), 'selected_event_count': len(selected_events),
                'conditions': CONDITIONS, 'core_conditions': CORE_CONDITIONS,
                'blind_sequences_excluded': sorted(blind), 'git_commit': os.popen('git rev-parse HEAD').read().strip(),
                'canonical_yaml_sha256': sha256_bytes(open(config_path, 'rb').read()),
                'checkpoint_sha256': sha256_bytes(open(args.checkpoint, 'rb').read()),
                'free_disk_gib_at_start': free_gb, 'cuda': torch.cuda.get_device_name(0),
                'torch': torch.__version__, 'python': sys.version, 'status': 'SMOKE_PASS' if args.mode == 'smoke' else 'PROVISIONAL'}
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        if args.mode == 'smoke' and (len(selected_events) < 8 or failures):
            raise RuntimeError('smoke failed: {} events or geometry failures={}'.format(len(selected_events), len(failures)))
        print('COMPLETED {}'.format(meta['status']), flush=True)
    finally:
        if os.path.exists(lock):
            os.unlink(lock)


if __name__ == '__main__':
    main()
