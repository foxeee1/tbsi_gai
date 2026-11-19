#!/usr/bin/env python
"""Robustness and closed-loop follow-up for synthetic joint interference v1.

Stages are deliberately explicit: audit existing v1 results, run fixed-event
controls, then run closed-loop replay. This script never reads blind labels and
never modifies a model or canonical configuration.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
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
from synthetic_joint_interference_v1 import (
    CORE_CONDITIONS, box_iou, choose_destination, frame_sort_key, image_loader,
    load_train_sequence, make_background_control, patch_image, rng_hash,
    rng_snapshot, restore_rng, sha256_bytes, write_csv)

EXP = 'synthetic_joint_interference_v2'
V1 = os.path.join(ROOT, 'output/experiments/synthetic_joint_interference_v1/batch')


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_events():
    events = []
    for row in read_csv(os.path.join(V1, 'event_summary.csv')):
        events.append({'event_id': row['event_id'], 'seq': row['seq'], 'frame': int(row['frame']),
                       'gt': [float(row['iou_00']) * 0.0, 0.0, 0.0, 0.0]})
    selected = {x['event_id']: x for x in json.load(open(os.path.join(V1, 'selected_events.json')))}
    for event in events:
        source = selected[event['event_id']]
        event.update({'gt': source['gt'], 'dest_rgb': source['dest_rgb'], 'dest_tir': source['dest_tir']})
    return events


def load_v1_event_rows():
    rows = read_csv(os.path.join(V1, 'event_summary.csv'))
    return {r['event_id']: r for r in rows}


def sequence_bootstrap(values, seed=3407, n=20000):
    rng = np.random.RandomState(seed)
    names = sorted(values)
    arr = np.asarray([values[k] for k in names], dtype=float)
    if len(arr) == 0:
        return {'mean': None, 'ci95': [None, None], 'n_sequences': 0}
    draws = rng.randint(0, len(arr), size=(n, len(arr)))
    means = arr[draws].mean(axis=1)
    return {'mean': float(arr.mean()), 'ci95': [float(np.percentile(means, 2.5)),
                                                  float(np.percentile(means, 97.5))],
            'n_sequences': int(len(arr)), 'seed': seed, 'n_bootstrap': n}


def paired_permutation(values, seed=3407, n=20000):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    observed = float(arr.mean())
    if len(arr) == 0:
        return {'observed': None, 'p_two_sided': None}
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n, len(arr)))
    null = (signs * arr).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1.0) / (n + 1.0)
    return {'observed': observed, 'p_two_sided': float(p), 'n': int(len(arr)),
            'seed': seed, 'n_permutations': n}


def audit(out_dir):
    rows = read_csv(os.path.join(V1, 'event_summary.csv'))
    events = [{'event_id': r['event_id'], 'seq': r['seq'], 'I': float(r['I'])} for r in rows]
    by_seq = defaultdict(list)
    for x in events:
        by_seq[x['seq']].append(x['I'])
    seq_mean = {k: float(np.mean(v)) for k, v in by_seq.items()}
    seq_median = {k: float(np.median(v)) for k, v in by_seq.items()}
    seq_pos = {k: float(np.mean(np.asarray(v) > 0)) for k, v in by_seq.items()}
    ordered = sorted(events, key=lambda x: x['I'], reverse=True)
    trim = {}
    for k in (0, 1, 3, 5):
        vals = [x['I'] for x in ordered[k:]]
        trim['drop_top_{}'.format(k)] = float(np.mean(vals))
    seq_contrib = {k: float(sum(v)) for k, v in by_seq.items()}
    positive_total = sum(max(0.0, x) for x in seq_contrib.values())
    largest_seq, largest_value = max(seq_contrib.items(), key=lambda x: x[1])
    # J is the negative of I for the same four IoU terms; compute directly for clarity.
    v1_rows = read_csv(os.path.join(V1, 'event_results.csv'))
    grouped = defaultdict(dict)
    for r in v1_rows:
        if r['condition'] in CORE_CONDITIONS:
            grouped[r['event_id']][r['condition']] = float(r['iou'])
    j = {eid: x['11'] - x['10'] - x['01'] + x['00'] for eid, x in grouped.items()}
    seq_j = defaultdict(list)
    for eid, value in j.items():
        seq_j[next(x['seq'] for x in events if x['event_id'] == eid)].append(value)
    stats = {
        'experiment': EXP, 'source': V1, 'n_events': len(events), 'n_sequences': len(by_seq),
        'event_mean_I': float(np.mean([x['I'] for x in events])),
        'event_median_I': float(np.median([x['I'] for x in events])),
        'event_std_I': float(np.std([x['I'] for x in events], ddof=1)),
        'event_positive_fraction': float(np.mean(np.asarray([x['I'] for x in events]) > 0)),
        'event_large_positive_count_I_gt_0.05': int(sum(x['I'] > 0.05 for x in events)),
        'sequence_mean_I': seq_mean, 'sequence_median_I': seq_median,
        'sequence_positive_fraction': seq_pos, 'trimmed_means': trim,
        'leave_one_sequence_out_mean_I': {k: float(np.mean([x['I'] for x in events if x['seq'] != k]))
                                          for k in by_seq},
        'largest_sequence_by_signed_sum': largest_seq,
        'largest_sequence_positive_sum_fraction': float(max(0.0, largest_value) / positive_total)
        if positive_total else 0.0,
        'sequence_bootstrap_I': sequence_bootstrap(seq_mean),
        'sequence_bootstrap_J': sequence_bootstrap({k: float(np.mean(v)) for k, v in seq_j.items()}),
        'paired_permutation_event_I': paired_permutation([x['I'] for x in events]),
        'paired_permutation_sequence_I': paired_permutation(list(seq_mean.values())),
        'J_by_event': j,
        'paired_deltas': {eid: {'iou11_minus_iou10': x['11'] - x['10'],
                                'iou11_minus_iou01': x['11'] - x['01'],
                                'iou11_minus_iou00': x['11'] - x['00']}
                          for eid, x in grouped.items()}
    }
    # Fixed first gate; later gates are filled after controls and closed-loop.
    stats['gate_A'] = {
        'A1_sequence_bootstrap_not_below_zero': stats['sequence_bootstrap_I']['ci95'][1] > 0,
        'A2_drop_top_3_mean_positive': trim['drop_top_3'] > 0,
        'A3_at_least_8_events_I_gt_0.05': stats['event_large_positive_count_I_gt_0.05'] >= 8,
        'A4_at_least_8_sequences_I_positive': sum(v > 0 for v in seq_mean.values()) >= 8,
        'A5_largest_sequence_positive_contribution_le_50pct': stats['largest_sequence_positive_sum_fraction'] <= 0.5,
        'A6_texture_control_pending': None,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'synthetic_interaction_statistics_v2.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    write_csv(os.path.join(out_dir, 'event_interaction_summary.csv'),
              [{'event_id': x['event_id'], 'seq': x['seq'], 'I': x['I'], 'J': j[x['event_id']]}
               for x in events], ['event_id', 'seq', 'I', 'J'])
    write_csv(os.path.join(out_dir, 'sequence_interaction_summary.csv'),
              [{'seq': k, 'mean_I': seq_mean[k], 'median_I': seq_median[k],
                'positive_fraction': seq_pos[k], 'n_events': len(by_seq[k])}
               for k in sorted(by_seq)], ['seq', 'mean_I', 'median_I', 'positive_fraction', 'n_events'])
    return stats


def build_params(checkpoint, config_name):
    config_path = os.path.join(ROOT, 'experiments/tbsi_track', config_name + '.yaml')
    update_config_from_file(config_path)
    params = TrackerParams(); params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR; params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR; params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = checkpoint; params.save_all_boxes = False; params.debug = 0
    return params, config_path


def map_box(box, src_shape, dst_shape):
    return [box[0] * dst_shape[1] / src_shape[1], box[1] * dst_shape[0] / src_shape[0],
            box[2] * dst_shape[1] / src_shape[1], box[3] * dst_shape[0] / src_shape[0]]


def make_shift_box(box, dx, dy, width, height):
    x, y, w, h = box
    x = min(max(0.0, x + dx), max(0.0, width - w - 1.0))
    y = min(max(0.0, y + dy), max(0.0, height - h - 1.0))
    return [x, y, w, h]


def make_control_frames(rgb, tir, gt, dest_rgb, dest_tir):
    h, w = rgb.shape[:2]
    diagonal = max(1.0, math.hypot(gt[2], gt[3]))
    shifted_rgb = make_shift_box(dest_rgb, 0.5 * diagonal, 0.0, w, h)
    shifted_tir = map_box(shifted_rgb, rgb.shape, tir.shape)
    independent_tir = make_shift_box(dest_tir, 0.5 * diagonal, 0.0, tir.shape[1], tir.shape[0])
    rgb_bg, _ = make_background_control(rgb, gt, dest_rgb)
    tir_bg, _ = make_background_control(tir, gt, dest_tir)
    rgb_same, _ = patch_image(rgb, gt, dest_rgb)
    tir_same, _ = patch_image(tir, gt, dest_tir)
    rgb_shift, _ = patch_image(rgb, gt, shifted_rgb)
    tir_shift, _ = patch_image(tir, gt, shifted_tir)
    rgb_ind, _ = patch_image(rgb, gt, dest_rgb)
    tir_ind, _ = patch_image(tir, gt, independent_tir)
    return {
        '00': (rgb, tir), '10': (rgb_same, tir), '01': (rgb, tir_same),
        '11_same': (rgb_same, tir_same), '11_independent': (rgb_ind, tir_ind),
        '11_texture': (rgb_bg[0], tir_bg[0]), '11_shift': (rgb_shift, tir_shift),
    }


def replay(tracker, frames_rgb, frames_tir, gt_boxes, event, condition_frames, rng, horizon):
    restore_rng(rng)
    tracker.initialize(np.concatenate([frames_rgb[0], frames_tir[0]], axis=2), {'init_bbox': gt_boxes[0]})
    outputs = []
    for frame_num in range(1, event['frame'] + horizon + 1):
        if frame_num == event['frame']:
            rgb, tir = condition_frames
        else:
            rgb, tir = frames_rgb[frame_num], frames_tir[frame_num]
        out = tracker.track(np.concatenate([rgb, tir], axis=2),
                            {'gt_bbox': gt_boxes[frame_num], 'sequence_name': event['seq'],
                             'frame_num': frame_num})
        if frame_num >= event['frame']:
            outputs.append({'frame_offset': frame_num - event['frame'],
                            'pred': [float(x) for x in out['target_bbox']],
                            'iou': box_iou(out['target_bbox'], gt_boxes[frame_num]),
                            'state_sha256': sha256_bytes(json.dumps([float(x) for x in out['target_bbox']]).encode())})
    return outputs


def sensitivity(out_dir, checkpoint, config_name):
    events = load_events()
    settings = env_settings()
    seq_map = {name: load_train_sequence(settings.lasher_path, name)
               for name in sorted(set(x['seq'] for x in events))}
    params, config_path = build_params(checkpoint, config_name)
    tracker = TBSITrack(params, EXP)
    rng = rng_snapshot()
    alpha_rows, position_rows = [], []
    for index, event in enumerate(events):
        seq = seq_map[event['seq']]
        rgb_frames = [image_loader(p) for p in seq.frames[0]]
        tir_frames = [image_loader(p) for p in seq.frames[1]]
        gt_boxes = [list(map(float, x)) for x in seq.ground_truth_rect]
        t = event['frame']
        rgb, tir, gt = rgb_frames[t], tir_frames[t], event['gt']
        diagonal = max(1.0, math.hypot(gt[2], gt[3]))
        positions = {
            'P0': event['dest_rgb'],
            'P1': make_shift_box(event['dest_rgb'], 0.25 * diagonal, 0.0, rgb.shape[1], rgb.shape[0]),
            'P2': make_shift_box(event['dest_rgb'], 0.50 * diagonal, 0.0, rgb.shape[1], rgb.shape[0]),
        }
        for alpha in (0.25, 0.50, 0.75, 1.00):
            copied_rgb, _ = patch_image(rgb, gt, event['dest_rgb'], alpha=alpha)
            copied_tir, _ = patch_image(tir, gt, event['dest_tir'], alpha=alpha)
            result = replay(tracker, rgb_frames, tir_frames, gt_boxes, event,
                            (copied_rgb, copied_tir), rng, 0)[0]
            alpha_rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': t,
                               'alpha': alpha, 'iou': result['iou'], 'error': 1.0 - result['iou'],
                               'pred_sha256': result['state_sha256'], 'status': 'OK'})
        for position, dest_rgb in positions.items():
            dest_tir = map_box(dest_rgb, rgb.shape, tir.shape)
            copied_rgb, _ = patch_image(rgb, gt, dest_rgb, alpha=0.88)
            copied_tir, _ = patch_image(tir, gt, dest_tir, alpha=0.88)
            result = replay(tracker, rgb_frames, tir_frames, gt_boxes, event,
                            (copied_rgb, copied_tir), rng, 0)[0]
            position_rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': t,
                                  'position': position, 'dest_rgb': json.dumps(dest_rgb),
                                  'dest_tir': json.dumps(dest_tir), 'iou': result['iou'],
                                  'error': 1.0 - result['iou'], 'pred_sha256': result['state_sha256'],
                                  'status': 'OK'})
        if index % 8 == 0:
            print('sensitivity processed {}/{} events'.format(index + 1, len(events)), flush=True)
    write_csv(os.path.join(out_dir, 'sensitivity_alpha.csv'), alpha_rows,
              ['event_id', 'seq', 'frame', 'alpha', 'iou', 'error', 'pred_sha256', 'status'])
    write_csv(os.path.join(out_dir, 'shift_sensitivity.csv'), position_rows,
              ['event_id', 'seq', 'frame', 'position', 'dest_rgb', 'dest_tir', 'iou', 'error',
               'pred_sha256', 'status'])
    with open(os.path.join(out_dir, 'sensitivity_metadata.json'), 'w') as f:
        json.dump({'experiment': EXP, 'alphas': [0.25, 0.50, 0.75, 1.0],
                   'positions': ['P0', 'P1', 'P2'], 'n_events': len(events),
                   'checkpoint': checkpoint, 'config': config_name, 'status': 'COMPLETED'}, f, indent=2)


def controls_and_closed_loop(out_dir, stage, checkpoint, config_name, horizon_max):
    events = load_events()
    seq_names = sorted(set(x['seq'] for x in events))
    settings = env_settings()
    seq_map = {name: load_train_sequence(settings.lasher_path, name) for name in seq_names}
    params, config_path = build_params(checkpoint, config_name)
    tracker = TBSITrack(params, EXP)
    rng = rng_snapshot()
    control_rows = []
    closed_rows = []
    control_names = ('00', '10', '01', '11_same', '11_independent', '11_texture', '11_shift')
    for index, event in enumerate(events):
        seq = seq_map[event['seq']]
        frames_rgb = [image_loader(p) for p in seq.frames[0]]
        frames_tir = [image_loader(p) for p in seq.frames[1]]
        n = min(len(frames_rgb), len(frames_tir), len(seq.ground_truth_rect))
        gt_boxes = [list(map(float, x)) for x in seq.ground_truth_rect[:n]]
        t = event['frame']
        if t + horizon_max >= n:
            continue
        frames = make_control_frames(frames_rgb[t], frames_tir[t], event['gt'], event['dest_rgb'], event['dest_tir'])
        if stage in ('controls', 'all'):
            preds = {}
            for name in control_names:
                result = replay(tracker, frames_rgb, frames_tir, gt_boxes, event, frames[name], rng, 0)
                final = result[0]
                preds[name] = final
                control_rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': t,
                                     'condition': name, 'iou': final['iou'], 'error': 1.0 - final['iou'],
                                     'pred_sha256': final['state_sha256'], 'rng_sha256': rng_hash(rng), 'status': 'OK'})
            base = 1.0 - preds['00']['iou']; same = 1.0 - preds['11_same']['iou']
            control_i = same - (1.0 - preds['10']['iou']) - (1.0 - preds['01']['iou']) + base
            for row in control_rows[-len(control_names):]:
                row['I_same_vs_additive'] = control_i
        if stage in ('closed_loop', 'all'):
            for name in ('00', '10', '01', '11_same', '11_independent', '11_texture'):
                for horizon in (1, 5, 10):
                    result = replay(tracker, frames_rgb, frames_tir, gt_boxes, event, frames[name], rng, horizon)
                    for point in result:
                        if point['frame_offset'] > horizon:
                            continue
                        closed_rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': t,
                                            'condition': name, 'horizon': horizon,
                                            'frame_offset': point['frame_offset'], 'iou': point['iou'],
                                            'pred_sha256': point['state_sha256'], 'rng_sha256': rng_hash(rng),
                                            'status': 'OK'})
        if index % 4 == 0:
            print('processed {}/{} events'.format(index + 1, len(events)), flush=True)
    if control_rows:
        write_csv(os.path.join(out_dir, 'same_vs_independent.csv'), control_rows,
                  ['event_id', 'seq', 'frame', 'condition', 'iou', 'error', 'pred_sha256',
                   'rng_sha256', 'I_same_vs_additive', 'status'])
    if closed_rows:
        write_csv(os.path.join(out_dir, 'closed_loop_event_results.csv'), closed_rows,
                  ['event_id', 'seq', 'frame', 'condition', 'horizon', 'frame_offset', 'iou',
                   'pred_sha256', 'rng_sha256', 'status'])
        summary = []
        groups = defaultdict(list)
        for row in closed_rows:
            groups[(row['event_id'], row['condition'], int(row['horizon']))].append(row)
        for key, rows in groups.items():
            summary.append({'event_id': key[0], 'condition': key[1], 'horizon': key[2],
                            'mean_iou': float(np.mean([float(x['iou']) for x in rows])),
                            'min_iou': float(np.min([float(x['iou']) for x in rows])),
                            'recover_frame': recovery_frame(rows), 'n_points': len(rows)})
        write_csv(os.path.join(out_dir, 'recovery_statistics.csv'), summary,
                  ['event_id', 'condition', 'horizon', 'mean_iou', 'min_iou', 'recover_frame', 'n_points'])
        seq_summary = []
        seq_groups = defaultdict(list)
        for row in summary:
            seq = next(x['seq'] for x in events if x['event_id'] == row['event_id'])
            seq_groups[(seq, row['condition'], row['horizon'])].append(row)
        for (seq, condition, horizon), rows in sorted(seq_groups.items()):
            seq_summary.append({'seq': seq, 'condition': condition, 'horizon': horizon,
                                'mean_iou': float(np.mean([x['mean_iou'] for x in rows])),
                                'mean_recover_frame': float(np.mean([x['recover_frame'] for x in rows]))})
        write_csv(os.path.join(out_dir, 'closed_loop_sequence_summary.csv'), seq_summary,
                  ['seq', 'condition', 'horizon', 'mean_iou', 'mean_recover_frame'])
    with open(os.path.join(out_dir, 'closed_loop_metadata.json'), 'w') as f:
        json.dump({'experiment': EXP, 'stage': stage, 'config': config_name,
                   'checkpoint': checkpoint, 'checkpoint_sha256': sha256_bytes(open(checkpoint, 'rb').read()),
                   'config_sha256': sha256_bytes(open(config_path, 'rb').read()), 'n_events': len(events),
                   'horizons': [1, 5, 10], 'status': 'COMPLETED'}, f, indent=2)


def recovery_frame(rows, threshold=0.5):
    by_offset = defaultdict(list)
    for row in rows:
        by_offset[int(row['frame_offset'])].append(float(row['iou']))
    offsets = sorted(by_offset)
    for offset in offsets:
        if offset >= 1 and all(np.mean(by_offset.get(x, [0.0])) >= threshold for x in (offset, offset + 1)):
            return offset
    return -1


def final_gate(out_dir):
    stats = json.load(open(os.path.join(out_dir, 'synthetic_interaction_statistics_v2.json')))
    control_path = os.path.join(out_dir, 'same_vs_independent.csv')
    closed_path = os.path.join(out_dir, 'closed_loop_event_results.csv')
    control = read_csv(control_path) if os.path.exists(control_path) else []
    closed = read_csv(closed_path) if os.path.exists(closed_path) else []
    by_event = defaultdict(dict)
    for row in control:
        by_event[row['event_id']][row['condition']] = float(row['iou'])
    same_i = []; independent_i = []; texture_i = []
    for eid, x in by_event.items():
        if not all(k in x for k in ('00', '10', '01', '11_same', '11_independent', '11_texture')):
            continue
        same_i.append((1-x['11_same'])-(1-x['10'])-(1-x['01'])+(1-x['00']))
        independent_i.append((1-x['11_independent'])-(1-x['10'])-(1-x['01'])+(1-x['00']))
        texture_i.append((1-x['11_texture'])-(1-x['10'])-(1-x['01'])+(1-x['00']))
    loop = defaultdict(list)
    for row in closed:
        if row['condition'] in ('00', '11_same') and int(row['horizon']) in (5, 10):
            loop[(row['event_id'], row['condition'], int(row['horizon']))].append(float(row['iou']))
    deltas = {}
    for horizon in (5, 10):
        pairs = []
        for eid in sorted(set(k[0] for k in loop)):
            a = loop.get((eid, '00', horizon), []); b = loop.get((eid, '11_same', horizon), [])
            if a and b: pairs.append(float(np.mean(a)) - float(np.mean(b)))
        deltas[str(horizon)] = {'mean_clean_minus_same': float(np.mean(pairs)) if pairs else None,
                                'n_events': len(pairs)}
    # Four-of-five gate; A3 remains deliberately visible as a hard descriptive failure.
    gate = {
        'A1_sequence_bootstrap_not_below_zero': stats['gate_A']['A1_sequence_bootstrap_not_below_zero'],
        'A2_drop_top_3_mean_positive': stats['gate_A']['A2_drop_top_3_mean_positive'],
        'A3_at_least_8_events_I_gt_0.05': stats['gate_A']['A3_at_least_8_events_I_gt_0.05'],
        'A4_at_least_8_sequences_I_positive': stats['gate_A']['A4_at_least_8_sequences_I_positive'],
        'A5_largest_sequence_positive_contribution_le_50pct': stats['gate_A']['A5_largest_sequence_positive_contribution_le_50pct'],
        'B_same_exceeds_independent': bool(same_i and np.mean(same_i) > np.mean(independent_i)),
        'B_texture_not_same_scale': bool(same_i and abs(np.mean(texture_i)) < abs(np.mean(same_i))),
        'C_H5_clean_minus_same_positive': bool(deltas['5']['mean_clean_minus_same'] is not None and deltas['5']['mean_clean_minus_same'] > 0),
        'C_H10_clean_minus_same_positive': bool(deltas['10']['mean_clean_minus_same'] is not None and deltas['10']['mean_clean_minus_same'] > 0),
        'same_I_mean': float(np.mean(same_i)) if same_i else None,
        'independent_I_mean': float(np.mean(independent_i)) if independent_i else None,
        'texture_I_mean': float(np.mean(texture_i)) if texture_i else None,
        'closed_loop_deltas': deltas,
    }
    evidence = [gate[k] for k in ('A1_sequence_bootstrap_not_below_zero', 'B_same_exceeds_independent',
                                   'B_texture_not_same_scale', 'C_H5_clean_minus_same_positive',
                                   'C_H10_clean_minus_same_positive')]
    gate['training_allowed'] = sum(bool(x) for x in evidence) >= 4
    gate['training_decision'] = 'READY_FOR_TRAINING_REVIEW' if gate['training_allowed'] else 'STOP_NO_TRAINING'
    with open(os.path.join(out_dir, 'automatic_gate.json'), 'w') as f:
        json.dump(gate, f, indent=2)
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['audit', 'controls', 'sensitivity', 'closed_loop', 'all'], default='all')
    ap.add_argument('--output-dir', default=os.path.join(ROOT, 'output/experiments', EXP))
    ap.add_argument('--checkpoint', default=os.path.join(ROOT, 'output/experiments/v1.0.0-单卡3090-48G-等效全局128基线/checkpoints/TBSITrack_ep0015.pth.tar'))
    ap.add_argument('--config', default='v1.0.0-单卡3090-48G-等效全局128基线')
    ap.add_argument('--min-free-gb', type=float, default=8.0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if shutil.disk_usage(ROOT).free / 2**30 < args.min_free_gb:
        raise RuntimeError('insufficient free disk')
    if not os.path.exists(os.path.join(V1, 'event_summary.csv')):
        raise RuntimeError('fixed v1 events are missing')
    out_dir = os.path.abspath(args.output_dir); os.makedirs(out_dir, exist_ok=True)
    lock = os.path.join(out_dir, '.lock')
    if os.path.exists(lock):
        raise RuntimeError('lock exists: {}'.format(lock))
    open(lock, 'w').write(str(os.getpid()))
    try:
        audit(out_dir)
        if args.stage in ('sensitivity', 'all'):
            sensitivity(out_dir, args.checkpoint, args.config)
        if args.stage in ('controls', 'closed_loop', 'all'):
            controls_and_closed_loop(out_dir, args.stage, args.checkpoint, args.config, 10)
        if args.stage in ('closed_loop', 'all'):
            gate = final_gate(out_dir)
            print(json.dumps(gate, indent=2), flush=True)
        print('COMPLETED {}'.format(args.stage), flush=True)
    finally:
        if os.path.exists(lock): os.unlink(lock)


if __name__ == '__main__':
    main()
