#!/usr/bin/env python
"""TCSI-P1 exact support audit.

This inference-only probe records real per-head/per-query search-to-template
attention support for the two TBSI directions, then tests common-coordinate
overlap, foreground enrichment, and fixed-size random controls. It does not
train, alter weights, or modify the canonical configuration.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import OrderedDict

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
from synthetic_joint_interference_v1 import image_loader, load_train_sequence

EXP = 'tcsi_support_probe_v1'
CANONICAL = 'v1.0.0-单卡3090-48G-等效全局128基线'
LAYERS = (3, 6, 9)
DIRECTIONS = ('RGB2F', 'TIR2F')
TOPKS = (8, 16, 32)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_params(config_name, checkpoint):
    config_path = os.path.join(ROOT, 'experiments', 'tbsi_track', config_name + '.yaml')
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


def select_events(manifest, events_per_sequence=2):
    grouped = OrderedDict()
    for event in manifest['events']:
        grouped.setdefault(event['seq'], []).append(event)
    selected = []
    for seq in sorted(grouped):
        selected.extend(sorted(grouped[seq], key=lambda item: item['frame'])[:events_per_sequence])
    return selected


def load_frames(settings, event):
    sequence = load_train_sequence(settings.lasher_path, event['seq'])
    rgb = [image_loader(path) for path in sequence.frames[0]]
    tir = [image_loader(path) for path in sequence.frames[1]]
    n = min(len(rgb), len(tir), len(sequence.ground_truth_rect))
    return rgb[:n], tir[:n], [[float(x) for x in box] for box in sequence.ground_truth_rect[:n]]


def box_iou(a, b):
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def token_mask(gt_box, state_before, resize_factor, grid_h=16, grid_w=16, ignore_radius=1):
    search_size = 256.0
    half_side = 0.5 * search_size / float(resize_factor)
    cx = float(state_before[0]) + 0.5 * float(state_before[2])
    cy = float(state_before[1]) + 0.5 * float(state_before[3])
    crop_x, crop_y = cx - half_side, cy - half_side
    gx, gy, gw, gh = [float(x) for x in gt_box]
    x0, y0 = (gx - crop_x) / (2.0 * half_side), (gy - crop_y) / (2.0 * half_side)
    x1, y1 = (gx + gw - crop_x) / (2.0 * half_side), (gy + gh - crop_y) / (2.0 * half_side)
    foreground = np.zeros((grid_h, grid_w), dtype=np.bool_)
    for row in range(grid_h):
        for col in range(grid_w):
            tx0, ty0 = col / float(grid_w), row / float(grid_h)
            tx1, ty1 = (col + 1) / float(grid_w), (row + 1) / float(grid_h)
            inter = max(0.0, min(tx1, x1) - max(tx0, x0)) * max(0.0, min(ty1, y1) - max(ty0, y0))
            if inter / (1.0 / float(grid_h * grid_w)) >= 0.25:
                foreground[row, col] = True
    ignore = np.zeros_like(foreground)
    if ignore_radius > 0:
        mask = foreground.astype(np.uint8) * 255
        kernel = np.ones((2 * ignore_radius + 1, 2 * ignore_radius + 1), dtype=np.uint8)
        ignore = cv2.dilate(mask, kernel).astype(bool) & (~foreground)
    return foreground.reshape(-1), ignore.reshape(-1)


def extract_topk(attention, topk):
    if attention.ndim == 3:
        attention = attention.unsqueeze(1)
    if attention.ndim != 4:
        raise ValueError('expected [B,H,Q,K], got {}'.format(tuple(attention.shape)))
    k = min(int(topk), attention.shape[-1])
    return torch.topk(attention, k=k, dim=-1, largest=True, sorted=True)


def support_stats(rgb_idx, tir_idx, foreground, ignore, topk, rng):
    rgb_idx, tir_idx = np.asarray(rgb_idx), np.asarray(tir_idx)
    if rgb_idx.shape != tir_idx.shape:
        raise ValueError('RGB/TIR index shape mismatch')
    valid_fg = np.asarray(foreground, dtype=bool) & ~np.asarray(ignore, dtype=bool)
    base_rate = float(valid_fg.mean())
    shared, rgb_comp, tir_comp, union = [], [], [], []
    rgb_fg, tir_fg, shared_fg, rgb_comp_fg, tir_comp_fg, random_fg = [], [], [], [], [], []
    for h in range(rgb_idx.shape[0]):
        for q in range(rgb_idx.shape[1]):
            rgb_set, tir_set = set(rgb_idx[h, q].tolist()), set(tir_idx[h, q].tolist())
            shared_set, rgb_set_only, tir_set_only = rgb_set & tir_set, rgb_set - tir_set, tir_set - rgb_set
            random_set = set(rng.choice(len(valid_fg), size=topk, replace=False).tolist())
            shared.append(len(shared_set) / float(topk))
            rgb_comp.append(len(rgb_set_only) / float(topk))
            tir_comp.append(len(tir_set_only) / float(topk))
            union.append(len(rgb_set | tir_set) / float(2 * topk))
            rgb_fg.append(np.mean([valid_fg[x] for x in rgb_set]))
            tir_fg.append(np.mean([valid_fg[x] for x in tir_set]))
            shared_fg.append(np.mean([valid_fg[x] for x in shared_set]) if shared_set else np.nan)
            rgb_comp_fg.append(np.mean([valid_fg[x] for x in rgb_set_only]) if rgb_set_only else np.nan)
            tir_comp_fg.append(np.mean([valid_fg[x] for x in tir_set_only]) if tir_set_only else np.nan)
            random_fg.append(np.mean([valid_fg[x] for x in random_set]))
    def finite_mean(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')
    return {
        'shared_ratio': float(np.mean(shared)), 'rgb_complementary_ratio': float(np.mean(rgb_comp)),
        'tir_complementary_ratio': float(np.mean(tir_comp)), 'union_ratio': float(np.mean(union)),
        'base_fg_rate': base_rate, 'rgb_fg_rate': float(np.mean(rgb_fg)), 'tir_fg_rate': float(np.mean(tir_fg)),
        'shared_fg_rate': finite_mean(shared_fg), 'rgb_comp_fg_rate': finite_mean(rgb_comp_fg),
        'tir_comp_fg_rate': finite_mean(tir_comp_fg), 'random_fg_rate': float(np.mean(random_fg)),
        'rgb_fg_enrichment': float(np.mean(rgb_fg) / max(base_rate, 1e-6)),
        'tir_fg_enrichment': float(np.mean(tir_fg) / max(base_rate, 1e-6)),
        'random_fg_enrichment': float(np.mean(random_fg) / max(base_rate, 1e-6)),
        'shared_fg_enrichment': float(finite_mean(shared_fg) / max(base_rate, 1e-6)),
        'rgb_comp_fg_enrichment': float(finite_mean(rgb_comp_fg) / max(base_rate, 1e-6)),
        'tir_comp_fg_enrichment': float(finite_mean(tir_comp_fg) / max(base_rate, 1e-6)),
    }


def run(args):
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TCSI_DUMP_ATTENTION'] = '1'
    manifest = json.load(open(args.manifest, 'r'))
    events = select_events(manifest, 2)
    if args.limit is not None:
        events = events[:args.limit]
    if args.limit is None and len(events) != 48:
        raise RuntimeError('expected 48 selected events, got {}'.format(len(events)))
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    lock_path = os.path.join(output_dir, '.lock')
    if os.path.exists(lock_path):
        raise RuntimeError('output lock exists')
    open(lock_path, 'w').write(str(os.getpid()))
    params, config_path = load_params(args.config, args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if shutil.disk_usage(ROOT).free / 2 ** 30 < args.min_free_gb:
        raise RuntimeError('insufficient free disk')
    rows, raw_records, sequence_summary = [], [], []
    try:
        settings = env_settings()
        grouped = OrderedDict()
        for event in events:
            grouped.setdefault(event['seq'], []).append(event)
        tracker = TBSITrack(params, EXP)
        for seq_name in sorted(grouped):
            rgb, tir, gt_boxes = load_frames(settings, grouped[seq_name][0])
            seq_events = sorted(grouped[seq_name], key=lambda item: item['frame'])
            tracker.initialize(np.concatenate([rgb[0], tir[0]], axis=2), {'init_bbox': gt_boxes[0]})
            for frame in range(1, seq_events[-1]['frame'] + 1):
                event = next((item for item in seq_events if item['frame'] == frame), None)
                state_before = list(tracker.state)
                image = np.concatenate([rgb[frame], tir[frame]], axis=2)
                height, width = image.shape[:2]
                # sample_target's resize factor is reproduced from the tracker crop contract.
                crop_side = max(math.ceil(math.sqrt(float(state_before[2]) * float(state_before[3])) *
                                          float(tracker.params.search_factor)), 1.0)
                resize_factor = float(tracker.params.search_size) / crop_side
                tracker.track(image, {'gt_bbox': gt_boxes[frame], 'sequence_name': seq_name, 'frame_num': frame})
                if event is None:
                    continue
                foreground, ignore = token_mask(gt_boxes[frame], state_before, resize_factor)
                layer_records = []
                for layer_id, layer in zip(LAYERS, tracker.network.backbone.tbsi_layers):
                    attentions = {
                        'RGB2F': getattr(layer.ca_s2t_v2f.attn_reshape, 'last_attention', None),
                        'TIR2F': getattr(layer.ca_s2t_i2f.attn_reshape, 'last_attention', None),
                    }
                    if any(value is None for value in attentions.values()):
                        raise RuntimeError('missing attention cache at {} layer {}'.format(event['event_id'], layer_id))
                    attentions = {direction: value.unsqueeze(1) if value.ndim == 3 else value
                                  for direction, value in attentions.items()}
                    shape = tuple(attentions['RGB2F'].shape)
                    if shape != tuple(attentions['TIR2F'].shape) or shape[-1] != 256:
                        raise RuntimeError('coordinate contract failed: {} / {}'.format(shape, tuple(attentions['TIR2F'].shape)))
                    raw_layer = {}
                    for direction in DIRECTIONS:
                        raw_layer[direction] = attentions[direction].numpy()[0]
                    for topk in TOPKS:
                        values_rgb, idx_rgb = extract_topk(attentions['RGB2F'], topk)
                        values_tir, idx_tir = extract_topk(attentions['TIR2F'], topk)
                        rng = np.random.RandomState(3407 + int(event['event_id'].split('-')[-1]) * 100 + layer_id * 10 + topk)
                        stats = support_stats(idx_rgb.numpy()[0], idx_tir.numpy()[0], foreground, ignore, topk, rng)
                        rows.append({'event_id': event['event_id'], 'seq': seq_name, 'frame': frame,
                                     'layer': layer_id, 'topk': topk, 'num_heads': shape[1], 'num_queries': shape[2],
                                     'attention_key_space': 'shared_search_grid_16x16',
                                     'attention_shape': json.dumps(list(shape)),
                                     'rgb_entropy': float((-(attentions['RGB2F'][0].clamp_min(1e-12) * attentions['RGB2F'][0].clamp_min(1e-12).log()).sum(-1).mean()).item()),
                                     'tir_entropy': float((-(attentions['TIR2F'][0].clamp_min(1e-12) * attentions['TIR2F'][0].clamp_min(1e-12).log()).sum(-1).mean()).item()),
                                     'rgb_topk_mass': float(values_rgb.sum(-1).mean().item()),
                                     'tir_topk_mass': float(values_tir.sum(-1).mean().item()),
                                     'gt_base_fg_rate': stats['base_fg_rate'], **stats})
                    for direction in DIRECTIONS:
                        for topk in TOPKS:
                            values, indices = extract_topk(attentions[direction], topk)
                            raw_records.append({'event_id': event['event_id'], 'seq': seq_name, 'frame': frame,
                                                'layer': layer_id, 'direction': direction, 'topk': topk,
                                                'top_values': values.numpy()[0], 'top_indices': indices.numpy()[0]})
                sequence_summary.append({'event_id': event['event_id'], 'seq': seq_name,
                                         'frame': frame, 'foreground_tokens': int(foreground.sum()),
                                         'ignored_tokens': int(ignore.sum()), 'state_before': state_before,
                                         'image_shape': [height, width]})
            print('processed sequence {} ({}/{})'.format(seq_name, len(sequence_summary), len(events)), flush=True)
        fields = list(rows[0].keys())
        with open(os.path.join(output_dir, 'support_summary.csv'), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        max_topk = max(TOPKS)
        padded_values = np.zeros((len(raw_records), raw_records[0]['top_values'].shape[0],
                                  raw_records[0]['top_values'].shape[1], max_topk), dtype=np.float32)
        padded_indices = np.full_like(padded_values, -1, dtype=np.int64)
        for record_index, record in enumerate(raw_records):
            width = int(record['topk'])
            padded_values[record_index, :, :, :width] = record['top_values']
            padded_indices[record_index, :, :, :width] = record['top_indices']
        np.savez_compressed(os.path.join(output_dir, 'raw_support_indices.npz'),
                            event_id=np.asarray([x['event_id'] for x in raw_records]),
                            seq=np.asarray([x['seq'] for x in raw_records]),
                            layer=np.asarray([x['layer'] for x in raw_records], dtype=np.int64),
                            direction=np.asarray([x['direction'] for x in raw_records]),
                            topk=np.asarray([x['topk'] for x in raw_records], dtype=np.int64),
                            top_values=padded_values,
                            top_indices=padded_indices,
                            padded_topk=np.asarray(TOPKS, dtype=np.int64))
        meta = {'experiment': EXP, 'manifest': os.path.abspath(args.manifest),
                'manifest_sha256': sha256_file(args.manifest), 'config': args.config,
                'config_sha256': sha256_file(config_path), 'checkpoint': os.path.abspath(args.checkpoint),
                'checkpoint_sha256': sha256_file(args.checkpoint), 'n_events': len(events),
                'layers': list(LAYERS), 'directions': list(DIRECTIONS), 'topk': list(TOPKS),
                'coordinate_space': 'common 16x16 search grid; both directions are s2t with 256 search keys',
                'raw_records': len(raw_records), 'gpu': torch.cuda.get_device_name(0),
                'torch': torch.__version__, 'fp16_env': os.environ['TBSI_INFER_FP16'], 'status': 'COMPLETED'}
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, 'event_geometry.json'), 'w') as handle:
            json.dump(sequence_summary, handle, ensure_ascii=False, indent=2)
    finally:
        if os.path.exists(lock_path):
            os.unlink(lock_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default=CANONICAL)
    parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'output/experiments', CANONICAL, 'checkpoints', 'TBSITrack_ep0015.pth.tar'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--min-free-gb', type=float, default=8.0)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
