#!/usr/bin/env python3
"""Collect deterministic image quality, scale, and motion covariates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def crop_gray(path, box):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    h, w = image.shape[:2]
    x, y, bw, bh = box
    x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
    x2, y2 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    if x2 <= x1 or y2 <= y1:
        return image, None
    return image, image[y1:y2, x1:x2]


def quality(image, crop):
    if crop is None or crop.size == 0:
        return {'lap_var': None, 'contrast': None, 'mean': None, 'gradient': None}
    lap = cv2.Laplacian(crop, cv2.CV_64F).var()
    gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
    return {'lap_var': float(lap), 'contrast': float(crop.std()), 'mean': float(crop.mean()),
            'gradient': float(np.mean(np.sqrt(gx * gx + gy * gy)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-frames-per-sequence', type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    with open(args.pairs, newline='') as handle:
        pairs = list(csv.DictReader(handle))
    previous = {}
    rows = []
    for pair in pairs:
        seq, frame = pair['seq'], int(pair['frame'])
        if args.max_frames_per_sequence > 0 and frame > args.max_frames_per_sequence:
            continue
        seq_dir = Path(args.dataset_dir) / seq
        visible = sorted((seq_dir / 'visible').glob('*.jpg'))
        infrared = sorted((seq_dir / 'infrared').glob('*.jpg'))
        if frame >= len(visible) or frame >= len(infrared):
            continue
        gt = [float(pair[key]) for key in ('gt_x', 'gt_y', 'gt_w', 'gt_h')]
        rgb_image, rgb_crop = crop_gray(visible[frame], gt)
        tir_image, tir_crop = crop_gray(infrared[frame], gt)
        rgb_q, tir_q = quality(rgb_image, rgb_crop), quality(tir_image, tir_crop)
        area = (gt[2] * gt[3]) / float(rgb_image.shape[0] * rgb_image.shape[1]) if rgb_image is not None else None
        center = (gt[0] + gt[2] / 2.0, gt[1] + gt[3] / 2.0)
        prior = previous.get(seq)
        motion = None if prior is None else float(np.hypot(center[0] - prior[0][0], center[1] - prior[0][1]) /
                                                    max(1.0, np.sqrt(rgb_image.shape[0] * rgb_image.shape[1])))
        scale = None if prior is None else float(abs(np.log(max(gt[2] * gt[3], 1e-8) / max(prior[1], 1e-8))) )
        previous[seq] = (center, gt[2] * gt[3])
        rows.append({
            'seq': seq, 'frame': frame, 'failure_label': pair['failure_label'],
            'delta_iou': pair['delta_iou'], 'target_area_ratio': area,
            'target_aspect_ratio': gt[2] / max(gt[3], 1e-8), 'motion_speed': motion,
            'scale_change': scale,
            'rgb_lap_var': rgb_q['lap_var'], 'tir_lap_var': tir_q['lap_var'],
            'rgb_contrast': rgb_q['contrast'], 'tir_contrast': tir_q['contrast'],
            'rgb_mean': rgb_q['mean'], 'tir_mean': tir_q['mean'],
            'rgb_gradient': rgb_q['gradient'], 'tir_gradient': tir_q['gradient'],
            'quality_delta_lap': None if rgb_q['lap_var'] is None or tir_q['lap_var'] is None else rgb_q['lap_var'] - tir_q['lap_var'],
            'quality_delta_contrast': None if rgb_q['contrast'] is None or tir_q['contrast'] is None else rgb_q['contrast'] - tir_q['contrast'],
        })
    fields = list(rows[0])
    with open(output / 'image_quality_stats.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = defaultdict(list)
    for row in rows:
        summary[row['failure_label']].append(row)
    summary_rows = []
    metrics = [field for field in fields if field not in ('seq', 'frame', 'failure_label')]
    for label, values in sorted(summary.items()):
        out = {'failure_label': label, 'rows': len(values)}
        for metric in metrics:
            numeric = [float(v[metric]) for v in values if v[metric] not in (None, '')]
            out[metric] = float(np.mean(numeric)) if numeric else None
        summary_rows.append(out)
    with open(output / 'image_quality_group_summary.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    with open(output / 'image_quality_manifest.json', 'w') as handle:
        json.dump({'experiment': 'mhsa_failure_analysis_v1', 'stage': 'C', 'rows': len(rows),
                   'source_pairs': args.pairs, 'dataset_dir': str(Path(args.dataset_dir).resolve()),
                   'prefix_sampling_inherited': True,
                   'max_frames_per_sequence': args.max_frames_per_sequence}, handle, indent=2)
    print(json.dumps({'rows': len(rows), 'labels': {key: len(value) for key, value in sorted(summary.items())}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
