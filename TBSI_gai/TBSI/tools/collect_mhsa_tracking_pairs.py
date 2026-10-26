#!/usr/bin/env python3
"""Build a frame-aligned, read-only M1/control tracking pair table."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_box(line):
    values = [float(value) for value in line.replace(',', '\t').split()]
    if len(values) < 4:
        raise ValueError('expected at least four numeric values')
    return values[:4]


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax) * max(0.0, ay2 - ay) + max(0.0, bx2 - bx) * max(0.0, by2 - by) - inter
    return inter / union if union > 0 else 0.0


def center_error(a, b):
    return float(np.hypot((a[0] + a[2] / 2) - (b[0] + b[2] / 2),
                          (a[1] + a[3] / 2) - (b[1] + b[3] / 2)))


def classify(delta_iou, iou_control, iou_m1):
    if delta_iou <= -0.10:
        return 'hard_regression'
    if delta_iou <= -0.05:
        return 'regression'
    if delta_iou >= 0.05:
        return 'gain'
    if max(iou_control, iou_m1) < 0.30:
        return 'common_failure'
    return 'neutral'


def read_boxes(path):
    with open(path) as handle:
        return [parse_box(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-dir', required=True)
    parser.add_argument('--m1-dir', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--control-name', default='canonical_v1.0.0')
    parser.add_argument('--m1-name', default='TBSI_E_MHSA_FULL_4090D_seed42')
    args = parser.parse_args()

    control_dir = Path(args.control_dir).resolve()
    m1_dir = Path(args.m1_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('refusing to reuse non-empty output directory: {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    control_files = {path.stem: path for path in control_dir.glob('*.txt') if not path.name.endswith('_time.txt')}
    m1_files = {path.stem: path for path in m1_dir.glob('*.txt') if not path.name.endswith('_time.txt')}
    names = sorted(set(control_files) & set(m1_files))
    if not names:
        raise RuntimeError('no common prediction files')

    rows = []
    sequence_rows = []
    skipped = []
    for seq in names:
        gt_path = gt_dir / seq / 'init.txt'
        if not gt_path.is_file():
            skipped.append({'sequence': seq, 'reason': 'missing_gt'})
            continue
        gt = np.loadtxt(str(gt_path), delimiter=',', ndmin=2)
        control = read_boxes(control_files[seq])
        m1 = read_boxes(m1_files[seq])
        if len(gt) != len(control) or len(gt) != len(m1):
            raise RuntimeError('frame count mismatch for {}: gt={}, control={}, m1={}'.format(
                seq, len(gt), len(control), len(m1)))
        count = len(gt)
        if count == 0:
            skipped.append({'sequence': seq, 'reason': 'empty_prediction_or_gt'})
            continue
        seq_rows = []
        visible_dir = gt_dir / seq / 'visible'
        visible_frames = sorted(visible_dir.glob('*.jpg'))
        image_area = None
        if visible_frames:
            image = cv2.imread(str(visible_frames[0]), cv2.IMREAD_UNCHANGED)
            if image is not None:
                image_area = float(image.shape[0] * image.shape[1])
        for frame in range(count):
            gt_box = [float(x) for x in gt[frame, :4]]
            c_box, m_box = control[frame], m1[frame]
            iou_c, iou_m = iou_xywh(c_box, gt_box), iou_xywh(m_box, gt_box)
            delta = iou_m - iou_c
            row = {
                'seq': seq, 'frame': frame,
                'gt_x': gt_box[0], 'gt_y': gt_box[1], 'gt_w': gt_box[2], 'gt_h': gt_box[3],
                'pred_c_x': c_box[0], 'pred_c_y': c_box[1], 'pred_c_w': c_box[2], 'pred_c_h': c_box[3],
                'pred_m_x': m_box[0], 'pred_m_y': m_box[1], 'pred_m_w': m_box[2], 'pred_m_h': m_box[3],
                'iou_c': iou_c, 'iou_m': iou_m, 'delta_iou': delta,
                'center_error_c': center_error(c_box, gt_box), 'center_error_m': center_error(m_box, gt_box),
                'target_area_ratio': (gt_box[2] * gt_box[3]) / image_area if image_area else 'NA',
                'target_aspect_ratio': gt_box[2] / max(gt_box[3], 1e-8),
                'failure_label': classify(delta, iou_c, iou_m),
            }
            rows.append(row)
            seq_rows.append(row)
        deltas = [row['delta_iou'] for row in seq_rows]
        sequence_rows.append({
            'seq': seq, 'frames_paired': count,
            'mean_iou_c': float(np.mean([row['iou_c'] for row in seq_rows])),
            'mean_iou_m': float(np.mean([row['iou_m'] for row in seq_rows])),
            'mean_delta_iou': float(np.mean(deltas)),
            'median_delta_iou': float(np.median(deltas)),
            'regression_rate': float(np.mean([row['failure_label'] == 'regression' for row in seq_rows])),
            'hard_regression_rate': float(np.mean([row['failure_label'] == 'hard_regression' for row in seq_rows])),
            'gain_rate': float(np.mean([row['failure_label'] == 'gain' for row in seq_rows])),
        })

    if not rows:
        raise RuntimeError('no frame pairs were produced')
    fields = list(rows[0])
    with open(output_dir / 'frame_level_pairs.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / 'sequence_level_pairs.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    manifest = {
        'experiment': 'mhsa_failure_analysis_v1', 'stage': 'A',
        'control_name': args.control_name, 'm1_name': args.m1_name,
        'control_dir': str(control_dir), 'm1_dir': str(m1_dir), 'gt_dir': str(gt_dir),
        'sequences_common': len(names), 'sequences_paired': len(sequence_rows),
        'frames_paired': len(rows), 'skipped': skipped,
        'thresholds': {'hard_regression': -0.10, 'regression': -0.05, 'gain': 0.05, 'common_failure_iou': 0.30},
        'read_only_inputs': True,
    }
    with open(output_dir.parent / 'manifest' / 'pair_collection_manifest.json', 'w') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
