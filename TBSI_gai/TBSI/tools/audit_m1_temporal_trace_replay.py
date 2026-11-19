#!/usr/bin/env python
"""Read-only audit of the frozen M1 temporal trace contract.

This is deliberately not a model re-forward. It checks that the stored
per-frame state/crop/raw-token assets can form a causal replay chain and
reports crop coverage for later G0 selection.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np


def close_list(a, b, tol=1e-5):
    return (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
            and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b)))


def intersection_area(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def audit(root):
    errors = []
    sequences = []
    rows_total = 0
    history_total = 0
    raw_link_ok = 0
    crop_inside = 0
    gt_valid = 0
    state_links = 0
    transform_checks = 0
    center_distances = []
    area_ratios = []

    for seq in sorted(os.listdir(root)):
        path = os.path.join(root, seq, 'frames.jsonl')
        if not os.path.isfile(path):
            continue
        rows = [json.loads(line) for line in open(path) if line.strip()]
        if not rows:
            errors.append('%s:empty' % seq)
            continue
        sequences.append(seq)
        rows_total += len(rows)
        raw_arrays = []
        for index, row in enumerate(rows):
            raw_rel = row.get('raw_head_file')
            if not raw_rel:
                errors.append('%s:%d:missing_raw_file_field' % (seq, index))
                raw_arrays.append(None)
                continue
            raw_path = os.path.join(root, seq, raw_rel)
            try:
                raw = np.load(raw_path, allow_pickle=False)
                digest = hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float32).tobytes()).hexdigest()
                if list(raw.shape) != [1, 256, 768] or digest != row.get('raw_head_sha256'):
                    errors.append('%s:%d:raw_mismatch' % (seq, index))
                raw_arrays.append(raw)
            except Exception as exc:
                errors.append('%s:%d:raw_read:%s' % (seq, index, exc))
                raw_arrays.append(None)

            crop = row.get('current_crop_xywh')
            transform = row.get('current_crop_transform')
            if isinstance(crop, list) and isinstance(transform, list) and len(transform) == 2:
                expected_scale = 256.0 / float(crop[2])
                if (abs(float(transform[0][0]) - expected_scale) > 1e-5
                        or abs(float(transform[1][1]) - expected_scale) > 1e-5
                        or abs(float(transform[0][2]) + expected_scale * float(crop[0])) > 1e-4
                        or abs(float(transform[1][2]) + expected_scale * float(crop[1])) > 1e-4):
                    errors.append('%s:%d:crop_transform_mismatch' % (seq, index))
                else:
                    transform_checks += 1

            gt = row.get('gt_box')
            if row.get('gt_valid') and isinstance(gt, list):
                gt_valid += 1
                if intersection_area(crop, gt) > 0.0:
                    crop_inside += 1
                cx = float(gt[0]) + 0.5 * float(gt[2])
                cy = float(gt[1]) + 0.5 * float(gt[3])
                ccx = float(crop[0]) + 0.5 * float(crop[2])
                ccy = float(crop[1]) + 0.5 * float(crop[3])
                center_distances.append(math.hypot(cx - ccx, cy - ccy) / max(float(crop[2]), 1.0) * 16.0)
                area_ratios.append((float(gt[2]) * float(gt[3])) / max(float(crop[2]) * float(crop[3]), 1.0))

            if index == 0:
                continue
            history_total += 1
            prev = rows[index - 1]
            if int(row.get('history_frame_id', -1)) != int(prev.get('frame_id', -2)):
                errors.append('%s:%d:history_frame_link' % (seq, index))
            elif not close_list(row.get('history_crop_xywh'), prev.get('current_crop_xywh')):
                errors.append('%s:%d:history_crop_link' % (seq, index))
            elif row.get('history_crop_transform') != prev.get('current_crop_transform'):
                errors.append('%s:%d:history_transform_link' % (seq, index))
            elif not close_list(row.get('tracker_state_before'), prev.get('tracker_state_after')):
                errors.append('%s:%d:state_link' % (seq, index))
            else:
                state_links += 1
            if raw_arrays[index] is not None and raw_arrays[index - 1] is not None:
                raw_link_ok += 1

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    summary = {
        'sequences': len(sequences),
        'rows': rows_total,
        'history_rows': history_total,
        'raw_history_links_checked': raw_link_ok,
        'state_links_pass': state_links,
        'transform_checks_pass': transform_checks,
        'gt_valid_rows': gt_valid,
        'gt_inside_current_crop': crop_inside,
        'gt_inside_crop_rate': (float(crop_inside) / gt_valid if gt_valid else None),
        'mean_center_distance_tokens': mean_or_none(center_distances),
        'mean_gt_crop_area_ratio': mean_or_none(area_ratios),
        'errors': errors,
        'model_reforward_performed': False,
    }
    summary['decision'] = 'REPLAY_CONTRACT_PASS' if not errors and rows_total > 0 else 'STOP_REPLAY_CONTRACT'
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.trace_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['decision'] == 'REPLAY_CONTRACT_PASS' else 2


if __name__ == '__main__':
    sys.exit(main())
