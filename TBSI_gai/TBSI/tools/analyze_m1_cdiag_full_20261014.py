"""Full per-sequence and observable information-flow analysis for M1 Cdiag.

This script keeps the official audit as the sequence-level metric source and
adds a deterministic frame-level comparison of the two prediction files.
It reports observable output changes; it does not claim access to hidden
inference activations.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def load_boxes(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                vals = [float(x) for x in line.replace(',', '\t').split()]
                rows.append(vals[:4])
    return np.asarray(rows, dtype=np.float64)


def load_gt(path):
    return np.loadtxt(path, delimiter=',', dtype=np.float64).reshape(-1, 4)


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1, iy1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = np.maximum(0, a[:, 2]) * np.maximum(0, a[:, 3])
    area_b = np.maximum(0, b[:, 2]) * np.maximum(0, b[:, 3])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/root/autodl-tmp/TBSI_gai/TBSI')
    ap.add_argument('--audit-dir', required=True)
    ap.add_argument('--candidate-dir', required=True)
    ap.add_argument('--baseline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    repo = Path(args.repo)
    audit = Path(args.audit_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(audit / 'per_sequence_official.csv')))
    detailed, frame_rows = [], []
    for r in rows:
        seq = r['sequence']
        auc_c = float(r['M1.official_sequence_curve_auc_sampled']) * 100.0
        auc_b = float(r['B0.official_sequence_curve_auc_sampled']) * 100.0
        da = auc_c - auc_b
        bucket = 'low' if auc_b < 30 else ('mid' if auc_b < 60 else 'high')
        p_c = Path(args.candidate_dir) / (seq + '.txt')
        p_b = Path(args.baseline_dir) / (seq + '.txt')
        gt_path = repo / 'data' / 'lasher' / 'testingset' / seq / 'init.txt'
        c, b, gt = load_boxes(p_c), load_boxes(p_b), load_gt(gt_path)
        n = min(len(c), len(b), len(gt))
        c, b, gt = c[:n], b[:n], gt[:n]
        # Official audit replaces frame 0 with GT. Exclude it from raw-output
        # information-flow comparison because both methods are identical there.
        ic, ib = iou(c, gt), iou(b, gt)
        changed = np.linalg.norm(c - b, axis=1) > 1e-6
        changed[0] = False
        d_iou = ic - ib
        rescue = changed & (d_iou > 0.05)
        harm = changed & (d_iou < -0.05)
        equal = changed & (np.abs(d_iou) <= 0.05)
        first = int(np.flatnonzero(changed)[0] + 1) if changed.any() else 0
        last = int(np.flatnonzero(changed)[-1] + 1) if changed.any() else 0
        # Preserve the official audit's exact sampled AUC ordering. The audit
        # labels M1=candidate and B0=fixed baseline; only exact equality is a tie.
        verdict = 'win' if da > 0 else ('loss' if da < 0 else 'tie')
        detailed.append({
            'sequence': seq, 'difficulty_bucket': bucket, 'verdict': verdict,
            'baseline_auc_percent': auc_b, 'candidate_auc_percent': auc_c,
            'auc_delta_percent': da,
            'baseline_mean_overlap': float(r['B0.official_mean_overlap_valid_frames']),
            'candidate_mean_overlap': float(r['M1.official_mean_overlap_valid_frames']),
            'mean_overlap_delta': float(r['M1.official_mean_overlap_valid_frames']) - float(r['B0.official_mean_overlap_valid_frames']),
            'baseline_failure_lt05': float(r['B0.frame_weighted_failure_rate_lt_0.5']),
            'candidate_failure_lt05': float(r['M1.frame_weighted_failure_rate_lt_0.5']),
            'failure_lt05_delta': float(r['M1.frame_weighted_failure_rate_lt_0.5']) - float(r['B0.frame_weighted_failure_rate_lt_0.5']),
            'baseline_persistent_segments': int(r['B0.persistent_failure_count']),
            'candidate_persistent_segments': int(r['M1.persistent_failure_count']),
            'persistent_segment_delta': int(r['M1.persistent_failure_count']) - int(r['B0.persistent_failure_count']),
            'baseline_longest_failure': int(r['B0.longest_failure_segment']),
            'candidate_longest_failure': int(r['M1.longest_failure_segment']),
            'longest_failure_delta': int(r['M1.longest_failure_segment']) - int(r['B0.longest_failure_segment']),
            'frames': n, 'changed_frames': int(changed.sum()),
            'changed_fraction_excl_init': float(changed[1:].mean()) if n > 1 else 0.0,
            'rescue_frames_iou_gt005': int(rescue.sum()),
            'harm_frames_iou_lt005': int(harm.sum()),
            'changed_neutral_frames': int(equal.sum()),
            'mean_iou_delta_changed': safe_mean(d_iou[changed]),
            'first_changed_frame': first, 'last_changed_frame': last,
        })
        for i in np.flatnonzero(changed):
            frame_rows.append({'sequence': seq, 'frame_1based': int(i + 1),
                               'baseline_iou': float(ib[i]), 'candidate_iou': float(ic[i]),
                               'iou_delta': float(d_iou[i]),
                               'box_l2_delta': float(np.linalg.norm(c[i] - b[i])),
                               'effect': 'rescue' if d_iou[i] > .05 else ('harm' if d_iou[i] < -.05 else 'neutral')})

    def write_csv(path, data):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()) if data else [])
            w.writeheader(); w.writerows(data)
    write_csv(out / 'per_sequence_detailed.csv', detailed)
    write_csv(out / 'changed_frames.csv', frame_rows)

    def agg(items):
        if not items: return {'n': 0}
        def mean(k): return safe_mean(np.asarray([x[k] for x in items], dtype=float))
        return {
            'n': len(items), 'wins': sum(x['verdict'] == 'win' for x in items),
            'losses': sum(x['verdict'] == 'loss' for x in items),
            'ties': sum(x['verdict'] == 'tie' for x in items),
            'mean_auc_delta_percent': mean('auc_delta_percent'),
            'median_auc_delta_percent': float(np.median([x['auc_delta_percent'] for x in items])),
            'mean_overlap_delta': mean('mean_overlap_delta'),
            'mean_failure_lt05_delta': mean('failure_lt05_delta'),
            'persistent_segment_delta_sum': sum(x['persistent_segment_delta'] for x in items),
            'mean_changed_fraction_excl_init': mean('changed_fraction_excl_init'),
            'rescue_frames': sum(x['rescue_frames_iou_gt005'] for x in items),
            'harm_frames': sum(x['harm_frames_iou_lt005'] for x in items),
            'neutral_changed_frames': sum(x['changed_neutral_frames'] for x in items),
            'mean_iou_delta_changed': mean('mean_iou_delta_changed'),
        }
    summary = {'contract': {'n_sequences': len(detailed), 'first_frame_excluded_from_output_change': True,
                            'rescue_threshold_iou_delta': 0.05, 'harm_threshold_iou_delta': -0.05,
                            'note': 'Frame comparison is observable output attribution, not hidden activation causality.'},
               'all': agg(detailed), 'buckets': {b: agg([x for x in detailed if x['difficulty_bucket'] == b]) for b in ['low','mid','high']},
               'verdict_groups': {v: agg([x for x in detailed if x['verdict'] == v]) for v in ['win','loss','tie']},
               'top_gains': sorted(detailed, key=lambda x: x['auc_delta_percent'], reverse=True)[:20],
               'top_losses': sorted(detailed, key=lambda x: x['auc_delta_percent'])[:20]}
    json.dump(summary, open(out / 'summary.json', 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
