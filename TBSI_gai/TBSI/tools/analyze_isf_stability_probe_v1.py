#!/usr/bin/env python
"""Offline statistics for ISF stability probe v1."""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr


def read_rows(path):
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('iou', 'clean_iou', 'iou_drop', 'msg_cos_drop', 'msg_rel_l2'):
            row[key] = float(row[key])
    return rows


def sequence_bootstrap(values, seqs, seed=42, reps=5000):
    values, seqs = np.asarray(values, float), np.asarray(seqs)
    unique = np.unique(seqs)
    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(reps):
        selected = rng.choice(unique, size=len(unique), replace=True)
        estimates.append(np.mean([values[seqs == seq].mean() for seq in selected]))
    return {'mean': float(values.mean()), 'ci95_low': float(np.percentile(estimates, 2.5)),
            'ci95_high': float(np.percentile(estimates, 97.5)), 'unit': 'sequence', 'n_sequences': int(len(unique))}


def summarize(rows):
    if len(rows) < 3:
        return {'n_events': len(rows), 'error': 'too_few_events'}
    x, y = np.asarray([r['msg_cos_drop'] for r in rows]), np.asarray([r['iou_drop'] for r in rows])
    pearson, pearson_p = pearsonr(x, y)
    spearman, spearman_p = spearmanr(x, y)
    q90 = np.quantile(y, 0.90)
    trimmed = [r for r in rows if r['iou_drop'] <= q90]
    tx, ty = np.asarray([r['msg_cos_drop'] for r in trimmed]), np.asarray([r['iou_drop'] for r in trimmed])
    return {'n_events': len(rows), 'n_sequences': len(set(r['seq'] for r in rows)),
            'msg_cos_drop': sequence_bootstrap(x, [r['seq'] for r in rows]),
            'iou_drop': sequence_bootstrap(y, [r['seq'] for r in rows]),
            'msg_rel_l2_mean': float(np.mean([r['msg_rel_l2'] for r in rows])),
            'pearson_r': float(pearson), 'pearson_p': float(pearson_p),
            'spearman_rho': float(spearman), 'spearman_p': float(spearman_p),
            'trimmed_n_events': len(trimmed),
            'trimmed_spearman_rho': float(spearmanr(tx, ty)[0]) if len(trimmed) >= 3 else None,
            'positive_direction_fraction': float(np.mean(y > 0.0))}


def analyze(input_path, output_dir, expected_events):
    rows = read_rows(input_path)
    events = sorted(set((r['event_id'], r['seq']) for r in rows))
    if len(events) != expected_events or set(r['condition'] for r in rows) != {'00', 'R_target_blur', 'T_target_blur', 'R_background_blur'}:
        raise ValueError('invalid coverage: events={} conditions={}'.format(len(events), sorted(set(r['condition'] for r in rows))))
    pairs = {}
    for row in rows:
        pairs[(row['event_id'], row['condition'])] = row
    summaries = {}
    for condition in ('R_target_blur', 'T_target_blur', 'R_background_blur'):
        paired = []
        for event_id, seq in events:
            clean, pert = pairs[(event_id, '00')], pairs[(event_id, condition)]
            item = dict(pert); item['seq'] = seq
            item['iou_drop'] = clean['iou'] - pert['iou']
            paired.append(item)
        summaries[condition] = summarize(paired)
    result = {'experiment': 'isf_stability_probe_v1', 'input_csv': os.path.abspath(input_path),
              'input_sha256': hashlib.sha256(open(input_path, 'rb').read()).hexdigest(),
              'expected_events': expected_events, 'conditions': summaries,
              'interpretation_rule': 'positive association of message change and IoU drop is required for mechanism support'}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'analysis.json'), 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-events', type=int, default=72)
    args = parser.parse_args()
    analyze(args.input, args.output_dir, args.expected_events)


if __name__ == '__main__':
    main()
