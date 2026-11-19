#!/usr/bin/env python3
"""Analyze FRCB admission evidence with sequence-clustered summaries."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def f(row, key):
    value = row.get(key, 'NA')
    return float(value) if value not in ('', 'NA', None) else np.nan


def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0: return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def cluster_bootstrap(rows, key, group, seed=42, n=2000):
    grouped = defaultdict(list)
    for row in rows:
        if row['group'] == group: grouped[row['sequence']].append(f(row, key))
    values = {seq: np.nanmean(v) for seq, v in grouped.items()}
    values = np.asarray([v for v in values.values() if np.isfinite(v)])
    if len(values) < 3: return {'n_sequences': int(len(values)), 'ci95': None}
    rng = np.random.RandomState(seed)
    samples = [np.mean(values[rng.randint(0, len(values), len(values))]) for _ in range(n)]
    return {'n_sequences': int(len(values)), 'mean': float(values.mean()),
            'ci95': [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.input)))
    for row in rows:
        current = f(row, 'current_iou')
        future = np.nanmean([f(row, 'next_{}_iou'.format(h)) for h in range(1, 6)])
        row['future_delta_iou'] = future - current if np.isfinite(future) else np.nan
        row['future_loss'] = current - future if np.isfinite(future) else np.nan
        row['hf_reliability'] = f(row, 'fg_LH_coherence') + f(row, 'fg_HL_coherence') + f(row, 'fg_HH_coherence')
        row['hf_aligned_reliability'] = f(row, 'fg_LH_aligned_coherence') + f(row, 'fg_HL_aligned_coherence') + f(row, 'fg_HH_aligned_coherence')
    result = {'protocol': 'FRCB-ADMISSION-V1', 'rows': len(rows),
              'sequences': len(set(r['sequence'] for r in rows)),
              'layers': sorted(set(r['layer'] for r in rows)),
              'directions': sorted(set(r['direction'] for r in rows)), 'bands': sorted(set(r['band'] for r in rows)),
              'gate': {'status': 'DO_NOT_TRAIN', 'reasons': []}, 'groups': {}}
    # The collector records fixed train events, not pre-labeled gain/regression groups.
    for layer in result['layers']:
        for direction in result['directions']:
            subset = [r for r in rows if r['layer'] == layer and r['direction'] == direction]
            x = np.asarray([r['hf_reliability'] for r in subset], dtype=float)
            y = np.asarray([r['future_loss'] for r in subset], dtype=float)
            result['groups']['layer{}_{}'.format(layer, direction)] = {
                'n_events': len(subset), 'mean_hf_fg': float(np.nanmean(x)),
                'mean_hf_bg': float(np.nanmean([f(r, 'bg_LH_coherence') + f(r, 'bg_HL_coherence') + f(r, 'bg_HH_coherence') for r in subset])),
                'mean_hf_fg_bg_discrepancy': float(np.nanmean([f(r, 'fg_LH_discrepancy') + f(r, 'fg_HL_discrepancy') + f(r, 'fg_HH_discrepancy') - f(r, 'bg_LH_discrepancy') - f(r, 'bg_HL_discrepancy') - f(r, 'bg_HH_discrepancy') for r in subset])),
                'aligned_minus_same_fg': float(np.nanmean([r['hf_aligned_reliability'] - r['hf_reliability'] for r in subset])),
                'corr_hf_reliability_future_loss': corr(x, y),
                'corr_aligned_minus_same_future_loss': corr(np.asarray([r['hf_aligned_reliability'] - r['hf_reliability'] for r in subset]), y),
            }
    reasons = []
    if result['sequences'] < 24:
        reasons.append('smoke_or_insufficient_sequence_clusters')
    if len(result['layers']) < 2: reasons.append('fewer_than_two_layers')
    correlations = [v['corr_hf_reliability_future_loss'] for v in result['groups'].values() if v['corr_hf_reliability_future_loss'] is not None]
    if not correlations or max(abs(x) for x in correlations) < 0.20:
        reasons.append('no_clear_future_iou_relation')
    if not any(v['aligned_minus_same_fg'] > 0.02 for v in result['groups'].values()):
        reasons.append('local_shift_does_not_recover_hf_correspondence')
    result['gate']['reasons'] = reasons
    result['gate']['status'] = 'CONDITIONAL_REVIEW' if not reasons else 'DO_NOT_TRAIN'
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
