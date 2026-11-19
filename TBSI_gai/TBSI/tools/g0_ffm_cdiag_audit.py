#!/usr/bin/env python3
"""Paired, sequence-level audit of the existing Cdiag versus C0 result."""
import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap(values, seed=42, n=20000):
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    draws = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return {'mean': float(x.mean()), 'lo95': float(np.quantile(draws, .025)),
            'hi95': float(np.quantile(draws, .975)), 'n_sequences': len(x)}


def summary(rows):
    x = np.asarray([r['delta_auc'] for r in rows], dtype=float)
    y = np.asarray([r['delta_iou'] for r in rows], dtype=float)
    order = np.argsort(x)
    k = max(1, int(len(x) * .10))
    trimmed = x[k:len(x) - k]
    max_gain_idx = int(np.argmax(x))
    without_max = np.delete(x, max_gain_idx)
    return {
        'mean_delta_auc': float(x.mean()),
        'median_delta_auc': float(np.median(x)),
        'paired_delta_auc_trimmed_mean_10pct': float(trimmed.mean()),
        'paired_delta_iou_trimmed_mean_10pct': float(np.sort(y)[k:len(y)-k].mean()),
        'bootstrap_delta_auc_sequence_95': bootstrap(x),
        'bootstrap_delta_iou_sequence_95': bootstrap(y, seed=43),
        'without_largest_gain_mean_delta_auc': float(without_max.mean()),
        'largest_gain_sequence': rows[max_gain_idx]['sequence'],
        'largest_gain_delta_auc': float(x[max_gain_idx]),
        'wins_auc': int((x > 0).sum()), 'losses_auc': int((x < 0).sum()),
        'ties_auc': int((x == 0).sum()),
        'regressions_ge_005': int((x <= -.05).sum()),
        'regressions_ge_010': int((x <= -.10).sum()),
        'largest_loss_delta_auc': float(x.min()),
        'largest_loss_sequence': rows[int(np.argmin(x))]['sequence'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', required=True)
    ap.add_argument('--windows', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    analysis = json.loads(Path(args.analysis).read_text())
    windows = json.loads(Path(args.windows).read_text())
    result = {
        'parent': 'G0-FFM-CLOSEDLOOP-STOP-20261005',
        'question': 'Is the observed Cdiag gain supported by paired robust sequence statistics rather than a few extreme trajectories?',
        'cdiag_vs_c0': summary(analysis['per_sequence_deltas']['Cdiag']),
        'event_window_cdiag_vs_c0': {
            'mean_delta_iou': float(np.mean([x['Cdiag']['mean_iou'] - x['C0']['mean_iou'] for x in windows['events']])),
            'wins_window_iou': int(sum(x['Cdiag']['mean_iou'] > x['C0']['mean_iou'] for x in windows['events'])),
            'losses_window_iou': int(sum(x['Cdiag']['mean_iou'] < x['C0']['mean_iou'] for x in windows['events'])),
            'recovery_rate_delta': float(windows['summary']['Cdiag']['recovery_rate'] - windows['summary']['C0']['recovery_rate']),
        },
        'decision': 'PROCEED_CDIAG_SEED3407_IF_PARITY_PASSES',
        'note': 'This is a development-set paired audit, not independent validation.'
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
