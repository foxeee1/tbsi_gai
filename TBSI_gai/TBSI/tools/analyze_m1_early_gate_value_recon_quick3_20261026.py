#!/usr/bin/env python3
"""Paired official-metric summary for the Value reconstruction quick3 screen."""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text

CONTROL = 'M1_EARLY_GATE_VALUE_RECON_QUICK3_CONTROL_20261026'
CANDIDATE = 'M1_EARLY_GATE_VALUE_RECON_QUICK3_CANDIDATE_20261026'
CONTROL_RUN = 'control_ep3'
CANDIDATE_RUN = 'candidate_ep3'


def measure(path, sequence):
    prediction = torch.as_tensor(load_text(str(path), delimiter=('\t', ','), dtype=np.float64), dtype=torch.float64)
    annotation = torch.as_tensor(sequence.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
    overlap, center, norm_center, valid = calc_seq_err_robust(prediction, annotation, sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    center_thresholds = torch.arange(0, 51, dtype=torch.float64)
    norm_thresholds = torch.arange(0, 51, dtype=torch.float64) / 100.0
    return {
        'auc': float((overlap[:, None] > thresholds[None, :]).double().mean(0).mean() * 100),
        'ao': float(overlap[valid].mean() * 100),
        'precision': float((center[:, None] <= center_thresholds[None, :]).double().mean(0)[20] * 100),
        'norm_precision': float((norm_center[:, None] <= norm_thresholds[None, :]).double().mean(0)[20] * 100),
        'op50': float((overlap > 0.50).double().mean() * 100),
        'op75': float((overlap > 0.75).double().mean() * 100),
    }


def main():
    manifest = ROOT / 'experiments/manifests/m1_dcvr_quick7_legacy_20261025.txt'
    os.environ['TBSI_RPP_TEST_LIST'] = str(manifest.resolve())
    sequences = get_dataset('rpp_lasher_test')
    if len(sequences) != 7:
        raise RuntimeError('expected 7 fixed development sequences, got {}'.format(len(sequences)))
    rows = []
    for sequence in sequences:
        base = ROOT / 'output/test/tracking_results/tbsi_track'
        control = measure(base / '{}_{}'.format(CONTROL, CONTROL_RUN) / (sequence.name + '.txt'), sequence)
        candidate = measure(base / '{}_{}'.format(CANDIDATE, CANDIDATE_RUN) / (sequence.name + '.txt'), sequence)
        row = {'sequence': sequence.name}
        for metric in control:
            row['control_' + metric] = control[metric]
            row['candidate_' + metric] = candidate[metric]
            row['delta_' + metric] = candidate[metric] - control[metric]
        rows.append(row)
    out = ROOT / 'experiments/snapshots/m1_early_gate_value_recon_quick3_20261026/analysis'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_sequence_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metrics = ('auc', 'ao', 'precision', 'norm_precision', 'op50', 'op75')
    summary = {'control': CONTROL, 'candidate': CANDIDATE, 'sequence_count': len(rows),
               'manifest': str(manifest), 'metrics_macro_mean': {},
               'auc_wins': 0, 'auc_losses': 0, 'auc_ties': 0}
    for metric in metrics:
        deltas = np.asarray([r['delta_' + metric] for r in rows], dtype=np.float64)
        summary['metrics_macro_mean'][metric] = {
            'control': float(np.mean([r['control_' + metric] for r in rows])),
            'candidate': float(np.mean([r['candidate_' + metric] for r in rows])),
            'delta': float(deltas.mean()),
        }
    for row in rows:
        summary['auc_wins'] += int(row['delta_auc'] > 1e-9)
        summary['auc_losses'] += int(row['delta_auc'] < -1e-9)
        summary['auc_ties'] += int(abs(row['delta_auc']) <= 1e-9)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
