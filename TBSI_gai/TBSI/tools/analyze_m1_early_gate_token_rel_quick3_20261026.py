#!/usr/bin/env python3
"""Paired official-metric summary for the Early-gate token residual quick7 screen."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/TBSI_gai/TBSI')
sys.path.insert(0, str(ROOT))

from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text


CONTROL = 'M1_EARLY_GATE_TOKEN_REL_QUICK3_CONTROL_RETRY1_20261026'
CANDIDATE = 'M1_EARLY_GATE_TOKEN_REL_QUICK3_CANDIDATE_RETRY1_20261026'
CONTROL_RUN = 'early_gate_sot_ep3'
CANDIDATE_RUN = 'reliability_sot_ep3'


def read_predictions(name, run_id, sequence):
    path = ROOT / 'output/test/tracking_results/tbsi_track/{}_{}'.format(name, run_id) / (sequence.name + '.txt')
    if not path.is_file():
        raise FileNotFoundError(str(path))
    values = load_text(str(path), delimiter=('\t', ','), dtype=np.float64)
    return torch.as_tensor(values, dtype=torch.float64)


def measure(prediction, sequence):
    annotation = torch.as_tensor(sequence.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
    overlap, center, norm_center, valid = calc_seq_err_robust(
        prediction, annotation, sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    center_thresholds = torch.arange(0, 51, dtype=torch.float64)
    norm_thresholds = torch.arange(0, 51, dtype=torch.float64) / 100.0
    auc_curve = (overlap[:, None] > thresholds[None, :]).double().mean(0)
    precision_curve = (center[:, None] <= center_thresholds[None, :]).double().mean(0)
    norm_precision_curve = (norm_center[:, None] <= norm_thresholds[None, :]).double().mean(0)
    return {
        'auc': float(auc_curve.mean() * 100.0),
        'ao': float(overlap[valid].mean() * 100.0),
        'precision': float(precision_curve[20] * 100.0),
        'norm_precision': float(norm_precision_curve[20] * 100.0),
        'op50': float((overlap > 0.50).double().mean() * 100.0),
        'op75': float((overlap > 0.75).double().mean() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-list', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    os.environ['TBSI_RPP_TEST_LIST'] = str(Path(args.sequence_list).resolve())
    sequences = get_dataset('rpp_lasher_test')
    if len(sequences) != 7:
        raise RuntimeError('expected 7 fixed development sequences, got {}'.format(len(sequences)))

    rows = []
    for sequence in sequences:
        control = measure(read_predictions(CONTROL, CONTROL_RUN, sequence), sequence)
        candidate = measure(read_predictions(CANDIDATE, CANDIDATE_RUN, sequence), sequence)
        row = {'sequence': sequence.name}
        for metric in control:
            row['control_' + metric] = control[metric]
            row['candidate_' + metric] = candidate[metric]
            row['delta_' + metric] = candidate[metric] - control[metric]
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'per_sequence_metrics.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = ('auc', 'ao', 'precision', 'norm_precision', 'op50', 'op75')
    summary = {
        'control': CONTROL,
        'candidate': CANDIDATE,
        'sequence_count': len(rows),
        'metrics_macro_mean': {},
        'paired_sequence_deltas': {},
        'auc_wins': 0,
        'auc_losses': 0,
        'auc_ties': 0,
        'sequence_manifest': str(Path(args.sequence_list).resolve()),
    }
    for metric in metric_names:
        deltas = np.asarray([row['delta_' + metric] for row in rows], dtype=np.float64)
        summary['metrics_macro_mean'][metric] = {
            'control': float(np.mean([row['control_' + metric] for row in rows])),
            'candidate': float(np.mean([row['candidate_' + metric] for row in rows])),
            'delta': float(deltas.mean()),
        }
        summary['paired_sequence_deltas'][metric] = [float(value) for value in deltas]
    for row in rows:
        delta = row['delta_auc']
        summary['auc_wins'] += int(delta > 1e-9)
        summary['auc_losses'] += int(delta < -1e-9)
        summary['auc_ties'] += int(abs(delta) <= 1e-9)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
