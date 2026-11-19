#!/usr/bin/env python3
"""Evaluate relation-bias quick3 on the frozen 12-sequence cohort."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / 'output/experiments/M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024'
MANIFEST = EXP / 'manifest.txt'
RESULTS = ROOT / 'output/test/tracking_results/tbsi_track/M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024_relation_bias_ep3'
BASELINE = ROOT / 'output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/per_sequence_metrics.csv'
os.environ['TBSI_TRACE_TRAIN_LIST'] = str(MANIFEST)

from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.utils.load_text import load_text


def evaluate(prediction, sequence):
    gt = torch.as_tensor(sequence.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
    overlap, center, center_norm, _ = calc_seq_err_robust(prediction, gt, sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0001, 0.05, dtype=torch.float64)
    success = (overlap[:, None] > thresholds[None, :]).double().mean(0)
    return {
        'auc': float(success.mean() * 100.0), 'op50': float(success[10] * 100.0),
        'op75': float(success[15] * 100.0), 'precision': float((center <= 20).double().mean() * 100.0),
        'norm_precision': float((center_norm <= 0.2).double().mean() * 100.0),
        'overlap': overlap.numpy(),
    }


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    expected = [x.strip() for x in MANIFEST.read_text().splitlines() if x.strip()]
    sequences = get_dataset('m1_trace_lasher_train')
    if [s.name for s in sequences] != expected:
        raise RuntimeError('manifest order mismatch')
    baseline = {}
    with BASELINE.open() as f:
        for row in csv.DictReader(f):
            if row['arm'] == 'lam0':
                baseline[row['sequence']] = {k: float(row[k]) for k in ('auc', 'op50', 'op75', 'precision', 'norm_precision')}
    rows, trajectory, deltas = [], {}, []
    for seq in sequences:
        path = RESULTS / (seq.name + '.txt')
        if not path.is_file():
            raise FileNotFoundError(str(path))
        boxes = np.asarray(load_text(str(path), delimiter=('\t', ','), dtype=np.float64))
        values = evaluate(torch.as_tensor(boxes, dtype=torch.float64), seq)
        row = {'sequence': seq.name, **{k: values[k] for k in ('auc', 'op50', 'op75', 'precision', 'norm_precision')}, 'frames': int(len(values['overlap']))}
        row.update({f'{k}_delta_vs_lam0': values[k] - baseline[seq.name][k] for k in ('auc', 'op50', 'op75', 'precision', 'norm_precision')})
        rows.append(row)
        deltas.append(row['auc_delta_vs_lam0'])
        trajectory[seq.name] = {'first_bbox_divergence_frame': None, 'changed_bbox_fraction': None}
    summary = {
        'sequence_count': len(sequences), 'manifest_sha256': sha256(MANIFEST),
        'candidate_macro_metrics': {k: float(np.mean([r[k] for r in rows])) for k in ('auc', 'op50', 'op75', 'precision', 'norm_precision')},
        'lam0_macro_metrics_from_record': {k: float(np.mean([baseline[s][k] for s in expected])) for k in ('auc', 'op50', 'op75', 'precision', 'norm_precision')},
        'paired_candidate_vs_lam0': {
            'mean_auc_delta': float(np.mean(deltas)), 'median_auc_delta': float(np.median(deltas)),
            'wins': int(sum(x > 1e-12 for x in deltas)), 'losses': int(sum(x < -1e-12 for x in deltas)),
            'ties': int(sum(abs(x) <= 1e-12 for x in deltas)), 'per_sequence_auc_delta': dict(zip(expected, map(float, deltas))),
        },
    }
    with (EXP / 'per_sequence_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (EXP / 'paired_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
