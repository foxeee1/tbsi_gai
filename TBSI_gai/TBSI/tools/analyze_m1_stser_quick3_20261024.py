#!/usr/bin/env python3
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP_PLAIN = ROOT / 'output/experiments/M1_STSER_PLAIN_T2S_QUICK3_20261024'
EXP_STSER = ROOT / 'output/experiments/M1_STSER_QUICK3_20261024'
MANIFEST = ROOT / 'output/experiments/M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024/rpp_v2_60_manifest.txt'
PLAIN_RESULTS = ROOT / 'output/test/tracking_results/tbsi_track/M1_STSER_PLAIN_T2S_QUICK3_20261024_plain_t2s_ep3'
STSER_RESULTS = ROOT / 'output/test/tracking_results/tbsi_track/M1_STSER_QUICK3_20261024_stser_ep3'

os.environ['TBSI_RPP_TEST_LIST'] = str(MANIFEST)
sys.path.insert(0, str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(sequence, result_path):
    boxes = np.asarray(load_text(str(result_path), delimiter=('\t', ','), dtype=np.float64))
    prediction = torch.as_tensor(boxes, dtype=torch.float64)
    ground_truth = torch.as_tensor(sequence.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
    overlap, center, center_norm, _ = calc_seq_err_robust(
        prediction, ground_truth, sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0001, 0.05, dtype=torch.float64)
    success = (overlap[:, None] > thresholds[None, :]).double().mean(0)
    return {
        'auc': float(success.mean() * 100.0),
        'op50': float(success[10] * 100.0),
        'op75': float(success[15] * 100.0),
        'precision': float((center <= 20).double().mean() * 100.0),
        'norm_precision': float((center_norm <= 0.2).double().mean() * 100.0),
        'mean_iou': float(overlap.double().mean()),
        'frames': int(overlap.numel()),
    }


def bootstrap(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    names = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    if len(names) != 60 or len(set(names)) != 60:
        raise RuntimeError('Expected a unique 60-sequence RPP-v2 manifest')
    sequences = get_dataset('rpp_lasher_test')
    if [sequence.name for sequence in sequences] != names:
        raise RuntimeError('Dataset order does not match the locked RPP-v2 manifest')

    rows = []
    for sequence in sequences:
        plain = metrics(sequence, PLAIN_RESULTS / (sequence.name + '.txt'))
        stser = metrics(sequence, STSER_RESULTS / (sequence.name + '.txt'))
        row = {'sequence': sequence.name, 'frames': stser['frames']}
        for key in ('auc', 'precision', 'norm_precision', 'op50', 'op75', 'mean_iou'):
            row['plain_' + key] = plain[key]
            row['stser_' + key] = stser[key]
            row['delta_' + key] = stser[key] - plain[key]
        rows.append(row)

    keys = ('auc', 'precision', 'norm_precision', 'op50', 'op75', 'mean_iou')
    summary = {
        'sequence_count': len(rows),
        'manifest_sha256': sha256(MANIFEST),
        'plain_checkpoint_sha256': sha256(EXP_PLAIN / 'checkpoints/TBSITrack_ep0003.pth.tar'),
        'stser_checkpoint_sha256': sha256(EXP_STSER / 'checkpoints/TBSITrack_ep0003.pth.tar'),
        'plain_mean': {key: float(np.mean([row['plain_' + key] for row in rows])) for key in keys},
        'stser_mean': {key: float(np.mean([row['stser_' + key] for row in rows])) for key in keys},
        'paired_stser_minus_plain': {},
    }
    for key in keys:
        deltas = np.asarray([row['delta_' + key] for row in rows], dtype=np.float64)
        summary['paired_stser_minus_plain'][key] = {
            'mean': float(deltas.mean()),
            'median': float(np.median(deltas)),
            'ci95_sequence_bootstrap': bootstrap(deltas),
            'wins': int((deltas > 1e-12).sum()),
            'losses': int((deltas < -1e-12).sum()),
            'ties': int((np.abs(deltas) <= 1e-12).sum()),
        }

    (EXP_STSER / 'paired_rpp60_per_sequence.csv').write_text('')
    with (EXP_STSER / 'paired_rpp60_per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (EXP_STSER / 'paired_rpp60_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
