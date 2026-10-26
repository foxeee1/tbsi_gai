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
CONTROL_NAME = 'M1_STSER_SOT_QUICK3_PLAIN_20261025'
CANDIDATE_NAME = 'M1_STSER_SOT_QUICK3_COMPETITION_20261025'
MANIFEST = ROOT / 'output/experiments/M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024/rpp_v2_60_manifest.txt'
CONTROL_RESULTS = ROOT / 'output/test/tracking_results/tbsi_track/{}_plain_sot_ep3'.format(CONTROL_NAME)
CANDIDATE_RESULTS = ROOT / 'output/test/tracking_results/tbsi_track/{}_competition_sot_ep3'.format(CANDIDATE_NAME)
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


def sequence_metrics(sequence, path):
    boxes = np.asarray(load_text(str(path), delimiter=('\t', ','), dtype=np.float64))
    prediction = torch.as_tensor(boxes, dtype=torch.float64)
    ground_truth = torch.as_tensor(sequence.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
    overlap, center, center_norm, _ = calc_seq_err_robust(prediction, ground_truth, sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0001, 0.05, dtype=torch.float64)
    success = (overlap[:, None] > thresholds[None, :]).double().mean(0)
    return {
        'auc': float(success.mean() * 100),
        'op50': float(success[10] * 100),
        'op75': float(success[15] * 100),
        'precision': float((center <= 20).double().mean() * 100),
        'norm_precision': float((center_norm <= 0.2).double().mean() * 100),
        'mean_iou': float(overlap.double().mean()),
        'frames': int(overlap.numel()),
    }


def bootstrap(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def main():
    names = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    sequences = get_dataset('rpp_lasher_test')
    if len(names) != 60 or [sequence.name for sequence in sequences] != names:
        raise RuntimeError('dataset order does not match the locked RPP-v2 manifest')

    keys = ('auc', 'precision', 'norm_precision', 'op50', 'op75', 'mean_iou')
    rows = []
    for sequence in sequences:
        control = sequence_metrics(sequence, CONTROL_RESULTS / (sequence.name + '.txt'))
        candidate = sequence_metrics(sequence, CANDIDATE_RESULTS / (sequence.name + '.txt'))
        row = {'sequence': sequence.name, 'frames': control['frames']}
        for key in keys:
            row['control_' + key] = control[key]
            row['candidate_' + key] = candidate[key]
            row['delta_' + key] = candidate[key] - control[key]
        rows.append(row)

    out = ROOT / 'output/experiments' / CANDIDATE_NAME
    summary = {
        'sequence_count': len(rows),
        'manifest_sha256': sha256(MANIFEST),
        'control_checkpoint_sha256': sha256(ROOT / 'output/experiments' / CONTROL_NAME / 'checkpoints/TBSITrack_ep0003.pth.tar'),
        'candidate_checkpoint_sha256': sha256(ROOT / 'output/experiments' / CANDIDATE_NAME / 'checkpoints/TBSITrack_ep0003.pth.tar'),
        'control_mean': {key: float(np.mean([row['control_' + key] for row in rows])) for key in keys},
        'candidate_mean': {key: float(np.mean([row['candidate_' + key] for row in rows])) for key in keys},
        'paired_candidate_minus_control': {},
    }
    for key in keys:
        delta = np.asarray([row['delta_' + key] for row in rows])
        summary['paired_candidate_minus_control'][key] = {
            'mean': float(delta.mean()), 'median': float(np.median(delta)),
            'ci95_sequence_bootstrap': bootstrap(delta),
            'wins': int((delta > 1e-12).sum()),
            'losses': int((delta < -1e-12).sum()),
            'ties': int((np.abs(delta) <= 1e-12).sum()),
        }
    with (out / 'paired_rpp60_per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'paired_rpp60_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
