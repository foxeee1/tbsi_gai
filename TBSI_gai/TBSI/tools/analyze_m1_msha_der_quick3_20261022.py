#!/usr/bin/env python3
"""Paired metrics for the DER quick3 train-side replay."""
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
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.utils.load_text import load_text

EXP = ROOT / 'output/experiments/M1_MSHA_DER_QUICK3_20261022'
MANIFEST = ROOT / 'output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/manifests/sequence_list.txt'
RESULTS = ROOT / 'output/test/tracking_results/tbsi_track'
ARMS = {
    'parent_lam0': RESULTS / 'M1_EARLY_GATE_STRENGTH_REPLAY_20261022_lam0',
    'der_ep3': RESULTS / 'M1_MSHA_DER_QUICK3_20261022_der_ep3',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate(prediction, sequence):
    gt = torch.as_tensor(sequence.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
    overlap, center, center_norm, _ = calc_seq_err_robust(prediction, gt, sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0001, 0.05, dtype=torch.float64)
    success = (overlap[:, None] > thresholds[None, :]).double().mean(0)
    return {
        'auc': float(success.mean() * 100.0),
        'op50': float(success[10] * 100.0),
        'op75': float(success[15] * 100.0),
        'precision': float((center <= 20).double().mean() * 100.0),
        'norm_precision': float((center_norm <= 0.2).double().mean() * 100.0),
        'overlap': overlap.numpy(),
    }


def main():
    os.environ['TBSI_TRACE_TRAIN_LIST'] = str(MANIFEST)
    sequences = get_dataset('m1_trace_lasher_train')
    expected = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    if len(expected) != 12 or [seq.name for seq in sequences] != expected:
        raise RuntimeError('dataset order or cohort does not match frozen 12-sequence manifest')

    rows, summaries, trajectory = [], {arm: [] for arm in ARMS}, {}
    for sequence in sequences:
        values, boxes = {}, {}
        for arm, folder in ARMS.items():
            path = folder / (sequence.name + '.txt')
            if not path.is_file():
                raise FileNotFoundError(str(path))
            boxes[arm] = np.asarray(load_text(str(path), delimiter=('\t', ','), dtype=np.float64))
            values[arm] = evaluate(torch.as_tensor(boxes[arm], dtype=torch.float64), sequence)
            summaries[arm].append(values[arm])
            rows.append({
                'sequence': sequence.name,
                'arm': arm,
                **{key: values[arm][key] for key in ('auc', 'op50', 'op75', 'precision', 'norm_precision')},
                'frames': int(values[arm]['overlap'].shape[0]),
            })
        base = boxes['parent_lam0']
        candidate = boxes['der_ep3']
        delta_iou = values['der_ep3']['overlap'] - values['parent_lam0']['overlap']
        changed = np.flatnonzero(np.max(np.abs(candidate - base), axis=1) > 1e-6)
        trajectory[sequence.name] = {
            'first_bbox_divergence_frame': int(changed[0]) if len(changed) else None,
            'changed_bbox_fraction': float(np.mean(np.max(np.abs(candidate - base), axis=1) > 1e-6)),
            'mean_delta_iou_after_init': float(np.mean(delta_iou[1:])) if len(delta_iou) > 1 else 0.0,
            'first5_delta_iou': float(np.mean(delta_iou[1:6])) if len(delta_iou) > 1 else 0.0,
            'first20_delta_iou': float(np.mean(delta_iou[1:min(21, len(delta_iou))])) if len(delta_iou) > 1 else 0.0,
        }

    summary = {
        'sequence_count': len(sequences),
        'manifest_sha256': sha256(MANIFEST),
        'arms': {
            arm: {key: float(np.mean([row[key] for row in summaries[arm]]))
                  for key in ('auc', 'op50', 'op75', 'precision', 'norm_precision')}
            for arm in ARMS
        },
        'paired_der_vs_parent': {},
        'trajectory': trajectory,
    }
    by_key = {(row['sequence'], row['arm']): row for row in rows}
    deltas = [by_key[(name, 'der_ep3')]['auc'] - by_key[(name, 'parent_lam0')]['auc'] for name in expected]
    summary['paired_der_vs_parent'] = {
        'mean_auc_delta': float(np.mean(deltas)),
        'median_auc_delta': float(np.median(deltas)),
        'wins': int(sum(delta > 0 for delta in deltas)),
        'losses': int(sum(delta < 0 for delta in deltas)),
        'ties': int(sum(delta == 0 for delta in deltas)),
        'per_sequence_auc_delta': {name: float(delta) for name, delta in zip(expected, deltas)},
    }
    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / 'per_sequence_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (EXP / 'trajectory_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence'] + list(next(iter(trajectory.values()))))
        writer.writeheader()
        writer.writerows({'sequence': name, **values} for name, values in trajectory.items())
    (EXP / 'paired_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
