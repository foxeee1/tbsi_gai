#!/usr/bin/env python3
"""Official-metric and trajectory summary for the early-gate strength replay."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.utils.load_text import load_text

EXP = ROOT / 'output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022'
MANIFEST = EXP / 'manifests/sequence_list.txt'
RESULTS = ROOT / 'output/test/tracking_results/tbsi_track'
ARMS = ('lam0', 'lam05', 'lam1')
STRENGTHS = {'lam0': 0.0, 'lam05': 0.5, 'lam1': 1.0}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(pred, seq):
    gt = torch.as_tensor(seq.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
    overlap, center, center_norm, valid = calc_seq_err_robust(pred, gt, seq.dataset, visible)
    thresholds = torch.arange(0.0, 1.0001, 0.05, dtype=torch.float64)
    success = (overlap[:, None] > thresholds[None, :]).double().mean(0)
    precision = (center <= 20).double().mean()
    norm_precision = (center_norm <= 0.2).double().mean()
    return {
        'auc': float(success.mean() * 100.0),
        'op50': float(success[10] * 100.0),
        'op75': float(success[15] * 100.0),
        'precision': float(precision * 100.0),
        'norm_precision': float(norm_precision * 100.0),
        'overlap': overlap.numpy(),
    }


def main():
    os.environ['TBSI_TRACE_TRAIN_LIST'] = str(MANIFEST)
    sequences = get_dataset('m1_trace_lasher_train')
    expected = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    if [seq.name for seq in sequences] != expected or len(sequences) != 12:
        raise RuntimeError('dataset sequence order does not match frozen manifest')

    per_sequence = []
    all_metrics = {arm: [] for arm in ARMS}
    trajectory = {}
    for seq in sequences:
        arm_values = {}
        arm_predictions = {}
        for arm in ARMS:
            path = RESULTS / ('M1_EARLY_GATE_STRENGTH_REPLAY_20261022_' + arm) / (seq.name + '.txt')
            if not path.is_file():
                raise FileNotFoundError(str(path))
            pred = torch.as_tensor(load_text(str(path), delimiter=('\t', ','), dtype=np.float64),
                                   dtype=torch.float64)
            arm_predictions[arm] = pred.numpy()
            values = metrics(pred, seq)
            all_metrics[arm].append(values)
            arm_values[arm] = values
            per_sequence.append({
                'sequence': seq.name, 'arm': arm, 'lambda': STRENGTHS[arm],
                **{key: values[key] for key in ('auc', 'op50', 'op75', 'precision', 'norm_precision')},
                'frames': int(values['overlap'].shape[0]),
            })
        base = arm_values['lam0']['overlap']
        for arm in ('lam05', 'lam1'):
            candidate = arm_values[arm]['overlap']
            delta = candidate - base
            pred_delta = arm_predictions[arm] - arm_predictions['lam0']
            changed = np.flatnonzero(np.max(np.abs(pred_delta), axis=1) > 1e-6)
            center_shift = np.linalg.norm(
                (arm_predictions[arm][:, :2] + 0.5 * arm_predictions[arm][:, 2:]
                 - arm_predictions['lam0'][:, :2] - 0.5 * arm_predictions['lam0'][:, 2:]), axis=1)
            first_sustained = next((index for index in range(1, max(1, len(delta) - 4))
                                    if np.mean(delta[index:index + 5]) < -0.25), None)
            trajectory[(seq.name, arm)] = {
                'first_bbox_divergence_frame': int(changed[0]) if len(changed) else None,
                'changed_bbox_fraction': float(np.mean(np.max(np.abs(pred_delta), axis=1) > 1e-6)),
                'mean_center_shift_px': float(np.mean(center_shift)),
                'max_center_shift_px': float(np.max(center_shift)),
                'delta_iou_first5': float(np.mean(delta[1:6])) if len(delta) > 1 else 0.0,
                'delta_iou_first20': float(np.mean(delta[1:min(len(delta), 21)])) if len(delta) > 1 else 0.0,
                'first_sustained_5frame_delta_below_minus_025': first_sustained,
                'candidate_iou_below_01_fraction': float(np.mean(candidate[1:] < 0.1)) if len(candidate) > 1 else 0.0,
                'lambda0_iou_below_01_fraction': float(np.mean(base[1:] < 0.1)) if len(base) > 1 else 0.0,
            }

    summary = {'sequence_count': len(sequences), 'manifest_sha256': sha256(MANIFEST), 'arms': {}}
    for arm in ARMS:
        summary['arms'][arm] = {
            'lambda': STRENGTHS[arm],
            'macro_metrics': {
                key: float(np.mean([row[key] for row in all_metrics[arm]]))
                for key in ('auc', 'op50', 'op75', 'precision', 'norm_precision')
            },
        }
    summary['paired_vs_lam0'] = {}
    by_key = {(row['sequence'], row['arm']): row for row in per_sequence}
    for arm in ('lam05', 'lam1'):
        deltas = [by_key[(seq.name, arm)]['auc'] - by_key[(seq.name, 'lam0')]['auc'] for seq in sequences]
        summary['paired_vs_lam0'][arm] = {
            'mean_auc_delta': float(np.mean(deltas)),
            'median_auc_delta': float(np.median(deltas)),
            'wins': int(sum(value > 0 for value in deltas)),
            'losses': int(sum(value < 0 for value in deltas)),
            'ties': int(sum(value == 0 for value in deltas)),
            'largest_losses': sorted(
                [{'sequence': seq.name, 'delta_auc': float(delta)} for seq, delta in zip(sequences, deltas)],
                key=lambda row: row['delta_auc'])[:5],
            'largest_gains': sorted(
                [{'sequence': seq.name, 'delta_auc': float(delta)} for seq, delta in zip(sequences, deltas)],
                key=lambda row: row['delta_auc'], reverse=True)[:5],
        }
    summary['trajectory'] = {
        sequence: {
            arm: trajectory[(sequence, arm)]
            for arm in ('lam05', 'lam1')
        }
        for sequence in expected
    }

    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / 'per_sequence_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sequence[0].keys()))
        writer.writeheader()
        writer.writerows(per_sequence)
    with (EXP / 'trajectory_summary.csv').open('w', newline='') as stream:
        fields = ['sequence', 'arm'] + list(next(iter(trajectory.values())).keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (sequence, arm), row in trajectory.items():
            writer.writerow({'sequence': sequence, 'arm': arm, **row})
    (EXP / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('per_sequence_metrics.csv:', EXP / 'per_sequence_metrics.csv')


if __name__ == '__main__':
    main()
