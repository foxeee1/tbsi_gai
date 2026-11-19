#!/usr/bin/env python3
"""Compare M1, gate-only, and trained target-margin gate on the frozen 12 clips."""
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

EXP = ROOT / 'output/experiments/M1_EARLY_GATE_MARGIN_OPERATOR_20261022'
MANIFEST = ROOT / 'output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/manifests/sequence_list.txt'
RESULTS = ROOT / 'output/test/tracking_results/tbsi_track'
DIRS = {
    'm1': 'M1_EARLY_GATE_STRENGTH_REPLAY_20261022_lam0',
    'gate': 'M1_EARLY_GATE_STRENGTH_REPLAY_20261022_lam1',
    'margin': 'M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022_margin_ep3',
}


def score(pred, seq):
    gt = torch.as_tensor(seq.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
    overlap, center, center_norm, _ = calc_seq_err_robust(pred, gt, seq.dataset, visible)
    thresholds = torch.arange(0.0, 1.0001, 0.05, dtype=torch.float64)
    curve = (overlap[:, None] > thresholds[None, :]).double().mean(0)
    values = {
        'auc': float(curve.mean() * 100.0),
        'op50': float(curve[10] * 100.0),
        'op75': float(curve[15] * 100.0),
        'precision': float((center <= 20).double().mean() * 100.0),
        'norm_precision': float((center_norm <= 0.2).double().mean() * 100.0),
    }
    return values, overlap.numpy()


def main():
    os.environ['TBSI_TRACE_TRAIN_LIST'] = str(MANIFEST)
    sequences = get_dataset('m1_trace_lasher_train')
    expected = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    if [seq.name for seq in sequences] != expected or len(sequences) != 12:
        raise RuntimeError('frozen sequence manifest mismatch')
    per_sequence, trajectories = [], {}
    for seq in sequences:
        values, predictions = {}, {}
        for arm, directory in DIRS.items():
            path = RESULTS / directory / (seq.name + '.txt')
            if not path.is_file():
                raise FileNotFoundError(str(path))
            pred = np.asarray(load_text(str(path), delimiter=('\t', ','), dtype=np.float64))
            predictions[arm] = pred
            metrics, overlap = score(torch.as_tensor(pred, dtype=torch.float64), seq)
            values[arm] = (metrics, overlap)
            per_sequence.append({'sequence': seq.name, 'arm': arm, **metrics, 'frames': len(pred)})
        base_box, cand_box = predictions['gate'], predictions['margin']
        changed = np.flatnonzero(np.max(np.abs(cand_box - base_box), axis=1) > 1e-6)
        center_delta = np.linalg.norm(
            cand_box[:, :2] + 0.5 * cand_box[:, 2:] -
            base_box[:, :2] - 0.5 * base_box[:, 2:], axis=1)
        trajectories[seq.name] = {
            'margin_minus_gate_auc': values['margin'][0]['auc'] - values['gate'][0]['auc'],
            'first_bbox_divergence_frame': int(changed[0]) if len(changed) else None,
            'changed_bbox_fraction': float(np.mean(np.max(np.abs(cand_box - base_box), axis=1) > 1e-6)),
            'mean_center_shift_px': float(np.mean(center_delta)),
            'max_center_shift_px': float(np.max(center_delta)),
            'mean_iou_delta_vs_gate': float(np.mean(values['margin'][1] - values['gate'][1])),
        }

    by = {(r['sequence'], r['arm']): r['auc'] for r in per_sequence}
    summary = {'sequence_count': len(sequences), 'arms': {}, 'paired': {}, 'trajectories': trajectories}
    for arm in DIRS:
        arm_rows = [next(row for row in per_sequence
                         if row['sequence'] == seq.name and row['arm'] == arm)
                    for seq in sequences]
        summary['arms'][arm] = {
            'macro_' + key: float(np.mean([row[key] for row in arm_rows]))
            for key in ('auc', 'op50', 'op75', 'precision', 'norm_precision')
        }
    for arm in ('gate', 'margin'):
        delta = [by[(seq.name, arm)] - by[(seq.name, 'm1')] for seq in sequences]
        summary['paired'][arm + '_vs_m1'] = {
            'mean_delta': float(np.mean(delta)), 'median_delta': float(np.median(delta)),
            'wins': int(sum(v > 1e-10 for v in delta)),
            'losses': int(sum(v < -1e-10 for v in delta)),
            'ties': int(sum(abs(v) <= 1e-10 for v in delta)),
        }
    margin_delta = [by[(seq.name, 'margin')] - by[(seq.name, 'gate')] for seq in sequences]
    summary['paired']['margin_vs_gate'] = {
        'mean_delta': float(np.mean(margin_delta)), 'median_delta': float(np.median(margin_delta)),
        'wins': int(sum(v > 1e-10 for v in margin_delta)),
        'losses': int(sum(v < -1e-10 for v in margin_delta)),
        'ties': int(sum(abs(v) <= 1e-10 for v in margin_delta)),
        'largest_losses': sorted([
            {'sequence': seq.name, 'delta': by[(seq.name, 'margin')] - by[(seq.name, 'gate')]}
            for seq in sequences], key=lambda r: r['delta'])[:4],
        'largest_gains': sorted([
            {'sequence': seq.name, 'delta': by[(seq.name, 'margin')] - by[(seq.name, 'gate')]}
            for seq in sequences], key=lambda r: r['delta'], reverse=True)[:4],
    }
    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / 'per_sequence_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sequence[0]))
        writer.writeheader(); writer.writerows(per_sequence)
    (EXP / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
