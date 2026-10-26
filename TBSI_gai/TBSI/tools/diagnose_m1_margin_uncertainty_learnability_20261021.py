#!/usr/bin/env python3
"""Leave-one-sequence-out learnability check for target/background margin uncertainty."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'output/experiments/M1_MSHA_UNCERTAINTY_GATE_50SEQ_20261021_RETRY/per_event.csv'
OUT = ROOT / 'output/experiments/M1_MSHA_MARGIN_UNCERTAINTY_LEARNABILITY_50SEQ_20261021'


def feature_names():
    names = []
    for loc in ('loc3', 'loc6', 'loc9'):
        for key in ('query_match_mean', 'reliability_mean', 'participation_mean'):
            names.append('{}_{}'.format(loc, key))
    return names


def ridge_fit_predict(x_train, y_train, x_test, alpha=1.0):
    mean = x_train.mean(0)
    std = x_train.std(0)
    std[std < 1e-6] = 1.0
    z_train = (x_train - mean) / std
    z_test = (x_test - mean) / std
    design = np.concatenate([np.ones((len(z_train), 1)), z_train], axis=1)
    test_design = np.concatenate([np.ones((len(z_test), 1)), z_test], axis=1)
    reg = np.eye(design.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y_train)
    return test_design @ beta


def main():
    rows = [r for r in csv.DictReader(INPUT.open()) if r.get('mode') == 'message_gate']
    names = feature_names()
    if len(rows) < 10:
        raise RuntimeError('too few fixed events: {}'.format(len(rows)))
    x = np.asarray([[float(r[k]) for k in names] for r in rows], dtype=np.float64)
    y = np.asarray([float(r['off_gt_hard_bg_margin']) for r in rows], dtype=np.float64)
    seq = np.asarray([r['sequence'] for r in rows])
    predictions = []
    for held in sorted(set(seq)):
        train = seq != held
        pred = ridge_fit_predict(x[train], y[train], x[~train], alpha=1.0)
        for r, value in zip([r for r in rows if r['sequence'] == held], pred):
            predictions.append({'sequence': held, 'frame': int(r['frame']),
                                'observed_margin': float(r['off_gt_hard_bg_margin']),
                                'predicted_margin': float(value),
                                'observed_positive': bool(float(r['off_gt_hard_bg_margin']) > 0),
                                'predicted_positive': bool(float(value) > 0)})
    obs = np.asarray([r['observed_margin'] for r in predictions])
    pred = np.asarray([r['predicted_margin'] for r in predictions])
    residual = obs - pred
    # Conservative uncertainty is estimated from training residuals per fold.
    sigmas = []
    for held in sorted(set(seq)):
        train = seq != held
        train_pred = ridge_fit_predict(x[train], y[train], x[train], alpha=1.0)
        sigma = float(np.sqrt(np.mean((y[train] - train_pred) ** 2) + 1e-8))
        sigmas.extend([sigma] * int((seq == held).sum()))
    sigmas = np.asarray(sigmas)
    lower = pred - 1.0 * sigmas
    corr = float(np.corrcoef(obs, pred)[0, 1]) if np.std(obs) > 1e-12 and np.std(pred) > 1e-12 else 0.0
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(obs)), np.argsort(np.argsort(pred)))[0, 1])
    summary = {
        'id': 'M1-MSHA-MARGIN-UNCERTAINTY-LEARNABILITY-50SEQ-20261021',
        'input': str(INPUT), 'n_events': len(rows), 'n_sequences': int(len(set(seq))),
        'features': names, 'target': 'off_gt_hard_bg_margin',
        'protocol': 'leave-one-sequence-out ridge; no GT-derived input features',
        'pearson': corr, 'rank_correlation': rank_corr,
        'sign_accuracy_mean_margin': float(np.mean((obs > 0) == (pred > 0))),
        'lower_bound_positive_rate': float(np.mean(lower > 0)),
        'observed_positive_rate': float(np.mean(obs > 0)),
        'rmse': float(np.sqrt(np.mean(residual ** 2))),
        'mean_observed_margin': float(obs.mean()),
        'mean_predicted_margin': float(pred.mean()),
        'decision': 'NOT-LEARNABLE-ENOUGH-FOR-GATE' if corr < 0.3 or np.mean((obs > 0) == (pred > 0)) < 0.65 else 'ADMIT-FIXED-STATE-GATE-AUDIT',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
