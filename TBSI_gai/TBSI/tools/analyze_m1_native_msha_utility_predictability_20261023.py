#!/usr/bin/env python3
"""Sequence-held-out, low-capacity prediction of native MSHA action utility."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/experiments'
ARMS = {
    'parent': 'M1_T2S_CONDITIONED_PARENT_CALIB46_20261023',
    'r2t_075': 'M1_NATIVE_MSHA_FIXED_R2T075_CALIB46_20261023',
    'r2t_125': 'M1_NATIVE_MSHA_FIXED_R2T125_CALIB46_20261023',
    't2r_075': 'M1_NATIVE_MSHA_FIXED_T2R075_CALIB46_20261023',
    't2r_125': 'M1_NATIVE_MSHA_FIXED_T2R125_CALIB46_20261023',
}
INTERVENTIONS = tuple(k for k in ARMS if k != 'parent')
FEATURES = (
    'center_gt_hard_bg_margin', 'center_raw_score_gt_hard_bg_margin',
    'center_gt_rank', 'center_top1_to_gt_center_cells',
    'postfusion_query_gt_hard_bg_margin', 'postfusion_query_gt_peak',
    'postfusion_query_gt_rank', 'postfusion_query_gt_strongest_bg_margin',
    'fusion_input_rgb_target_bg_cosine', 'fusion_input_tir_target_bg_cosine',
    'fusion_input_rgb_target_bg_l2', 'fusion_input_tir_target_bg_l2',
    'fusion_output_target_bg_cosine', 'fusion_output_target_bg_l2',
)


def load(directory):
    with (OUT / directory / 'per_event.csv').open(newline='') as f:
        return {r['event_id']: r for r in csv.DictReader(f)}


def fit_ridge(x, y, lam=1.0):
    # Standardized features and an unpenalized intercept are fitted on each train fold.
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    z = (x - mu) / sd
    z1 = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    reg = np.eye(z1.shape[1]) * lam
    reg[0, 0] = 0.0
    beta = np.linalg.solve(z1.T.dot(z1) + reg, z1.T.dot(y))
    return mu, sd, beta


def predict(x, model):
    mu, sd, beta = model
    z = (x - mu) / sd
    return np.concatenate([np.ones((len(z), 1)), z], axis=1).dot(beta)


def summarize(values):
    values = list(values)
    return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
            'median': float(np.median(values)) if values else None,
            'positive': int(sum(v > 0 for v in values)),
            'negative': int(sum(v < 0 for v in values)),
            'zero': int(sum(v == 0 for v in values))}


def run(rows, held_out_sequences, threshold):
    train = [r for r in rows if r['sequence'] not in held_out_sequences]
    test = [r for r in rows if r['sequence'] in held_out_sequences]
    x_train = np.array([[r['features'][k] for k in FEATURES] for r in train], dtype=float)
    x_test = np.array([[r['features'][k] for k in FEATURES] for r in test], dtype=float)
    predictions = {}
    for arm in INTERVENTIONS:
        y = np.array([r['deltas'][arm] for r in train], dtype=float)
        predictions[arm] = predict(x_test, fit_ridge(x_train, y))
    chosen, utility = [], []
    for i, r in enumerate(test):
        scores = {arm: predictions[arm][i] for arm in INTERVENTIONS}
        arm = max(scores, key=scores.get)
        if scores[arm] <= threshold:
            chosen.append('no_op')
            utility.append(0.0)
        else:
            chosen.append(arm)
            utility.append(r['deltas'][arm])
    return test, chosen, utility, predictions


def main():
    arms = {name: load(directory) for name, directory in ARMS.items()}
    rows = []
    for event_id, parent in arms['parent'].items():
        rows.append({
            'event_id': event_id,
            'sequence': parent['sequence'],
            'phase': parent['phase'],
            'features': {k: float(parent[k]) for k in FEATURES},
            'deltas': {arm: float(arms[arm][event_id]['fixed_replay_iou']) - float(parent['fixed_replay_iou'])
                       for arm in INTERVENTIONS},
        })
    thresholds = [0.0, 0.001, 0.003]
    datasets = {
        'all': rows,
        'without_two_outlier_states': [r for r in rows if r['event_id'] not in {
            'bike150:severe_onset:144', 'theleftestrunningboy:pre_onset:17'}],
    }
    summary = {'features': list(FEATURES), 'datasets': {}}
    detail = []
    for dataset_name, dataset_rows in datasets.items():
        sequences = sorted({r['sequence'] for r in dataset_rows})
        summary['datasets'][dataset_name] = {'n_events': len(dataset_rows),
                                              'n_sequences': len(sequences),
                                              'sequence_loso': {}}
        for threshold in thresholds:
            selected, utilities = [], []
            for seq in sequences:
                test, chosen, utility, predictions = run(dataset_rows, {seq}, threshold)
                for i, r in enumerate(test):
                    detail.append({'dataset': dataset_name, 'threshold': threshold,
                                   'event_id': r['event_id'], 'sequence': seq,
                                   'chosen': chosen[i], 'utility': utility[i],
                                   'oracle_arm': max(r['deltas'], key=r['deltas'].get),
                                   'oracle_delta': max(r['deltas'].values()),
                                   'predicted_scores': {a: float(predictions[a][i]) for a in INTERVENTIONS}})
                selected.extend(chosen)
                utilities.extend(utility)
            summary['datasets'][dataset_name]['sequence_loso'][str(threshold)] = {
                'utility': summarize(utilities),
                'coverage': float(sum(x != 'no_op' for x in selected) / len(selected)),
                'action_counts': {a: selected.count(a) for a in ('no_op',) + INTERVENTIONS},
                'oracle_utility': summarize([max(r['deltas'].values()) for r in dataset_rows]),
                'fixed_action_utility': {a: summarize([r['deltas'][a] for r in dataset_rows]) for a in INTERVENTIONS},
            }
    out = OUT / 'M1_NATIVE_MSHA_UTILITY_PREDICTABILITY_20261023'
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'sequence_loso_predictions.jsonl').open('w') as f:
        for row in detail:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
