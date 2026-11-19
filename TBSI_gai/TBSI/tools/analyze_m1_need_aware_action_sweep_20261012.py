#!/usr/bin/env python3
"""Sequence-held-out screening of runtime signals for M1 action utility."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


FEATURES_PRE = [
    'rgb_support_peak', 'tir_support_peak',
    'rgb_support_top_margin', 'tir_support_top_margin',
    'rgb_support_entropy', 'tir_support_entropy',
    'support_top_margin_gap_rgb_minus_tir', 'support_peak_gap_rgb_minus_tir',
    'support_entropy_gap_rgb_minus_tir', 'support_peak_distance', 'support_map_correlation',
    'rgb_local_energy_at_center_peak', 'tir_local_energy_at_center_peak',
    'local_energy_gap_rgb_minus_tir', 'rgb_crop_luma_mean', 'rgb_crop_luma_std',
    'tir_crop_luma_mean', 'tir_crop_luma_std', 'crop_luma_mean_gap_rgb_minus_tir',
    'state_area_fraction', 'state_aspect_ratio', 'state_motion_norm', 'state_log_scale_change',
]
FEATURES_ALL = FEATURES_PRE + ['center_entropy', 'center_top_margin']
ACTION_NAMES = ['rgb_rho_0.05', 'rgb_rho_0.10', 'rgb_rho_0.15',
                'tir_rho_0.05', 'tir_rho_0.10', 'tir_rho_0.15']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError('refusing to write empty CSV: {}'.format(path))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ridge_predict(x_train, y_train, x_test, alpha):
    mean = x_train.mean(0)
    std = x_train.std(0)
    std[std < 1e-6] = 1.
    xtr = np.clip((x_train - mean) / std, -8., 8.)
    xte = np.clip((x_test - mean) / std, -8., 8.)
    ymean = float(y_train.mean())
    yc = y_train - ymean
    gram = xtr.T @ xtr + alpha * np.eye(xtr.shape[1])
    weight = np.linalg.solve(gram, xtr.T @ yc)
    fitted = xtr @ weight + ymean
    residual = y_train - fitted
    sigma = float(np.sqrt(np.mean(residual ** 2)))
    return xte @ weight + ymean, sigma


def rank_values(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = .5 * (i + j - 1)
        i = j
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) < 1e-10 or np.std(b[valid]) < 1e-10:
        return None
    return float(np.corrcoef(rank_values(a[valid]), rank_values(b[valid]))[0, 1])


def bootstrap(values, seed=3407, n=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(values, size=len(values), replace=True).mean()
                          for _ in range(n)])
    return float(values.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signals', required=True)
    ap.add_argument('--actions', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--ridge-alpha', type=float, default=10.)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty analysis directory: {}'.format(output))
    output.mkdir(parents=True, exist_ok=True)
    signals = read_csv(args.signals)
    actions = read_csv(args.actions)
    keys = [(r['sequence'], int(r['frame'])) for r in signals]
    if len(keys) != len(set(keys)):
        raise RuntimeError('duplicate signal states')
    signal_by_key = {(r['sequence'], int(r['frame'])): r for r in signals}
    action_by_key = {}
    for row in actions:
        action_by_key.setdefault((row['sequence'], int(row['frame'])), {})[row['arm']] = row
    if set(signal_by_key) != set(action_by_key):
        raise RuntimeError('signal/action state keys differ')
    sequences = sorted(set(r['sequence'] for r in signals))
    if len(sequences) < 24 or len(signals) < 30:
        raise RuntimeError('preregistered sample minimum failed: {} states / {} sequences'.format(
            len(signals), len(sequences)))
    for key, row in action_by_key.items():
        missing = {'no_op'}.union(ACTION_NAMES).difference(row)
        if missing:
            raise RuntimeError('state {} missing actions {}'.format(key, sorted(missing)))

    def feature_matrix(names):
        return np.asarray([[float(signal_by_key[key][name]) for name in names] for key in keys],
                          dtype=np.float64)

    utilities = np.asarray([[float(action_by_key[key][name]['delta_iou_vs_no_op'])
                             for name in ACTION_NAMES] for key in keys], dtype=np.float64)
    actual_best = np.maximum(0., utilities.max(1))
    rgb_best = utilities[:, :3].max(1)
    tir_best = utilities[:, 3:].max(1)

    decisions = {}
    for model_name, feature_names in (('prefusion_signals', FEATURES_PRE),
                                      ('prefusion_plus_center_risk', FEATURES_ALL)):
        x = feature_matrix(feature_names)
        predictions = np.zeros_like(utilities)
        lcb_predictions = np.zeros_like(utilities)
        for held in sequences:
            test_idx = np.asarray([i for i, key in enumerate(keys) if key[0] == held], dtype=int)
            train_idx = np.asarray([i for i, key in enumerate(keys) if key[0] != held], dtype=int)
            if not len(test_idx) or len(train_idx) < 10:
                raise RuntimeError('insufficient leave-one-sequence-out training states')
            for action_idx in range(len(ACTION_NAMES)):
                pred, sigma = ridge_predict(x[train_idx], utilities[train_idx, action_idx],
                                            x[test_idx], args.ridge_alpha)
                predictions[test_idx, action_idx] = pred
                lcb_predictions[test_idx, action_idx] = pred - 1.645 * sigma

        raw_choice = np.argmax(np.column_stack((np.zeros(len(keys)), predictions)), axis=1) - 1
        lcb_choice = np.argmax(np.column_stack((np.zeros(len(keys)), lcb_predictions)), axis=1) - 1
        decisions[model_name] = {'raw': raw_choice, 'lcb': lcb_choice,
                                 'raw_predictions': predictions, 'lcb_predictions': lcb_predictions}

    def evaluate_choices(choice):
        selected = np.where(choice < 0, 0., utilities[np.arange(len(keys)), np.maximum(choice, 0)])
        selected_action = np.where(choice < 0, 'no_op', np.asarray(ACTION_NAMES)[np.maximum(choice, 0)])
        oracle_noop = actual_best <= .005
        return {
            'state_utility': selected,
            'selected_action': selected_action,
            'mean_utility': float(selected.mean()),
            'oracle_noop_specificity': float(np.mean(choice[oracle_noop] < 0)) if oracle_noop.any() else None,
            'oracle_noop_states': int(oracle_noop.sum()),
            'harmful_intervention_rate': float(np.mean(selected[choice >= 0] < -.01)) if np.any(choice >= 0) else 0.,
            'intervention_coverage': float(np.mean(choice >= 0)),
            'mean_regret_to_tested_action_oracle': float(np.mean(actual_best - selected)),
        }

    result_rows = []
    for model_name, result in decisions.items():
        for rule in ('raw', 'lcb'):
            report = evaluate_choices(result[rule])
            for i, key in enumerate(keys):
                result_rows.append({'sequence': key[0], 'frame': key[1], 'label': signal_by_key[key]['label'],
                                    'model': model_name, 'rule': rule,
                                    'selected_action': report['selected_action'][i],
                                    'selected_utility': float(report['state_utility'][i]),
                                    'oracle_best_utility': float(actual_best[i]),
                                    'regret': float(actual_best[i] - report['state_utility'][i]),
                                    'oracle_noop': int(actual_best[i] <= .005)})
    write_csv(output / 'per_state_decisions.csv', result_rows)

    all_arms = ['no_op'] + ACTION_NAMES + ['random_rho_0.10']
    per_sequence = []
    for seq in sequences:
        rows = [i for i, key in enumerate(keys) if key[0] == seq]
        item = {'sequence': seq, 'n_states': len(rows)}
        for arm in all_arms:
            vals = [float(action_by_key[keys[i]][arm]['delta_iou_vs_no_op']) for i in rows]
            item[arm + '_delta_iou'] = float(np.mean(vals))
        for model_name, result in decisions.items():
            for rule in ('raw', 'lcb'):
                report = evaluate_choices(result[rule])
                item[model_name + '_' + rule + '_delta_iou'] = float(np.mean(report['state_utility'][rows]))
                item[model_name + '_' + rule + '_intervention_coverage'] = float(np.mean(result[rule][rows] >= 0))
        per_sequence.append(item)
    write_csv(output / 'per_sequence.csv', per_sequence)

    action_summary = []
    for arm in all_arms:
        per_seq_delta = np.asarray([row[arm + '_delta_iou'] for row in per_sequence])
        mean, low, high = bootstrap(per_seq_delta, args.seed)
        action_summary.append({'condition': arm, 'mean_iou_delta_vs_no_op': mean,
                               'paired_sequence_bootstrap95_low': low,
                               'paired_sequence_bootstrap95_high': high,
                               'sequence_wins': int(np.sum(per_seq_delta > 0)),
                               'sequence_losses': int(np.sum(per_seq_delta < 0)),
                               'sequence_ties': int(np.sum(per_seq_delta == 0))})
    gate_summary = {}
    for model_name, result in decisions.items():
        gate_summary[model_name] = {}
        for rule in ('raw', 'lcb'):
            report = evaluate_choices(result[rule])
            seq_delta = np.asarray([row[model_name + '_' + rule + '_delta_iou'] for row in per_sequence])
            scalar_report = {key: value for key, value in report.items()
                             if not isinstance(value, np.ndarray)}
            scalar_report['sequence_macro_iou_delta_ci95'] = list(bootstrap(seq_delta, args.seed))
            gate_summary[model_name][rule] = scalar_report

    signal_screen = []
    best_action_utility = actual_best
    direction_gap = rgb_best - tir_best
    for feature in FEATURES_ALL:
        values = np.asarray([float(signal_by_key[key][feature]) for key in keys])
        signal_screen.append({'signal': feature,
                              'spearman_with_best_positive_action_utility': spearman(values, best_action_utility),
                              'spearman_with_rgb_vs_tir_best_utility': spearman(values, direction_gap),
                              'mean': float(values.mean()), 'std': float(values.std())})
    signal_screen.sort(key=lambda row: abs(row['spearman_with_best_positive_action_utility'] or 0.), reverse=True)
    write_csv(output / 'signal_screen.csv', signal_screen)

    label_summary = []
    labels = sorted(set(row['label'] for row in signals))
    for label in labels:
        idx = np.asarray([i for i, key in enumerate(keys)
                          if signal_by_key[key]['label'] == label], dtype=int)
        item = {'label': label, 'n_states': len(idx),
                'n_sequences': len(set(keys[i][0] for i in idx))}
        for arm in all_arms:
            item[arm + '_mean_delta_iou'] = float(np.mean([
                float(action_by_key[keys[i]][arm]['delta_iou_vs_no_op']) for i in idx]))
        for model_name, result in decisions.items():
            for rule in ('raw', 'lcb'):
                report = evaluate_choices(result[rule])
                item[model_name + '_' + rule + '_mean_delta_iou'] = float(
                    np.mean(report['state_utility'][idx]))
                item[model_name + '_' + rule + '_coverage'] = float(
                    np.mean(result[rule][idx] >= 0))
        label_summary.append(item)
    write_csv(output / 'by_label.csv', label_summary)

    risk_coverage = []
    for model_name, result in decisions.items():
        confidence = result['lcb_predictions'].max(1)
        action_choice = result['lcb_predictions'].argmax(1)
        eligible = np.flatnonzero(confidence > 0.)
        ranked = eligible[np.argsort(confidence[eligible])[::-1]]
        for target_coverage in (.05, .10, .20):
            count = min(len(ranked), max(1, int(np.ceil(len(keys) * target_coverage))))
            selected_idx = ranked[:count]
            selected_utilities = utilities[selected_idx, action_choice[selected_idx]]
            oracle_regret = actual_best[selected_idx] - selected_utilities
            risk_coverage.append({
                'model': model_name,
                'target_coverage': target_coverage,
                'actual_coverage': float(count / len(keys)),
                'n_selected_states': int(count),
                'mean_selected_utility': float(selected_utilities.mean()) if count else None,
                'harmful_rate_below_minus_0.01': float(np.mean(selected_utilities < -.01)) if count else None,
                'mean_regret_to_tested_action_oracle': float(oracle_regret.mean()) if count else None,
                'minimum_selected_lcb': float(confidence[selected_idx].min()) if count else None,
            })
    write_csv(output / 'risk_coverage.csv', risk_coverage)

    summary = {
        'id': 'M1-NEED-AWARE-ACTION-SWEEP-20261012',
        'status': 'EXPLORATORY-COMPLETE',
        'n_states': len(keys), 'n_sequences': len(sequences),
        'label_counts': {label: sum(row['label'] == label for row in signals)
                         for label in sorted(set(row['label'] for row in signals))},
        'action_definitions': ACTION_NAMES + ['random_rho_0.10'],
        'fixed_action_results_sequence_macro': action_summary,
        'signal_gate_results': gate_summary,
        'risk_coverage_curve': risk_coverage,
        'oracle_positive_action_rate': float(np.mean(actual_best > .005)),
        'oracle_best_utility_mean': float(actual_best.mean()),
        'oracle_rgb_best_mean': float(rgb_best.mean()),
        'oracle_tir_best_mean': float(tir_best.mean()),
        'sequence_bootstrap': {'unit': 'sequence', 'resamples': 10000, 'seed': args.seed},
        'ridge': {'alpha': args.ridge_alpha, 'split': 'leave-one-sequence-out',
                  'uncertainty_proxy': 'train-fold residual RMSE times 1.645; not calibrated'},
        'limitations': [
            'Exploratory train-side cohort; M1 checkpoint training-sequence exposure cannot be reconstructed.',
            'Action utility is offline same-crop decoded IoU, not closed-loop AUC or long-horizon return.',
            'The selected action family is reciprocal RGB/TIR scaling; conclusions do not generalize to arbitrary learned residual operators.',
            'Input signals and ridge family were predeclared, but the cohort is small and sequence-correlated states are averaged by sequence.',
            'The uncertainty proxy is residual RMSE and is not a calibrated predictive interval.',
            'A positive result would only justify a new frozen-state replication/short-rollout stage.'
        ],
        'inputs_sha256': {'signals': sha256(args.signals), 'actions': sha256(args.actions)},
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
