#!/usr/bin/env python3
"""Sequence-held-out selector for decoded readout oracle utilities."""
import argparse, csv, json
from pathlib import Path
import numpy as np


def fit(x, y, alpha=10.0):
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-6] = 1.0
    z = np.clip((x - mu) / sd, -8.0, 8.0)
    w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return mu, sd, float(y.mean()), w


def predict(model, x):
    mu, sd, bias, w = model
    return bias + np.clip((x - mu) / sd, -8.0, 8.0) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoded', required=True)
    ap.add_argument('--signals', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--alpha', type=float, default=10.0)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing nonempty output directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)

    decoded = list(csv.DictReader(open(args.decoded)))
    decoded = [r for r in decoded if r['action'] in ('no_op', 'readout_offset_oracle', 'readout_size_oracle')]
    by_key = {}
    for r in decoded:
        by_key[(r['sequence'], r['frame'], r['action'])] = r
    signals = list(csv.DictReader(open(args.signals)))
    signal_map = {(r['sequence'], r['frame']): r for r in signals}
    keys = sorted((s, f) for s, f in signal_map if (s, f, 'no_op') in by_key)
    if len(keys) != len(signals):
        raise RuntimeError('decoded/signal key mismatch: {} vs {}'.format(len(keys), len(signals)))

    feature_sets = {
        'base': ['base_gt_rank', 'base_margin'],
        'evidence': ['base_gt_rank', 'base_margin', 'energy_gap_rgb_minus_tir',
                     'template_support_gap_rgb_minus_tir', 'template_psr_gap_rgb_minus_tir', 'conflict'],
        'all': ['base_gt_rank', 'base_margin', 'energy_gap_rgb_minus_tir',
                'template_support_gap_rgb_minus_tir', 'template_psr_gap_rgb_minus_tir', 'conflict',
                'rgb_rank_worsening_net', 'tir_rank_worsening_net', 'label_severe'],
    }
    thresholds = [0.0, 0.0025, 0.005, 0.01, 0.02]
    records = []
    aggregate = {}
    sequences = sorted(set(s for s, _ in keys))
    for model_name, fields in feature_sets.items():
        x = np.asarray([[float(signal_map[(s, f)][name]) for name in fields] for s, f in keys], dtype=float)
        y_offset = np.asarray([float(by_key[(s, f, 'readout_offset_oracle')]['utility']) for s, f in keys])
        y_size = np.asarray([float(by_key[(s, f, 'readout_size_oracle')]['utility']) for s, f in keys])
        for threshold in thresholds:
            selected = []
            oracle = []
            used = []
            harmful = []
            rows = []
            for heldout in sequences:
                test = np.asarray([s == heldout for s, _ in keys])
                train = ~test
                mo = fit(x[train], y_offset[train], args.alpha)
                mz = fit(x[train], y_size[train], args.alpha)
                po = predict(mo, x[test])
                pz = predict(mz, x[test])
                test_keys = [keys[i] for i in np.where(test)[0]]
                for j, (s, f) in enumerate(test_keys):
                    pred = [0.0, float(po[j]), float(pz[j])]
                    actual = [0.0, float(y_offset[np.where(test)[0][j]]), float(y_size[np.where(test)[0][j]])]
                    action = int(np.argmax(pred))
                    use = pred[action] > threshold
                    value = actual[action] if use else 0.0
                    selected.append(value); oracle.append(max(actual)); used.append(use); harmful.append(use and value < -1e-6)
                    rows.append({'sequence': s, 'frame': f, 'threshold': threshold,
                                 'selected_action': ['noop', 'offset_oracle', 'size_oracle'][action] if use else 'noop',
                                 'predicted_utility': pred[action], 'selected_utility': value,
                                 'offset_utility': actual[1], 'size_utility': actual[2],
                                 'oracle_utility': max(actual)})
            by_seq = {s: [r for r in rows if r['sequence'] == s] for s in sequences}
            seq_selected = [np.mean([r['selected_utility'] for r in rr]) for rr in by_seq.values()]
            seq_oracle = [np.mean([r['oracle_utility'] for r in rr]) for rr in by_seq.values()]
            key = '{}@{}'.format(model_name, threshold)
            aggregate[key] = {
                'model': model_name, 'threshold': threshold, 'states': len(rows), 'sequences': len(sequences),
                'selected_mean_utility': float(np.mean(selected)),
                'sequence_macro_selected_utility': float(np.mean(seq_selected)),
                'oracle_mean_utility': float(np.mean(oracle)),
                'sequence_macro_oracle_utility': float(np.mean(seq_oracle)),
                'coverage': float(np.mean(used)),
                'harmful_rate_all': float(np.mean(harmful)),
                'harmful_rate_covered': float(np.sum(harmful) / max(np.sum(used), 1)),
                'oracle_fraction': float(np.mean(selected) / max(np.mean(oracle), 1e-12)),
            }
            records.extend([{**r, 'model': model_name} for r in rows])

    with (out / 'per_state_selection.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    (out / 'summary.json').write_text(json.dumps({
        'id': 'M1-DECODED-READOUT-SELECTOR-SCREEN-20261014',
        'decoded': args.decoded, 'signals': args.signals, 'sequence_holdout': True,
        'models': aggregate,
        'limitations': ['readout action values are GT regression oracles',
                         'selector utility is fixed-state decoded IoU, not closed-loop tracking',
                         'no predicted offset/size value is deployed'],
    }, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
