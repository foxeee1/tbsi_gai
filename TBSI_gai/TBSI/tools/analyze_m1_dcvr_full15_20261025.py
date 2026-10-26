#!/usr/bin/env python3
"""Offline paired attribution for the DCVR full-15 LasHeR evaluation."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text


def read_predictions(result_dir, seq):
    path = os.path.join(result_dir, seq.name + '.txt')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.as_tensor(load_text(path, delimiter=('\t', ','), dtype=np.float64))


def metric_row(seq, result_dir):
    anno = torch.as_tensor(np.asarray(seq.ground_truth_rect), dtype=torch.float64)
    pred = read_predictions(result_dir, seq)
    overlap, center, center_norm, valid = calc_seq_err_robust(
        pred, anno, seq.dataset, None)
    n = anno.shape[0]
    overlap_thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    center_thresholds = torch.arange(0, 51, dtype=torch.float64)
    center_norm_thresholds = center_thresholds / 100.0
    success = (overlap[:, None] > overlap_thresholds[None, :]).sum(0).double() / n
    precision = (center[:, None] <= center_thresholds[None, :]).sum(0).double() / n
    norm_precision = (center_norm[:, None] <= center_norm_thresholds[None, :]).sum(0).double() / n
    return {
        'sequence': seq.name,
        'frames': n,
        'valid_frames': int(valid.sum().item()),
        'ao': float(overlap[valid].mean().item()),
        'auc': float(success.mean().item() * 100.0),
        'op50': float(success[10].item() * 100.0),
        'op75': float(success[15].item() * 100.0),
        'precision': float(precision[20].item() * 100.0),
        'norm_precision': float(norm_precision[20].item() * 100.0),
    }


def paired_bootstrap(delta, rng, reps=20000):
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.size
    values = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 1000):
        size = min(1000, reps - start)
        ids = rng.integers(0, n, size=(size, n))
        values[start:start + size] = delta[ids].mean(axis=1)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def summarize_diagnostics(path):
    fields = [
        'rgb_entropy_mean', 'tir_entropy_mean',
        'rgb_competition_mean', 'tir_competition_mean',
        'agreement_mean', 'rgb_source_scale_mean', 'tir_source_scale_mean',
        'rgb_source_scale_std', 'tir_source_scale_std',
    ]
    count = 0
    fallback = 0
    sums = {k: 0.0 for k in fields}
    mins = {k: float('inf') for k in fields}
    maxs = {k: float('-inf') for k in fields}
    alphas = {}
    malformed_lines = 0
    recovered_fragments = 0
    decoder = json.JSONDecoder()

    def add_row(row):
        nonlocal count, fallback
        count += 1
        fallback += int(bool(row.get('fg_mask_fallback_uniform', False)))
        alpha = str(row.get('alpha', 'NA'))
        alphas[alpha] = alphas.get(alpha, 0) + 1
        for key in fields:
            if key not in row:
                continue
            value = float(row[key])
            sums[key] += value
            mins[key] = min(mins[key], value)
            maxs[key] = max(maxs[key], value)

    carry = ''
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            text = carry + line
            carry = ''
            pos = 0
            line_failed = False
            last_incomplete = None
            while pos < len(text):
                start = text.find('{', pos)
                if start < 0:
                    break
                try:
                    row, end = decoder.raw_decode(text, start)
                except json.JSONDecodeError:
                    line_failed = True
                    last_incomplete = start
                    pos = start + 1
                    continue
                if isinstance(row, dict) and row.get('mode') == 'dcvr':
                    add_row(row)
                    if start > 0:
                        recovered_fragments += 1
                pos = end
            if line_failed:
                malformed_lines += 1
            if last_incomplete is not None and len(text) - last_incomplete <= 8192:
                carry = text[last_incomplete:]
    return {
        'records': count,
        'malformed_source_lines': malformed_lines,
        'recovered_objects_after_interleaving': recovered_fragments,
        'fg_mask_fallback_uniform_fraction': fallback / count if count else None,
        'alphas': alphas,
        'mean': {k: sums[k] / count for k in fields if count and mins[k] != float('inf')},
        'min': {k: mins[k] for k in fields if mins[k] != float('inf')},
        'max': {k: maxs[k] for k in fields if maxs[k] != float('-inf')},
        'sequence_frame_keys_present': False,
    }


def summarize_training(run_dir):
    log_dir = os.path.join(run_dir, 'logs')
    loss_path = os.path.join(log_dir, 'loss_components.csv')
    grad_path = os.path.join(log_dir, 'gradient_flow.csv')
    epochs = {}
    with open(loss_path, newline='') as f:
        for row in csv.DictReader(f):
            epoch = int(row['epoch'])
            bucket = epochs.setdefault(epoch, {})
            for key in ('total', 'giou', 'l1', 'location', 'iou'):
                bucket.setdefault(key, []).append(float(row[key]))
    grad_rows = []
    with open(grad_path, newline='') as f:
        for row in csv.DictReader(f):
            grad_rows.append(row)
    return {
        'loss_by_epoch_sampled_mean': {
            str(epoch): {key: float(np.mean(vals)) for key, vals in metrics.items()}
            for epoch, metrics in epochs.items()
        },
        'gradient_records': len(grad_rows),
        'zero_grad_count_values': sorted(set(int(r['zero_grad_count']) for r in grad_rows)),
        'total_norm_values': sorted(set(r['total_norm'] for r in grad_rows)),
        'nan_gradient_records': sum(
            any(r[k].lower() == 'nan' for k in ('total_norm', 'backbone_mean', 'head_mean'))
            for r in grad_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='M1_DCVR_FULL15_SOT_20261025')
    parser.add_argument('--parent', default='TBSI_E_MHSA_FULL_4090D_seed42')
    parser.add_argument('--dataset', default='lasher_test')
    parser.add_argument('--out-dir', default='output/experiments/M1_DCVR_FULL15_SOT_20261025/attribution')
    parser.add_argument('--diagnostics', default='/autodl-fs/data/tbsi_dcvr/full15_20261025/dcvr_test_stats.jsonl')
    parser.add_argument('--train-diagnostics', default='/autodl-fs/data/tbsi_dcvr/full15_20261025/dcvr_train_stats.jsonl')
    parser.add_argument('--seed', type=int, default=20261026)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    base = os.path.join(ROOT, 'output', 'test', 'tracking_results', 'tbsi_track')
    parent_dir = os.path.join(base, args.parent)
    candidate_dir = os.path.join(base, args.candidate)
    parent_rows = [metric_row(seq, parent_dir) for seq in dataset]
    cand_rows = [metric_row(seq, candidate_dir) for seq in dataset]
    rows = []
    metric_keys = ('ao', 'auc', 'op50', 'op75', 'precision', 'norm_precision')
    for p, c in zip(parent_rows, cand_rows):
        row = {'sequence': p['sequence'], 'frames': p['frames'], 'valid_frames': p['valid_frames']}
        for k in metric_keys:
            row['parent_' + k] = p[k]
            row['candidate_' + k] = c[k]
            row['delta_' + k] = c[k] - p[k]
        rows.append(row)

    rng = np.random.default_rng(args.seed)
    summary = {'dataset': args.dataset, 'sequences': len(rows), 'metrics': {},
               'paired_sequence_bootstrap_95ci': {}, 'sequence_counts': {}, 'seed': args.seed}
    for k in metric_keys:
        d = np.array([r['delta_' + k] for r in rows])
        summary['metrics'][k] = {
            'parent_mean': float(np.mean([r['parent_' + k] for r in rows])),
            'candidate_mean': float(np.mean([r['candidate_' + k] for r in rows])),
            'delta_mean': float(d.mean()),
            'delta_median': float(np.median(d)),
            'positive': int((d > 1e-9).sum()),
            'negative': int((d < -1e-9).sum()),
            'tied': int((np.abs(d) <= 1e-9).sum()),
        }
        summary['paired_sequence_bootstrap_95ci'][k] = paired_bootstrap(d, rng)
    protected_anchor = {
        'auc': 56.21,
        'op50': 68.33,
        'op75': 47.13,
        'precision': 70.21,
        'norm_precision': 66.20,
    }
    anchor_tolerance = 0.05
    current_parent = {
        key: summary['metrics'][key]['parent_mean']
        for key in protected_anchor
    }
    anchor_mismatches = {
        key: {'protected': protected_anchor[key], 'recomputed': current_parent[key]}
        for key in protected_anchor
        if abs(protected_anchor[key] - current_parent[key]) > anchor_tolerance
    }
    summary['pairing_audit'] = {
        'status': 'PROTECTED-M1-ANCHOR-MATCH' if not anchor_mismatches else
                  'PROTECTED-M1-COMPARABILITY-UNRESOLVED',
        'paired_sequence_inference_valid': not bool(anchor_mismatches),
        'anchor_tolerance_pp': anchor_tolerance,
        'protected_m1_reference': protected_anchor,
        'current_m1_result_directory_recomputed': current_parent,
        'anchor_mismatches': anchor_mismatches,
        'reason': ('Current M1 result-directory metrics do not reproduce the protected M1 '
                   'reference; paired sequence deltas and bootstrap intervals are exploratory only.'
                   if anchor_mismatches else
                   'Current M1 result-directory metrics match the protected reference within tolerance.'),
    }
    ao_delta = np.array([r['delta_ao'] for r in rows])
    summary['sequence_counts'] = {
        'ao_gain': int((ao_delta > 1e-9).sum()),
        'ao_loss': int((ao_delta < -1e-9).sum()),
        'ao_tie': int((np.abs(ao_delta) <= 1e-9).sum()),
        'top5_gain_share_of_net_delta': float(np.sort(ao_delta)[-5:].sum() / ao_delta.sum()) if ao_delta.sum() else None,
        'top5_loss_share_of_net_delta': float(np.sort(ao_delta)[:5].sum() / ao_delta.sum()) if ao_delta.sum() else None,
    }
    summary['top_gains_auc'] = sorted(rows, key=lambda r: r['delta_auc'], reverse=True)[:10]
    summary['top_losses_auc'] = sorted(rows, key=lambda r: r['delta_auc'])[:10]
    summary['training'] = summarize_training(os.path.join(ROOT, 'output', 'experiments', args.candidate))
    summary['train_dcvr_diagnostics'] = summarize_diagnostics(args.train_diagnostics)
    summary['test_dcvr_diagnostics'] = summarize_diagnostics(args.diagnostics)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'per_sequence_paired.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(out_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps({'summary': json_path, 'per_sequence': csv_path,
                      'metrics': summary['metrics'],
                      'sequence_counts': summary['sequence_counts'],
                      'train_fallback': summary['train_dcvr_diagnostics']['fg_mask_fallback_uniform_fraction'],
                      'test_fallback': summary['test_dcvr_diagnostics']['fg_mask_fallback_uniform_fraction']},
                     indent=2))


if __name__ == '__main__':
    main()
