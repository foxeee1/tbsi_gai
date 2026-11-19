#!/usr/bin/env python3
"""Score MR delta-strength screen outputs with the repository evaluator."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--baseline-predictions', required=True)
    parser.add_argument('--baseline-metrics', required=True)
    parser.add_argument('--candidate', action='append', required=True,
                        help='name=prediction-directory')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    cohort_path = Path(args.cohort).resolve()
    baseline_predictions = Path(args.baseline_predictions).resolve()
    baseline_metrics = Path(args.baseline_metrics).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(workspace))
    os.chdir(workspace)
    from lib.test.evaluation import get_dataset
    from lib.test.analysis.extract_results import calc_seq_err_robust

    names = cohort_path.read_text().splitlines()
    dataset = get_dataset('lasher_test')
    by_name = {seq.name: seq for seq in dataset}
    if len(names) != 30 or any(name not in by_name for name in names):
        raise ValueError('The locked cohort does not resolve to 30 LasHeR test sequences')
    old_rows = {r['sequence']: r for r in csv.DictReader(baseline_metrics.open(newline=''))}
    specs = {}
    for item in args.candidate:
        name, path = item.split('=', 1)
        specs[name] = Path(path).resolve()
    specs = {'parent_025': baseline_predictions, **specs}
    groups = {}
    cohort_info = json.loads((cohort_path.parent / 'cohort_manifest.json').read_text())
    for group, data in cohort_info['groups'].items():
        groups.update({name: group for name in data['selected_sequences']})

    output_rows = []
    for sequence_name in names:
        seq = by_name[sequence_name]
        anno = torch.as_tensor(seq.ground_truth_rect)
        visible = torch.as_tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
        scores = {}
        for arm, pred_dir in specs.items():
            pred_path = pred_dir / (sequence_name + '.txt')
            pred = torch.as_tensor(np.loadtxt(str(pred_path), delimiter='\t', ndmin=2), dtype=anno.dtype)
            overlap, center, center_norm, valid = calc_seq_err_robust(pred, anno, seq.dataset, visible)
            length = anno.shape[0]
            auc = float(((overlap[:, None] > torch.arange(0.0, 1.01, 0.05).view(1, -1)).float().mean(0).mean() * 100).item())
            precision = float(((center[:, None] <= torch.arange(0, 51).view(1, -1)).float().mean(0)[20] * 100).item())
            norm_precision = float(((center_norm[:, None] <= (torch.arange(0, 51).float() / 100.0).view(1, -1)).float().mean(0)[20] * 100).item())
            scores[arm] = {'auc': auc, 'precision': precision,
                           'norm_precision': norm_precision,
                           'op50': float((overlap > 0.50).float().mean().item() * 100),
                           'op75': float((overlap > 0.75).float().mean().item() * 100),
                           'frames': int(length)}
        parent = scores['parent_025']
        row = {'sequence': sequence_name, 'stratum': groups[sequence_name],
               'historical_mr_minus_b0_auc_pp': float(old_rows[sequence_name]['delta_auc'])}
        for arm, values in scores.items():
            for metric, value in values.items():
                row[arm + '_' + metric] = value
        for arm in scores:
            if arm == 'parent_025':
                continue
            for metric in ('auc', 'precision', 'norm_precision', 'op50', 'op75'):
                row[arm + '_minus_parent_' + metric + '_pp'] = scores[arm][metric] - parent[metric]
        output_rows.append(row)

    per_sequence = out / 'per_sequence_metrics.csv'
    with per_sequence.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    arm_names = list(specs)
    summary = {'cohort_sequences': len(output_rows), 'cohort_frames': sum(r['parent_025_frames'] for r in output_rows),
               'arms': {}, 'comparison': 'Each candidate minus max_delta=0.25, same MR checkpoint and fixed frames.'}
    for arm in arm_names:
        summary['arms'][arm] = {}
        for group in ('all_30_balanced_macro', 'improved', 'regressed', 'neutral'):
            selected = output_rows if group == 'all_30_balanced_macro' else [r for r in output_rows if r['stratum'] == group]
            summary['arms'][arm][group] = {
                metric: float(np.mean([r[arm + '_' + metric] for r in selected]))
                for metric in ('auc', 'precision', 'norm_precision', 'op50', 'op75')}
    for arm in arm_names:
        if arm == 'parent_025':
            continue
        summary['arms'][arm]['delta_vs_parent_by_group'] = {}
        for group in ('all_30_balanced_macro', 'improved', 'regressed', 'neutral'):
            selected = output_rows if group == 'all_30_balanced_macro' else [r for r in output_rows if r['stratum'] == group]
            summary['arms'][arm]['delta_vs_parent_by_group'][group] = {
                metric: float(np.mean([r[arm + '_' + metric] - r['parent_025_' + metric] for r in selected]))
                for metric in ('auc', 'precision', 'norm_precision', 'op50', 'op75')}
        auc_deltas = [r[arm + '_auc'] - r['parent_025_auc'] for r in output_rows]
        summary['arms'][arm]['sequence_auc_win_tie_loss_vs_parent'] = {
            'wins': sum(x > 1e-9 for x in auc_deltas),
            'ties': sum(abs(x) <= 1e-9 for x in auc_deltas),
            'losses': sum(x < -1e-9 for x in auc_deltas),
        }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
