import csv
import json
import os
import sys

import torch

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.analysis.extract_results import extract_results


def evaluate(dataset, name, parameter, run_id, report):
    tracker = Tracker('tbsi_track', parameter, 'lasher_test', run_id)
    return extract_results([tracker], dataset, report)


def metrics(eval_data):
    overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'], dtype=torch.float64)[:, 0, :]
    center = torch.tensor(eval_data['ave_success_rate_plot_center'], dtype=torch.float64)[:, 0, :]
    center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'], dtype=torch.float64)[:, 0, :]
    valid = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    out = {}
    for i, seq in enumerate(eval_data['sequences']):
        if not valid[i]:
            continue
        out[seq] = {
            'auc': float(overlap[i].mean() * 100.0),
            'op50': float(overlap[i, 10] * 100.0),
            'op75': float(overlap[i, 15] * 100.0),
            'precision': float(center[i, 20] * 100.0),
            'norm_precision': float(center_norm[i, 20] * 100.0),
        }
    return out


def main():
    dataset = get_dataset('lasher_test')
    methods = {
        'm1': ('TBSI_E_MHSA_FULL_4090D_seed42', None, 'amcm_pair_m1'),
        'e2_seed42': ('M1_AMCM_E2_10EP_FULL244_20261017', None, 'amcm_pair_seed42'),
        'e2_seed3407': ('M1_AMCM_E2_SEED3407_10EP_FULL244_20261018', 20261018, 'amcm_pair_seed3407'),
    }
    all_metrics = {}
    for key, (parameter, run_id, report) in methods.items():
        all_metrics[key] = metrics(evaluate(dataset, key, parameter, run_id, report))

    seqs = sorted(set(all_metrics['m1']) & set(all_metrics['e2_seed42']) & set(all_metrics['e2_seed3407']))
    out_dir = 'output/experiments/M1_AMCM_E2_SEED3407_10EP_FULL244_20261018/analysis'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'per_sequence_comparison.csv')
    fields = ['sequence']
    for method in ('m1', 'e2_seed42', 'e2_seed3407'):
        fields += [method + '_' + metric for metric in ('auc', 'op50', 'op75', 'precision', 'norm_precision')]
    for metric in ('auc', 'op50', 'op75', 'precision', 'norm_precision'):
        fields += ['seed42_delta_' + metric, 'seed3407_delta_' + metric, 'seed3407_vs_seed42_delta_' + metric]
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for seq in seqs:
            row = {'sequence': seq}
            for method in ('m1', 'e2_seed42', 'e2_seed3407'):
                for metric in ('auc', 'op50', 'op75', 'precision', 'norm_precision'):
                    row[method + '_' + metric] = all_metrics[method][seq][metric]
            for metric in ('auc', 'op50', 'op75', 'precision', 'norm_precision'):
                row['seed42_delta_' + metric] = all_metrics['e2_seed42'][seq][metric] - all_metrics['m1'][seq][metric]
                row['seed3407_delta_' + metric] = all_metrics['e2_seed3407'][seq][metric] - all_metrics['m1'][seq][metric]
                row['seed3407_vs_seed42_delta_' + metric] = all_metrics['e2_seed3407'][seq][metric] - all_metrics['e2_seed42'][seq][metric]
            writer.writerow(row)

    summary = {'num_sequences': len(seqs), 'methods': {}}
    for method in ('m1', 'e2_seed42', 'e2_seed3407'):
        summary['methods'][method] = {metric: sum(all_metrics[method][s][metric] for s in seqs) / len(seqs)
                                      for metric in ('auc', 'op50', 'op75', 'precision', 'norm_precision')}
    for seed_key, label in (('e2_seed42', 'seed42_delta'), ('e2_seed3407', 'seed3407_delta')):
        summary[label] = {}
        for metric in ('auc', 'op50', 'op75', 'precision', 'norm_precision'):
            deltas = [all_metrics[seed_key][s][metric] - all_metrics['m1'][s][metric] for s in seqs]
            summary[label][metric] = {
                'mean': sum(deltas) / len(deltas),
                'improved': sum(d > 1e-9 for d in deltas),
                'degraded': sum(d < -1e-9 for d in deltas),
                'tie': sum(abs(d) <= 1e-9 for d in deltas),
                'max_gain': sorted(((d, s) for d, s in zip(deltas, seqs)), reverse=True)[:5],
                'max_loss': sorted(((d, s) for d, s in zip(deltas, seqs)))[0:5],
            }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print('CSV:', csv_path)


if __name__ == '__main__':
    main()
