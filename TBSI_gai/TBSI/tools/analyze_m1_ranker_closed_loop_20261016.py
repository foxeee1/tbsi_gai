import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.evaluation import get_dataset, trackerlist
from lib.test.analysis.extract_results import extract_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    args = ap.parse_args()
    os.environ['TBSI_RPP_TEST_LIST'] = os.path.abspath(args.manifest)
    dataset = get_dataset('rpp_lasher_test')
    trackers = trackerlist('tbsi_track', args.config, 'rpp_lasher_test',
                           run_ids=[None, 102, 103],
                           display_name='M1|output_only|coupled')
    # M1 is the protected full-244 result directory; the two new dirs use run ids.
    eval_data = extract_results(trackers, dataset,
                                'm1_ranker_closed_loop_union40_20261016')
    overlap = np.asarray(eval_data['ave_success_rate_plot_overlap'], dtype=np.float64)
    center = np.asarray(eval_data['ave_success_rate_plot_center'], dtype=np.float64)
    center_norm = np.asarray(eval_data['ave_success_rate_plot_center_norm'], dtype=np.float64)
    avg_overlap = np.asarray(eval_data['avg_overlap_all'], dtype=np.float64)
    thresholds = np.asarray(eval_data['threshold_set_overlap'], dtype=np.float64)
    # Match print_results/get_auc_curve: mean over the sampled overlap thresholds.
    auc = overlap.mean(axis=2) * 100.0
    op50 = overlap[:, :, int(np.argmin(np.abs(thresholds - 0.50)))]
    op75 = overlap[:, :, int(np.argmin(np.abs(thresholds - 0.75)))]
    precision = center[:, :, 20]
    norm_precision = center_norm[:, :, 20]
    names = [s.name for s in dataset]
    normal = np.arange(len(names)) < 20
    # The manifest is ordered as normal20 followed by difficult20.
    rows = []
    for i, name in enumerate(names):
        rows.append({'sequence': name, 'group': 'normal20' if normal[i] else 'difficult20',
                     'm1_auc': float(auc[i, 0]), 'output_only_auc': float(auc[i, 1]),
                     'coupled_auc': float(auc[i, 2]),
                     'output_only_delta': float(auc[i, 1] - auc[i, 0]),
                     'coupled_delta': float(auc[i, 2] - auc[i, 0]),
                     'm1_avg_iou': float(avg_overlap[i, 0]),
                     'output_only_avg_iou': float(avg_overlap[i, 1]),
                     'coupled_avg_iou': float(avg_overlap[i, 2]),
                     'm1_op50': float(op50[i, 0]), 'output_only_op50': float(op50[i, 1]), 'coupled_op50': float(op50[i, 2]),
                     'm1_precision': float(precision[i, 0]), 'output_only_precision': float(precision[i, 1]), 'coupled_precision': float(precision[i, 2]),
                     'm1_norm_precision': float(norm_precision[i, 0]), 'output_only_norm_precision': float(norm_precision[i, 1]), 'coupled_norm_precision': float(norm_precision[i, 2])})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.with_suffix('.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {'sequences': len(rows), 'methods': ['m1', 'output_only', 'coupled'], 'groups': {}}
    for group, mask in [('all', np.ones(len(rows), dtype=bool)),
                        ('normal20', normal), ('difficult20', ~normal)]:
        summary['groups'][group] = {}
        for j, method in enumerate(['m1', 'output_only', 'coupled']):
            summary['groups'][group][method] = {
                'auc': float(auc[mask, j].mean()),
                'avg_iou': float(avg_overlap[mask, j].mean()),
                'op50': float(op50[mask, j].mean()),
                'precision': float(precision[mask, j].mean()),
                'norm_precision': float(norm_precision[mask, j].mean()),
                'wins_vs_m1': int((auc[mask, j] > auc[mask, 0] + 1e-12).sum()) if j else 0,
                'losses_vs_m1': int((auc[mask, j] < auc[mask, 0] - 1e-12).sum()) if j else 0,
            }
    summary['delta_output_only_minus_coupled_auc'] = float(auc[:, 1].mean() - auc[:, 2].mean())
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
