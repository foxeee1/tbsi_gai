#!/usr/bin/env python3
"""Evaluate the independent FP32 adaptive full-244 run against protected M1."""
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist
from analyze_m1_amcm_adaptive_full244_20261021 import (
    bbox_iou, paired_bootstrap, sequence_scores,
)


ROOT = Path(__file__).resolve().parents[1]
NAME = 'M1_AMCM_ADAPTIVE_FP32_OMP1_FULL244_20261021'
RESULTS = ROOT / 'output/test/tracking_results/tbsi_track' / NAME
M1_RESULTS = ROOT / 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_1030'
M1_EVAL = ROOT / 'output/test/result_plots/M1_S1_FULL244_CONFIRM_20261014/eval_data.pkl'
EVAL = ROOT / 'output/test/result_plots' / NAME / 'eval_data.pkl'
OUT = ROOT / 'output/experiments' / NAME
DATASET = ROOT.parent / 'data/lasher/testingset'
METRICS = ('auc', 'precision', 'norm_precision', 'op50', 'op75', 'average_overlap')


def load_boxes(path):
    return np.atleast_2d(np.loadtxt(str(path), dtype=np.float64))


def main():
    sequence_names = [name.strip() for name in
                      (ROOT.parent / 'data/lasher/testingsetList.txt').read_text().splitlines()
                      if name.strip()]
    sequence_names = [name for name in sequence_names
                      if name not in {'advancedredcup', 'cameraman_1202', 'mirroratleft'}]
    if len(sequence_names) != 244 or len(set(sequence_names)) != 244:
        raise RuntimeError('unexpected testingset manifest')
    for name in sequence_names:
        for suffix in ('.txt', '_time.txt'):
            if not (RESULTS / (name + suffix)).is_file():
                raise RuntimeError('missing result {}'.format(name + suffix))

    trackers = trackerlist('tbsi_track', NAME, 'lasher_test', run_ids=None,
                           display_name=NAME)
    dataset = get_dataset('lasher_test')
    print_results(trackers, dataset, NAME, merge_results=True,
                  plot_types=('success', 'norm_prec', 'prec'), force_evaluation=True)
    with M1_EVAL.open('rb') as stream:
        m1_eval = pickle.load(stream)
    with EVAL.open('rb') as stream:
        candidate_eval = pickle.load(stream)
    if set(m1_eval['sequences']) != set(candidate_eval['sequences']):
        raise RuntimeError('M1 and candidate evaluator sequence sets differ')
    if candidate_eval['trackers'][0]['param'] != NAME:
        raise RuntimeError('candidate evaluator loaded the wrong tracker')
    candidate_indices = {name: i for i, name in enumerate(candidate_eval['sequences'])}
    rows = []
    for i, name in enumerate(m1_eval['sequences']):
        j = candidate_indices[name]
        gt = np.atleast_2d(np.loadtxt(str(DATASET / name / 'init.txt'), delimiter=','))
        m1_boxes = load_boxes(M1_RESULTS / (name + '.txt'))
        candidate_boxes = load_boxes(RESULTS / (name + '.txt'))
        if len(gt) != len(m1_boxes) or len(gt) != len(candidate_boxes):
            raise RuntimeError('frame count mismatch for {}'.format(name))
        parent = sequence_scores(m1_eval, i, 0)
        candidate = sequence_scores(candidate_eval, j, 0)
        m1_iou = bbox_iou(m1_boxes, gt)
        candidate_iou = bbox_iou(candidate_boxes, gt)
        rows.append({
            'sequence': name, 'frames': len(gt),
            **{'m1_' + key: value for key, value in parent.items()},
            **{'candidate_' + key: value for key, value in candidate.items()},
            **{'delta_' + key: candidate[key] - parent[key] for key in METRICS},
            'm1_first_update_iou': float(m1_iou[1]),
            'candidate_first_update_iou': float(candidate_iou[1]),
            'candidate_first5_update_iou': float(candidate_iou[1:6].mean()),
            'candidate_first_box': json.dumps(candidate_boxes[1].tolist()),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'per_sequence_m1_vs_candidate.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    auc_delta = np.asarray([row['delta_auc'] for row in rows])
    summary = {
        'id': 'M1-AMCM-ADAPTIVE-FP32-OMP1-FULL244-20261021',
        'sequence_count': len(rows), 'frame_count': sum(row['frames'] for row in rows),
        'metrics': {metric: {
            'm1': float(np.mean([row['m1_' + metric] for row in rows])),
            'candidate': float(np.mean([row['candidate_' + metric] for row in rows])),
            'delta': float(np.mean([row['delta_' + metric] for row in rows])),
        } for metric in METRICS},
        'auc_wins': int((auc_delta > 1e-8).sum()),
        'auc_losses': int((auc_delta < -1e-8).sum()),
        'auc_ties': int((np.abs(auc_delta) <= 1e-8).sum()),
        'auc_delta_median': float(np.median(auc_delta)),
        'auc_mean_delta_paired_bootstrap_95_ci': paired_bootstrap(auc_delta, samples=20000),
        'auc_loss_over_5': int((auc_delta < -5).sum()),
        'auc_gain_over_5': int((auc_delta > 5).sum()),
        'first_update_collapse_count': sum(
            row['candidate_first_update_iou'] < 0.05 and row['m1_first_update_iou'] > 0.5
            for row in rows),
        'largest_losses': sorted(rows, key=lambda row: row['delta_auc'])[:12],
        'largest_gains': sorted(rows, key=lambda row: row['delta_auc'], reverse=True)[:12],
        'evaluator_cache': str(EVAL.relative_to(ROOT)),
        'candidate_results': str(RESULTS.relative_to(ROOT)),
        'm1_results': str(M1_RESULTS.relative_to(ROOT)),
    }
    (OUT / 'paired_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in (
        'sequence_count', 'frame_count', 'metrics', 'auc_wins', 'auc_losses',
        'auc_ties', 'auc_delta_median', 'auc_mean_delta_paired_bootstrap_95_ci',
        'auc_loss_over_5', 'auc_gain_over_5', 'first_update_collapse_count')}, indent=2))


if __name__ == '__main__':
    main()
