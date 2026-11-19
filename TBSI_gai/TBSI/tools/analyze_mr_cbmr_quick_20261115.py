#!/usr/bin/env python3
"""Analyze the paired 24-sequence MR-CBMR quick evaluation."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text

RUN = ROOT / 'output/experiments/MSHA-MR-CBMR-QUICK-S42-20261115'
LIST = RUN / 'contracts/eval_flip_gain_regression_24.txt'


def metrics(result_dir, sequence):
    pred = torch.tensor(load_text(str(result_dir / (sequence.name + '.txt')),
                                  delimiter=('\t', ','), dtype=np.float64))
    anno = torch.tensor(sequence.ground_truth_rect)
    visible = (torch.tensor(sequence.target_visible, dtype=torch.uint8)
               if sequence.target_visible is not None else None)
    overlap, center, center_norm, _ = calc_seq_err_robust(
        pred, anno, sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    return {
        'sequence': sequence.name,
        'frames': int(anno.shape[0]),
        'auc': float(((overlap[:, None] > thresholds[None, :]).sum(0).double() /
                      anno.shape[0]).mean() * 100),
        'precision': float((center <= 20).sum() / anno.shape[0] * 100),
        'npr': float((center_norm <= 0.2).sum() / anno.shape[0] * 100),
        'op50': float((overlap > 0.5).sum() / anno.shape[0] * 100),
        'op75': float((overlap > 0.75).sum() / anno.shape[0] * 100),
    }


def mean(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ('auc', 'precision', 'npr', 'op50', 'op75')}


def delta_rows(candidate, control):
    output = []
    for cand, ctrl in zip(candidate, control):
        if cand['sequence'] != ctrl['sequence']:
            raise RuntimeError('sequence ordering mismatch')
        row = {'sequence': cand['sequence'], 'frames': cand['frames']}
        for metric in ('auc', 'precision', 'npr', 'op50', 'op75'):
            row['control_' + metric] = ctrl[metric]
            row['candidate_' + metric] = cand[metric]
            row['delta_' + metric] = cand[metric] - ctrl[metric]
        output.append(row)
    return output


def stats(rows, metric='auc'):
    values = np.asarray([row['delta_' + metric] for row in rows], dtype=float)
    return {
        'mean_pp': float(values.mean()), 'median_pp': float(np.median(values)),
        'wins': int((values > 1e-9).sum()), 'ties': int((np.abs(values) <= 1e-9).sum()),
        'losses': int((values < -1e-9).sum()), 'min_pp': float(values.min()),
        'max_pp': float(values.max()), 'p25_pp': float(np.percentile(values, 25)),
        'p75_pp': float(np.percentile(values, 75)),
    }


def main():
    names = [x.strip() for x in LIST.read_text().splitlines() if x.strip()]
    dataset = {s.name: s for s in get_dataset('lasher_test')}
    sequences = [dataset[name] for name in names]
    control_dir = RUN / 'eval_control'
    candidate_dir = RUN / 'eval_candidate'
    control = [metrics(control_dir, s) for s in sequences]
    candidate = [metrics(candidate_dir, s) for s in sequences]
    deltas = delta_rows(candidate, control)
    groups = json.loads((RUN / 'contracts/eval_groups.json').read_text()) if \
        (RUN / 'contracts/eval_groups.json').exists() else {}
    summary = {
        'run_id': RUN.name, 'sequence_count': len(sequences),
        'frames': int(sum(row['frames'] for row in candidate)),
        'control': mean(control), 'candidate': mean(candidate),
        'candidate_minus_control': {m: float(mean(candidate)[m] - mean(control)[m])
                                    for m in ('auc', 'precision', 'npr', 'op50', 'op75')},
        'auc_delta_stats': stats(deltas), 'groups': groups,
        'eval_protocol': {'precision': 'FP32', 'tracker_threads': 0,
                          'template_update': False, 'seed': 42,
                          'cohort': 'fixed 24 exploratory MR quick cohort'},
        'throughput_fps': {
            'control': json.loads((RUN / 'eval_control/completed.json').read_text())['throughput_fps'],
            'candidate': json.loads((RUN / 'eval_candidate/completed.json').read_text())['throughput_fps'],
        },
    }
    analysis = RUN / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / 'per_sequence.csv').open('w', newline='') as handle:
        fields = list(deltas[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(deltas)
    (analysis / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    report = ['# MR-CBMR quick paired analysis', '',
              '| metric | control | CBMR candidate | delta |', '|---|---:|---:|---:|']
    for m in ('auc', 'precision', 'npr', 'op50', 'op75'):
        report.append('| %s | %.4f | %.4f | %+.4f |' %
                      (m, summary['control'][m], summary['candidate'][m],
                       summary['candidate_minus_control'][m]))
    s = summary['auc_delta_stats']
    report += ['', '## AUC sequence comparison', '',
               '| mean pp | median pp | wins | ties | losses | min pp | max pp |',
               '|---:|---:|---:|---:|---:|---:|---:|',
               '| %+.4f | %+.4f | %d | %d | %d | %+.4f | %+.4f |' %
               (s['mean_pp'], s['median_pp'], s['wins'], s['ties'], s['losses'],
                s['min_pp'], s['max_pp']), '',
               'This is a single-seed, 24-sequence exploratory result; it does not establish full244 or seed robustness.']
    (analysis / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
