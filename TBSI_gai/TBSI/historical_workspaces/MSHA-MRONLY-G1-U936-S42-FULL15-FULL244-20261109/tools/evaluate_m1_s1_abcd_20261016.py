#!/usr/bin/env python3
"""Evaluate A-D state/output counterfactuals on the frozen 244-sequence set."""
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/experiments/M1_S1_STATE_ONLY_FULL244_20261016'
SEQ = OUT / 'sequence_list.txt'
if not SEQ.exists():
    SEQ.write_text((ROOT / 'output/experiments/M1_S1_FULL244_CONFIRM_20261014/sequence_list.txt').read_text())
os.environ['TBSI_RPP_TEST_LIST'] = str(SEQ)

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker

CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
RUNS = [('A_M1_M1STATE', '1030'), ('B_S1_M1STATE', '1044'),
        ('C_M1_S1STATE', '1050'), ('D_S1_S1STATE', '1031')]
LOW = set((ROOT / 'output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/s1_full62/sequence_list.txt').read_text().splitlines())
NORMAL = set((ROOT / 'output/experiments/M1_SINGLE_SIGNAL_NORMAL20_20261012/s1_full20/sequence_list.txt').read_text().splitlines())

dataset = get_dataset('rpp_lasher_test')
trackers = [Tracker('tbsi_track', CONFIG, 'rpp_lasher_test', run) for _, run in RUNS]
data = extract_results(trackers, dataset, 'M1_S1_ABCD_20261016', skip_missing_seq=False)
overlap = np.asarray(data['ave_success_rate_plot_overlap'], dtype=np.float64)
center = np.asarray(data['ave_success_rate_plot_center'], dtype=np.float64)
ncenter = np.asarray(data['ave_success_rate_plot_center_norm'], dtype=np.float64)
metric_arrays = {'AUC': overlap.mean(axis=2) * 100, 'PR': center[:, :, 20] * 100,
                 'NPR': ncenter[:, :, 20] * 100, 'OP50': overlap[:, :, 10] * 100,
                 'OP75': overlap[:, :, 15] * 100}
names = list(data['sequences'])
labels = [x[0] for x in RUNS]
rows = []
for i, name in enumerate(names):
    row = {'sequence': name, 'cohort': 'low_auc62' if name in LOW else ('normal20' if name in NORMAL else 'remaining162')}
    for metric, vals in metric_arrays.items():
        for j, label in enumerate(labels):
            row[label + '_' + metric] = float(vals[i, j])
        for j in range(1, 4):
            row[labels[j] + '_minus_A_' + metric] = float(vals[i, j] - vals[i, 0])
    rows.append(row)
with (OUT / 'abcd_per_sequence.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

summary = {'id': 'M1-S1-ABCD-FULL244-20261016', 'status': 'EVALUATION-COMPLETE',
           'arms': dict(RUNS), 'n_sequences': len(names), 'metrics': {}, 'cohorts': {}}
cohorts = {'low_auc62': LOW, 'normal20': NORMAL, 'remaining162': set(names) - LOW - NORMAL, 'all244': set(names)}
for metric, vals in metric_arrays.items():
    summary['metrics'][metric] = {labels[j]: float(vals[:, j].mean()) for j in range(4)}
for cohort, members in cohorts.items():
    idx = [i for i, n in enumerate(names) if n in members]
    summary['cohorts'][cohort] = {'n_sequences': len(idx), 'metrics': {}}
    for metric, vals in metric_arrays.items():
        summary['cohorts'][cohort]['metrics'][metric] = {labels[j]: float(vals[idx, j].mean()) for j in range(4)}
(OUT / 'abcd_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
