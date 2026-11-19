#!/usr/bin/env python3
"""Officially evaluate and compare the S1 admission-policy full-244 arms."""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
SEQUENCES = ROOT / 'output/experiments/M1_S1_FULL244_CONFIRM_20261014/sequence_list.txt'
OUT = ROOT / 'output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/evaluation'
ARMS = [('M1', '1030'), ('S1_PRIOR', '1031'), ('S1_CONTROL', '1041'),
        ('S1_SELECTIVE_RISK', '1042'), ('S1_MOTION_INNOVATION', '1043')]


def main():
    sys.path.insert(0, str(ROOT))
    sequences = [line.strip() for line in SEQUENCES.read_text().splitlines() if line.strip()]
    if len(sequences) != 244:
        raise RuntimeError('expected 244 official sequences, got {}'.format(len(sequences)))
    for arm, run_id in ARMS:
        path = ROOT / 'output/test/tracking_results/tbsi_track' / (CONFIG + '_' + run_id)
        found = {p.stem for p in path.glob('*.txt') if not p.name.endswith('_time.txt')}
        if found != set(sequences):
            raise RuntimeError('{} incomplete: {}/244'.format(arm, len(found)))
    os.environ['TBSI_RPP_TEST_LIST'] = str(SEQUENCES)
    from lib.test.analysis.extract_results import extract_results
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    trackers = [Tracker('tbsi_track', CONFIG, 'rpp_lasher_test', run_id)
                for _, run_id in ARMS]
    data = extract_results(trackers, get_dataset('rpp_lasher_test'),
                           'M1_S1_ADMISSION_METHODS_FULL244_20261014', skip_missing_seq=False)
    names = data['sequences']
    if len(names) != 244:
        raise RuntimeError('official evaluator returned {} sequences'.format(len(names)))
    overlap = np.asarray(data['ave_success_rate_plot_overlap'], dtype=np.float64)
    center = np.asarray(data['ave_success_rate_plot_center'], dtype=np.float64)
    center_norm = np.asarray(data['ave_success_rate_plot_center_norm'], dtype=np.float64)
    metrics = {'AUC': overlap.mean(axis=2) * 100.0,
               'PR': center[:, :, 20] * 100.0,
               'NPR': center_norm[:, :, 20] * 100.0,
               'OP50': overlap[:, :, 10] * 100.0,
               'OP75': overlap[:, :, 15] * 100.0}
    arm_names = [arm for arm, _ in ARMS]
    rows = []
    summary = {'n_sequences': len(names), 'arms': arm_names, 'metrics': {}}
    for i, name in enumerate(names):
        row = {'sequence': name}
        for metric, values in metrics.items():
            for j, arm in enumerate(arm_names):
                row[arm + '_' + metric] = float(values[i, j])
            for arm in arm_names[2:]:
                row[arm + '_minus_S1_CONTROL_' + metric] = float(values[i, arm_names.index(arm)] - values[i, 2])
                row[arm + '_minus_M1_' + metric] = float(values[i, arm_names.index(arm)] - values[i, 0])
        rows.append(row)
    for metric, values in metrics.items():
        summary['metrics'][metric] = {}
        for j, arm in enumerate(arm_names):
            summary['metrics'][metric][arm] = float(values[:, j].mean())
        for j, arm in enumerate(arm_names[2:], start=2):
            delta_s1 = values[:, j] - values[:, 2]
            delta_m1 = values[:, j] - values[:, 0]
            summary['metrics'][metric][arm + '_minus_S1_CONTROL'] = float(delta_s1.mean())
            summary['metrics'][metric][arm + '_minus_M1'] = float(delta_m1.mean())
            summary['metrics'][metric][arm + '_wins_losses_ties_vs_S1_CONTROL'] = [
                int((delta_s1 > 0).sum()), int((delta_s1 < 0).sum()), int((delta_s1 == 0).sum())]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'per_sequence_official.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'official_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
