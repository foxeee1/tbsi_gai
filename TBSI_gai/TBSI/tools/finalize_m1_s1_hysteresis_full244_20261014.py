#!/usr/bin/env python3
"""Resume official evaluation and analysis from completed run1032 predictions."""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
OUT = ROOT / 'output/experiments/M1_S1_HYSTERESIS_FULL244_20261014'
SEQUENCES = ROOT / 'output/experiments/M1_S1_FULL244_CONFIRM_20261014/sequence_list.txt'
RUN_ID = '1032'


def main():
    sys.path.insert(0, str(ROOT))
    from lib.test.evaluation.lasherdataset import LasHeRDataset
    sequences = [x.strip() for x in SEQUENCES.read_text().splitlines() if x.strip()]
    pred_dir = ROOT / 'output/test/tracking_results/tbsi_track' / (CONFIG + '_' + RUN_ID)
    found = {p.stem for p in pred_dir.glob('*.txt') if not p.name.endswith('_time.txt')}
    if len(sequences) != 244 or found != set(sequences):
        raise RuntimeError('run1032 prediction coverage mismatch: {}'.format(len(found)))

    os.environ['TBSI_RPP_TEST_LIST'] = str(SEQUENCES)
    from lib.test.analysis.extract_results import extract_results
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    official_dataset = get_dataset('rpp_lasher_test')
    trackers = [Tracker('tbsi_track', CONFIG, 'rpp_lasher_test', rid)
                for rid in ('1030', '1031', RUN_ID)]
    plot_name = 'M1_S1_HYSTERESIS_FULL244_20261014'
    data = extract_results(trackers, official_dataset, plot_name, skip_missing_seq=False)
    names = data['sequences']
    if len(names) != 244:
        raise RuntimeError('evaluator returned {} sequences'.format(len(names)))
    overlap = np.asarray(data['ave_success_rate_plot_overlap'], dtype=np.float64)
    center = np.asarray(data['ave_success_rate_plot_center'], dtype=np.float64)
    center_norm = np.asarray(data['ave_success_rate_plot_center_norm'], dtype=np.float64)
    values = {'AUC': overlap.mean(axis=2) * 100., 'PR': center[:, :, 20] * 100.,
              'NPR': center_norm[:, :, 20] * 100., 'OP50': overlap[:, :, 10] * 100.,
              'OP75': overlap[:, :, 15] * 100.}
    summary = {'n_sequences': len(names), 'arms': ['M1', 'S1', 'S1_HYSTERESIS'], 'metrics': {}}
    per_seq = []
    for i, name in enumerate(names):
        row = {'sequence': name}
        for metric, arr in values.items():
            for arm, idx in [('M1', 0), ('S1', 1), ('S1_HYSTERESIS', 2)]:
                row[arm + '_' + metric] = float(arr[i, idx])
            row['HYSTERESIS_minus_M1_' + metric] = float(arr[i, 2] - arr[i, 0])
            row['HYSTERESIS_minus_S1_' + metric] = float(arr[i, 2] - arr[i, 1])
        per_seq.append(row)
    for metric, arr in values.items():
        d_m1, d_s1 = arr[:, 2] - arr[:, 0], arr[:, 2] - arr[:, 1]
        summary['metrics'][metric] = {
            'M1': float(arr[:, 0].mean()), 'S1': float(arr[:, 1].mean()),
            'S1_HYSTERESIS': float(arr[:, 2].mean()),
            'hysteresis_minus_M1': float(d_m1.mean()),
            'hysteresis_minus_S1': float(d_s1.mean()),
            'wins_losses_ties_vs_M1': [int((d_m1 > 0).sum()), int((d_m1 < 0).sum()), int((d_m1 == 0).sum())],
            'wins_losses_ties_vs_S1': [int((d_s1 > 0).sum()), int((d_s1 < 0).sum()), int((d_s1 == 0).sum())],
        }
    with (OUT / 'per_sequence_official.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seq[0])); writer.writeheader(); writer.writerows(per_seq)
    (OUT / 'official_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    action_counts, frames = {}, 0
    for path in (OUT / 'action_audit').glob('*/actions.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip():
                action = json.loads(line)['action']
                action_counts[action] = action_counts.get(action, 0) + 1
                frames += 1
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['stages'] = {
        'tracking': {'status': 'COMPLETE', 'run_id': RUN_ID, 'prediction_files': len(found),
                     'log': str(OUT / 'S1_HYSTERESIS.log')},
        'official_evaluation': {'status': 'COMPLETE', 'sequences': len(names),
                                'summary': str(OUT / 'official_summary.json'),
                                'eval_data': str(ROOT / 'output/test/result_plots' / plot_name / 'eval_data.pkl')},
        'action_audit': {'status': 'COMPLETE', 'sequences': 244, 'frames': frames,
                         'counts': action_counts},
        'initial_evaluator_attempt': {'status': 'FAILED-CONNECTION',
                                      'reason': 'TBSI_RPP_TEST_LIST was not set in runner process; tracking outputs preserved'},
    }
    manifest.update({'prediction_files': len(found), 'action_frames': frames,
                     'action_counts': action_counts,
                     'official_summary': str(OUT / 'official_summary.json'),
                     'per_sequence': str(OUT / 'per_sequence_official.csv'),
                     'finalizer': 'tools/finalize_m1_s1_hysteresis_full244_20261014.py',
                     'status': 'COMPLETE / OFFICIAL-FULL244-RECORDED'})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
