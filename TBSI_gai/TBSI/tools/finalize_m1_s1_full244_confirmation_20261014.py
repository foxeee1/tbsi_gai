#!/usr/bin/env python3
"""Finalize official metrics for completed M1/S1 full-244 predictions."""
import csv, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/experiments/M1_S1_FULL244_CONFIRM_20261014'
SEQ = OUT / 'sequence_list.txt'
RESULT_ROOT = ROOT / 'output/test/tracking_results/tbsi_track'
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
M1 = RESULT_ROOT / (CONFIG + '_1030')
S1 = RESULT_ROOT / (CONFIG + '_1031')

def main():
    if not SEQ.is_file() or len([x for x in SEQ.read_text().splitlines() if x.strip()]) != 244:
        raise RuntimeError('full-244 manifest missing or incomplete')
    for d in (M1, S1):
        if len([p for p in d.glob('*.txt') if not p.name.endswith('_time.txt')]) != 244:
            raise RuntimeError('prediction directory incomplete: {}'.format(d))
    os.environ['TBSI_RPP_TEST_LIST'] = str(SEQ)
    sys.path.insert(0, str(ROOT))
    from lib.test.analysis.extract_results import extract_results
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    dataset = get_dataset('rpp_lasher_test')
    trackers = [Tracker('tbsi_track', CONFIG, 'rpp_lasher_test', '1030'),
                Tracker('tbsi_track', CONFIG, 'rpp_lasher_test', '1031')]
    report = 'M1_S1_FULL244_CONFIRM_20261014'
    data = extract_results(trackers, dataset, report, skip_missing_seq=False)
    overlap = np.asarray(data['ave_success_rate_plot_overlap'], dtype=float)
    center = np.asarray(data['ave_success_rate_plot_center'], dtype=float)
    norm = np.asarray(data['ave_success_rate_plot_center_norm'], dtype=float)
    names = data['sequences']
    metrics = {'AUC': overlap.mean(axis=2) * 100, 'PR': center[:, :, 20] * 100,
               'NPR': norm[:, :, 20] * 100, 'OP50': overlap[:, :, 10] * 100,
               'OP75': overlap[:, :, 15] * 100}
    summary = {}
    rows = []
    for i, name in enumerate(names):
        row = {'sequence': name}
        for metric, value in metrics.items():
            row['M1_' + metric] = float(value[i, 0]); row['S1_' + metric] = float(value[i, 1])
            row['delta_' + metric] = float(value[i, 1] - value[i, 0])
        rows.append(row)
    for metric, value in metrics.items():
        delta = value[:, 1] - value[:, 0]
        summary[metric] = {'M1': float(value[:, 0].mean()), 'S1': float(value[:, 1].mean()),
                           'delta': float(delta.mean()), 'wins': int((delta > 0).sum()),
                           'losses': int((delta < 0).sum()), 'ties': int((delta == 0).sum())}
    with (OUT / 'per_sequence_official.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / 'official_summary.json').write_text(json.dumps({'n_sequences': len(names), 'metrics': summary,
        'official_eval_data': str(ROOT / 'output/test/result_plots' / report / 'eval_data.pkl')}, indent=2) + '\n')
    audit = OUT / 'persistence_audit'
    audit.mkdir(exist_ok=True)
    env = os.environ.copy(); env['TBSI_RPP_TEST_LIST'] = str(SEQ)
    cmd = [sys.executable, '-u', 'tools/m1_official_metric_and_persistence_audit.py', '--repo', str(ROOT),
           '--pred-dir', str(S1), '--baseline-dir', str(M1), '--dataset', 'rpp_lasher_test',
           '--output-dir', str(audit)]
    with (OUT / 'persistence_audit.log').open('w') as log:
        subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    manifest['stages']['official_evaluation'] = {'status':'COMPLETE', 'sequences':len(names),
                                                 'summary':str(OUT/'official_summary.json')}
    manifest['stages']['persistence_audit'] = {'status':'COMPLETE', 'output':str(audit)}
    manifest['status'] = 'COMPLETE / ANALYSIS-RECORDED'
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
