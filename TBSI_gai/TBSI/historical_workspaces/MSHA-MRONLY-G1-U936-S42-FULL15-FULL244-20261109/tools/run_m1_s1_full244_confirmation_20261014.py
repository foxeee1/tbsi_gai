#!/usr/bin/env python3
"""Run a matched current-M1/S1 full LasHeR test and official metric audit."""
from __future__ import print_function

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('TBSI_PYTHON', '/root/autodl-tmp/conda_envs/tbsi/bin/python')
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
CHECKPOINT = ROOT / 'output/experiments' / CONFIG / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_PATH = ROOT / 'experiments/tbsi_track' / (CONFIG + '.yaml')
TRACKER_PATH = ROOT / 'lib/test/tracker/tbsi_track.py'
PREDICTOR = ROOT / 'experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json'
POLICY = ROOT / 'experiments/artifacts/m1_readout_independent_20261014.json'
OUT = ROOT / 'output/experiments/M1_S1_FULL244_CONFIRM_20261014'
RUNS = [('M1_CURRENT', '1030'), ('S1_FROZEN', '1031')]
RESULT_ROOT = ROOT / 'output/test/tracking_results/tbsi_track'
EXPECTED_CONFIG_SHA = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
EXPECTED_CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
EXPECTED_PREDICTOR_SHA = 'b97bf2fe4d576a447f6cabe62a7523a2276f6837ffcc8761714b181119d79c6c'
EXPECTED_POLICY_SHA = 'ff4cbef73341eae20c726911f916b25cd80b6a61458bba0a1e5196004840c098'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_stage(command, env, log_path):
    started = time.time()
    with open(log_path, 'w') as stream:
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return time.time() - started


def main():
    sys.path.insert(0, str(ROOT))
    from lib.test.evaluation.lasherdataset import LasHeRDataset

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('refusing to overwrite nonempty output: {}'.format(OUT))
    if any((RESULT_ROOT / (CONFIG + '_' + runid)).exists() for _, runid in RUNS):
        raise RuntimeError('refusing to overwrite an existing result directory')
    if (sha(CONFIG_PATH) != EXPECTED_CONFIG_SHA or sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(PREDICTOR) != EXPECTED_PREDICTOR_SHA or sha(POLICY) != EXPECTED_POLICY_SHA):
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not Path(PYTHON).is_file():
        raise RuntimeError('configured Python executable is missing: {}'.format(PYTHON))

    dataset = LasHeRDataset('testingset')
    sequences = list(dataset.sequence_list)
    if len(sequences) != 244 or len(set(sequences)) != 244:
        raise RuntimeError('expected 244 unique official test sequences; got {}'.format(len(sequences)))
    for sequence in sequences:
        sequence_path = Path(dataset.base_path) / sequence
        if (not (sequence_path / 'init.txt').is_file() or
                not list((sequence_path / 'visible').glob('*.jpg')) or
                not list((sequence_path / 'infrared').glob('*.jpg'))):
            raise RuntimeError('incomplete test sequence: {}'.format(sequence))

    OUT.mkdir(parents=True)
    manifest_path = OUT / 'sequence_list.txt'
    manifest_path.write_text('\n'.join(sequences) + '\n')
    env = os.environ.copy()
    env.update({'TBSI_RPP_TEST_LIST': str(manifest_path), 'TBSI_INFER_FP16': '0',
                'TBSI_CHECKPOINT_OVERRIDE': str(CHECKPOINT), 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'})
    for key in ('TBSI_BOUNDED_READOUT_PREDICTOR', 'TBSI_READOUT_POLICY', 'TBSI_READOUT_AUDIT_DIR'):
        env.pop(key, None)

    manifest = {
        'id': 'M1-S1-FULL244-CONFIRM-20261014',
        'parent': CONFIG,
        'seed': 42,
        'checkpoint': str(CHECKPOINT),
        'checkpoint_sha256': sha(CHECKPOINT),
        'config': str(CONFIG_PATH),
        'config_sha256': sha(CONFIG_PATH),
        'tracker_code': str(TRACKER_PATH),
        'tracker_code_sha256': sha(TRACKER_PATH),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT.parents[1])).decode().strip(),
        'dataset': 'LasHeR official testingset; 244 sequences; test list excludes repository SKIP_SEQUENCES',
        'dataset_path': dataset.base_path,
        'sequence_manifest': str(manifest_path),
        'sequence_manifest_sha256': sha(manifest_path),
        'candidate_predictor': str(PREDICTOR),
        'candidate_predictor_sha256': sha(PREDICTOR),
        'candidate_policy': str(POLICY),
        'candidate_policy_sha256': sha(POLICY),
        'precision': 'FP32',
        'order': ['M1_CURRENT', 'S1_FROZEN'],
        'training': 'none; evaluate the already fitted train-side predictor and frozen threshold',
        'environment': {},
        'commands': {},
        'stages': {},
    }
    import platform
    import torch
    manifest['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
                               'cuda_runtime': torch.version.cuda,
                               'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).decode().strip()}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    for label, runid in RUNS:
        result_dir = RESULT_ROOT / (CONFIG + '_' + runid)
        command = [PYTHON, '-u', 'tracking/test.py', 'tbsi_track', CONFIG,
                   '--runid', runid, '--dataset_name', 'rpp_lasher_test',
                   '--threads', '0', '--num_gpus', '1']
        stage_env = env.copy()
        if label == 'S1_FROZEN':
            stage_env['TBSI_BOUNDED_READOUT_PREDICTOR'] = str(PREDICTOR)
            stage_env['TBSI_READOUT_POLICY'] = str(POLICY)
            stage_env['TBSI_READOUT_AUDIT_DIR'] = str(OUT / 'action_audit' / label)
        manifest['commands'][label] = command
        manifest['stages'][label] = {'status': 'RUNNING', 'log': str(OUT / (label + '.log')),
                                     'result_dir': str(result_dir)}
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('START {} runid={} at {}'.format(label, runid,
              time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), flush=True)
        duration = run_stage(command, stage_env, OUT / (label + '.log'))
        found = {p.stem for p in result_dir.glob('*.txt') if not p.name.endswith('_time.txt')}
        if found != set(sequences):
            raise RuntimeError('{} predictions mismatch: expected {}, found {}'.format(label, len(sequences), len(found)))
        manifest['stages'][label].update({'status': 'COMPLETE', 'seconds': duration,
                                          'prediction_files': len(found)})
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('FINISH {} seconds={} predictions={}'.format(label, duration, len(found)), flush=True)

    # Use the repository evaluator for success, precision, and normalized precision curves.
    from lib.test.analysis.extract_results import extract_results
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    official_dataset = get_dataset('rpp_lasher_test')
    trackers = [Tracker('tbsi_track', CONFIG, 'rpp_lasher_test', runid) for _, runid in RUNS]
    plot_name = 'M1_S1_FULL244_CONFIRM_20261014'
    data = extract_results(trackers, official_dataset, plot_name, skip_missing_seq=False)
    overlap = np.asarray(data['ave_success_rate_plot_overlap'], dtype=np.float64)
    center = np.asarray(data['ave_success_rate_plot_center'], dtype=np.float64)
    center_norm = np.asarray(data['ave_success_rate_plot_center_norm'], dtype=np.float64)
    names = data['sequences']
    if len(names) != 244 or overlap.shape[:2] != (244, 2):
        raise RuntimeError('official evaluator did not return a complete 244x2 result')
    metric_arrays = {'AUC': overlap.mean(axis=2) * 100.0,
                     'PR': center[:, :, 20] * 100.0,
                     'NPR': center_norm[:, :, 20] * 100.0,
                     'OP50': overlap[:, :, 10] * 100.0,
                     'OP75': overlap[:, :, 15] * 100.0}
    summaries = {}
    per_sequence = []
    for i, name in enumerate(names):
        row = {'sequence': name}
        for metric, values in metric_arrays.items():
            row['M1_' + metric] = float(values[i, 0])
            row['S1_' + metric] = float(values[i, 1])
            row['delta_' + metric] = float(values[i, 1] - values[i, 0])
        per_sequence.append(row)
    for metric, values in metric_arrays.items():
        delta = values[:, 1] - values[:, 0]
        summaries[metric] = {'M1': float(values[:, 0].mean()), 'S1': float(values[:, 1].mean()),
                             'delta': float(delta.mean()), 'wins': int((delta > 0).sum()),
                             'losses': int((delta < 0).sum()), 'ties': int((delta == 0).sum())}
    with (OUT / 'per_sequence_official.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sequence[0]))
        writer.writeheader()
        writer.writerows(per_sequence)
    (OUT / 'official_summary.json').write_text(json.dumps({'n_sequences': len(names), 'metrics': summaries,
                                                           'official_report': str(ROOT / 'output/test/result_plots' / plot_name / 'eval_data.pkl')}, indent=2) + '\n')

    audit_dir = OUT / 'persistence_audit'
    audit_command = [PYTHON, '-u', 'tools/m1_official_metric_and_persistence_audit.py',
                     '--repo', str(ROOT), '--pred-dir', str(RESULT_ROOT / (CONFIG + '_1031')),
                     '--baseline-dir', str(RESULT_ROOT / (CONFIG + '_1030')),
                     '--dataset', 'rpp_lasher_test', '--output-dir', str(audit_dir)]
    manifest['commands']['official_persistence_audit'] = audit_command
    manifest['stages']['official_evaluation'] = {'status': 'COMPLETE', 'sequences': len(names),
                                                  'summary': str(OUT / 'official_summary.json')}
    manifest['stages']['persistence_audit'] = {'status': 'RUNNING', 'log': str(OUT / 'persistence_audit.log')}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    duration = run_stage(audit_command, env, OUT / 'persistence_audit.log')
    manifest['stages']['persistence_audit'].update({'status': 'COMPLETE', 'seconds': duration})
    manifest['status'] = 'COMPLETE / ANALYSIS-RECORDED'
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
