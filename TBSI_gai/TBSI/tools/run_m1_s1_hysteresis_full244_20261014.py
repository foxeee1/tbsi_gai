#!/usr/bin/env python3
"""Run S1 with only temporal action hysteresis changed, then compare full-244."""
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
REPO = ROOT.parent
PYTHON = os.environ.get('TBSI_PYTHON', '/root/autodl-tmp/conda_envs/tbsi/bin/python')
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
CHECKPOINT = ROOT / 'output/experiments' / CONFIG / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_PATH = ROOT / 'experiments/tbsi_track' / (CONFIG + '.yaml')
PREDICTOR = ROOT / 'experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json'
POLICY = ROOT / 'experiments/artifacts/m1_readout_hysteresis_confirm2_release2_20261014.json'
OUT = ROOT / 'output/experiments/M1_S1_HYSTERESIS_FULL244_20261014'
RESULT_ROOT = ROOT / 'output/test/tracking_results/tbsi_track'
RUN_ID = '1032'
RUN_RESULT = RESULT_ROOT / (CONFIG + '_' + RUN_ID)
M1_RESULT = RESULT_ROOT / (CONFIG + '_1030')
S1_RESULT = RESULT_ROOT / (CONFIG + '_1031')
SEQUENCE_SOURCE = ROOT / 'output/experiments/M1_S1_FULL244_CONFIRM_20261014/sequence_list.txt'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run(command, env, log):
    with open(log, 'w') as f:
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('refusing nonempty output directory: {}'.format(OUT))
    if RUN_RESULT.exists():
        raise RuntimeError('refusing existing tracker result: {}'.format(RUN_RESULT))
    if not all(p.is_dir() for p in (M1_RESULT, S1_RESULT)):
        raise RuntimeError('full-244 M1/S1 comparison results are missing')
    if not Path(PYTHON).is_file():
        raise RuntimeError('missing Python executable: {}'.format(PYTHON))
    if sha(CONFIG_PATH) != 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e':
        raise RuntimeError('protected M1 config hash mismatch')
    if sha(CHECKPOINT) != '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef':
        raise RuntimeError('protected M1 checkpoint hash mismatch')

    sys.path.insert(0, str(ROOT))
    from lib.test.evaluation.lasherdataset import LasHeRDataset
    dataset = LasHeRDataset('testingset')
    sequences = list(dataset.sequence_list)
    expected = [x.strip() for x in SEQUENCE_SOURCE.read_text().splitlines() if x.strip()]
    if len(sequences) != 244 or sequences != expected:
        raise RuntimeError('official full-244 sequence manifest differs from S1 confirmation')

    OUT.mkdir(parents=True)
    manifest = {
        'id': 'M1-S1-HYSTERESIS-FULL244-20261014',
        'parent': 'M1-S1-FULL244-CONFIRM-20261014 S1_FROZEN run1031',
        'm1_parent': CONFIG,
        'seed': 42,
        'sequence_count': len(sequences),
        'sequence_manifest_source': str(SEQUENCE_SOURCE),
        'sequence_manifest_sha256': sha(SEQUENCE_SOURCE),
        'config_sha256': sha(CONFIG_PATH),
        'checkpoint_sha256': sha(CHECKPOINT),
        'predictor': str(PREDICTOR),
        'predictor_sha256': sha(PREDICTOR),
        'policy': str(POLICY),
        'policy_sha256': sha(POLICY),
        'single_changed_variable': 'temporal action hysteresis: confirm_frames=2, release_frames=2',
        'fixed_s1_parameters': {'rho': 0.5, 'utility_threshold': 0.08,
                                'predictor': 'frozen train-side S1 predictor'},
        'comparators': {'M1': str(M1_RESULT), 'S1': str(S1_RESULT)},
        'precision': 'FP32',
        'environment': {},
        'command': [],
        'status': 'PREFLIGHT-COMPLETE / RUNNING',
    }
    import platform
    import torch
    manifest['environment'] = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).decode().strip(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
        'tracker_sha256': sha(ROOT / 'lib/test/tracker/tbsi_track.py'),
    }
    env = os.environ.copy()
    env.update({'TBSI_RPP_TEST_LIST': str(SEQUENCE_SOURCE), 'TBSI_INFER_FP16': '0',
                'TBSI_CHECKPOINT_OVERRIDE': str(CHECKPOINT), 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1',
                'TBSI_BOUNDED_READOUT_PREDICTOR': str(PREDICTOR),
                'TBSI_READOUT_POLICY': str(POLICY),
                'TBSI_READOUT_AUDIT_DIR': str(OUT / 'action_audit')})
    command = [PYTHON, '-u', 'tracking/test.py', 'tbsi_track', CONFIG,
               '--runid', RUN_ID, '--dataset_name', 'rpp_lasher_test',
               '--threads', '0', '--num_gpus', '1']
    manifest['command'] = command
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    started = time.time()
    run(command, env, OUT / 'S1_HYSTERESIS.log')
    prediction_files = {p.stem for p in RUN_RESULT.glob('*.txt') if not p.name.endswith('_time.txt')}
    if prediction_files != set(sequences):
        raise RuntimeError('candidate prediction coverage mismatch: {}'.format(len(prediction_files)))

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
        w = csv.DictWriter(f, fieldnames=list(per_seq[0])); w.writeheader(); w.writerows(per_seq)
    (OUT / 'official_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    action_counts = {}
    frames = 0
    for path in (OUT / 'action_audit').glob('*/actions.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip():
                action = json.loads(line)['action']
                action_counts[action] = action_counts.get(action, 0) + 1
                frames += 1
    manifest.update({'tracking_seconds': time.time() - started,
                     'prediction_files': len(prediction_files), 'action_frames': frames,
                     'action_counts': action_counts,
                     'official_summary': str(OUT / 'official_summary.json'),
                     'per_sequence': str(OUT / 'per_sequence_official.csv'),
                     'status': 'COMPLETE / OFFICIAL-FULL244-RECORDED'})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
