#!/usr/bin/env python3
"""Run C: M1 output with S1 internal state (full LasHeR test set)."""
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('TBSI_PYTHON', '/root/autodl-tmp/conda_envs/tbsi/bin/python')
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
CHECKPOINT = ROOT / 'output/experiments' / CONFIG / 'checkpoints/TBSITrack_ep0015.pth.tar'
PREDICTOR = ROOT / 'experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json'
POLICY = ROOT / 'experiments/artifacts/m1_readout_independent_20261014.json'
DATASET = ROOT / 'output/experiments/M1_S1_FULL244_CONFIRM_20261014/sequence_list.txt'
OUT = ROOT / 'output/experiments/M1_S1_STATE_ONLY_FULL244_20261016'
RESULT_ROOT = ROOT / 'output/test/tracking_results/tbsi_track'
RUN_ID = '1050'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('refusing nonempty output: ' + str(OUT))
    if not DATASET.is_file() or len(DATASET.read_text().splitlines()) != 244:
        raise RuntimeError('expected frozen 244-sequence manifest')
    result_dir = RESULT_ROOT / (CONFIG + '_' + RUN_ID)
    if result_dir.exists():
        raise RuntimeError('refusing existing result directory: ' + str(result_dir))
    OUT.mkdir(parents=True)
    manifest = {
        'id': 'M1-S1-STATE-ONLY-FULL244-20261016',
        'parent': 'M1-S1-FULL244-CONFIRM-20261014',
        'hypothesis': 'If state feedback causes remaining162 degradation, M1 output with S1 state will retain the degradation; comparison with B isolates whether output and state effects differ.',
        'one_changed_variable': 'TBSI_READOUT_STATE_MODE=state_only',
        'config': str(ROOT / 'experiments/tbsi_track' / (CONFIG + '.yaml')),
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha(CHECKPOINT),
        'predictor': str(PREDICTOR), 'predictor_sha256': sha(PREDICTOR),
        'policy': str(POLICY), 'policy_sha256': sha(POLICY),
        'dataset_manifest': str(DATASET), 'sequences': 244,
        'environment': {'python': platform.python_version(), 'workers': 0, 'tracker_processes': 1},
        'status': 'RUNNING', 'run_id': RUN_ID,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = os.environ.copy()
    env.update({
        'TBSI_RPP_TEST_LIST': str(DATASET), 'TBSI_INFER_FP16': '0',
        'TBSI_CHECKPOINT_OVERRIDE': str(CHECKPOINT),
        'TBSI_BOUNDED_READOUT_PREDICTOR': str(PREDICTOR),
        'TBSI_READOUT_POLICY': str(POLICY),
        'TBSI_READOUT_STATE_MODE': 'state_only',
        'TBSI_READOUT_AUDIT_DIR': str(OUT / 'action_audit'),
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1',
    })
    command = [PYTHON, '-u', 'tracking/test.py', 'tbsi_track', CONFIG,
               '--runid', RUN_ID, '--dataset_name', 'rpp_lasher_test',
               '--threads', '0', '--num_gpus', '1']
    manifest['command'] = command
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    started = time.time()
    with (OUT / 'tracking.log').open('w') as log:
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    found = {p.stem for p in result_dir.glob('*.txt') if not p.name.endswith('_time.txt')}
    if found != set(DATASET.read_text().splitlines()):
        raise RuntimeError('prediction coverage mismatch: {}'.format(len(found)))
    manifest['status'] = 'TRACKING-COMPLETE'
    manifest['elapsed_seconds'] = time.time() - started
    manifest['prediction_files'] = len(found)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'seconds': manifest['elapsed_seconds'], 'predictions': len(found)}))


if __name__ == '__main__':
    main()
