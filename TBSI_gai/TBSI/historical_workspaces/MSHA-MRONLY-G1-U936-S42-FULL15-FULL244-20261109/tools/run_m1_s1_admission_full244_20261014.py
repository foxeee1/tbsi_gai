#!/usr/bin/env python3
"""Run one isolated S1 admission-policy arm on a fixed LasHeR sequence list."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
CHECKPOINT = ROOT / 'output/experiments' / CONFIG / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_PATH = ROOT / 'experiments/tbsi_track' / (CONFIG + '.yaml')
PREDICTOR = ROOT / 'experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json'
S1_POLICY = ROOT / 'experiments/artifacts/m1_readout_independent_20261014.json'
EXPERIMENT = ROOT / 'output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014'
DEFAULT_SEQUENCES = ROOT / 'output/experiments/M1_S1_FULL244_CONFIRM_20261014/sequence_list.txt'
RESULT_ROOT = ROOT / 'output/test/tracking_results/tbsi_track'


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['s1_control', 'selective_risk', 'motion_innovation'], required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--sequence-list', default=str(DEFAULT_SEQUENCES))
    args = ap.parse_args()
    sequence_file = Path(args.sequence_list).resolve()
    sequences = [line.strip() for line in sequence_file.read_text().splitlines() if line.strip()]
    arm_dir = EXPERIMENT / 'arms' / (args.arm + '_run' + args.run_id)
    result_dir = RESULT_ROOT / (CONFIG + '_' + args.run_id)
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise RuntimeError('refusing existing arm output: {}'.format(arm_dir))
    if result_dir.exists():
        raise RuntimeError('refusing existing tracker result: {}'.format(result_dir))
    if sha(CONFIG_PATH) != 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e':
        raise RuntimeError('protected M1 config hash mismatch')
    if sha(CHECKPOINT) != '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef':
        raise RuntimeError('protected M1 checkpoint hash mismatch')
    arm_dir.mkdir(parents=True)
    policy = S1_POLICY
    if args.arm == 'selective_risk':
        policy = EXPERIMENT / 'training_locked/policy_selective_risk.json'
    elif args.arm == 'motion_innovation':
        policy = EXPERIMENT / 'training_locked/policy_motion_innovation.json'
    for path in (PREDICTOR, policy, sequence_file):
        if not path.is_file():
            raise RuntimeError('missing run input: {}'.format(path))

    env = os.environ.copy()
    env.update({'TBSI_RPP_TEST_LIST': str(sequence_file), 'TBSI_INFER_FP16': '0',
                'TBSI_CHECKPOINT_OVERRIDE': str(CHECKPOINT), 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1',
                'TBSI_BOUNDED_READOUT_PREDICTOR': str(PREDICTOR),
                'TBSI_READOUT_POLICY': str(policy),
                'TBSI_READOUT_AUDIT_DIR': str(arm_dir / 'action_audit')})
    command = ['/root/autodl-tmp/conda_envs/tbsi/bin/python', '-u', 'tracking/test.py',
               'tbsi_track', CONFIG, '--runid', args.run_id,
               '--dataset_name', 'rpp_lasher_test', '--threads', '0', '--num_gpus', '1']
    manifest = {'id': 'M1-S1-ADMISSION-{}-FULL244-20261014'.format(args.arm.upper()),
                'arm': args.arm, 'run_id': args.run_id, 'parent': 'M1-S1-FULL244-CONFIRM-20261014',
                'seed': 42, 'dataset': 'LasHeR official testingset',
                'sequence_count': len(sequences), 'sequence_list': str(sequence_file),
                'sequence_list_sha256': sha(sequence_file), 'precision': 'FP32',
                'config': str(CONFIG_PATH), 'config_sha256': sha(CONFIG_PATH),
                'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha(CHECKPOINT),
                'predictor': str(PREDICTOR), 'predictor_sha256': sha(PREDICTOR),
                'policy': str(policy), 'policy_sha256': sha(policy),
                'tracker_sha256': sha(ROOT / 'lib/test/tracker/tbsi_track.py'),
                'single_changed_variable': args.arm,
                'fixed_s1': {'rho': 0.5, 'utility_threshold': 0.08},
                'command': command, 'status': 'PREFLIGHT-COMPLETE / RUNNING'}
    (arm_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    started = time.time()
    with (arm_dir / 'tracking.log').open('w') as log:
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    found = {p.stem for p in result_dir.glob('*.txt') if not p.name.endswith('_time.txt')}
    if found != set(sequences):
        raise RuntimeError('prediction coverage mismatch: expected {}, found {}'.format(len(sequences), len(found)))
    counts = {}
    total_frames = 0
    for audit in (arm_dir / 'action_audit').glob('*/actions.jsonl'):
        for line in audit.read_text().splitlines():
            if line.strip():
                action = json.loads(line)['action']
                counts[action] = counts.get(action, 0) + 1
                total_frames += 1
    manifest.update({'status': 'TRACKING-COMPLETE', 'prediction_files': len(found),
                     'tracking_seconds': time.time() - started, 'action_frames': total_frames,
                     'action_counts': counts, 'result_dir': str(result_dir)})
    (arm_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'arm': args.arm, 'sequences': len(found),
                      'frames': total_frames, 'actions': counts,
                      'tracking_seconds': manifest['tracking_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
