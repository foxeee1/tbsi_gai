#!/usr/bin/env python3
"""Run the registered E10 spatial-token-bridge amplitude dose-response replay."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'
OUT = ROOT / 'output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024'
PARENT_EVENTS = ROOT / 'output/experiments/M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_E10_OFF_FIXED46_20261024/per_event.csv'
RUNNER = ROOT / 'tools/run_m1_screening_fixed_state_flow_20261022.py'
CONFIG = EXPERIMENTS / 'tbsi_track/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_20261024.yaml'
CHECKPOINT = ROOT / 'output/experiments/M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024/checkpoints/TBSITrack_ep0010.pth.tar'
ENV_EVENTS = OUT / 'events.jsonl'
SEQUENCE_LIST = OUT / 'sequence_list.txt'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_events():
    grouped = {}
    with PARENT_EVENTS.open(newline='') as stream:
        for row in csv.DictReader(stream):
            event = grouped.setdefault(row['sequence'], {'sequence': row['sequence']})
            if row['phase'] == 'pre_onset':
                event.update({
                    'pre_frame': int(row['frame']),
                    'pre_iou': float(row['trace_iou']),
                    'pre_state_before': json.loads(row['trace_state_before']),
                    'pre_gt_box': json.loads(row['gt_box']),
                    'pre_crop_xywh': json.loads(row['trace_crop_xywh']),
                })
            else:
                event.update({
                    'frame': int(row['frame']),
                    'm1_iou': float(row['trace_iou']),
                    'state_before': json.loads(row['trace_state_before']),
                    'gt_box': json.loads(row['gt_box']),
                    'crop_xywh': json.loads(row['trace_crop_xywh']),
                })
    events = [grouped[name] for name in sorted(grouped)]
    fixed_states = sum(1 + int('pre_frame' in event) for event in events)
    if fixed_states != 46 or len(events) != 23:
        raise RuntimeError('parent calibration replay must resolve to 46 states and 23 sequences')
    OUT.mkdir(parents=True, exist_ok=False)
    ENV_EVENTS.write_text(''.join(json.dumps(event) + '\n' for event in events))
    SEQUENCE_LIST.write_text(''.join(name + '\n' for name in
                                     sorted(set(event['sequence'] for event in events))))
    return events


def main():
    if OUT.exists():
        raise RuntimeError('refusing to overwrite amplitude diagnostic output: {}'.format(OUT))
    for path in (PARENT_EVENTS, RUNNER, CONFIG, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    events = reconstruct_events()
    arms = [('off', 1.0)] + [(mode, scale) for scale in (1.0, 32.0, 128.0)
                            for mode in ('target', 'shifted')]
    planned_manifest = {
        'experiment': 'M1-SPATIAL-TOKEN-BRIDGE-AMPLITUDE-FIXED46-RETRY1-20261024',
        'status': 'RUNNING',
        'parent': 'M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024 epoch10',
        'parent_events': str(PARENT_EVENTS),
        'parent_events_sha256': sha256(PARENT_EVENTS),
        'events_sha256': sha256(ENV_EVENTS),
        'sequence_list_sha256': sha256(SEQUENCE_LIST),
        'config': str(CONFIG), 'config_sha256': sha256(CONFIG),
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha256(CHECKPOINT),
        'runner': str(RUNNER), 'runner_sha256': sha256(RUNNER),
        'wrapper': str(Path(__file__).resolve()), 'wrapper_sha256': sha256(__file__),
        'source_snapshot': str(EXPERIMENTS / 'snapshots/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_20261024/executed_sources_retry1.tar.gz'),
        'source_snapshot_sha256': sha256(EXPERIMENTS / 'snapshots/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_20261024/executed_sources_retry1.tar.gz'),
        'split': 'LasHeR train-side calibration cohort; identical fixed states inherited from E10 off replay',
        'fixed_states': 46, 'sequences': 23, 'precision': 'FP32',
        'torch_threads': 1,
        'planned_arms': [{'mode': mode, 'scale': scale} for mode, scale in arms],
        'scale_semantics': 'Post-bridge output delta multiplier; diagnostic intervention only; scale=1 is native learned message; off is zero-message parent control.',
        'decision_target': 'Dose response of localization readouts and target-vs-equal-density-shifted correspondence; not a performance claim.',
    }
    (OUT / 'run_manifest.json').write_text(json.dumps(planned_manifest, indent=2) + '\n')
    run_log = OUT / 'commands_and_logs.txt'
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'PYTHONUNBUFFERED': '1', 'TBSI_INFER_FP16': '0'})
    command_rows = []
    for mode, scale in arms:
        label = '{}_x{:g}'.format(mode, scale).replace('.', 'p')
        output_dir = OUT / 'runs' / label
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise RuntimeError('refusing to overwrite {}'.format(output_dir))
        command = [sys.executable, str(RUNNER), '--events', str(ENV_EVENTS),
                   '--sequence-list', str(SEQUENCE_LIST), '--output-dir', str(output_dir),
                   '--config', str(CONFIG), '--checkpoint', str(CHECKPOINT),
                   '--spatial-token-bridge-mode', mode,
                   '--spatial-token-bridge-scale', str(scale)]
        command_rows.append({'arm': label, 'command': command, 'output_dir': str(output_dir)})
        print('[start] {}: {}'.format(label, ' '.join(command)), flush=True)
        with run_log.open('a') as log:
            log.write('[start] {}\n$ {}\n'.format(label, ' '.join(command)))
            log.flush()
            result = subprocess.run(command, cwd=str(ROOT), env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            log.write('[exit] {} code={}\n'.format(label, result.returncode))
        if result.returncode != 0:
            raise RuntimeError('{} failed with exit code {}'.format(label, result.returncode))
        print('[done] {}'.format(label), flush=True)

    manifest = dict(planned_manifest)
    manifest['status'] = 'COMPLETE'
    manifest['arms'] = command_rows
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'fixed_states': 46, 'arms_completed': len(arms),
                      'output_dir': str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
