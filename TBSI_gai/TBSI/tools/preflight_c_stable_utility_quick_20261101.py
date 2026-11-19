#!/usr/bin/env python3
"""Validate the isolated C stable-utility quick run contract."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/early_gate_closed_loop_20261026/c_stable_utility_quick_20261101'
RUN_ID = os.environ.get('C_STABLE_UTILITY_RUN_ID', 'EG_C_STABLE_UTILITY_QUICK_S42_20261101')
OUT = ROOT / 'output/experiments' / RUN_ID
CONTROL = ROOT / 'experiments/tbsi_track/EG_C_STABLE_UTILITY_QUICK_CONTROL_S42_20261101.yaml'
CANDIDATE = ROOT / 'experiments/tbsi_track/EG_C_STABLE_UTILITY_QUICK_CANDIDATE_S42_20261101.yaml'
PUBLIC = ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'
P1 = ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not str(ROOT.resolve()).startswith('/root/autodl-tmp/'):
        raise RuntimeError('Project root escaped /root/autodl-tmp')
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Refusing reused output identity: ' + str(OUT))
    if not PUBLIC.is_file() or sha(PUBLIC) != '1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b':
        raise RuntimeError('Public SOT checkpoint identity mismatch')
    control, candidate = yaml.safe_load(CONTROL.read_text()), yaml.safe_load(CANDIDATE.read_text())
    candidate_copy = json.loads(json.dumps(candidate))
    candidate_copy['TRAIN']['MSHA_SOURCE_UTILITY_WEIGHT'] = control['TRAIN']['MSHA_SOURCE_UTILITY_WEIGHT']
    if candidate_copy != control:
        raise RuntimeError('Configs differ outside the utility-loss coefficient')
    for config in (control, candidate):
        if config['MODEL']['INIT_CHECKPOINT'] != '' or config['MODEL']['PRETRAIN_FILE'] != PUBLIC.name:
            raise RuntimeError('Invalid initialization contract')
        if config['MODEL']['BACKBONE']['EARLY_GATE_SEARCH_CONTEXT_MODE'] != 'cross':
            raise RuntimeError('Both arms must use C search context')
        if not config['TRAIN']['MSHA_SOURCE_UTILITY_ENABLE'] or config['TRAIN']['MSHA_SOURCE_UTILITY_MODE'] != 'stable_region_pair':
            raise RuntimeError('Both arms must use identical stable utility taps')
        if (config['TRAIN']['NUM_WORKER'], config['TRAIN']['PREFETCH_FACTOR'],
                config['TRAIN']['PERSISTENT_WORKERS']) != (4, 2, False):
            raise RuntimeError('Training worker contract changed')
    manifest = json.loads((P1 / 'data_manifest.json').read_text())
    sequence_names = (P1 / 'screen96_max2000.txt').read_text().splitlines()
    if len(manifest['train_sequences']) != 977 or len(sequence_names) != 96:
        raise RuntimeError('Frozen train/evaluation manifests do not match the contract')
    free = shutil.disk_usage('/root/autodl-tmp').free
    if free < 4 * 1024**3:
        raise RuntimeError('Insufficient project-disk headroom before start: %d bytes' % free)
    record = {
        'run_id': RUN_ID,
        'run_root': str(OUT.resolve()),
        'project_root': str(ROOT.resolve()),
        'initialization': {'path': str(PUBLIC.resolve()), 'sha256': sha(PUBLIC), 'stage': 'public SOT single-stage'},
        'seed': 42, 'updates': 468, 'checkpoint_milestones': [468],
        'milestone_deviation': 'Only u0468 retained because four full optimizer checkpoints (u0234/u0468 x two arms) exceed available project-disk space; the fixed primary quick endpoint remains u0468.',
        'train_manifest': {'path': str((P1 / 'data_manifest.json').resolve()), 'sha256': sha(P1 / 'data_manifest.json'), 'sequences': 977},
        'screen_manifest': {'path': str((P1 / 'screen96_max2000.txt').resolve()), 'sha256': sha(P1 / 'screen96_max2000.txt'), 'sequences': 96},
        'configs': {str(CONTROL): sha(CONTROL), str(CANDIDATE): sha(CANDIDATE)},
        'stable_utility': 'pairwise source-gain ranking only for token pairs ordered consistently by center-peak and target-box-peak Hann margins',
        'code': {str(ROOT / 'lib/models/layers/msha_source_utility_loss.py'): sha(ROOT / 'lib/models/layers/msha_source_utility_loss.py'),
                 str(ROOT / 'lib/train/actors/tbsi_track.py'): sha(ROOT / 'lib/train/actors/tbsi_track.py'),
                 str(ROOT / 'tools/run_early_gate_p1_v2.py'): sha(ROOT / 'tools/run_early_gate_p1_v2.py')},
        'free_project_bytes_before_start': free,
        'autodl_fs_used': False,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    manifest_name = 'run_manifest.json' if RUN_ID == 'EG_C_STABLE_UTILITY_QUICK_S42_20261101' else 'run_manifest_retry1.json'
    (RUN / manifest_name).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'valid': True, 'free_project_bytes': free,
                      'train_sequences': 977, 'screen_sequences': 96,
                      'updates_per_arm': 468, 'only_config_delta': 'TRAIN.MSHA_SOURCE_UTILITY_WEIGHT'}, indent=2))


if __name__ == '__main__':
    main()
