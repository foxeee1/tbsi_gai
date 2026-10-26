"""Explicit, one-phase-at-a-time runner for the STSER content pair."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/root/autodl-tmp/conda_envs/tbsi/bin/python')
PREFIX = 'M1_STSER_CONTENT'
SOT = ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'
TRAIN_LIST = ROOT / 'experiments/protocols/m1_screening_v1_20261022/quick_train_sequences.txt'
TEST_LIST = ROOT / 'output/experiments/M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024/rpp_v2_60_manifest.txt'
FULL_TRAIN_LIST = ROOT / 'data/lasher/trainingsetList.txt'
FULL_VAL_LIST = ROOT / 'data/lasher/testingsetList.txt'
FULL_TEST_LIST = ROOT.parent / 'data/lasher/testingsetList.txt'
DATA_ROOTS = {'train': ROOT / 'data/lasher/trainingset',
              'validation': ROOT / 'data/lasher/testingset',
              'evaluation': ROOT.parent / 'data/lasher/testingset'}
M1_CONFIG = ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'
M1_WEIGHT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
PROTECTED = {M1_CONFIG: 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e',
             M1_WEIGHT: '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'}
SOURCE = ('lib/models/layers/search_conditioned_t2s_router.py',
          'lib/models/layers/tbsi_layer.py', 'lib/models/tbsi_track/vit_tbsi_care.py',
          'lib/models/tbsi_track/tbsi_track.py', 'lib/train/actors/tbsi_track.py',
          'lib/train/trainers/diagnostics.py', 'lib/train/trainers/ltr_trainer.py',
          'lib/config/tbsi_track/config.py', 'lib/test/tracker/tbsi_track.py',
          'lib/test/parameter/tbsi_track.py', 'tracking/test.py',
          'tools/prepare_m1_stser_content_pair_20261025.py',
          'tools/preflight_m1_stser_content_20261025.py',
          'tools/run_m1_stser_content_campaign_20261025.py',
          'tools/analyze_m1_stser_content_pair_20261025.py',
          'tools/analyze_m1_stser_content_probes_20261025.py',
          'tools/audit_m1_stser_content_checkpoint_20261025.py',
          'tools/probe_m1_stser_content_fixed_state_20261025.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def process_gate():
    # Worker-4 training has five actual processes and must run alone.
    output = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    patterns = ('lib/train/run_training.py', 'tracking/test.py',
                'tracking/analysis_results.py', 'tools/preflight_m1_stser_content_',
                'tools/probe_m1_stser_content_fixed_state_')
    active = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or int(parts[0]) == os.getpid():
            continue
        executable = Path(parts[1]).name
        if executable.startswith('python') and any(p in parts[2] for p in patterns):
            active.append(line)
    require(not active, 'resource gate: existing experiment process:\n' + '\n'.join(active))
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True)
    require(gpu.returncode == 0, 'cannot inspect existing GPU processes')
    gpu_pids = {int(value.strip()) for value in gpu.stdout.splitlines() if value.strip().isdigit()}
    require(not (gpu_pids - {os.getpid()}),
            'resource gate: GPU is already occupied by {}'.format(sorted(gpu_pids)))


def environment(scope):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('TBSI_'):
            env.pop(key)
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg',
                'TBSI_INFER_FP16': '0', 'TBSI_TEMPLATE_UPDATE_MODE': 'off',
                'TBSI_DISABLE_TRACKER_CACHE': '1'})
    if scope == 'quick3':
        env['TBSI_RPP_TRAIN_LIST'] = str(TRAIN_LIST)
        env['TBSI_RPP_TEST_LIST'] = str(TEST_LIST)
    else:
        env.pop('TBSI_RPP_TRAIN_LIST', None)
        env.pop('TBSI_RPP_TEST_LIST', None)
    return env


def contracts(scope):
    require(PYTHON.is_file(), 'Python environment missing')
    require(all(path.is_dir() for path in DATA_ROOTS.values()),
            'LasHeR image root is missing')
    for path, expected in PROTECTED.items():
        require(path.is_file() and sha(path) == expected,
                'protected M1 asset missing or changed: ' + str(path))
    require(SOT.is_file() and sha(SOT) ==
            '1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b',
            'SOT checkpoint hash mismatch')
    names = {arm: '{}_{}_{}_20261025'.format(PREFIX, scope.upper(), arm.upper())
             for arm in ('control', 'candidate')}
    paths = {arm: ROOT / 'experiments/tbsi_track/{}.yaml'.format(name)
             for arm, name in names.items()}
    configs = {arm: yaml.safe_load(path.read_text()) for arm, path in paths.items()}
    plain = copy.deepcopy(configs['candidate'])
    strength = plain['MODEL']['BACKBONE']['SEARCH_CONDITIONED_T2S_CONTENT_STRENGTH']
    plain['MODEL']['BACKBONE']['SEARCH_CONDITIONED_T2S_CONTENT_STRENGTH'] = 0.0
    require(plain == configs['control'] and strength == 1.0,
            'paired configs must differ only in content strength 0/1')
    cfg = configs['control']
    expected_epoch = 3 if scope == 'quick3' else 15
    require(cfg['TRAIN']['EPOCH'] == expected_epoch and cfg['TEST']['EPOCH'] == expected_epoch,
            'checkpoint epoch mismatch')
    require(cfg['TRAIN']['NUM_WORKER'] == 4 and cfg['TRAIN']['BATCH_SIZE'] == 16 and
            cfg['TRAIN']['GRAD_ACCUM_STEPS'] == 8 and cfg['TRAIN']['MATCHED_SEED'] == 42 and
            cfg['TRAIN']['AMP'] is True, 'M1 training contract changed')
    require(cfg['MODEL']['INIT_CHECKPOINT_STRICT'] is True and
            cfg['MODEL']['INIT_CHECKPOINT_SOT_PROVENANCE'] is True and
            cfg['MODEL']['BACKBONE']['SEARCH_CONDITIONED_T2S_MODE'] ==
            'stser_competition_content', 'paired initialization contract changed')
    init = ROOT / cfg['MODEL']['INIT_CHECKPOINT']
    require(init.resolve() == (ROOT / 'experiments/snapshots/m1_stser_content_pair_{}_20261025/sot_initialized_shared_net.pth.tar'.format(scope)).resolve(),
            'paired initialization path changed')
    if scope == 'quick3':
        require(TRAIN_LIST.is_file() and TEST_LIST.is_file() and
                len(TRAIN_LIST.read_text().splitlines()) == 176 and
                len(TEST_LIST.read_text().splitlines()) == 60 and
                sha(TEST_LIST) == 'ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8',
                'quick manifest missing or altered')
    else:
        require(all(path.is_file() for path in (FULL_TRAIN_LIST, FULL_VAL_LIST, FULL_TEST_LIST)),
                'full LasHeR train/validation/evaluation lists are missing')
    require(shutil.disk_usage(ROOT).free >= 5_000_000_000, 'less than 5 GB free')
    return names, paths, init


def command(argv, log, env):
    require(not log.exists(), 'refusing to overwrite log: ' + str(log))
    with log.open('x') as handle:
        handle.write('command=' + json.dumps([str(x) for x in argv]) + '\n')
        handle.flush()
        subprocess.run([str(x) for x in argv], cwd=str(ROOT), env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('quick3', 'full15'), required=True)
    parser.add_argument('--phase', choices=('prepare', 'preflight', 'train-control',
                                           'train-candidate', 'test-control',
                                           'test-candidate', 'resume-test-control',
                                           'resume-test-candidate', 'analyze', 'probe-off',
                                           'retry-train-control', 'retry-train-candidate',
                                           'retry2-train-control', 'retry2-train-candidate',
                                           'retry3-train-control', 'retry3-train-candidate',
                                           'probe-roll-mask', 'probe-roll-competition',
                                           'probe-scale-only', 'analyze-probes'), required=True)
    args = parser.parse_args()
    scope, phase = args.scope, args.phase
    names, configs, init = contracts(scope)
    campaign = ROOT / 'experiments/snapshots/m1_stser_content_campaign_{}_20261025'.format(scope)
    lock = campaign / 'contract.json'
    if phase == 'prepare':
        require(not campaign.exists() and not init.parent.exists(),
                'campaign or paired initialization already exists')
        for name in names.values():
            require(not (ROOT / 'output/experiments' / name).exists(),
                    'run output already exists: ' + name)
        process_gate()
        campaign.mkdir(parents=True)
        try:
            command([PYTHON, 'tools/prepare_m1_stser_content_pair_20261025.py',
                     '--scope', scope], campaign / 'prepare.log', environment(scope))
            manifest = json.loads((init.parent / 'initialization_manifest.json').read_text())
            require(sha(init) == manifest['shared_initial_sha256'], 'initial state hash mismatch')
            locked = {'scope': scope, 'prepared_utc': now(),
                      'shared_initial_sha256': sha(init),
                      'config_sha256': {arm: sha(path) for arm, path in configs.items()},
                      'source_sha256': {path: sha(ROOT / path) for path in SOURCE},
                      'sot_sha256': sha(SOT),
                      'train_manifest': str(TRAIN_LIST) if scope == 'quick3' else 'LasHeR_train full',
                      'test_manifest': str(TEST_LIST) if scope == 'quick3' else 'LasHeR_test full',
                      'quick_train_manifest_sha256': sha(TRAIN_LIST) if scope == 'quick3' else None,
                      'quick_test_manifest_sha256': sha(TEST_LIST) if scope == 'quick3' else None,
                      'full_list_sources': {label: {'path': str(path), 'sha256': sha(path)}
                                            for label, path in (('train', FULL_TRAIN_LIST),
                                                                ('validation', FULL_VAL_LIST),
                                                                ('evaluation', FULL_TEST_LIST))}
                      if scope == 'full15' else None,
                      'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
                      'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=str(ROOT), text=True),
                      'python': str(PYTHON), 'worker_process_contract': 'one main plus four workers, alone'}
            locked['dataset'] = {'name': 'LasHeR', 'version': 'local testingset/trainingset snapshot',
                                 'roots': {label: str(path.resolve()) for label, path in DATA_ROOTS.items()}}
            if scope == 'full15':
                skip = {'advancedredcup', 'cameraman_1202', 'mirroratleft'}
                for label, source in (('train', FULL_TRAIN_LIST), ('validation', FULL_VAL_LIST),
                                      ('evaluation', FULL_TEST_LIST)):
                    members = [line.strip().split(',')[0] for line in source.read_text().splitlines()
                               if line.strip() and line.strip().split(',')[0] not in skip]
                    resolved = campaign / ('resolved_{}_sequences.txt'.format(label))
                    resolved.write_text('\n'.join(members) + '\n')
                    locked['full_list_sources'][label]['resolved_path'] = str(resolved)
                    locked['full_list_sources'][label]['resolved_sha256'] = sha(resolved)
            lock.write_text(json.dumps(locked, indent=2) + '\n')
        except Exception:
            # Preserve the failed preparation as evidence; never reuse it.
            raise
        print(lock)
        return

    require(lock.is_file() and init.is_file(), 'run prepare first')
    locked = json.loads(lock.read_text())
    manifest = json.loads((init.parent / 'initialization_manifest.json').read_text())
    require(locked['scope'] == scope and sha(init) == locked['shared_initial_sha256'] ==
            manifest['shared_initial_sha256'], 'paired initialization changed')
    for arm, path in configs.items():
        require(sha(path) == locked['config_sha256'][arm] == manifest['configs'][names[arm]],
                'config changed after preparation: ' + arm)
    for path, expected in locked['source_sha256'].items():
        # Model and data-contract sources are immutable after preparation.
        # Runner/report utilities may be repaired after a failed phase while
        # preserving the same initialized model pair.
        if path.startswith('lib/') or path == 'tracking/test.py':
            require(sha(ROOT / path) == expected, 'model source changed after preparation: ' + path)
    if scope == 'quick3':
        require(sha(TRAIN_LIST) == locked['quick_train_manifest_sha256'] and
                sha(TEST_LIST) == locked['quick_test_manifest_sha256'], 'manifest changed')
    else:
        for source in locked['full_list_sources'].values():
            require(sha(source['path']) == source['sha256'] and
                    sha(source['resolved_path']) == source['resolved_sha256'],
                    'full data member list changed')
    env = environment(scope)
    if phase == 'preflight':
        process_gate()
        preflight_log = campaign / 'preflight.log'
        if preflight_log.exists() and 'operator_and_full_model_preflight' not in preflight_log.read_text():
            preflight_log = campaign / 'preflight_retry1.log'
        command([PYTHON, 'tools/preflight_m1_stser_content_20261025.py',
                 '--device', 'cuda', '--scope', scope], preflight_log, env)
        return
    if phase == 'analyze':
        process_gate()
        command([PYTHON, 'tools/analyze_m1_stser_content_pair_20261025.py',
                 '--scope', scope], campaign / 'analyze.log', env)
        return
    if phase == 'analyze-probes':
        process_gate()
        command([PYTHON, 'tools/analyze_m1_stser_content_probes_20261025.py',
                 '--scope', scope], campaign / 'analyze-probes.log', env)
        return
    process_gate()
    probe = phase.startswith('probe-')
    resume = phase.startswith('resume-test-')
    retry_train = (phase.startswith('retry-train-') or phase.startswith('retry2-train-') or
                   phase.startswith('retry3-train-'))
    action, arm = ('test', 'candidate') if probe else (
        ('train', phase.split('-')[-1]) if retry_train else
        ('test', phase.split('-')[-1]) if resume else phase.split('-'))
    name = names[arm]
    run = ROOT / 'output/experiments' / name
    epoch = 3 if scope == 'quick3' else 15
    checkpoint = run / 'checkpoints/TBSITrack_ep{:04d}.pth.tar'.format(epoch)
    phase_log = campaign / (phase + '.json')
    require(not phase_log.exists(), 'phase already started: ' + phase)
    if action == 'train':
        preflight_logs = list(campaign.glob('preflight*.log'))
        require(any('operator_and_full_model_preflight' in path.read_text()
                    for path in preflight_logs),
                'operator preflight has not completed')
        if retry_train:
            previous = campaign / ('train-' + arm + '.json')
            require(previous.is_file() and run.is_dir() and not checkpoint.is_file(),
                    'retry requires the interrupted run without a checkpoint')
            env['TBSI_NUM_WORKER_OVERRIDE'] = '1'
            train_log_name = ('train_retry3.log' if phase.startswith('retry3-') else
                              'train_retry2.log' if phase.startswith('retry2-') else
                              'train_retry1.log')
            if phase.startswith('retry3-'):
                env['TBSI_IMAGE_LOADER'] = 'jpeg4py'
        else:
            require(not run.exists(), 'refusing to overwrite run directory: ' + str(run))
            run.mkdir(parents=True)
            (run / 'logs').mkdir()
            train_log_name = 'train.log'
        argv = [PYTHON, '-u', 'lib/train/run_training.py', '--script', 'tbsi_track',
                '--config', name, '--save_dir', 'output', '--seed', '42', '--use_lmdb', '0']
        log = run / 'logs/{}'.format(train_log_name)
    else:
        require(checkpoint.is_file(), 'explicit checkpoint missing: ' + str(checkpoint))
        probe_mode = phase[len('probe-'):].replace('-', '_') if probe else 'correct'
        runid = 'content_{}'.format(probe_mode) if probe else None
        results = ROOT / 'output/test/tracking_results/tbsi_track' / (
            name + '_' + runid if runid else name)
        if resume:
            original = campaign / ('test-' + arm + '.json')
            require(original.is_file() and results.is_dir(),
                    'resume requires an interrupted original test with partial predictions')
            original_status = json.loads(original.read_text())
            require(original_status['status'] != 'complete' and
                    original_status['checkpoint_sha256_before'] == sha(checkpoint),
                    'completed test or checkpoint mismatch; refusing resume')
        else:
            require(not results.exists(), 'refusing to overwrite predictions: ' + str(results))
        env['TBSI_CHECKPOINT_OVERRIDE'] = str(checkpoint)
        env['TBSI_STSER_CONTENT_PROBE'] = probe_mode
        argv = [PYTHON, '-u', 'tracking/test.py', 'tbsi_track', name,
                '--dataset_name', 'rpp_lasher_test' if scope == 'quick3' else 'lasher_test',
                '--threads', '0', '--num_gpus', '1']
        if runid:
            argv.extend(['--runid', runid])
        log = run / 'logs/{}.log'.format(phase)
    record = {'phase': phase, 'start_utc': now(), 'argv': [str(x) for x in argv],
              'config_sha256': sha(configs[arm]), 'initial_sha256': sha(init),
              'checkpoint_sha256_before': sha(checkpoint) if action == 'test' else None,
              'environment': {key: env.get(key) for key in ('TBSI_RPP_TRAIN_LIST',
                  'TBSI_RPP_TEST_LIST', 'TBSI_INFER_FP16', 'TBSI_TEMPLATE_UPDATE_MODE',
                  'TBSI_CHECKPOINT_OVERRIDE', 'TBSI_STSER_CONTENT_PROBE',
                  'OMP_NUM_THREADS')},
              'gpu': subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout,
              'python_runtime': subprocess.run([str(PYTHON), '-c',
                  'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda)'],
                  capture_output=True, text=True).stdout}
    phase_log.write_text(json.dumps(record, indent=2) + '\n')
    try:
        command(argv, log, env)
        require(checkpoint.is_file(), 'specified checkpoint was not produced')
        if action == 'train':
            command([PYTHON, 'tools/audit_m1_stser_content_checkpoint_20261025.py',
                     '--checkpoint', checkpoint, '--output', campaign / (phase + '_checkpoint.json'),
                     '--expected-epoch', str(epoch)],
                    campaign / (phase + '_checkpoint.log'), env)
        if action == 'test':
            require(results.is_dir() and any(results.glob('*.txt')),
                    'no tracking predictions were produced')
        record.update({'status': 'complete', 'end_utc': now(),
                       'checkpoint_sha256_after': sha(checkpoint)})
    except Exception as error:
        record.update({'status': 'failed', 'end_utc': now(), 'error': str(error)})
        raise
    finally:
        phase_log.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
