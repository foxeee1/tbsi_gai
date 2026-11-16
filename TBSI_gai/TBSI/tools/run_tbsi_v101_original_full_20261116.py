#!/usr/bin/env python3
"""Run the original v1.0.1 TBSI baseline under the candidate's exact protocol.

This run is intentionally independent from the MR candidate. It starts from
the public SOT checkpoint, uses the same paired LasHeR manifest and seed, and
forces the same update-based scheduler milestone (u936). The v1.0.1 YAML is
copied unchanged into the isolated workspace used by the tracker.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/experiments/TBSI-V101-ORIGINAL-FULL15-FULL244-20261116'
SOURCE = ROOT / 'output/experiments/MSHA-B0-U936-S42-FULL15-FULL244-20261109/workspace'
CONFIG_SOURCE = ROOT / 'experiments/主实验/配置/v1.0.1-单卡4090D-24G-等效全局128加速.yaml'
CONFIG_NAME = 'TBSI-V101-ORIGINAL-FULL15-FULL244-20261116.yaml'
PUBLIC = ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'
PUBLIC_SHA = '1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b'
MANIFEST = ROOT / 'output/experiments/MR-G1-FGLOSS-U936-S42-FULL15-FULL244-20261116/candidate/workspace/experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json'
TEST_LIST = MANIFEST.parent / 'test_sequences.txt'
TRAIN_LIST = MANIFEST.parent / 'train_sequences.txt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def project_boundary(path):
    p = Path(path).resolve()
    root = Path('/root/autodl-tmp').resolve()
    if p != root and root not in p.parents:
        raise RuntimeError('Execution/output escaped project disk: ' + str(p))
    return p


def validate_inputs():
    project_boundary(ROOT)
    if sha(PUBLIC) != PUBLIC_SHA:
        raise RuntimeError('Public checkpoint hash changed')
    if sha(CONFIG_SOURCE) != '6f8ae936e3ab724a1316132d5cb92f68bda09f86065f9efba3b7e1e079267d68':
        raise RuntimeError('v1.0.1 configuration changed')
    names = json.loads(MANIFEST.read_text())
    if len(names['train_sequences']) != 977 or len(names['test_sequences']) != 244:
        raise RuntimeError('Expected train977/test244 manifest')
    return {'manifest': sha(MANIFEST), 'train_list': sha(TRAIN_LIST),
            'test_list': sha(TEST_LIST), 'config': sha(CONFIG_SOURCE),
            'public': PUBLIC_SHA}


def sync_asset(status, conclusion, **updates):
    identity = 'TBSI-V101-ORIGINAL-FULL15-FULL244-20261116'
    csvpath = ROOT / 'experiments/experiment_asset_registry.csv'
    mdpath = ROOT / 'experiments/实验资产数据库.md'
    fields = None
    with (RUN / 'registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with csvpath.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        prior = next((r for r in rows if r['ID'] == identity), None)
        row = dict(prior) if prior else {key: 'NA' for key in fields}
        row.update({
            'ID': identity, 'Date': '20261116',
            'Parent': 'v1.0.1-单卡4090D-24G-等效全局128加速',
            'CommitOrSnapshot': str(RUN / 'workspace_manifest.json'),
            'Config': str(RUN / 'workspace/experiments/tbsi_track' / CONFIG_NAME) + ' SHA256 ' + sha(RUN / 'workspace/experiments/tbsi_track' / CONFIG_NAME),
            'Seed': '42',
            'Dataset': 'LasHeR train977/test244; paired manifest shared with MR candidate',
            'EvalScope': '7020 successful updates; u936 decay; epoch15 checkpoint; full244/219292 frames FP32 threads0 template-update off',
            'AssumptionOrHypothesis': 'Original v1.0.1 TBSI baseline under the exact candidate train/test protocol.',
            'Conclusion': conclusion, 'Status': status,
            'Evidence': str(RUN),
            'Notes': 'Original v1.0.1 YAML copied byte-for-byte; public SOT fresh single-stage; seed42; same paired sampler/manifest, B16 accum8 AMP workers4; scheduler forced to Mstep milestone u936 for protocol parity. This is the sole newly trained baseline comparator; historical strongest MR remains the primary reference.'})
        if 'environment' in updates:
            env = updates.pop('environment')
            row['GPU'] = str(env['gpu_count']) + 'x ' + env['gpu']
            row['HardwareConfig'] = 'Python ' + env['python'].split()[0] + '/PyTorch ' + env['pytorch'] + '/CUDA ' + str(env['cuda']) + '; B16 accum8 AMP workers4 prefetch2 persistentfalse; fresh AdamW; u936'
        row.update({key: str(value) for key, value in updates.items() if key in fields})
        rows = [r for r in rows if r['ID'] != identity] + [row]
        temporary = csvpath.with_suffix('.tbsi-v101.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(csvpath)
        begin, end = '<!-- ' + identity + ' BEGIN -->', '<!-- ' + identity + ' END -->'
        block = begin + '\n### ' + identity + '\n\n' + ''.join('- **' + key + '**：' + row[key] + '\n' for key in fields) + end
        text = mdpath.read_text(encoding='utf-8')
        if begin in text:
            a = text.index(begin); b = text.index(end, a) + len(end)
            text = text[:a] + block + text[b:]
        else:
            text += '\n\n' + block + '\n'
        mdtmp = mdpath.with_suffix('.tbsi-v101.tmp')
        mdtmp.write_text(text, encoding='utf-8'); mdtmp.replace(mdpath)
        write_json(RUN / 'asset_record.json', row)


def prepare():
    pins = validate_inputs()
    RUN.mkdir(parents=True, exist_ok=True)
    work = RUN / 'workspace'
    if not work.exists():
        shutil.copytree(SOURCE, work, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        config_dir = work / 'experiments/tbsi_track'
        config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CONFIG_SOURCE, config_dir / CONFIG_NAME)
    config = work / 'experiments/tbsi_track' / CONFIG_NAME
    if sha(config) != pins['config']:
        raise RuntimeError('Copied v1.0.1 config differs')
    manifest = {str(p.relative_to(work)): sha(p) for p in sorted(work.rglob('*'))
                if p.is_file() and not p.is_symlink()}
    write_json(RUN / 'workspace_manifest.json', manifest)
    write_json(RUN / 'identity.json', {'run': str(RUN), 'pins': pins, 'config_name': CONFIG_NAME,
                                       'seed': 42, 'protocol': 'train977/15epochs/7020u; Full244 FP32 threads0'})
    if not (RUN / 'status.json').exists():
        write_json(RUN / 'status.json', {'status': 'PREPARED / NOT-STARTED'})
        sync_asset('PREPARED / NOT-STARTED', 'Independent v1.0.1 baseline prepared; no training/test yet.')


def resource_guard():
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if gpu:
        raise RuntimeError('CUDA process active: ' + gpu.replace('\n', ','))
    me = str(os.getpid())
    for line in subprocess.check_output(['ps', '-eo', 'pid=,pcpu=,rss=,args='], text=True).splitlines():
        vals = line.split(None, 3)
        if len(vals) < 4 or vals[0] == me:
            continue
        cmd = vals[3]
        if 'run_mr_source_foreground_full_20261116' in cmd or 'run_tbsi_v101_original_full_20261116' in cmd:
            raise RuntimeError('Another paired experiment is active: ' + line.strip())


def load_runner():
    work = RUN / 'workspace'
    sys.path.insert(0, str(work))
    spec = importlib.util.spec_from_file_location('tbsi_v101_runner', work / 'tools/run_dual_message_p1.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    return runner, work


def recorder(output, config, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    sync_asset(status, conclusion, Checkpoint=checkpoint, GPUHours=gpu_hours)


def run_train():
    resource_guard()
    runner, work = load_runner()
    import lib.train.base_functions as base_functions
    original = base_functions.get_optimizer_scheduler
    def fixed_scheduler(net, cfg):
        cfg.TRAIN.SCHEDULER.TYPE = 'Mstep'
        cfg.TRAIN.SCHEDULER.MILESTONES = [2]
        cfg.TRAIN.SCHEDULER.GAMMA = 0.1
        return original(net, cfg)
    base_functions.get_optimizer_scheduler = fixed_scheduler
    output = RUN / 'train'
    config = work / 'experiments/tbsi_track' / CONFIG_NAME
    args = SimpleNamespace(config=str(config), output=str(output), seed=42, arm='original_v101',
                           manifest=str(MANIFEST), updates=7020, milestones='7020',
                           diagnostic_only=False, shared_init_snapshot=None, export_shared_init=None)
    write_json(RUN / 'status.json', {'status': 'RUNNING / FULL15-TRAIN'})
    try:
        runner.train(args, recorder=recorder)
        done = json.loads((output / 'completed.json').read_text())
        if done['successful_updates'] != 7020 or done['samples'] != 898560 or done['unique_sequences'] != 977:
            raise RuntimeError('Training contract incomplete')
        write_json(RUN / 'checkpoint_identity.json', {'path': str(output / 'checkpoint_u7020.pth.tar'),
                   'sha256': sha(output / 'checkpoint_u7020.pth.tar'), 'successful_updates': 7020})
        write_json(RUN / 'status.json', {'status': 'TRAINING-COMPLETE / FULL244-PENDING'})
    except Exception as exc:
        write_json(RUN / 'failure_train.json', {'type': type(exc).__name__, 'error': str(exc), 'time': time.time()})
        sync_asset('EXECUTION-FAILED / NO-NEW-EFFICACY-CLAIM', 'Training failed: ' + type(exc).__name__ + ': ' + str(exc))
        raise


def run_eval():
    resource_guard()
    runner, work = load_runner()
    output = RUN / 'full244'
    config = work / 'experiments/tbsi_track' / CONFIG_NAME
    checkpoint = RUN / 'train/checkpoint_u7020.pth.tar'
    identity = json.loads((RUN / 'checkpoint_identity.json').read_text())
    if sha(checkpoint) != identity['sha256']:
        raise RuntimeError('Explicit checkpoint identity changed')
    write_json(RUN / 'status.json', {'status': 'RUNNING / FULL244-EVALUATION'})
    sync_asset('RUNNING / FULL244-EVALUATION', 'Explicit u7020 checkpoint, FP32 full244.')
    args = SimpleNamespace(config=str(config), output=str(output), seed=42, checkpoint=str(checkpoint), list=str(TEST_LIST))
    try:
        runner.evaluate(args)
        done = json.loads((output / 'completed.json').read_text())
        if done['sequences'] != 244 or done['frames'] != 219292:
            raise RuntimeError('Full244 evaluation incomplete')
        write_json(RUN / 'status.json', {'status': 'FULL15-FULL244-COMPLETE'})
        sync_asset('FULL15-FULL244-COMPLETE / ANALYSIS-PENDING', 'Training and Full244 complete; paired analysis pending.',
                   FPS=done['throughput_fps'], Checkpoint=str(checkpoint))
    except Exception as exc:
        write_json(RUN / 'failure_evaluate.json', {'type': type(exc).__name__, 'error': str(exc), 'time': time.time()})
        sync_asset('EXECUTION-FAILED / NO-NEW-EFFICACY-CLAIM', 'Evaluation failed: ' + type(exc).__name__ + ': ' + str(exc))
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['prepare', 'train', 'evaluate', 'all'], default='prepare')
    args = p.parse_args(); prepare()
    if args.stage in ('train', 'all'): run_train()
    if args.stage in ('evaluate', 'all'): run_eval()


if __name__ == '__main__':
    main()
