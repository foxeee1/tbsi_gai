#!/usr/bin/env python3
"""Prepare/run an isolated exact-MR pair; only foreground-loss weight differs.

Default action prepares files without training. No tracker checkpoint is used
for initialization. Each resource-intensive stage runs serially in a fresh
process using a copy of the strongest MR's archived code, not today's model.
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
FAMILY = 'MR-G1-FGLOSS-U936-S42-FULL15-FULL244-20261116'
RUN = ROOT / 'output/experiments' / FAMILY
PLAN = ROOT / 'experiments/mr_source_foreground_full_20261116'
PARENT_ID = 'MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109'
PARENT = ROOT / 'output/experiments' / PARENT_ID
B0 = ROOT / 'output/experiments/MSHA-B0-U936-S42-FULL15-FULL244-20261109'
SOURCE = PARENT / 'workspace'
PUBLIC = ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'
PUBLIC_SHA = '1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b'
LOSS_SOURCE = ROOT / 'lib/models/layers/mr_source_foreground_loss.py'
CONFIGS = {arm: ROOT / 'experiments/tbsi_track' /
           ('MR-G1-FGLOSS-U936-S42-FULL15-' + arm.upper() + '-20261116.yaml')
           for arm in ('control', 'candidate')}
WEIGHTS = {'control': 0., 'candidate': 0.05}
ENVIRONMENT = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def project_boundary(path):
    real = Path(path).resolve()
    if Path('/root/autodl-tmp') not in real.parents and real != Path('/root/autodl-tmp'):
        raise RuntimeError('Project execution/output must stay under /root/autodl-tmp: ' + str(real))
    return real


def validate_inputs():
    import yaml
    project_boundary(__file__)
    project_boundary(Path.cwd())
    project_boundary(RUN)
    pins = json.loads((PLAN / 'source_pins.json').read_text())
    for relative, expected in pins['sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('Pinned input changed: ' + relative)
    if sha(PUBLIC) != PUBLIC_SHA:
        raise RuntimeError('Wrong public SOT checkpoint')
    parent = yaml.safe_load((PARENT / 'contracts/mr_only_g1.yaml').read_text())
    for arm, config in CONFIGS.items():
        current = yaml.safe_load(config.read_text())
        weight = current['TRAIN'].pop('MR_SOURCE_FOREGROUND_WEIGHT')
        threshold = current['TRAIN'].pop('MR_SOURCE_FOREGROUND_THRESHOLD')
        if weight != WEIGHTS[arm] or threshold != 0.5 or current != parent:
            raise RuntimeError('Only the declared loss settings may differ from historical MR')
    contract = json.loads((PARENT / 'run_contract.json').read_text())
    manifest = SOURCE / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json'
    checks = [(manifest, 'train_manifest_sha256'),
              (manifest.parent / 'train_sequences.txt', 'train_list_sha256'),
              (manifest.parent / 'test_sequences.txt', 'test_list_sha256')]
    for path, key in checks:
        if sha(path) != contract[key]:
            raise RuntimeError('Historical data definition mismatch: ' + str(path))
    names = json.loads(manifest.read_text())
    if len(names['train_sequences']) != 977 or len(names['test_sequences']) != 244:
        raise RuntimeError('Expected train977/test244 membership')
    return pins


def sync_asset(arm, status, conclusion, **updates):
    """Update the same unique arm ID in both authoritative tables."""
    identity = FAMILY + '-' + arm.upper()
    env = ENVIRONMENT
    csvpath = ROOT / 'experiments/experiment_asset_registry.csv'
    mdpath = ROOT / 'experiments/实验资产数据库.md'
    with (PLAN / 'registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with csvpath.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        prior = next((r for r in rows if r['ID'] == identity), None)
        row = dict(prior) if prior else {key: 'NA' for key in fields}
        row.update({
            'ID': identity, 'Date': '20261116', 'Parent': PARENT_ID,
            'CommitOrSnapshot': str(PLAN / 'before.json') + '; ' + str(RUN / arm / 'workspace_manifest.json'),
            'Config': str(CONFIGS[arm]) + ' SHA256 ' + sha(CONFIGS[arm]), 'Seed': '42',
            'Dataset': 'LasHeR train977/test244; identical archived manifests; 59904 pairs/epoch x15 =898560 pairs',
            'EvalScope': '7020 successful updates; u936 decay; epoch15 checkpoint only; full244/219292 frames FP32 threads0 template-update off',
            'AssumptionOrHypothesis': 'A capped GT foreground-source constraint can stabilize original MR allocation without changing architecture or inference.',
            'Conclusion': conclusion, 'Status': status,
            'Evidence': str(RUN / arm) + '; ' + str(PLAN / 'PROTOCOL.md'),
            'Notes': 'Control lambda0; candidate lambda0.05; threshold0.5 shared. Only original allocator parameters receive auxiliary gradients via detached-input recomputation. Public SOT fresh single-stage; reconstruct deleted B0 pre-optimizer cache and verify historical tensor hash. No trained tracker initialization; no claim of multi-seed robustness.'})
        if env:
            row['GPU'] = str(env['gpu_count']) + 'x ' + env['gpu']
            row['HardwareConfig'] = ('Python ' + env['python'].split()[0] + '/PyTorch ' + env['pytorch'] +
                '/CUDA ' + str(env['cuda']) + '; B16 accum8 AMP workers4 prefetch2 persistentfalse; fresh AdamW; historical u936')
        row.update({key: str(value) for key, value in updates.items() if key in fields})
        rows = [r for r in rows if r['ID'] != identity] + [row]
        temporary = csvpath.with_suffix('.fg-loss.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csvpath)
        begin, end = '<!-- ' + identity + ' BEGIN -->', '<!-- ' + identity + ' END -->'
        block = begin + '\n### ' + identity + '\n\n' + ''.join(
            '- **' + key + '**：' + row[key] + '\n' for key in fields) + end
        text = mdpath.read_text(encoding='utf-8')
        if begin in text:
            a = text.index(begin)
            b = text.index(end, a) + len(end)
            text = text[:a] + block + text[b:]
        else:
            text += '\n\n' + block + '\n'
        mdtmp = mdpath.with_suffix('.fg-loss.tmp')
        mdtmp.write_text(text, encoding='utf-8')
        mdtmp.replace(mdpath)
        write_json(RUN / arm / 'asset_record.json', row)


def prepare():
    pins = validate_inputs()
    RUN.mkdir(parents=True, exist_ok=True)
    identity = RUN / 'identity.json'
    expected = {'family': FAMILY, 'parent': PARENT_ID, 'pins': pins,
                'loss_weight': WEIGHTS, 'threshold': 0.5, 'seed': 42}
    if identity.exists() and json.loads(identity.read_text()) != expected:
        raise RuntimeError('Refusing to reuse a run with different pinned inputs')
    write_json(identity, expected)
    for arm in ('control', 'candidate'):
        directory = RUN / arm
        work = directory / 'workspace'
        if not work.exists():
            shutil.copytree(SOURCE, work, symlinks=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copyfile(LOSS_SOURCE, work / 'lib/models/layers/mr_source_foreground_loss.py')
            shutil.copyfile(CONFIGS[arm], work / 'experiments/tbsi_track' / CONFIGS[arm].name)
            # Store a complete manifest of actual copied regular code/data-list files.
            manifest = {str(p.relative_to(work)): sha(p) for p in sorted(work.rglob('*'))
                        if p.is_file() and not p.is_symlink()}
            write_json(directory / 'workspace_manifest.json', manifest)
        manifest = json.loads((directory / 'workspace_manifest.json').read_text())
        for relative, expected_hash in manifest.items():
            if sha(work / relative) != expected_hash:
                raise RuntimeError('Copied workspace changed: ' + relative)
        for link in work.rglob('*'):
            if link.is_symlink():
                project_boundary(link)
        for relative, expected_hash in pins['sha256'].items():
            source_prefix = str(SOURCE.relative_to(ROOT)) + '/'
            if relative.startswith(source_prefix):
                if sha(work / relative[len(source_prefix):]) != expected_hash:
                    raise RuntimeError('Archived model/runner copy differs')
        if sha(work / 'lib/models/layers/mr_source_foreground_loss.py') != sha(LOSS_SOURCE):
            raise RuntimeError('Loss implementation copy differs')
        if sha(work / 'experiments/tbsi_track' / CONFIGS[arm].name) != sha(CONFIGS[arm]):
            raise RuntimeError('Candidate/control config copy differs')
        if not (directory / 'status.json').exists():
            write_json(directory / 'status.json', {'status': 'PREPARED / NOT-STARTED'})
            sync_asset(arm, 'PREPARED / NOT-STARTED', 'Scripts/configuration prepared; no training, model validation or test result.', GPUHours=0)
    return RUN


def resource_guard():
    """Worker-4 means exclusive train; evaluate uses one actual process."""
    gpu_pids = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    other = [pid.strip() for pid in gpu_pids.splitlines() if pid.strip() and pid.strip() != str(os.getpid())]
    if other:
        raise RuntimeError('Other CUDA processes must finish first: ' + ','.join(other))
    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent > 1:
        ancestors.add(parent)
        try:
            parent = int(Path('/proc/%d/stat' % parent).read_text().split(') ', 1)[1].split()[1])
        except (FileNotFoundError, ValueError, IndexError):
            break
    tokens = ('tracking/train.py', 'tracking/test.py', 'run_training.py', 'analysis_results.py',
              '--internal-stage train', '--internal-stage evaluate', 'run_dual_message_p1.py',
              'run_ablation_full15_full244.py')
    for line in subprocess.check_output(['ps', '-eo', 'pid=,pcpu=,rss=,args='], text=True).splitlines():
        values = line.split(None, 3)
        if len(values) < 4 or int(values[0]) in ancestors:
            continue
        pid, cpu, rss, command = values
        project_python = 'python' in command and ('TBSI' in command or 'tools/' in command)
        busy = project_python and (float(cpu) > 5 or int(rss) > 1024 * 1024)
        if any(token in command for token in tokens) or busy:
            raise RuntimeError('Another resource-intensive experiment is active: ' + line.strip())


def load_runner(arm):
    work = RUN / arm / 'workspace'
    sys.path.insert(0, str(work))
    spec = importlib.util.spec_from_file_location('exact_mr_foreground_runner', work / 'tools/run_dual_message_p1.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import lib.config.tbsi_track.config as config_module
    config_module.cfg.TRAIN.MR_SOURCE_FOREGROUND_WEIGHT = 0.
    config_module.cfg.TRAIN.MR_SOURCE_FOREGROUND_THRESHOLD = 0.5
    import lib.models.tbsi_track as model_package
    if project_boundary(model_package.__file__) != (work / 'lib/models/tbsi_track/__init__.py').resolve():
        raise RuntimeError('Imported the live model instead of archived MR')
    return runner, work


def reconstruct_public_initial_state(runner):
    """Rebuild the deleted cache, never substitute a trained B0 checkpoint."""
    import copy
    import torch
    import lib.models.tbsi_track as model_package
    import lib.config.tbsi_track.config as config_module
    target = RUN / 'pairing/reconstructed_b0_public_initial_state.pt'
    expected = json.loads((PARENT / 'train/initial_state.json').read_text())['paired_common_state_sha256']
    if target.exists():
        snapshot = torch.load(str(target), map_location='cpu')
        if runner.state_hash(snapshot['state']) != expected:
            raise RuntimeError('Reconstructed public initial cache changed')
        return target
    pristine = copy.deepcopy(config_module.cfg)
    config_module.update_config_from_file(str(B0 / 'contracts/b0.yaml'))
    runner.setup(42)
    net = model_package.build_tbsi_track(config_module.cfg, training=True)
    state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    actual = runner.state_hash(state)
    if actual != expected:
        raise RuntimeError('Public B0 reconstruction does not match historical tensor hash: ' + actual)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state': state, 'source_shared_state_sha256': actual,
                'source_identity': {'semantics': 'Reconstructed public SOT pre-optimizer initial state; not a trained tracker',
                                    'public_pretrain_sha256': PUBLIC_SHA, 'seed': 42}}, str(target))
    write_json(target.with_suffix('.json'), {'state_sha256': actual, 'file_sha256': sha(target),
               'historical_tensor_hash_matched': True, 'tensor_count': len(state)})
    config_module.cfg.clear()
    config_module.cfg.update(pristine)
    del net, state
    return target


def install_exact_training(runner, arm, common_initial):
    import torch
    import lib.models.tbsi_track as model_package
    import lib.train.base_functions as base_functions
    import lib.train.actors as actors
    from lib.models.layers.mr_source_foreground_loss import MRSourceForegroundLoss
    original_build = model_package.build_tbsi_track
    expected = json.loads((PARENT / 'train/initial_state.json').read_text())['shared_state_sha256']

    def build(cfg, *args, **kwargs):
        net = original_build(cfg, *args, **kwargs)
        layers = [m for m in net.modules() if getattr(m, 'mass_reallocation_enabled', False)]
        if len(layers) != 3:
            raise RuntimeError('Wrong archived pair MR execution branch')
        for layer in layers:
            with torch.no_grad():
                layer.early_gate_scale.zero_()
            layer.early_gate_scale.requires_grad_(False)
            for p in layer.early_gate.parameters():
                p.requires_grad_(False)
        snapshot = torch.load(str(common_initial), map_location='cpu')
        state = net.state_dict()
        for key, value in snapshot['state'].items():
            if key not in state or state[key].shape != value.shape:
                raise RuntimeError('Public common-state tensor contract changed: ' + key)
            state[key].copy_(value)
        net.load_state_dict(state, strict=True)
        shared = {key: value for key, value in state.items()
                  if '.value_direction.' not in key and '.mass_reallocator_' not in key}
        if runner.state_hash(shared) != expected:
            raise RuntimeError('Shared initial state differs from strongest historical MR')
        full_hash = runner.state_hash(state)
        pair_file = RUN / 'pairing/full_mr_initial_state.json'
        if pair_file.exists() and json.loads(pair_file.read_text())['full_state_sha256'] != full_hash:
            raise RuntimeError('Candidate/control complete MR initial states differ')
        if not pair_file.exists():
            write_json(pair_file, {'full_state_sha256': full_hash, 'historical_shared_state_sha256': expected})
        write_json(RUN / arm / 'exact_initial_contract.json', {
            'full_state_sha256': full_hash, 'historical_shared_state_sha256': expected,
            'gate_scales': [float(m.early_gate_scale) for m in layers],
            'all_gates_frozen': all(not p.requires_grad for m in layers for p in m.early_gate.parameters()),
            'parameter_count': sum(p.numel() for p in net.parameters()),
            'trainable_parameter_count': sum(p.numel() for p in net.parameters() if p.requires_grad),
            'new_model_parameters': 0})
        return net

    model_package.build_tbsi_track = build
    original_optimizer = base_functions.get_optimizer_scheduler

    def optimizer_scheduler(net, cfg):
        cfg.TRAIN.SCHEDULER.MILESTONES = [2]
        cfg.TRAIN.SCHEDULER.GAMMA = 0.1
        return original_optimizer(net, cfg)

    base_functions.get_optimizer_scheduler = optimizer_scheduler
    base_actor = actors.TBSITrackActor

    class ForegroundActor(base_actor):
        def __init__(self, net, objective, loss_weight, settings, cfg=None):
            super().__init__(net, objective, loss_weight, settings, cfg)
            self.fg_weight = float(cfg.TRAIN.MR_SOURCE_FOREGROUND_WEIGHT)
            self.fg_aux = MRSourceForegroundLoss(net, cfg.TRAIN.MR_SOURCE_FOREGROUND_THRESHOLD) if self.fg_weight else None
            self.fg_batches = 0
            self.fg_file = (RUN / arm / 'train/foreground_batches.jsonl').open('w') if self.fg_aux else None

        def __call__(self, data):
            if self.fg_aux is None:
                return super().__call__(data)
            self.fg_aux.begin_batch()
            prediction = self.forward_pass(data)
            task_loss, stats = self.compute_losses(prediction, data['visible'])
            boxes = torch.stack((data['visible']['template_anno'][0].view(-1, 4),
                                 data['infrared']['template_anno'][0].view(-1, 4)), dim=1)
            auxiliary, extra = self.fg_aux(boxes)
            if self.fg_batches == 0:
                readouts = [m.readout[-1].weight for _, m, _, _ in self.fg_aux.records]
                gradients = torch.autograd.grad(auxiliary, readouts, retain_graph=True,
                                                create_graph=False, allow_unused=True)
                write_json(RUN / arm / 'auxiliary_first_batch_gradient.json', {
                    'auxiliary_finite': bool(torch.isfinite(auxiliary)),
                    'readout_gradient_norms': [None if g is None else float(g.detach().float().norm())
                                               for g in gradients],
                    'auxiliary_updates_only_allocator_by_detached_input_construction': True,
                    'new_model_parameters': 0})
            total = task_loss + self.fg_weight * auxiliary
            extra.update({'microbatch_call': self.fg_batches, 'epoch': int(data['epoch']),
                          'Loss/tracking': float(task_loss.detach()),
                          'Loss/mr_source_foreground_weighted': float(auxiliary.detach() * self.fg_weight)})
            self.fg_file.write(json.dumps(extra, allow_nan=False) + '\n')
            self.fg_file.flush()
            self.fg_batches += 1
            stats.update(extra)
            stats['Loss/total'] = float(total.detach())
            return total, stats

    actors.TBSITrackActor = ForegroundActor


def metrics(runner, work, arm):
    import numpy as np
    import torch
    from lib.test.analysis.extract_results import calc_seq_err_robust
    directory = RUN / arm
    evaluation = json.loads((directory / 'full244/completed.json').read_text())
    if evaluation['sequences'] != 244 or evaluation['frames'] != 219292:
        raise RuntimeError('Incomplete full244 evaluation')
    names = (work / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3/test_sequences.txt').read_text().splitlines()
    rows = []
    for name in names:
        prediction = np.loadtxt(directory / 'full244' / (name + '.txt'), ndmin=2, dtype=np.float64)
        target = np.loadtxt(work / 'data/lasher/testingset' / name / 'init.txt', delimiter=',', ndmin=2, dtype=np.float64)
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise RuntimeError('Invalid prediction/GT contract: ' + name)
        overlap, center, normalized, _ = calc_seq_err_robust(torch.tensor(prediction), torch.tensor(target), 'lasher')
        overlap, center, normalized = overlap.numpy(), center.numpy(), normalized.numpy()
        rows.append(dict(sequence=name, frames=len(target),
            auc=float(100 * np.mean([(overlap > t).mean() for t in np.arange(0, 1.00001, .05)])),
            precision=float(100 * (center <= 20).mean()), norm_precision=float(100 * (normalized <= .2).mean()),
            op50=float(100 * (overlap > .5).mean()), op75=float(100 * (overlap > .75).mean())))
    analysis = directory / 'analysis'
    analysis.mkdir(exist_ok=True)
    write_csv(analysis / 'per_sequence_metrics.csv', rows)
    result = {key: float(np.mean([r[key] for r in rows])) for key in ('auc', 'precision', 'norm_precision', 'op50', 'op75')}
    write_json(analysis / 'summary.json', {'metrics': result, 'sequence_count': 244, 'frames': 219292,
               'checkpoint_sha256': sha(directory / 'train/checkpoint_u7020.pth.tar'),
               'metric_definition': 'Archived MR official robust errors, equal-sequence aggregation; NPR normalized distance <=0.20',
               'evaluation': evaluation})
    completed = json.loads((directory / 'train/completed.json').read_text())
    sync_asset(arm, 'FULL15-FULL244-COMPLETE / COMPARISON-PENDING', 'Training and all244 predictions complete; comparison pending.',
               AUC=result['auc'], PR=result['precision'], NPR=result['norm_precision'], OP50=result['op50'], OP75=result['op75'],
               FPS=evaluation['throughput_fps'], GPUHours=completed['gpu_wall_hours'],
               Checkpoint=directory / 'train/checkpoint_u7020.pth.tar')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare():
    import numpy as np
    # No comparison ranking is permitted with different initial state or data.
    a = json.loads((RUN / 'control/exact_initial_contract.json').read_text())
    b = json.loads((RUN / 'candidate/exact_initial_contract.json').read_text())
    if a != b or sha(RUN / 'control/train/batches.jsonl') != sha(RUN / 'candidate/train/batches.jsonl'):
        raise RuntimeError('Pairing confound: repair before attributing differences to loss')
    out = RUN / 'comparison'
    out.mkdir(exist_ok=True)
    def read(path):
        with Path(path).open(newline='') as stream:
            return {r['sequence']: r for r in csv.DictReader(stream)}
    candidate = read(RUN / 'candidate/analysis/per_sequence_metrics.csv')
    references = {'matched_mr': read(RUN / 'control/analysis/per_sequence_metrics.csv'),
                  'historical_best_mr': read(PARENT / 'analysis/per_sequence_metrics.csv'),
                  'historical_b0': read(B0 / 'analysis/per_sequence_metrics.csv')}
    summaries = {}
    for label, reference in references.items():
        if set(candidate) != set(reference) or len(candidate) != 244:
            raise RuntimeError('Comparison membership mismatch')
        rows = []
        for name in sorted(candidate):
            row = {'sequence': name}
            for field in ('auc', 'precision', 'norm_precision', 'op50', 'op75'):
                row['candidate_' + field] = float(candidate[name][field])
                row['reference_' + field] = float(reference[name][field])
                row['delta_' + field] = row['candidate_' + field] - row['reference_' + field]
            row['auc_result_025pp'] = 'win' if row['delta_auc'] > .25 else 'loss' if row['delta_auc'] < -.25 else 'tie'
            rows.append(row)
        write_csv(out / ('per_sequence_vs_' + label + '.csv'), rows)
        delta = np.asarray([r['delta_auc'] for r in rows])
        ordered = np.sort(delta)
        rng = np.random.RandomState(20261116)
        ci = np.percentile(rng.choice(delta, size=(20000, 244), replace=True).mean(1), [2.5, 97.5]).tolist()
        summaries[label] = dict(mean_delta_pp=float(delta.mean()), median_delta_pp=float(np.median(delta)),
            wins=int((delta > .25).sum()), ties=int((abs(delta) <= .25).sum()), losses=int((delta < -.25).sum()),
            worst_decile_mean_pp=float(ordered[:25].mean()), without_top3_mean_pp=float(ordered[:-3].mean()),
            without_top5_mean_pp=float(ordered[:-5].mean()),
            top5_share_of_net=(float(ordered[-5:].sum() / delta.sum()) if abs(delta.sum()) > 1e-12 else None),
            bootstrap95_sequence_only=ci)
    write_json(out / 'summary.json', summaries)
    # Reproduction agreement is reported separately from candidate efficacy.
    control = read(RUN / 'control/analysis/per_sequence_metrics.csv')
    historical = references['historical_best_mr']
    reproduction_delta = np.asarray([float(control[n]['auc']) - float(historical[n]['auc'])
                                     for n in sorted(control)])
    write_json(out / 'control_reproduction.json', {
        'auc_mean_delta_pp': float(reproduction_delta.mean()),
        'max_abs_sequence_auc_delta_pp': float(abs(reproduction_delta).max()),
        'all_prediction_files_identical': all(
            sha(RUN / 'control/full244' / (n + '.txt')) == sha(PARENT / 'full244' / (n + '.txt'))
            for n in control),
        'interpretation': 'If reproduction differs, matched control remains direct loss comparator; disclose differences from historical best.'})
    lines = ['# 固定g=1 MR 来源前景监督全量比较', '',
             '唯一候选变量：训练loss权重0→0.05；推理结构不变。固定seed42，不能作为多seed稳定结论。', '',
             '| 参照 | ΔAUC pp | 中位数 pp | ±0.25pp 胜/平/负 | 最差10%均值 pp | 去前三 pp | 去前五 pp |',
             '|---|---:|---:|---|---:|---:|---:|']
    for label, item in summaries.items():
        lines.append('| %s | %.4f | %.4f | %d/%d/%d | %.4f | %.4f | %.4f |' % (
            label, item['mean_delta_pp'], item['median_delta_pp'], item['wins'], item['ties'], item['losses'],
            item['worst_decile_mean_pp'], item['without_top3_mean_pp'], item['without_top5_mean_pp']))
    lines += ['', '序列bootstrap只描述固定checkpoint的序列不确定性。历史最好MR为附加参照，匹配新MR为loss的直接对照。',
              '所有逐序列五指标见本目录CSV。历史B0并非新重训对照。',
              '前景来源占比提升不自动证明定位改善；需结合保存的foreground_batches读数和尾部/净胜结果解释。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    for arm in ('control', 'candidate'):
        sync_asset(arm, 'FULL15-FULL244-COMPLETE / ANALYZED', 'Full paired and historical comparisons saved under ' + str(out))


def worker(stage, arm):
    global ENVIRONMENT
    resource_guard()
    runner, work = load_runner(arm)
    if stage == 'analyze':
        compare()
        return
    import torch
    runner.setup(42)
    ENVIRONMENT = runner.environment()
    historic_env = json.loads((PARENT / 'train/started.json').read_text())['environment']
    for field in ('pytorch', 'cuda', 'gpu', 'gpu_count'):
        if ENVIRONMENT[field] != historic_env[field]:
            raise RuntimeError('Historical environment differs: ' + field)
    if ENVIRONMENT['python'].split()[0] != historic_env['python'].split()[0]:
        raise RuntimeError('Historical Python version differs')
    directory = RUN / arm
    config = work / 'experiments/tbsi_track' / CONFIGS[arm].name
    manifest = work / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json'
    write_json(directory / ('environment_' + stage + '.json'), ENVIRONMENT)
    try:
        if stage == 'train':
            if (directory / 'train/started.json').exists():
                raise RuntimeError('Refusing training restart/overwrite; use a new run identity')
            if shutil.disk_usage(ROOT).free < 8 * 1024**3:
                raise RuntimeError('At least8 GiB free space required for full pair/checkpoints')
            common = reconstruct_public_initial_state(runner)
            install_exact_training(runner, arm, common)
            def recorder(output, cfg, seed, recorded_arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
                sync_asset(arm, status, conclusion, Checkpoint=checkpoint, GPUHours=gpu_hours)
            args = SimpleNamespace(config=str(config), output=str(directory / 'train'), seed=42, arm=arm,
                manifest=str(manifest), updates=7020, milestones='7020', diagnostic_only=False,
                shared_init_snapshot=str(common), export_shared_init=None, expected_shared_state_sha256=None)
            write_json(directory / 'status.json', {'status': 'RUNNING / FULL15-TRAIN'})
            runner.train(args, recorder=recorder)
            completed = json.loads((directory / 'train/completed.json').read_text())
            if completed['successful_updates'] != 7020 or completed['samples'] != 898560 or completed['unique_sequences'] != 977:
                raise RuntimeError('Full15 successful-update/data coverage contract not met')
            write_json(directory / 'checkpoint_identity.json', {'path': str(directory / 'train/checkpoint_u7020.pth.tar'),
                       'sha256': sha(directory / 'train/checkpoint_u7020.pth.tar'), 'successful_updates': 7020})
            if sha(directory / 'train/batches.jsonl') != sha(PARENT / 'train/batches.jsonl'):
                write_json(directory / 'historical_input_comparison.json', {'identical': False})
                raise RuntimeError('Input stream differs from strongest historical MR; pause attribution')
            write_json(directory / 'historical_input_comparison.json', {'identical': True})
            write_json(directory / 'status.json', {'status': 'TRAINING-COMPLETE / FULL244-PENDING'})
        elif stage == 'evaluate':
            identity = json.loads((directory / 'checkpoint_identity.json').read_text())
            checkpoint = directory / 'train/checkpoint_u7020.pth.tar'
            if sha(checkpoint) != identity['sha256']:
                raise RuntimeError('Explicit epoch15 checkpoint identity changed')
            # Inference has no loss hooks. Existing checkpoint/parameter loader
            # reads the copied independent config; no YAML rewriting occurs.
            sync_asset(arm, 'RUNNING / FULL244-EVALUATION', 'Explicit update7020 checkpoint, FP32 full244.')
            args = SimpleNamespace(config=str(config), output=str(directory / 'full244'), seed=42,
                checkpoint=str(checkpoint), list=str(manifest.parent / 'test_sequences.txt'))
            runner.evaluate(args)
            metrics(runner, work, arm)
            write_json(directory / 'status.json', {'status': 'FULL15-FULL244-COMPLETE'})
    except Exception as exc:
        failure = {'status': 'FAILED / NO-NEW-EFFICACY-CLAIM', 'stage': stage,
                   'type': type(exc).__name__, 'error': str(exc), 'time_unix': time.time()}
        write_json(directory / ('failure_' + stage + '.json'), failure)
        sync_asset(arm, failure['status'], stage + ': ' + failure['error'])
        raise


def launch(stage, arm):
    resource_guard()
    log = RUN / 'logs' / (arm + '_' + stage + '.log')
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--internal-stage', stage, '--arm', arm]
    write_json(RUN / 'logs' / (arm + '_' + stage + '_command.json'), {'argv': command, 'cwd': str(ROOT)})
    with log.open('a') as stream:
        subprocess.run(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                PYTHONUNBUFFERED='1', TBSI_INFER_FP16='0'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'train', 'evaluate', 'analyze', 'all'], default='prepare')
    parser.add_argument('--arm', choices=['pair', 'control', 'candidate'], default='pair')
    parser.add_argument('--internal-stage', choices=['train', 'evaluate', 'analyze'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_stage:
        validate_inputs()
        if args.arm == 'pair':
            raise ValueError('Internal workers execute exactly one arm')
        try:
            worker(args.internal_stage, args.arm)
        except Exception as exc:
            sync_asset(args.arm, 'EXECUTION-FAILED / NO-NEW-EFFICACY-CLAIM',
                       args.internal_stage + ': ' + type(exc).__name__ + ': ' + str(exc))
            raise
        return
    prepare()
    if args.stage == 'prepare':
        print('Prepared only; no training/test started:', RUN)
        return
    arms = ('control', 'candidate') if args.arm == 'pair' else (args.arm,)
    if args.stage == 'analyze':
        launch('analyze', 'candidate')
        return
    for arm in arms:
        if args.stage in ('train', 'all'):
            launch('train', arm)
        if args.stage in ('evaluate', 'all'):
            launch('evaluate', arm)
    if args.stage == 'all' and args.arm == 'pair':
        launch('analyze', 'candidate')


if __name__ == '__main__':
    main()
