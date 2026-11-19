#!/usr/bin/env python3
"""Resource-serial fresh SOT quick experiment for transport-gradient isolation.

The waiting supervisor imports no torch. Its train child starts a single fresh
joint-learning run after the existing full244 evaluation finishes. No resume,
best-checkpoint selection, or legacy admission scripts are used.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/early_gate_closed_loop_20261026/evdr_transport_grad_20261027'
P1 = RUN.parent / 'p1_sot_v3'
CONFIG = ROOT / 'experiments/tbsi_track/EG_EVDR_TGRAD_P1_SOT_S42_20261027.yaml'
OUTPUT = ROOT / 'output/experiments/EG_EVDR_TGRAD_P1_SOT_S42_20261027'
TEST = ROOT / 'output/test/evdr_transport_grad_p1/u0468_screen96_20261027'
ASSET = 'EG-EVDR-TGRAD-P1-SOT-S42-20261027'
DEPENDENCY = ROOT / 'output/test/evdr_aligned_full15/full244_20261027'


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(output, cfg, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    from record_early_gate_p1_v2 import register
    register(ASSET, {
        'Date': '20261027',
        'Parent': 'Direct operator parent: EVDR aligned SOT quick; efficacy parent: matched Early gate SOT quick. Fresh public SOT only.',
        'CommitOrSnapshot': str(RUN / 'source_at_queue/manifest.json'),
        'Config': str(CONFIG), 'Seed': 42,
        'Dataset': 'LasHeR eligible train977/51strata;59904pairs;468successful updates;fixed96<=2000-frame development test54666frames',
        'EvalScope': 'Fixed468 endpoint; same frozen low/middle/normal groups;FP32 threads0;no template update; no full or independent-validation claim',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB (training/evaluation target; actual environment recorded at launch)',
        'HardwareConfig': 'Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8;AMP B16 accum8;workers4,prefetch2,persistentFalse;fresh AdamW/StepLR;LR drop full epoch10',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'Only detach attention weights in the extra receiver-condition route; preserve forward, early gate, original attention-readout gradient, rank8,max_step0.1,layers3/6/9 and all training rules. Test gradient co-adaptation as one possible cause of normal-group harm.',
        'Conclusion': conclusion, 'Status': status,
        'Evidence': str(RUN) + '; ' + str(output) + '; ' + str(TEST),
        'Notes': 'Not a reliability gate, backbone freeze, smaller step or multistage training. No M1/full15/quick continuation. CPU synthetic contracts with explicit checker-only GELU layout correction; real-GT/CUDA evidence comes from fresh run. Dataset is developed cohort; normal target>=0pp vs Early gate is exploratory, not a universal safety guarantee.'})


def resource_free():
    proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True)
    gpu = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if gpu:
        return False, {'gpu_pids': gpu}
    proc = subprocess.run(['ps', '-eo', 'pid=,ppid=,args='], capture_output=True, text=True, check=True)
    busy = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or int(parts[0]) == os.getpid():
            continue
        command = parts[2]
        # Treat known training/testing entry points as busy even during CPU
        # loading, before they allocate GPU. Idle wait supervisors do not count.
        if ('python' in command and
            any(token in command for token in ['tools/run_early_gate_p1_v2.py', 'tools/run_evdr_aligned', 'tracking/test.py', 'tracking/train.py'])
            and not command.startswith(('/bin/bash', 'bash ', 'sh '))):
            busy.append({'pid': int(parts[0]), 'command': command[:250]})
    return not busy, {'gpu_pids': [], 'experiment_processes': busy}


def verify_source():
    manifest = json.loads((RUN / 'source_at_queue/manifest.json').read_text())
    changed = [r['path'] for r in manifest['files'] if sha(ROOT/r['path']) != r['sha256']]
    if changed:
        raise RuntimeError('Queued source identity changed: ' + repr(changed))


def train_once():
    from run_early_gate_p1_v2 import train
    verify_source()
    free, state = resource_free()
    if not free:
        raise RuntimeError('Refusing concurrent worker4 training: ' + repr(state))
    # Check the unmodified CUDA backend before spending the training budget.
    # No checker hooks and no optimizer state are reused by train().
    import torch
    from lib.models.layers.early_conditioned_value_direction_aligned import AttentionAlignedValueDirection
    from lib.models.layers.early_conditioned_value_direction_transport_grad import TransportGradientIsolatedValueDirection
    torch.set_num_threads(1)
    torch.manual_seed(42)
    parent = AttentionAlignedValueDirection(48, 3, rank=4).cuda()
    candidate = TransportGradientIsolatedValueDirection(48, 3, rank=4).cuda()
    with torch.no_grad():
        parent.up.normal_(std=0.07)
    candidate.load_state_dict(parent.state_dict(), strict=True)
    v = torch.randn(2, 3, 5, 16, device='cuda')
    src, rec = torch.randn(2, 5, 48, device='cuda'), torch.randn(2, 7, 48, device='cuda')
    evidence = torch.randn(2, 5, 3, device='cuda')
    attention = torch.rand(2, 3, 7, 5, device='cuda', requires_grad=True)
    weight = torch.randn_like(v)
    yp, yc = parent(v, src, rec, evidence, attention), candidate(v, src, rec, evidence, attention)
    if not torch.equal(yp, yc):
        raise RuntimeError('CUDA nonzero-state forward differs')
    gp = torch.autograd.grad((yp*weight).sum(), attention)[0]
    (yc*weight).sum().backward()
    if attention.grad is not None or not torch.isfinite(gp).all() or not gp.abs().max() > 0:
        raise RuntimeError('CUDA attention-gradient isolation/positive control failed')
    grad_norms = {name: float(p.grad.double().norm()) for name, p in candidate.named_parameters() if p.grad is not None}
    if len(grad_norms) != 5 or not all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None) or not all(x > 0 for x in grad_norms.values()):
        raise RuntimeError('CUDA predictor gradient contract failed')
    write(RUN/'cuda_contract.json', {'scope': 'synthetic CUDA check only; real-GT training follows',
          'pytorch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'forward_exact': True, 'extra_attention_gradient_isolated': True,
          'predictor_gradient_norms': grad_norms, 'cpu_gelu_hook_used': False})
    del parent, candidate, v, src, rec, evidence, attention, weight, yp, yc, gp
    torch.cuda.empty_cache()
    args = SimpleNamespace(config=str(CONFIG), output=str(OUTPUT), seed=42,
                           arm='transport_gradient_isolated', manifest=str(P1/'data_manifest.json'),
                           updates=468, milestones='234,468')
    train(args, recorder=record)


def analyze():
    import numpy as np
    import torch
    # Reuse the current, independently cross-checked metric reader only;
    # do not invoke old screening thresholds, compare(), or admission tools.
    from analyze_early_gate_p1_v2 import read_result, validate_pair, summarize
    torch.set_num_threads(1)
    frames = {r['sequence']: r['frames'] for r in json.loads((P1/'data_manifest.json').read_text())['test_sequences']}
    frozen = json.loads((P1/'frozen_groups.json').read_text())['groups']
    membership = {name: group for group, names in frozen.items() for name in names}
    names = (P1/'screen96_max2000.txt').read_text().splitlines()
    comparisons = {}
    for label, train_dir, test_dir in [
        ('early_gate', ROOT/'output/experiments/EG_EVDR_P1_SOT_V3_CONTROL_S42_20261026', ROOT/'output/test/early_gate_p1_sot_v3/control_u0468_screen96_max2000_20261026'),
        ('aligned', ROOT/'output/experiments/EG_EVDR_ALIGNED_P1_SOT_S42_20261026', ROOT/'output/test/evdr_aligned_p1/aligned_u0468_screen96_max2000_20261026')]:
        out = RUN / ('analysis_vs_' + label)
        out.mkdir(parents=True, exist_ok=True)
        pairing = validate_pair(str(train_dir), str(OUTPUT), out/'pairing.json')
        rows = []
        for name in names:
            control, ciou = read_result(test_dir, name, frames[name])
            candidate, diou = read_result(TEST, name, frames[name])
            row = {'sequence': name, 'frames': frames[name], 'group': membership[name]}
            for metric in control:
                row[metric+'_control'] = control[metric]
                row[metric+'_candidate'] = candidate[metric]
                row[metric+'_delta'] = candidate[metric] - control[metric]
            row['new_failure_from_control_good_frames'] = int(((ciou >= .5) & (diou < .1)).sum())
            row['recovered_control_failure_frames'] = int(((ciou < .1) & (diou >= .5)).sum())
            rows.append(row)
        with (out/'per_sequence.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        summary = {'all_evaluated': summarize(rows, None),
                   **{group: summarize(rows, members) for group, members in frozen.items()},
                   'pairing': pairing,
                   'scope': 'seed42 fixed96 development cohort; no independent/full/seed-robust claim; no legacy admission threshold'}
        write(out/'summary.json', summary)
        comparisons[label] = summary
    e, a = comparisons['early_gate'], comparisons['aligned']
    restored = e['normal_protection']['auc_delta'] >= 0
    low_positive = e['low_target']['auc_delta'] > 0
    overall_positive = e['all_evaluated']['auc_delta'] > 0
    status = 'DEVELOPMENT-TARGET-OBSERVED / CONFIRMATION-NEEDED' if restored and low_positive and overall_positive else 'DEVELOPMENT-MIXED-OR-NEGATIVE / CAUSE-UNRESOLVED'
    done = json.loads((OUTPUT/'completed.json').read_text())
    tested = json.loads((TEST/'completed.json').read_text())
    times = [float(x) for p in TEST.glob('*_time.txt') for x in p.read_text().split()]
    fps = len(times)/sum(times) if times else 'NA'
    from record_early_gate_p1_v2 import register
    metrics = e['all_evaluated']['metrics']
    conclusion = ('Transport-gradient single variable: vs Early gate overall/low/normal ΔAUC=%+.6f/%+.6f/%+.6fpp; vs aligned=%+.6f/%+.6f/%+.6fpp. %s; sequence intervals do not represent seed uncertainty.' %
                  (e['all_evaluated']['auc_delta'], e['low_target']['auc_delta'], e['normal_protection']['auc_delta'],
                   a['all_evaluated']['auc_delta'], a['low_target']['auc_delta'], a['normal_protection']['auc_delta'], status))
    register(ASSET, {'Status': 'SCREEN-COMPLETE / '+status,
                    'AUC': metrics['auc']['candidate'], 'PR': metrics['precision']['candidate'],
                    'NPR': metrics['norm_precision']['candidate'], 'OP50': metrics['op50']['candidate'],
                    'OP75': metrics['op75']['candidate'], 'FPS': fps,
                    'GPUHours': done['gpu_wall_hours'] + tested['seconds']/3600,
                    'Checkpoint': str(OUTPUT/'checkpoint_u0468.pth.tar')+' SHA256 '+sha(OUTPUT/'checkpoint_u0468.pth.tar')+';468successful updates/epoch1',
                    'Conclusion': conclusion})
    result = {'status': status, 'normal_mean_recovered_vs_early_gate': restored,
              'low_positive_vs_early_gate': low_positive, 'overall_positive_vs_early_gate': overall_positive,
              'comparisons': comparisons, 'no_automatic_full_promotion': True}
    write(RUN/'result.json', result)
    lines = ['# Receiver运输梯度隔离：固定468更新结果', '', conclusion, '',
             '| 对照/组 | n | ΔAUC pp | 90%序列区间 | >5pp损失 |', '|---|---:|---:|---|---:|']
    for label, comparison in comparisons.items():
        for group in ['all_evaluated', 'low_target', 'middle', 'normal_protection']:
            s = comparison[group]
            lines.append('| %s/%s | %d | %+.4f | [%+.4f,%+.4f] | %d |' %
                         (label, group, s['n'], s['auc_delta'], *s['sequence_bootstrap_90pct'], s['losses_over_5pp']))
    lines += ['', '同96开发集合，保留所有序列和冻结分组；不据单序列恢复或平均数跨零宣称安全。区间、负尾、逐序列、配对证据见analysis_vs_*。若只修复正常组但低组/整体退化，不算完成效用目标；若没有改善，不能永久否定方向机制。没有full/多seed结论。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def supervise():
    while not (DEPENDENCY/'completed.json').exists():
        if (DEPENDENCY/'failure.json').exists():
            raise RuntimeError('Full244 dependency failed; do not overlap/restart it automatically')
        write(RUN/'queue_state.json', {'state': 'WAITING_FOR_FULL244', 'time': time.time(), 'dependency': str(DEPENDENCY)})
        time.sleep(30)
    while True:
        free, resources = resource_free()
        if free:
            break
        write(RUN/'queue_state.json', {'state': 'WAITING_FOR_RESOURCES', 'time': time.time(), 'resources': resources})
        time.sleep(30)
    verify_source()
    if (OUTPUT/'started.json').exists():
        raise RuntimeError('Refusing reused training identity; no resume permitted')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TBSI_INFER_FP16='0', PYTHONUNBUFFERED='1')
    write(RUN/'queue_state.json', {'state': 'TRAINING', 'time': time.time()})
    with (RUN/'train.log').open('ab', buffering=0) as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', 'train'],
                       cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Child exit has joined its workers; do not start tracking until idle.
    free, resources = resource_free()
    if not free:
        raise RuntimeError('Resources busy after train child exited: '+repr(resources))
    write(RUN/'queue_state.json', {'state': 'EVALUATING_FIXED96', 'time': time.time()})
    command = [sys.executable, 'tools/run_early_gate_p1_v2.py', 'evaluate',
               '--config', str(CONFIG), '--output', str(TEST), '--checkpoint', str(OUTPUT/'checkpoint_u0468.pth.tar'),
               '--list', str(P1/'screen96_max2000.txt')]
    with (RUN/'evaluate.log').open('ab', buffering=0) as log:
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (RUN/'analysis.log').open('ab', buffering=0) as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', 'analyze'],
                       cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(RUN/'queue_state.json', {'state': 'COMPLETE', 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['wait', 'train', 'analyze'], default='wait')
    stage = parser.parse_args().stage
    try:
        {'wait': supervise, 'train': train_once, 'analyze': analyze}[stage]()
    except Exception as exc:
        write(RUN/('failure_'+stage+'.json'), {'type': type(exc).__name__, 'error': str(exc),
                                             'traceback': traceback.format_exc(), 'time': time.time()})
        record(OUTPUT, CONFIG, 42, 'transport_gradient_isolated',
               'EXECUTION-FAILED / '+stage.upper()+' / NO-UTILITY-CONCLUSION',
               type(exc).__name__+': '+str(exc))
        raise
