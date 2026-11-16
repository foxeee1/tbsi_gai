#!/usr/bin/env python3
"""Matched full15 run for the selected MR receiver-message residual.

Candidate and control are launched separately. The candidate exports a
pre-optimizer state; the control imports compatible public tensors from that
snapshot so the matched pair does not depend on construction-time RNG order.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from run_dual_message_p1 import train, sha, write_json, state_hash

RUN = ROOT / 'output/experiments/MR-RECEIVER-RESIDUAL-FULL15-S42-20261116'
MANIFEST = ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json'
CANDIDATE_CONFIG = ROOT / 'experiments/tbsi_track/MR-RECEIVER-RESIDUAL-FULL15-S42-20261115.yaml'
CONTROL_CONFIG = ROOT / 'experiments/tbsi_track/MR-RECEIVER-RESIDUAL-FULL15-CONTROL-S42-20261115.yaml'
PUBLIC_SHA = '1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b'


def register(output, config, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    output = Path(output)
    identity = 'MR-RECEIVER-RESIDUAL-FULL15-S42-20261116-' + arm.upper()
    row = dict(ID=identity, Date='20261116',
        Parent='MR-RECEIVER-RESIDUAL-FULL-CANDIDATE-SELECTION-20261115; selected original receiver message residual',
        CommitOrSnapshot=str(RUN / 'source/full_launch_snapshot.tar.gz'),
        Config=str(config) + ' SHA256 ' + sha(config), Seed=str(seed),
        Dataset='LasHeR train977 sequences, 59904 pairs/epoch; 15 epochs, 7020 updates, 898560 sampled pairs; full lasher_test 244 sequences',
        EvalScope='Fresh public SOT single-stage full15; epoch15 checkpoint; candidate/control matched pre-optimizer state and train order; FP32 full244 test, threads0, template update off',
        AUC='NA', PR='NA', NPR='NA', OP50='NA', OP75='NA', FPS='NA',
        GPU='1x NVIDIA GeForce RTX 4090 D 24 GiB',
        HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118; seed42; AMP; B16 accum8; NUM_WORKER4; prefetch2; persistentfalse; fresh AdamW/StepLR',
        GPUHours=str(gpu_hours), Checkpoint=str(checkpoint),
        AssumptionOrHypothesis='A bounded receiver-conditioned residual after cross-Value aggregation can correct content direction for the receiving template while preserving MR allocation and improve full-sequence tracking.',
        Conclusion=conclusion, Status=status, Evidence=str(output),
        Notes='Candidate has RECEIVER_MESSAGE_RESIDUAL=true; control=false. No auxiliary target-direction loss, inference GT, online template update or quick-checkpoint continuation. Full result pending until both arms and full244 evaluation complete.')
    csvpath = ROOT / 'experiments/experiment_asset_registry.csv'
    mdpath = ROOT / 'experiments/实验资产数据库.md'
    with csvpath.open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    full = {k: row.get(k, 'NA') for k in fields}
    matches = [i for i, old in enumerate(rows) if old['ID'] == identity]
    if matches:
        rows[matches[0]] = full
    else:
        rows.append(full)
    temporary = csvpath.with_suffix('.fullrun.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(csvpath)
    begin, end = '<!-- ' + identity + ' BEGIN -->', '<!-- ' + identity + ' END -->'
    block = begin + '\n### ' + identity + '\n\n' + ''.join('- **' + k + '**：' + full[k] + '\n' for k in fields) + end
    text = mdpath.read_text()
    if begin in text:
        i = text.index(begin); j = text.index(end, i) + len(end); text = text[:i] + block + text[j:]
    else:
        text += '\n\n' + block + '\n'
    mdpath.write_text(text)
    write_json(output / 'run_record.json', full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['candidate', 'control'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = CANDIDATE_CONFIG if args.arm == 'candidate' else CONTROL_CONFIG
    shared = RUN / 'candidate/shared_init_preoptimizer.pth.tar'
    args.config = str(config); args.seed = 42; args.arm = args.arm
    args.manifest = str(MANIFEST); args.updates = 7020; args.milestones = '3510,7020'
    args.checkpoint = str(output / 'checkpoint_u7020.pth.tar')
    args.list = None; args.diagnostic_only = False; args.checkpoint_weights_only = True
    args.expected_shared_state_sha256 = None
    args.export_shared_init = str(shared) if args.arm == 'candidate' else None
    args.shared_init_snapshot = None if args.arm == 'candidate' else str(shared)
    args.process_pid = __import__('os').getpid()
    protocol = dict(run_identity='MR-RECEIVER-RESIDUAL-FULL15-S42-20261116', arm=args.arm,
                    config=str(config), config_sha256=sha(config), public_pretrain_sha256=PUBLIC_SHA,
                    manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST), seed=42,
                    updates=7020, samples=898560, epochs=15, batch=16, grad_accum=8,
                    checkpoint=str(output / 'checkpoint_u7020.pth.tar'),
                    initialization='public SOT fresh single-stage; candidate exports pre-optimizer state; control imports it; no optimizer continuation',
                    test='all 244 lasher_test sequences, FP32, tracker threads0, template update off')
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'process.json', dict(pid=args.process_pid, arm=args.arm, started=time.time()))
    try:
        if args.arm == 'control' and not shared.exists():
            raise RuntimeError('Candidate pre-optimizer shared snapshot is required before control')
        train(args, recorder=register)
    except Exception as exc:
        write_json(output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'time': time.time()})
        register(output, config, args.seed, args.arm, 'EXECUTION-FAILED / NO-UTILITY-CONCLUSION', type(exc).__name__ + ': ' + str(exc), args.checkpoint)
        raise


if __name__ == '__main__':
    main()
