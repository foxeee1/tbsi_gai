#!/usr/bin/env python3
"""Finalize paired Early gate fixed-source assets after analysis."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from record_early_gate_fixed_20261028 import register

EXP = ROOT / 'experiments/early_gate_closed_loop_20261026/early_gate_fixed_control_20261028'
RUN = Path('/autodl-fs/data/early_gate_fixed_control_20261028/runs/FIXED')
P0 = Path('/autodl-fs/data/early_gate_fixed_control_20261028/runs/P0')
CONFIG = ROOT / 'experiments/tbsi_track/EG_EARLY_GATE_FIXED_SOT_QUICK_S42_20261028.yaml'


def main():
    report = json.loads((EXP/'analysis_u0468/summary.json').read_text())
    train = json.loads((RUN/'completed.json').read_text())
    p0 = json.loads((P0/'completed.json').read_text())
    evaluate = json.loads((RUN/'eval96/completed.json').read_text())
    checkpoint = RUN/'checkpoint_u0468.pth.tar'
    checkpoint_meta = json.loads((RUN/'checkpoint_u0468.json').read_text())
    all_result = report['results']['all']
    metrics = all_result['metrics']
    fixed_gains = [x for layer in report['fixed_source_gains_by_fusion_layer']
                   for x in layer['gains_rgb_source_then_tir_source']]
    p0_id = 'EG-EARLY-GATE-FIXED-P0-OPERATOR-CONTRACT-S42-20261028'
    register(p0_id, {
        'Date': '20261028',
        'Parent': 'Fixed-source Early gate candidate; public SOT initialization snapshot exported before any optimization',
        'CommitOrSnapshot': str(EXP/'source_before/code_before.tar.gz') + '; P0 logs and operator_contract.json retained',
        'Config': str(CONFIG) + ' SHA256=' + __import__('hashlib').sha256(CONFIG.read_bytes()).hexdigest(),
        'Seed': 42,
        'Dataset': 'LasHeR train; one real batch-window update; 128 sampled pairs from 128 unique sequences',
        'EvalScope': 'No tracking evaluation; gradient/transmission check only',
        'AUC': 'NA', 'PR': 'NA', 'NPR': 'NA', 'OP50': 'NA', 'OP75': 'NA', 'FPS': 'NA',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python 3.8.20 / PyTorch 2.0.0+cu118 / CUDA 11.8; AMP; batch16/accum8; workers4; one update',
        'GPUHours': p0['gpu_wall_hours'],
        'Checkpoint': 'NA; diagnostic run intentionally wrote no checkpoint',
        'AssumptionOrHypothesis': 'Zero fixed source logits preserve unit gain and receive finite task gradients on a real LasHeR update.',
        'Conclusion': 'Passed: one successful update; all three fusion layers had finite nonzero source-logit gradients and updated parameters; template boxes matched across modalities; 282 common initialized tensors matched the P snapshot.',
        'Status': 'P0-PASS / TRANSMISSION-ONLY / NO-UTILITY-CLAIM',
        'Evidence': str(P0),
        'Notes': 'This is one update over 128 samples, not the 468-update quick run. No tracking evaluation or model checkpoint was produced.'})

    p1_id = 'EG-EARLY-GATE-FIXED-S42-20261028'
    register(p1_id, {
        'Date': '20261028',
        'Parent': 'EG-DUALMSG-V2-P-S42-20261027 dynamic Early gate; both arms single-stage from the public SOT pretrained checkpoint, no trained-tracker continuation',
        'CommitOrSnapshot': str(EXP/'source_manifest.json') + '; initial source archive ' + str(EXP/'source_before/code_before.tar.gz'),
        'Config': str(CONFIG) + ' SHA256=' + __import__('hashlib').sha256(CONFIG.read_bytes()).hexdigest() + '; only EARLY_GATE_SOURCE_MODE=fixed',
        'Seed': 42,
        'Dataset': 'LasHeR train 977 eligible sequences / 59904 pairs / 468 successful updates; fixed screen96_max2000 test cohort',
        'EvalScope': '96 sequences / 54666 frames; FP32; tracker threads=0; template update off; same fixed panel as dynamic P',
        'AUC': metrics['auc']['fixed'],
        'PR': metrics['precision']['fixed'],
        'NPR': metrics['norm_precision']['fixed'],
        'OP50': metrics['op50']['fixed'],
        'OP75': metrics['op75']['fixed'],
        'FPS': report['runtime']['fps'],
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python 3.8.20 / PyTorch 2.0.0+cu118 / CUDA 11.8; train AMP B16/accum8/workers4/prefetch2/persistentFalse; eval FP32 threads0; serial',
        'GPUHours': train['gpu_wall_hours'] + evaluate['seconds']/3600.0,
        'Checkpoint': str(checkpoint) + ' SHA256=' + checkpoint_meta['sha256'] + '; epoch 1 / update 468',
        'AssumptionOrHypothesis': 'Per-token early evidence conditioning improves cross-source Value amplitude control beyond a learned constant source gain per fusion layer.',
        'Conclusion': 'Dynamic P outperformed fixed on the reused screen96: fixed-minus-dynamic ΔAUC -1.3014pp (90% sequence interval [-2.3939,-0.3640]); low/middle/normal deltas -1.1925/-2.6611/-0.0507pp. Fixed gains stayed near 1. The result favors dynamic Early gate under this contract but does not isolate evidence from predictor capacity.',
        'Status': 'QUICK-TRAIN-AND-SCREEN-COMPLETE / DYNAMIC-PREFERRED-IN-SCREEN / CAPACITY-CONFOUNDED / NO-FULL-PROMOTION',
        'Evidence': str(RUN) + '; ' + str(EXP/'REPORT.md') + '; ' + str(EXP/'analysis_u0468'),
        'Notes': 'Common initialization hash and all 3744 augmented microbatch records matched; both arms 468 updates, no skipped attempts, 977/977 train sequence coverage. Fixed arm has 6 active source logits total versus 582 dynamic-MLP parameters and is lower capacity, not a matched-capacity control. screen96 is a reused development panel; sequence bootstrap is not seed uncertainty. No full-data or novel-scene claim.'})

    attempt_id = 'EG-EARLY-GATE-FIXED-ANALYSIS-ATTEMPT0-20261028'
    incident = {
        'status': 'ANALYSIS-TOOL-FAILURE / NO-MODEL-FAILURE',
        'trigger': 'Initial offline analysis rejected infinite center error on 15 zero-area GT frames in ballshootatthebasket3times.',
        'cause': 'The repository robust evaluator intentionally marks invalid GT frames as Inf center error, -1 overlap, and valid=false; the first analyzer required all center errors finite.',
        'repair': 'Use the evaluator valid mask; retain expected Inf for invalid frames; cross-check direct IoU only on valid GT frames.',
        'model_runs_changed': False,
        'final_analysis': str(EXP/'analysis_u0468/summary.json'),
    }
    (EXP/'analysis_attempt0.json').write_text(json.dumps(incident, ensure_ascii=False, indent=2)+'\n')
    register(attempt_id, {
        'Date': '20261028', 'Parent': p1_id,
        'CommitOrSnapshot': str(EXP/'source_manifest.json') + '; corrected analyzer and incident record retained',
        'Config': 'CPU offline analysis of fixed u0468 and dynamic P u0468 outputs; no model/config change',
        'Seed': 'Existing seed42 checkpoints; no new training seed',
        'Dataset': 'Existing frozen screen96 predictions; one sequence exposed invalid GT frame semantics',
        'EvalScope': 'CPU metric validation only; no inference or training',
        'AUC': 'NA', 'PR': 'NA', 'NPR': 'NA', 'OP50': 'NA', 'OP75': 'NA', 'FPS': 'NA',
        'GPU': 'NA; CPU-only analysis', 'HardwareConfig': 'Python 3.8.20; NumPy/PyTorch CPU metric check',
        'GPUHours': 0, 'Checkpoint': 'No new checkpoint; original P and FIXED u0468 unchanged',
        'AssumptionOrHypothesis': 'Metric checker must follow repository handling for zero-area GT frames and still independently verify IoU on valid frames.',
        'Conclusion': 'First analyzer pass stopped on expected Inf center error from 15 invalid GT frames; corrected valid-mask handling, then all 96 sequences and 54666 frames passed metric cross-check.',
        'Status': 'ANALYSIS-TOOL-INCIDENT-RECORDED / CORRECTED / FINAL-ANALYSIS-COMPLETE',
        'Evidence': str(EXP/'analysis_attempt0.json') + '; ' + str(EXP/'analysis_u0468/summary.json'),
        'Notes': 'This was an analysis-contract bug only. No model, checkpoint, prediction, or experimental conclusion changed.'})

    alias_id = 'EG-DUALMSG-P1-FIXED-S42-20261027'
    register(alias_id, {
        'Conclusion': 'Runner default recorder initially wrote this run under the wrong dual-message-stage ID. The single fixed-source run is canonically registered as EG-EARLY-GATE-FIXED-S42-20261028; this row is retained only as a registration alias and contains no separate experiment.',
        'Status': 'REGISTRY-ALIAS / SEE EG-EARLY-GATE-FIXED-S42-20261028',
        'Evidence': str(EXP/'REPORT.md') + '; canonical run: ' + str(RUN),
        'Notes': 'Do not count this alias as a second run. All final metrics and provenance live under EG-EARLY-GATE-FIXED-S42-20261028.'})


if __name__ == '__main__':
    main()
