#!/usr/bin/env python3
"""One narrowed EVDR receiver intervention, identical SOT-v3 paired trainer.

Only fresh public-pretraining training is exposed.  No resume or checkpoint
initialization argument is accepted.  Evaluation uses the common native runner.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

from run_early_gate_p1_v2 import train, ROOT, write_json
from record_early_gate_p1_v2 import register

ASSET = 'EG-EVDR-ALIGNED-P1-SOT-S42-20261026'
RUN = ROOT / 'experiments/early_gate_closed_loop_20261026/evdr_scope_v2_20261026'
CFG = ROOT / 'experiments/tbsi_track/EG_EVDR_ALIGNED_P1_SOT_S42_20261026.yaml'
OUTPUT = ROOT / 'output/experiments/EG_EVDR_ALIGNED_P1_SOT_S42_20261026'


def record(output, cfg, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    register(ASSET, {
        'Date': '20261026',
        'Parent': 'Early gate matched SOT-v3 control; direct operator parent EVDR-v1. Both are fresh SOT single-stage runs; no tracker continuation',
        'CommitOrSnapshot': str(RUN / 'source_at_launch/manifest.json'),
        'Config': str(cfg), 'Seed': seed,
        'Dataset': 'LasHeR train977,51strata; same manifest and deterministic sampling as SOT-v3;59904pairs/468 successful updates',
        'EvalScope': 'Fixed screen96_max2000 development cohort;54666frames;FP32 threads0 no template update;468 endpoint. No independent-validation or full244 claim',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8;AMP B16 accum8;workers4 prefetch2 persistentFalse;fresh AdamW/StepLR;LR-drop full epoch10',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'Single variable: replace colocated receiver conditioning by column-normalized existing cross-attention transport, keeping source predictor,rank8,max_step0.1,layers3/6/9,Early gate,loss,optimizer,data and budget fixed',
        'Conclusion': conclusion, 'Status': status, 'Evidence': str(RUN) + '; ' + str(output),
        'Notes': 'One exploratory candidate. Public SOT SHA1f6ebf59... only. Shared-state/input hashes must match existing control before ranking. No quick checkpoint or optimizer continuation. No proven quick/full direction calibration.'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--updates', type=int, default=468)
    # An explicit smaller budget creates a different identity; this entry is
    # intentionally fixed to the predeclared pair rather than silently reusing it.
    args = p.parse_args()
    if args.updates != 468:
        raise ValueError('This identity is fixed to 468 successful updates')
    args.config = str(CFG); args.output = str(OUTPUT)
    args.seed = 42; args.arm = 'aligned'
    args.manifest = str(ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json')
    args.milestones = '234,468'
    try:
        train(args, recorder=record)
    except Exception as exc:
        write_json(OUTPUT / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                   'traceback': traceback.format_exc(), 'time': time.time()})
        record(OUTPUT, CFG, 42, 'aligned', 'EXECUTION-FAILED / NO-UTILITY-CONCLUSION',
               type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()
