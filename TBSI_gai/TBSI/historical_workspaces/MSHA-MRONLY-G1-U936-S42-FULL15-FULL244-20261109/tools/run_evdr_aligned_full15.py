#!/usr/bin/env python3
"""Full 15-epoch EVDR receiver-transport run from public SOT pretraining."""
import argparse
import json
import time
import traceback
from pathlib import Path

from run_early_gate_p1_v2 import train, ROOT, write_json
from record_early_gate_p1_v2 import register

CONFIG = ROOT / 'experiments/tbsi_track/EG_EVDR_ALIGNED_FULL15_SOT_S42_20261027.yaml'
MANIFEST = ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json'
OUTPUT = ROOT / 'output/experiments/EG_EVDR_ALIGNED_FULL15_SOT_S42_20261027'
RUN = ROOT / 'experiments/early_gate_closed_loop_20261026/evdr_scope_v2_20261026'
ASSET = 'EG-EVDR-ALIGNED-FULL15-SOT-S42-20261027'


def record(output, cfg, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    register(ASSET, {
        'Date': '20261027',
        'Parent': 'Early gate full15 architecture; receiver-transport EVDR aligned quick; fresh public SOT initialization only',
        'CommitOrSnapshot': str(RUN / 'full15_source_at_launch/manifest.json'),
        'Config': str(CONFIG), 'Seed': seed,
        'Dataset': 'LasHeR train977 eligible sequences,51 strata; 59904 pairs/epoch x15 = 898560 samples; 7020 successful updates',
        'EvalScope': 'Full training endpoint u07020; test all native lasher_test sequences expected244; FP32 threads0 template update off; metrics pending',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8; AMP; B16 accum8; NUM_WORKER4/prefetch2/persistentFalse; fresh AdamW/StepLR; LR drop epoch10',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'The receiver-transport Value condition that partially mitigated EVDR-v1 in the 468-update development cohort preserves direction and protects normal sequences after the full 15-epoch public-SOT training schedule.',
        'Conclusion': conclusion, 'Status': status, 'Evidence': str(RUN) + '; ' + str(output),
        'Notes': 'Independent full run from public SOT SHA2561f6ebf59...; no M1/full15/quick checkpoint or optimizer continuation. Full result is compared to the same-protocol Early gate full run only with explicit data/software differences recorded. Test uses the previously verified serial threads=0 configuration.'
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=7020)
    args = parser.parse_args()
    if args.updates != 7020:
        raise ValueError('Full identity is fixed to 15 epochs / 7020 successful updates')
    args.config = str(CONFIG)
    args.output = str(OUTPUT)
    args.seed = 42
    args.arm = 'aligned_full15'
    args.manifest = str(MANIFEST)
    args.milestones = '3510,7020'
    try:
        train(args, recorder=record)
    except Exception as exc:
        write_json(OUTPUT / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                   'traceback': traceback.format_exc(), 'time': time.time()})
        record(OUTPUT, CONFIG, 42, 'aligned_full15', 'EXECUTION-FAILED / NO-UTILITY-CONCLUSION',
               type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()
