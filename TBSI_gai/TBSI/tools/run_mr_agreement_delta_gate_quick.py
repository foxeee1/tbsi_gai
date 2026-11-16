#!/usr/bin/env python3
"""Paired quick runner for the MR source-agreement delta gate."""

import argparse
import json
import time
from pathlib import Path

from run_dual_message_p1 import evaluate, train


RUN_ID = 'MSHA-MR-AGREEMENT-DELTA-GATE-QUICK-S42-20261113'


def record(output, cfg, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    payload = {
        'time': time.time(), 'run_id': RUN_ID, 'arm': str(arm), 'seed': int(seed),
        'status': str(status), 'conclusion': str(conclusion),
        'config': str(Path(cfg).resolve()), 'checkpoint': str(checkpoint),
        'gpu_hours': str(gpu_hours),
        'hypothesis': 'Agreement-gated MR score corrections reduce harmful source re-ranking without changing cross-row mass.',
        'scope': '468 updates; public pretrained single-stage; fixed 24-sequence exploratory cohort; template update off.'
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_record.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'evaluate'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--updates', type=int, default=468)
    parser.add_argument('--milestones', default='234,468')
    parser.add_argument('--checkpoint')
    parser.add_argument('--list')
    parser.add_argument('--shared-init-snapshot')
    parser.add_argument('--export-shared-init')
    parser.add_argument('--diagnostic-only', action='store_true')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args, recorder=record)
    else:
        if not args.checkpoint:
            parser.error('evaluate requires --checkpoint')
        evaluate(args)


if __name__ == '__main__':
    main()
