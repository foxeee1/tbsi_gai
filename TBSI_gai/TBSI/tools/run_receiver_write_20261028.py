#!/usr/bin/env python3
"""Fresh public-SOT quick run for receiver-side cross-message acceptance."""
import argparse
import json
import time
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_dual_message_p1 import train, evaluate
from record_receiver_write_20261028 import run_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'evaluate'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arm', default='RECEIVER_WRITE')
    parser.add_argument('--manifest', default=str(ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json'))
    parser.add_argument('--updates', type=int, default=468)
    parser.add_argument('--milestones', default='234,468')
    parser.add_argument('--checkpoint')
    parser.add_argument('--list')
    parser.add_argument('--shared-init-snapshot', default='/autodl-fs/data/early_gate_closed_loop/dual_message_p1_v2_20261027/shared_public_sot_init.pt')
    parser.add_argument('--diagnostic-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.mode == 'train':
            train(args, recorder=run_record)
        else:
            if not args.checkpoint:
                parser.error('evaluate requires --checkpoint')
            evaluate(args)
    except Exception as exc:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'failure.json').write_text(json.dumps({
            'type': type(exc).__name__, 'error': str(exc),
            'traceback': traceback.format_exc(), 'time': time.time()},
            ensure_ascii=False, indent=2) + '\n')
        if args.mode == 'train':
            run_record(output, args.config, args.seed,
                       'P0' if args.diagnostic_only else args.arm,
                       'P0-EXECUTION-FAILED' if args.diagnostic_only else 'EXECUTION-FAILED',
                       type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()
