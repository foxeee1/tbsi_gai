#!/usr/bin/env python3
"""Normalization-only CBMR refinement; explicit independent run identity."""
import argparse
import json
import time
from pathlib import Path

from run_dual_message_p1 import train, evaluate


def record(output, cfg, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'run_record.json').write_text(json.dumps(dict(
        run_id='MSHA-MR-CYCLE-DIFFERENCE-QUICK-S42-20261115', time=time.time(),
        config=str(Path(cfg).resolve()), seed=seed, arm=arm, status=status,
        conclusion=conclusion, checkpoint=checkpoint, gpu_hours=gpu_hours,
        hypothesis='Row-normalized message differences remove degree-induced constant drift; test learned utility.',
        scope='468 successful updates; public SOT fresh; reused original matched control subject to parity and all input hashes; original24 exploratory'),
        ensure_ascii=False, indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode', choices=['train','evaluate'])
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--updates', type=int, default=468)
    p.add_argument('--milestones', default='468')
    p.add_argument('--checkpoint')
    p.add_argument('--list')
    p.add_argument('--shared-init-snapshot')
    p.add_argument('--export-shared-init')
    p.add_argument('--diagnostic-only',action='store_true')
    p.add_argument('--checkpoint-weights-only',action='store_true')
    args=p.parse_args()
    if args.mode=='train':
        train(args, recorder=record)
    else:
        if not args.checkpoint:
            p.error('evaluate requires an explicit checkpoint')
        evaluate(args)


if __name__=='__main__':
    main()
