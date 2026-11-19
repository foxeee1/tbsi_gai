#!/usr/bin/env python
"""Merge a TAM-only audit state into a copy of its exact M1 parent checkpoint."""
import argparse
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', required=True)
    ap.add_argument('--tam-checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(args.output)
    parent = torch.load(args.parent, map_location='cpu')
    tam = torch.load(args.tam_checkpoint, map_location='cpu')
    state = parent['net']
    for name, value in tam['tam'].items():
        key = 'causal_tam.' + name
        if key in state:
            raise ValueError('adapter key unexpectedly exists in M1 parent: ' + key)
        state[key] = value
    parent['net'] = state
    parent['tam_parent_checkpoint'] = os.path.abspath(args.parent)
    parent['tam_audit_checkpoint'] = os.path.abspath(args.tam_checkpoint)
    parent['tam_audit_step'] = int(tam['step'])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(parent, args.output)
    print('MERGED_CHECKPOINT', os.path.abspath(args.output))
    print('TAM_STEP', int(tam['step']))
    print('TAM_KEYS', len(tam['tam']))


if __name__ == '__main__':
    main()
