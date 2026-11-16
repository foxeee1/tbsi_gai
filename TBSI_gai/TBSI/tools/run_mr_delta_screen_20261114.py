#!/usr/bin/env python3
"""Evaluate one trained MR checkpoint at a runtime-only delta bound."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--runner', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--list', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-delta', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.max_delta < 0 or args.max_delta > 1:
        raise ValueError('max-delta must be in [0, 1] for this bounded screen')

    workspace = Path(args.workspace).resolve()
    runner_path = Path(args.runner).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    list_path = Path(args.list).resolve()
    output_path = Path(args.output).resolve()
    if workspace.parent.parent.name != 'experiments':
        raise ValueError('Unexpected project workspace: ' + str(workspace))
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TBSI_INFER_FP16'] = '0'

    module_name = 'mr_delta_parent_runner'
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    from lib.models.layers.receiver_source_mass_reallocator import ReceiverSourceMassReallocator
    original_init = ReceiverSourceMassReallocator.__init__
    constructed = []

    def bounded_init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        self.max_delta = args.max_delta
        constructed.append(self)
    ReceiverSourceMassReallocator.__init__ = bounded_init

    run_args = SimpleNamespace(
        seed=args.seed, config=str(config_path), checkpoint=str(checkpoint_path),
        list=str(list_path), output=str(output_path))
    runner.evaluate(run_args)
    observed = sorted(set(float(module.max_delta) for module in constructed))
    if not constructed or observed != [float(args.max_delta)]:
        raise RuntimeError('MR delta override verification failed: %r' % observed)
    (output_path / 'parameter_override.json').write_text(json.dumps({
        'requested_max_delta': float(args.max_delta),
        'constructed_allocator_count': len(constructed),
        'observed_max_delta_values': observed,
        'checkpoint': str(checkpoint_path),
        'status': 'verified after evaluation model construction',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
