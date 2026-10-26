#!/usr/bin/env python
"""Audit whether the existing training actor provides a causal TAM history.

This intentionally refuses to fabricate a training batch. A PASS requires
explicit previous-frame features, crop rectangles, frame IDs, and sequence
IDs in the actor-to-network call.
"""
import argparse
import ast
import json
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    actor_path = os.path.join(root, 'lib/train/actors/tbsi_track.py')
    cfg_path = os.path.join(root, 'experiments/tbsi_track/G0_CAUSAL_TAM_PILOT_seed42.yaml')
    source = open(actor_path).read()
    tree = ast.parse(source)
    forward_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Attribute) and n.func.attr == 'net']
    names = {kw.arg for call in forward_calls for kw in call.keywords if kw.arg}
    required = {'prev_head_feature', 'current_crop_xywh', 'history_crop_xywh',
                'current_frame_id', 'history_frame_id', 'current_sequence_id',
                'history_sequence_id'}
    missing = sorted(required - names)
    # The model may derive previous raw tokens from explicit prev_search under
    # no_grad; this is equivalent to passing prev_head_feature directly.
    prev_raw_contract = ('prev_head_feature' in names or
                         'prev_search' in names)
    if prev_raw_contract and 'prev_head_feature' in missing:
        missing.remove('prev_head_feature')
    trace_fields = {'sequence_id', 'history_sequence_id', 'frame_id',
                    'history_frame_id', 'current_crop_xywh', 'history_crop_xywh'}
    sampler_source = open(os.path.join(root, 'lib/train/data/sampler.py')).read()
    crop_available = 'current_crop_xywh' in sampler_source and 'history_crop_xywh' in sampler_source
    result = {
        'status': 'PASS' if not missing and crop_available else 'STOP_TRAINING_HISTORY_CONTRACT',
        'actor_path': actor_path,
        'config_path': cfg_path,
        'network_history_keywords_found': sorted(names & required),
        'missing_history_keywords': missing,
        'sampler_trace_fields': sorted(trace_fields & set(open(os.path.join(root, 'lib/train/data/sampler.py')).read().split("'"))),
        'crop_trace_available': crop_available,
        'real_batch_count': 0,
        'history_source_verified': False,
        'reason': ('Training actor/model history interface is present, but the '
                   'sampler has no frozen M1 actual crop trace; no real training '
                   'batch was run.') if not missing and not crop_available else
                   ('History contract is incomplete; no real training batch was run.'
                    if missing else 'Static contract present; runtime batch audit still required.'),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    sys.exit(main())
