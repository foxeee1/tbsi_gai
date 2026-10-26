#!/usr/bin/env python
"""Run the causal TAM adapter on real M1 trace tensors.

This is an intermediate gate: it validates real deployed raw-head tokens and
crop/frame provenance without claiming that the training actor is wired or
that tracking improves.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lib.models.layers.causal_tam_adapter import CausalTemporalFusionResidual


def load_pairs(root, max_per_sequence=1):
    pairs = []
    for seq in sorted(os.listdir(root)):
        path = os.path.join(root, seq, 'frames.jsonl')
        if not os.path.isfile(path):
            continue
        rows = [json.loads(line) for line in open(path) if line.strip()]
        count = 0
        for index in range(1, len(rows)):
            row, prev = rows[index], rows[index - 1]
            if int(row['history_frame_id']) != int(prev['frame_id']):
                continue
            cur_path = os.path.join(root, seq, row['raw_head_file'])
            old_path = os.path.join(root, seq, prev['raw_head_file'])
            cur = np.load(cur_path, allow_pickle=False).astype('float32')
            old = np.load(old_path, allow_pickle=False).astype('float32')
            if list(cur.shape) != [1, 256, 768] or list(old.shape) != [1, 256, 768]:
                raise ValueError('unexpected raw shape in %s' % seq)
            pairs.append((seq, row, prev, torch.from_numpy(cur), torch.from_numpy(old)))
            count += 1
            if count >= max_per_sequence:
                break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    pairs = load_pairs(args.trace_dir)
    if len(pairs) < 3:
        raise RuntimeError('need at least three sequence-isolated trace pairs')

    adapter = CausalTemporalFusionResidual(dim=768, hidden=192, heads=4,
                                            grid_size=16, local_radius=2,
                                            max_message_ratio=0.10)
    adapter.train()
    results = []
    all_finite = True
    all_ratio_ok = True
    gate_grad_ok = True
    branch_grad_ok = True
    sequence_ids = []
    for seq, row, prev, cur, old in pairs[:3]:
        sequence_ids.append(seq)
        cur = cur.clone().requires_grad_(False)
        old = old.clone().requires_grad_(False)
        kwargs = dict(
            current_rect=[row['current_crop_xywh']],
            history_rect=[prev['current_crop_xywh']],
            current_frame_id=[row['frame_id']],
            history_frame_id=[prev['frame_id']],
            current_sequence_id=[seq],
            history_sequence_id=[seq],
        )
        adapter.zero_grad()
        out, stats = adapter(cur, old, **kwargs)
        loss = out.square().mean()
        loss.backward()
        gate_ok = adapter.gamma.grad is not None and torch.isfinite(adapter.gamma.grad).all()
        gate_ok = bool(gate_ok and adapter.gamma.grad.abs().item() > 0)
        gate_grad_ok = gate_grad_ok and gate_ok

        with torch.no_grad():
            adapter.gamma.fill_(0.1)
        adapter.zero_grad()
        out_open, stats_open = adapter(cur, old, **kwargs)
        out_open.square().mean().backward()
        branch = [adapter.q.weight.grad, adapter.k.weight.grad,
                  adapter.v.weight.grad, adapter.out.weight.grad]
        branch_ok = all(x is not None and torch.isfinite(x).all() and x.abs().sum().item() > 0
                        for x in branch)
        branch_grad_ok = branch_grad_ok and bool(branch_ok)
        finite = bool(torch.isfinite(out_open).all() and all(
            np.isfinite(float(v)) for v in stats_open.values() if isinstance(v, (float, int))))
        ratio_ok = float(stats_open.get('actual_ratio_max', 1.0)) <= 0.10 + 1e-6
        all_finite = all_finite and finite
        all_ratio_ok = all_ratio_ok and ratio_ok
        results.append({
            'sequence_id': seq,
            'current_frame_id': int(row['frame_id']),
            'history_frame_id': int(prev['frame_id']),
            'coverage': stats_open.get('coverage'),
            'query_usable_fraction': stats_open.get('query_usable_fraction'),
            'actual_ratio_mean': stats_open.get('actual_ratio_mean'),
            'actual_ratio_max': stats_open.get('actual_ratio_max'),
            'gate_gradient_ok': gate_ok,
            'branch_gradients_ok': bool(branch_ok),
            'finite_ok': finite,
            'ratio_ok': ratio_ok,
        })

    result = {
        'decision': ('TRACE_BATCH_ADAPTER_PASS' if all_finite and all_ratio_ok
                     and gate_grad_ok and branch_grad_ok else 'STOP_TRACE_BATCH_ADAPTER'),
        'pairs': len(results),
        'sequence_ids': sequence_ids,
        'results': results,
        'real_m1_raw_tokens': True,
        'real_m1_crop_metadata': True,
        'training_actor_exercised': False,
        'optimizer_step_performed': False,
        'tracking_effect_claim': False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['decision'] == 'TRACE_BATCH_ADAPTER_PASS' else 2


if __name__ == '__main__':
    sys.exit(main())
