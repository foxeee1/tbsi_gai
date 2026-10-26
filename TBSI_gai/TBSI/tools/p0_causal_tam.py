#!/usr/bin/env python
"""P0 contract checks for the causal TAM tracker adapter.

This is an operator/integration check only. It does not run tracking or
modify any baseline artifact.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from lib.models.layers.causal_tam_adapter import CausalTemporalFusionResidual


def check(device):
    torch.manual_seed(42)
    b, s, d = 2, 16, 32
    model = CausalTemporalFusionResidual(dim=d, hidden=16, heads=4,
                                        grid_size=4, local_radius=1).to(device)
    cur = torch.randn(b, s, d, device=device)
    old = torch.randn_like(cur)
    rect = torch.tensor([[0., 0., 160., 160.], [20., 30., 160., 160.]], device=device)
    results = {}

    out, stats = model(cur, old, rect, rect, [2, 2], [1, 1])
    results['gamma_zero_identity'] = bool(torch.equal(out, cur))
    results['identity_coverage'] = stats['coverage']
    out_no_history, _ = model(cur)
    results['no_history_identity'] = bool(torch.equal(out_no_history, cur))

    with torch.no_grad():
        model.gamma.fill_(0.5)
    out, stats = model(cur, old, rect, rect, [2, 2], [1, 1])
    results['nonzero_changes_output'] = bool((out - cur).abs().max().item() > 0)
    results['message_ratio'] = stats['message_ratio']
    results['message_ratio_bounded'] = bool(stats['message_ratio'] <= 0.10 + 1e-6)

    far = rect.clone()
    far[:, 0] += 2000
    out_far, far_stats = model(cur, old, rect, far, [2, 2], [1, 1])
    results['no_overlap_identity'] = bool(torch.equal(out_far, cur))
    results['no_overlap_coverage'] = far_stats['coverage']

    try:
        model(cur, old, rect, rect, [1, 2], [2, 1])
        results['future_frame_rejected'] = False
    except ValueError:
        results['future_frame_rejected'] = True

    model.zero_grad()
    with torch.no_grad():
        model.gamma.fill_(0.0)
    out, _ = model(cur, old, rect, rect, [2, 2], [1, 1])
    out.square().mean().backward()
    results['gate_gradient_at_zero'] = bool(model.gamma.grad is not None and
                                            model.gamma.grad.abs().item() > 0)
    model.zero_grad()
    with torch.no_grad():
        model.gamma.fill_(0.1)
    out, _ = model(cur, old, rect, rect, [2, 2], [1, 1])
    out.square().mean().backward()
    results['branch_gradient_after_gate'] = bool(model.k.weight.grad is not None and
                                                  model.k.weight.grad.abs().sum().item() > 0)
    results['finite'] = bool(torch.isfinite(out).all())
    results['parameter_count'] = sum(p.numel() for p in model.parameters())
    results['pass'] = all(results[k] for k in (
        'gamma_zero_identity', 'no_history_identity', 'nonzero_changes_output',
        'message_ratio_bounded', 'no_overlap_identity', 'future_frame_rejected',
        'gate_gradient_at_zero', 'branch_gradient_after_gate', 'finite'))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    result = check(torch.device(args.device))
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        with open(args.output, 'w') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
