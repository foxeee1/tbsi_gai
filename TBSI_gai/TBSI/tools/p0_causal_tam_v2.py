#!/usr/bin/env python
"""Strict P0 checks for the causal TAM v2 contract."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch

from lib.models.layers.causal_tam_adapter import CausalTemporalFusionResidual


def run(device):
    torch.manual_seed(42)
    model = CausalTemporalFusionResidual(dim=8, hidden=8, heads=2, grid_size=4,
                                        local_radius=1, max_message_ratio=0.10).to(device)
    cur = torch.randn(2, 16, 8, device=device)
    old = torch.randn_like(cur)
    same = torch.tensor([[0., 0., 160., 160.], [0., 0., 160., 160.]], device=device)
    far = same.clone(); far[1, 0] += 1000.
    result = {}

    with torch.no_grad():
        model.gamma.fill_(0.1)
    out, stats = model(cur, old, same, far, [2, 2], [1, 1], ['s', 's'], ['s', 's'])
    result['mixed_batch_no_overlap_identity'] = bool(torch.equal(out[1], cur[1]))
    result['mixed_batch_overlap_changes'] = bool((out[0] - cur[0]).abs().max().item() > 0)
    result['mixed_batch_second_coverage'] = float(stats['coverage'])

    encoded = torch.arange(16, device=device, dtype=torch.float32).view(1, 16, 1).repeat(1, 1, 8)
    aligned, valid = model._align_history(encoded, same[:1], same[:1])
    result['same_crop_valid_fraction'] = float(valid.float().mean().item())
    result['same_crop_alignment_max_error'] = float((aligned - encoded).abs().max().item())

    with torch.no_grad():
        model.gamma.fill_(1.0)
    out_partial, stats_partial = model(cur[:1], old[:1], same[:1],
                                        torch.tensor([[0., 0., 80., 160.]], device=device),
                                        [2], [1], ['s'], ['s'])
    invalid = ~model._align_history(old[:1], same[:1],
                                    torch.tensor([[0., 0., 80., 160.]], device=device))[1].squeeze(0)
    result['partial_invalid_queries_identity'] = bool(torch.equal(out_partial[0, invalid], cur[0, invalid]))
    result['partial_valid_fraction'] = float(stats_partial['query_usable_fraction'])

    ratios = []
    for gamma in (0.1, 1.0, 5.0):
        with torch.no_grad(): model.gamma.fill_(gamma)
        _, s = model(cur[:1], old[:1], same[:1], same[:1], [2], [1], ['s'], ['s'])
        ratios.append(float(s['actual_ratio_max']))
    result['actual_ratio_max_by_gamma'] = ratios
    result['actual_ratio_bounded'] = bool(max(ratios) <= 0.10 + 1e-6)

    model.zero_grad(); model.gamma.data.zero_()
    out, _ = model(cur[:1], old[:1], same[:1], same[:1], [2], [1], ['s'], ['s'])
    out.square().mean().backward()
    result['gate_gradient_at_zero'] = bool(model.gamma.grad is not None and model.gamma.grad.abs().item() > 0)
    model.zero_grad()
    with torch.no_grad(): model.gamma.fill_(0.1)
    out, _ = model(cur[:1], old[:1], same[:1], same[:1], [2], [1], ['s'], ['s'])
    out.square().mean().backward()
    result['qkv_out_gradients_finite_nonzero'] = all(
        getattr(model, name).weight.grad is not None and
        torch.isfinite(getattr(model, name).weight.grad).all() and
        getattr(model, name).weight.grad.abs().sum().item() > 0
        for name in ('q', 'k', 'v', 'out'))

    try:
        model(cur[:1], old[:1], same[:1], same[:1], [2], [1], ['a'], ['b'])
        result['cross_sequence_rejected'] = False
    except ValueError:
        result['cross_sequence_rejected'] = True
    result['finite'] = bool(torch.isfinite(out).all())
    result['pass'] = all(result[k] for k in (
        'mixed_batch_no_overlap_identity', 'mixed_batch_overlap_changes',
        'partial_invalid_queries_identity', 'actual_ratio_bounded',
        'gate_gradient_at_zero', 'qkv_out_gradients_finite_nonzero',
        'cross_sequence_rejected', 'finite'))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    result = run(torch.device(args.device))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f: json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
