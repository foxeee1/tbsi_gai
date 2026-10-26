#!/usr/bin/env python3
"""G1 P0 audit for the isolated target-conditioned local search adapter."""
import argparse
import json
import time
from pathlib import Path

import torch

from lib.models.layers.target_conditioned_local_search import TargetConditionedLocalSearchAdapter


def max_abs(a, b):
    return float((a - b).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    b, nt, ns, c = 2, 64, 256, 768
    template_rgb = torch.randn(b, nt, c, device=device)
    template_tir = torch.randn(b, nt, c, device=device)
    search_rgb = torch.randn(b, ns, c, device=device)
    search_tir = torch.randn(b, ns, c, device=device)
    model = TargetConditionedLocalSearchAdapter(dim=c, use_similarity=True).to(device)
    with torch.no_grad():
        model.gamma_rgb.fill_(0.1)
        model.gamma_tir.fill_(0.1)
    model.eval()
    with torch.no_grad():
        out_rgb, out_tir, info = model(template_rgb, template_tir, search_rgb, search_tir)
    zero_model = TargetConditionedLocalSearchAdapter(dim=c, use_similarity=True).to(device)
    zero_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        zero_model.gamma_rgb.zero_()
        zero_model.gamma_tir.zero_()
        zero_rgb, zero_tir, _ = zero_model(template_rgb, template_tir, search_rgb, search_tir)
    shape_ok = out_rgb.shape == search_rgb.shape and out_tir.shape == search_tir.shape
    finite = bool(torch.isfinite(out_rgb).all() and torch.isfinite(out_tir).all())
    parity = max(max_abs(zero_rgb, search_rgb), max_abs(zero_tir, search_tir))
    changed = max(max_abs(out_rgb, search_rgb), max_abs(out_tir, search_tir))

    train_model = TargetConditionedLocalSearchAdapter(dim=c, use_similarity=True).to(device)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4)
    grad_rows = []
    for step in range(2):
        optimizer.zero_grad()
        pred_rgb, pred_tir, _ = train_model(template_rgb, template_tir, search_rgb, search_tir)
        loss = (pred_rgb.square().mean() + pred_tir.square().mean())
        loss.backward()
        grad_rows.append({
            'step': step + 1,
            'loss': float(loss.item()),
            'gamma_rgb_grad': float(train_model.gamma_rgb.grad.abs().item()),
            'gamma_tir_grad': float(train_model.gamma_tir.grad.abs().item()),
            'finite_grads': all(p.grad is None or torch.isfinite(p.grad).all().item()
                                for p in train_model.parameters()),
        })
        optimizer.step()

    with torch.no_grad():
        swapped_rgb, swapped_tir, _ = model(template_tir, template_rgb, search_rgb, search_tir)
        shuffled = search_tir[:, torch.randperm(ns, device=device), :]
        shuffled_rgb, shuffled_tir, _ = model(template_rgb, template_tir, search_rgb, shuffled)
    result = {
        'status': 'PASS' if shape_ok and finite and parity == 0.0 and changed > 0 and
                  all(x['finite_grads'] for x in grad_rows) else 'FAIL',
        'device': str(device), 'input_shapes': {'template': [b, nt, c], 'search': [b, ns, c]},
        'operator': {'num_connected': 5, 'conv_res': False, 'search_grid': [16, 16]},
        'shape_ok': shape_ok, 'finite_forward': finite, 'gamma_zero_max_abs_diff': parity,
        'nonzero_gamma_output_max_abs_diff': changed,
        'swap_output_max_abs_diff': max(max_abs(swapped_rgb, out_rgb), max_abs(swapped_tir, out_tir)),
        'shuffle_output_max_abs_diff': max(max_abs(shuffled_rgb, out_rgb), max_abs(shuffled_tir, out_tir)),
        'gradient_steps': grad_rows,
        'elapsed_note': 'P0 synthetic tensors only; no tracker training or test data used',
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
