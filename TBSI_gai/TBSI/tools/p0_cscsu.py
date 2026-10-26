#!/usr/bin/env python3
"""P0 contract checks for the exploratory SCSU early-search residual."""
import json
from pathlib import Path

import torch

from lib.models.layers.early_search_residual import EarlySearchResidual


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b, n, c = 2, 256, 768
    late_rgb = torch.randn(b, n, c, device=device)
    late_tir = torch.randn(b, n, c, device=device)
    early_rgb = torch.randn(b, n, c, device=device)
    early_tir = torch.randn(b, n, c, device=device)
    module = EarlySearchResidual(dim=c, use_scsu=True).to(device).train()

    zero = EarlySearchResidual(dim=c, force_zero=True, use_scsu=True).to(device)
    zero.load_state_dict(module.state_dict())
    zero.force_zero = True
    with torch.no_grad():
        zero_rgb, zero_tir = zero(late_rgb, late_tir, early_rgb, early_tir)
    zero_diff = max((zero_rgb - late_rgb).abs().max().item(),
                    (zero_tir - late_tir).abs().max().item())

    out_rgb, out_tir = module(late_rgb, late_tir, early_rgb, early_tir)
    finite = bool(torch.isfinite(out_rgb).all() and torch.isfinite(out_tir).all())
    loss = (out_rgb.square().mean() + out_tir.square().mean())
    loss.backward()
    grad_names = [n for n, p in module.named_parameters()
                  if p.requires_grad and p.grad is not None]
    finite_grads = all(torch.isfinite(p.grad).all() for p in module.parameters()
                       if p.requires_grad and p.grad is not None)
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    result = {
        'device': str(device), 'input_shape': [b, n, c],
        'zero_gate_max_abs_diff': zero_diff, 'zero_gate_passed': zero_diff == 0.0,
        'enabled_output_finite': finite, 'gradient_names': grad_names,
        'gradient_nonempty': bool(grad_names), 'gradient_finite': bool(finite_grads),
        'trainable_parameters': trainable,
        'passed': bool(zero_diff == 0.0 and finite and grad_names and finite_grads),
        'note': 'Synthetic operator contract only; full tracker parity is checked after Cscsu inference.'
    }
    out = Path('/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'p0_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
