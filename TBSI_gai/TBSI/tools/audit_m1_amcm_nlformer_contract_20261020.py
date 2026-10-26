#!/usr/bin/env python3
"""Contract audit for the official NLformer core at the AMCM boundary."""
import json
from pathlib import Path
import torch

from lib.models.layers.amcm_reliability import AMCMLiteReliability


def main():
    torch.manual_seed(20261020)
    device = torch.device('cpu')
    c = 768
    module = AMCMLiteReliability(c, hidden_dim=32, mode='nlformer').to(device)
    module.eval()
    rgb = torch.randn(1, c, 16, 16, device=device, requires_grad=True)
    tir = torch.randn(1, c, 16, 16, device=device, requires_grad=True)
    trgb = torch.randn(1, 64, c, device=device, requires_grad=True)
    ttir = torch.randn(1, 64, c, device=device, requires_grad=True)

    with torch.no_grad():
        y0r, y0t, _ = module(rgb, tir, trgb, ttir)
    identity = max((y0r - rgb).abs().max().item(), (y0t - tir).abs().max().item())

    module.nlformer_alpha.data.fill_(0.1)
    yr, yt, stats = module(rgb, tir, trgb, ttir)
    loss = yr.square().mean() + yt.square().mean()
    loss.backward()
    names = ('nlformer_core.qkv.weight', 'nlformer_core.qkv_c.weight',
             'nlformer_core.phi.weight', 'nlformer_template.weight',
             'nlformer_alpha')
    grads = {n: float(dict(module.named_parameters())[n].grad.abs().mean()) for n in names}
    residual = torch.cat((yr - rgb, yt - tir), dim=1).detach()
    spatial_std = float(residual.mean(dim=1).std())
    state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    restored = AMCMLiteReliability(c, hidden_dim=32, mode='nlformer').to(device)
    restored.load_state_dict(state)
    with torch.no_grad():
        rr, rt, _ = restored(rgb.detach(), tir.detach(), trgb.detach(), ttir.detach())
    reload_error = max((rr - yr.detach()).abs().max().item(),
                       (rt - yt.detach()).abs().max().item())
    result = {
        'module': 'NLformer Attention_block adapted at AMCM boundary',
        'input_shapes': [[1, c, 16, 16], [1, c, 16, 16], [1, 64, c], [1, 64, c]],
        'output_shapes': [list(yr.shape), list(yt.shape)],
        'parameter_count': sum(p.numel() for p in module.parameters()),
        'identity_max_abs_error': identity,
        'enabled_residual_l2': float(residual.norm()),
        'enabled_spatial_std': spatial_std,
        'gradient_mean_abs': grads,
        'reload_max_abs_error': reload_error,
        'branch_norms': {
            k: float(stats[k].detach().norm())
            for k in ('nlformer_local_live', 'nlformer_channel_live',
                      'nlformer_complementary_live')
        },
        'status': 'PASS' if identity == 0.0 and reload_error < 1e-5 and
                  all(v > 0.0 for v in grads.values()) and spatial_std > 0.0
                  else 'FAIL',
    }
    out = Path(__file__).resolve().parents[1] / 'output/experiments/M1_AMCM_NLFORMER_CONTRACT_AUDIT_20261020'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'contract_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
