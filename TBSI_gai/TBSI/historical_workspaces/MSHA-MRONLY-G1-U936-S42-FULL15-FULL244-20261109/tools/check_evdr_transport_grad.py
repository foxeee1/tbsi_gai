#!/usr/bin/env python3
"""Independent, bounded CPU contracts; no legacy admission or GPU code."""
import copy
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from lib.models.layers.early_conditioned_value_direction_aligned import (
    AttentionAlignedValueDirection, install_aligned_value_direction,
)
from lib.models.layers.early_conditioned_value_direction_transport_grad import (
    TransportGradientIsolatedValueDirection,
    install_transport_gradient_isolated_value_direction,
)
from lib.models.layers.tbsi_layer import V2TemplateFusion


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # This environment's CPU GELU backward misreads a noncontiguous upstream
    # gradient. Verify and localize the backend independently before applying
    # a layout-only hook in this checker. Production operators are untouched.
    x = torch.randn(2, 5, 3, 4, requires_grad=True)
    upstream = torch.randn(2, 3, 5, 4).permute(0, 2, 1, 3)
    expected = upstream * (0.5 * (1 + torch.erf(x / 2**0.5))
                           + x * torch.exp(-x*x/2) / (2*torch.pi)**0.5)
    direct = torch.autograd.grad(F.gelu(x), x, upstream)[0]
    contiguous = torch.autograd.grad(F.gelu(x), x, upstream.contiguous())[0]
    require(torch.allclose(contiguous, expected, atol=1e-6, rtol=1e-6), 'CPU contiguous GELU derivative control failed')
    cpu_backend = {'noncontiguous_derivative_matches_analytic': bool(torch.allclose(direct, expected, atol=1e-6, rtol=1e-6)),
                   'contiguous_derivative_matches_analytic': True,
                   'noncontiguous_maxabs_error': float((direct-expected).abs().max()),
                   'production_code_modified_for_backend': False}
    native_gelu = F.gelu
    def cpu_checked_gelu(tensor, *args, **kwargs):
        out = native_gelu(tensor, *args, **kwargs)
        if tensor.device.type == 'cpu' and out.requires_grad:
            out.register_hook(lambda gradient: gradient.contiguous())
        return out
    F.gelu = cpu_checked_gelu
    parent = AttentionAlignedValueDirection(48, 3, rank=4)
    candidate = TransportGradientIsolatedValueDirection(48, 3, rank=4)
    candidate.load_state_dict(parent.state_dict(), strict=True)
    v, src, rec = torch.randn(2, 3, 5, 16), torch.randn(2, 5, 48), torch.randn(2, 7, 48)
    evidence = torch.randn(2, 5, 3)
    attention = torch.rand(2, 3, 7, 5, requires_grad=True)
    zero = candidate(v, src, rec, evidence, attention)
    require(torch.equal(zero, v), 'zero-up identity failed')
    with torch.no_grad():
        parent.up.normal_(std=0.07)
    candidate.load_state_dict(parent.state_dict(), strict=True)
    yp = parent(v, src, rec, evidence, attention)
    yc = candidate(v, src, rec, evidence, attention)
    require(torch.equal(yp, yc), 'nonzero learned forward differs')
    weight = torch.randn_like(v)
    gp = torch.autograd.grad((yp * weight).sum(), attention, retain_graph=True)[0]
    gc = torch.autograd.grad((yc * weight).sum(), attention, allow_unused=True)[0]
    require(torch.isfinite(gp).all() and gp.abs().max() > 0, 'parent positive attention-gradient control failed')
    require(gc is None, 'candidate extra condition gradient remains connected')
    require(torch.equal(candidate(v, src, rec, evidence, attention, enabled=False), v), 'disabled path differs')
    zeros = candidate(torch.zeros_like(v), src, rec, evidence, torch.zeros_like(attention))
    require(torch.equal(zeros, torch.zeros_like(v)), 'zero Value/empty attention instability')
    norm_error = (yc.norm(dim=-1) - v.norm(dim=-1)).abs().max().item()
    require(norm_error < 2e-6, 'source norm changed')
    buffer = io.BytesIO()
    torch.save(candidate.state_dict(), buffer)
    buffer.seek(0)
    loaded = TransportGradientIsolatedValueDirection(48, 3, rank=4)
    loaded.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)
    require(torch.equal(loaded(v, src, rec, evidence, attention), yc), 'serialization parity failed')

    # Host-level check: detached context must not detach original QK readout.
    host = V2TemplateFusion(48, 3, early_conditioned_cross_gate=True)
    host_candidate = copy.deepcopy(host)
    rng = torch.get_rng_state().clone()
    install_aligned_value_direction(host, rank=4)
    require(torch.equal(torch.get_rng_state(), rng), 'parent installer consumed RNG')
    install_transport_gradient_isolated_value_direction(host_candidate, rank=4)
    require(torch.equal(torch.get_rng_state(), rng), 'candidate installer consumed RNG')
    require(list(host.state_dict()) == list(host_candidate.state_dict()), 'state keys differ')
    require(all(torch.equal(t, host_candidate.state_dict()[k]) for k, t in host.state_dict().items()), 'same-seed installed state differs')
    with torch.no_grad():
        host.value_direction.up.normal_(std=0.07)
    host_candidate.load_state_dict(host.state_dict(), strict=True)
    rgb, tir = torch.randn(2, 5, 48), torch.randn(2, 5, 48)
    context = (rgb, tir, torch.randn(2, 9, 48), torch.randn(2, 9, 48))
    hp = host(rgb, tir, context)
    hc = host_candidate(rgb, tir, context)
    require(all(torch.equal(a, b) for a, b in zip(hp, hc)), 'host forward not exact')
    target = [torch.randn_like(a) for a in hc]
    sum((a * w).sum() for a, w in zip(hp, target)).backward()
    sum((a * w).sum() for a, w in zip(hc, target)).backward()
    qk_grad = host_candidate.qkv.weight.grad[:96]
    require(torch.isfinite(qk_grad).all() and qk_grad.abs().max() > 0, 'original host QK path frozen')
    require(not torch.equal(host.qkv.weight.grad, host_candidate.qkv.weight.grad), 'single gradient intervention not visible')
    operator_grads = {name: float(p.grad.norm()) for name, p in host_candidate.value_direction.named_parameters() if p.grad is not None}
    require(len(operator_grads) == 5 and all(x > 0 for x in operator_grads.values()), 'new parameters not trainable after nonzero up')
    nonfinite = [name for name, p in host_candidate.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    require(not nonfinite, 'nonfinite host gradients: ' + repr(nonfinite))
    # Check task-like directional supervision can update the zero-up path.
    learner = TransportGradientIsolatedValueDirection(48, 3, rank=4)
    opt = torch.optim.AdamW(learner.parameters(), lr=1e-4)
    learning = []
    for step in range(2):
        opt.zero_grad()
        loss = (learner(v, src, rec, evidence, attention) * weight).mean()
        loss.backward()
        grads = {name: float(p.grad.norm()) for name, p in learner.named_parameters() if p.grad is not None}
        require(grads['up'] > 0 and all(torch.isfinite(p.grad).all() for p in learner.parameters() if p.grad is not None), 'zero-init update unreachable')
        if step == 1:
            require(all(x > 0 for x in grads.values()), 'predictor gradient not reached after up update')
        opt.step()
        learning.append({'step': step + 1, 'gradient_norms': grads})
    F.gelu = native_gelu
    result = {'scope': 'small synthetic CPU contracts with explicit GELU-backward layout hook in checker only; not real-GT training, not an efficacy test',
              'cpu_backend': cpu_backend,
              'device': 'cpu', 'threads': 1, 'pytorch': torch.__version__,
              'zero_init_exact': True, 'nonzero_forward_parent_exact': True,
              'disabled_exact': True, 'zero_value_empty_attention_exact': True,
              'source_norm_maxabs': norm_error, 'serialization_exact': True,
              'parent_extra_attention_gradient_maxabs': float(gp.abs().max()),
              'candidate_extra_attention_gradient': None,
              'host_forward_exact': True, 'host_original_qk_gradient_finite_nonzero': True,
              'host_gradient_change_observed': True, 'state_keys_and_initial_parameters_exact': True,
              'installers_preserve_rng': True, 'operator_gradients_after_nonzero_up': operator_grads,
              'two_synthetic_updates': learning, 'parameter_count': sum(p.numel() for p in candidate.parameters()),
              'added_parameters_vs_aligned': 0,
              'real_gt_check': 'pending resource-serial fresh candidate run; no previous training state reused',
              'performance': 'not measured'}
    out = ROOT / 'experiments/early_gate_closed_loop_20261026/evdr_transport_grad_20261027'
    (out / 'cpu_contract.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
