"""Small CPU contract checks, not tracking efficacy or real-sample fitting."""
import argparse
import copy
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from lib.models.layers.tbsi_layer import V2TemplateFusion, TBSILayer
from lib.models.layers.early_conditioned_msha_gate import early_msha_gate_evidence
from lib.models.layers.early_conditioned_value_direction import (
    EarlyConditionedValueDirection, install_value_direction,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    report = {'scope': 'CPU synthetic tensor contracts only; no tracking claim',
              'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_build': torch.version.cuda, 'checks': {}}
    checks = report['checks']
    b, n, c, h = 2, 4, 32, 4
    rgb, tir = torch.randn(b, n, c), torch.randn(b, n, c)
    context = (torch.randn_like(rgb), torch.randn_like(tir),
               torch.randn(b, 9, c), torch.randn(b, 9, c))
    fusion = V2TemplateFusion(c, h, qkv_bias=True,
                              early_conditioned_cross_gate=True).eval()
    # Nonuniform gates make a source/query broadcasting bug observable.
    with torch.no_grad():
        fusion.early_gate[2].weight.normal_(std=1)
        fusion.early_gate[2].bias.copy_(torch.tensor([0.8, -0.4]))
    x = torch.cat((rgb, tir), 1)
    qkv = fusion.qkv(fusion.norm(x)).reshape(b, 2*n, 3, h, c//h)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)
    a = ((q @ k.transpose(-1, -2)) * fusion.scale).softmax(-1)
    evidence, _, _ = early_msha_gate_evidence(context, rgb, tir)
    gate = torch.exp(torch.tanh(fusion.early_gate_scale) *
                     torch.tanh(fusion.early_gate(evidence))).clamp(0.25, 4)
    vr, vt = v[:, :, :n], v[:, :, n:]
    mr = a[:, :, :n, :n] @ vr + a[:, :, :n, n:] @ (vt * gate[:, None, :, 1:2])
    mt = a[:, :, n:, n:] @ vt + a[:, :, n:, :n] @ (vr * gate[:, None, :, 0:1])
    expected = x + fusion.proj(torch.cat((mr, mt), 2).transpose(1, 2).reshape(b, 2*n, c))
    actual = torch.cat(fusion(rgb, tir, early_context=context), 1)
    require(torch.equal(expected, actual), 'source-axis formula mismatch')
    query_wrong = a[:, :, :n, :n] @ vr + (a[:, :, :n, n:] * gate[:, None, :, 1:2]) @ vt
    require(not torch.allclose(mr, query_wrong, atol=1e-6), 'gate fixture not discriminative')
    checks['source_gate_axis'] = 'PASS; nonuniform gate rejects query-axis variant'

    # Installing the candidate must preserve both parent state and RNG.
    common = {k: v.clone() for k, v in fusion.state_dict().items()}
    rng = torch.get_rng_state().clone()
    install_value_direction(fusion, rank=4)
    require(torch.equal(rng, torch.get_rng_state()), 'installer changed RNG')
    require(all(torch.equal(v, fusion.state_dict()[k]) for k, v in common.items()),
            'installer changed common parameters or buffers')
    zero = torch.cat(fusion(rgb, tir, early_context=context), 1)
    require(torch.equal(actual, zero), 'zero-init is not exact at fusion boundary')
    checks['common_state_rng_zero_parity'] = 'PASS; bitwise CPU fusion equality'
    fusion.value_direction_enabled = False
    require(torch.equal(actual, torch.cat(fusion(rgb, tir, early_context=context), 1)),
            'disabled path changed parent')
    fusion.value_direction_enabled = True
    # Generic host Linear init must not destroy the identity output transform.
    fusion.value_direction.apply(lambda m: nn.init.normal_(m.weight)
                                if isinstance(m, nn.Linear) else None)
    require(torch.count_nonzero(fusion.value_direction.up).item() == 0,
            'generic Linear initializer overwrote zero up')
    checks['outer_linear_init_identity'] = 'PASS'

    op = EarlyConditionedValueDirection(c, h, rank=4)
    values = torch.randn(b, h, n, c//h)
    values[0, 0, 0] = 0
    target = torch.randn_like(values)
    optimizer = torch.optim.AdamW(op.parameters(), lr=0.01, weight_decay=0)
    first_up_grad = None
    for step in range(3):
        optimizer.zero_grad()
        y = op(values, rgb, tir, evidence)
        ((y-target)**2).mean().backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all()
                    for p in op.parameters()), 'missing or nonfinite gradient')
        if step == 0:
            first_up_grad = op.up.grad.norm().item()
            require(first_up_grad > 0, 'zero initialization blocks up gradient')
        optimizer.step()
    require(all(p.grad.norm().item() > 0 for p in op.parameters()),
            'predictor did not receive gradients after up began learning')
    y = op(values, rgb, tir, evidence)
    norms = values.norm(dim=-1)
    norm_error = (y.norm(dim=-1) - norms).abs().max().item()
    relative_delta = ((y-values).norm(dim=-1)/norms.clamp_min(1e-10)).max().item()
    require(norm_error < 2e-6, 'Value norm not preserved')
    require(0 < relative_delta <= 0.10001, 'direction step bound violated or inactive')
    require(torch.equal(y[0, 0, 0], values[0, 0, 0]), 'zero Value not stable')
    require(torch.isfinite(y).all(), 'nonfinite output')
    checks['geometry_and_learning'] = dict(norm_error=norm_error,
        max_relative_delta=relative_delta, initial_up_grad=first_up_grad,
        note='synthetic target; not true GT overfit; 3 successful CPU updates')
    radial = op(values, rgb, tir, evidence, mode='radial')
    cosine = torch.nn.functional.cosine_similarity(values, radial, dim=-1)
    require(torch.allclose(cosine[norms > 0], torch.ones_like(cosine[norms > 0]), atol=2e-6),
            'radial control changes content direction')
    require(torch.equal(op(values, None, None, None, enabled=False), values),
            'disabled operator is not identity')
    loaded = EarlyConditionedValueDirection(c, h, rank=4)
    loaded.load_state_dict(op.state_dict(), strict=True)
    require(torch.equal(y, loaded(values, rgb, tir, evidence)), 'save/load mismatch')
    require(len(dict(op.named_buffers())) == 0, 'unexpected running state')
    checks['radial_save_load_disabled_stateless'] = 'PASS'
    # Use trained/nonzero parameters in host integration to expose scope bugs.
    fusion.value_direction.load_state_dict(op.state_dict())
    seen = []
    def observe(_module, inputs, output):
        seen.append((inputs[0].detach().clone(), output.detach().clone()))
    handle = fusion.value_direction.register_forward_hook(observe)
    with torch.no_grad():
        changed = torch.cat(fusion(rgb, tir, early_context=context), 1)
    handle.remove()
    require(len(seen) == 2, 'operator must run once per source modality')
    cv_rgb, cv_tir = seen[0][1], seen[1][1]
    mr = a[:, :, :n, :n] @ vr + a[:, :, :n, n:] @ (cv_tir * gate[:, None, :, 1:2])
    mt = a[:, :, n:, n:] @ vt + a[:, :, n:, :n] @ (cv_rgb * gate[:, None, :, 0:1])
    scoped = x + fusion.proj(torch.cat((mr, mt), 2).transpose(1, 2).reshape(b, 2*n, c))
    require(torch.equal(changed, scoped), 'candidate altered QK/self path')
    require(not torch.equal(changed, actual), 'enabled candidate has no host effect')
    checks['cross_only_scope_nonzero'] = 'PASS; independent four-message formula'

    layer = TBSILayer(c, h, fusion_mode='tbsi_e',
                      early_conditioned_cross_gate=True).eval()
    require(all(not hasattr(layer,name) for name in
        ('t_fusion','ca_s2t_i2f','ca_s2t_v2f','ca_t2s_f2v',
         'ca_t2s_f2i','ca_t2t_f2v','ca_t2t_f2i')),
        'TBSI-E must not construct dead legacy bridge state')
    srgb, stir = torch.randn(b, 9, c), torch.randn(b, 9, c)
    out_r, out_t = layer(torch.cat((rgb,srgb),1), torch.cat((tir,stir),1),
                         n, early_context=context)
    require(torch.equal(out_r[:, n:],srgb) and torch.equal(out_t[:, n:],stir),
            'TBSI-E changed search tokens directly')
    require(layer.get_smsa_stats() is None, 'TBSI-E diagnostics reached legacy state')
    checks['tbsi_e_template_only_entry'] = 'PASS; six legacy bridges absent'
    report['status'] = 'PASS-CPU-CONTRACT / NO-TRACKING-EFFICACY'
    report['parameters_per_dim768_heads12_rank8'] = sum(
        p.numel() for p in EarlyConditionedValueDirection(768,12).parameters())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
