"""CPU contract checks for current-layer search-conditioned Value repair.

This script validates implementation and learning reachability only. It does
not run tracking and cannot support an efficacy claim.
"""
import argparse
import copy
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from lib.models.layers.tbsi_layer import V2TemplateFusion
from lib.models.layers.early_conditioned_msha_gate import early_msha_gate_evidence
from lib.models.layers.search_conditioned_value_direction import (
    SearchConditionedValueDirection,
    install_search_conditioned_value_direction,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    report = {'scope': 'CPU synthetic contracts only; no tracking efficacy',
              'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_build': torch.version.cuda, 'checks': {}}
    checks = report['checks']
    b, n, ns, c, h = 2, 4, 9, 32, 4
    rgb, tir = torch.randn(b, n, c), torch.randn(b, n, c)
    early = (torch.randn(b, n, c), torch.randn(b, n, c),
             torch.randn(b, ns, c), torch.randn(b, ns, c))
    current = tuple(x.clone() for x in early)
    fusion = V2TemplateFusion(c, h, qkv_bias=True,
                              early_conditioned_cross_gate=True).eval()
    common = {k: v.clone() for k, v in fusion.state_dict().items()}
    rng = torch.get_rng_state().clone()
    parent = torch.cat(fusion(rgb, tir, early_context=early), 1)
    install_search_conditioned_value_direction(fusion, rank=4, max_step=0.10)
    require(torch.equal(rng, torch.get_rng_state()), 'installer changed CPU RNG')
    require(all(torch.equal(v, fusion.state_dict()[k]) for k, v in common.items()),
            'installer changed common parent state')
    zero = torch.cat(fusion(rgb, tir, early_context=early,
                            fusion_context=current), 1)
    require(torch.equal(parent, zero), 'zero-init host path is not exact identity')
    checks['zero_init_parity'] = 'PASS; bitwise CPU fusion equality'

    op = SearchConditionedValueDirection(c, h, rank=4, max_step=0.10)
    values = torch.randn(b, h, n, c // h)
    values[0, 0, 0] = 0
    receiver_attention = torch.randn(b, h, n, n).softmax(dim=-1)
    evidence = torch.randn(b, n, 3)
    target = torch.randn_like(values)
    optimizer = torch.optim.AdamW(op.parameters(), lr=0.01, weight_decay=0)
    initial_up_grad = None
    for step in range(3):
        optimizer.zero_grad()
        output = op(values, rgb, tir, evidence, receiver_attention,
                    current[3])
        ((output - target) ** 2).mean().backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all()
                    for p in op.parameters()), 'missing or nonfinite gradient')
        if step == 0:
            initial_up_grad = op.up.grad.norm().item()
            require(initial_up_grad > 0, 'zero up blocks the first gradient')
        optimizer.step()
    output = op(values, rgb, tir, evidence, receiver_attention, current[3])
    norms = values.norm(dim=-1)
    norm_error = (output.norm(dim=-1) - norms).abs().max().item()
    relative_delta = ((output - values).norm(dim=-1) /
                      norms.clamp_min(1.e-10)).max().item()
    require(norm_error < 2.e-6, 'Value norm is not preserved')
    require(0 < relative_delta <= 0.10001, 'bounded direction step violated')
    require(torch.equal(output[0, 0, 0], values[0, 0, 0]),
            'zero Value is not stable')
    checks['geometry_and_learning'] = dict(
        norm_error=norm_error, max_relative_delta=relative_delta,
        initial_up_grad=initial_up_grad,
        note='synthetic target; not true GT overfit; 3 CPU updates')

    loaded = SearchConditionedValueDirection(c, h, rank=4, max_step=0.10)
    loaded.load_state_dict(op.state_dict(), strict=True)
    require(torch.equal(output, loaded(values, rgb, tir, evidence,
                                       receiver_attention, current[3])),
            'save/load output mismatch')
    require(torch.equal(op(values, None, None, None, None, None, enabled=False),
                        values), 'disabled operator is not identity')
    checks['save_load_disabled'] = 'PASS; stateless disabled path'

    fusion.value_direction.load_state_dict(op.state_dict())
    calls = []
    handle = fusion.value_direction.register_forward_hook(
        lambda _module, inputs, output: calls.append((inputs[0].clone(), output.clone())))
    changed = torch.cat(fusion(rgb, tir, early_context=early,
                               fusion_context=current), 1)
    handle.remove()
    require(len(calls) == 2, 'operator must run once per source modality')
    require(not torch.equal(changed, parent), 'enabled operator has no host effect')
    require(torch.equal(current[0], early[0]) and torch.equal(current[1], early[1]),
            'test fixture unexpectedly changed template condition')
    checks['host_scope'] = 'PASS; two cross-source calls, no direct search write'
    report['status'] = 'PASS-CPU-CONTRACT / NO-TRACKING-EFFICACY'
    report['parameters_dim768_heads12_rank8'] = sum(
        p.numel() for p in SearchConditionedValueDirection(768, 12).parameters())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
