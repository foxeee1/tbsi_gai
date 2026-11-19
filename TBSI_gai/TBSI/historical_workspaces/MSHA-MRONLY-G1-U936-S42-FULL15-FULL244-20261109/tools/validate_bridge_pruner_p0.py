#!/usr/bin/env python
"""P0 checks for the fixed-length bridge token pruner.

This is a numerical/integration diagnostic only. It does not train or test on
LasHeR and does not touch any canonical output directory.
"""

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.models.layers.tbsi_layer import TBSILayer


def max_abs(a, b):
    return float((a - b).abs().max().item())


def main():
    torch.manual_seed(42)
    dim, heads, batch, tokens, lens_z = 32, 4, 2, 20, 4
    x_v = torch.randn(batch, tokens, dim)
    x_i = torch.randn(batch, tokens, dim)

    common = dict(dim=dim, num_heads=heads, drop_path=0.0)
    base = TBSILayer(**common)
    disabled = TBSILayer(**common, use_bridge_token_pruner=False)
    identity = TBSILayer(
        **common,
        use_bridge_token_pruner=True,
        pruner_cfg={"dim": dim, "template_keep": 1.0, "search_keep": 1.0},
    )
    candidate = TBSILayer(
        **common,
        use_bridge_token_pruner=True,
        pruner_cfg={"dim": dim, "template_keep": 0.75, "search_keep": 0.75},
    )
    disabled.load_state_dict(base.state_dict())
    identity.load_state_dict(base.state_dict(), strict=False)
    candidate.load_state_dict(base.state_dict(), strict=False)

    base.eval()
    disabled.eval()
    identity.eval()
    candidate.train()

    with torch.no_grad():
        base_out = base(x_v.clone(), x_i.clone(), lens_z)
        disabled_out = disabled(x_v.clone(), x_i.clone(), lens_z)
        identity_out = identity(x_v.clone(), x_i.clone(), lens_z)
        candidate_eval_out = candidate(x_v.clone(), x_i.clone(), lens_z)

    parity_disabled = max(max_abs(base_out[0], disabled_out[0]),
                          max_abs(base_out[1], disabled_out[1]))
    parity_identity = max(max_abs(base_out[0], identity_out[0]),
                          max_abs(base_out[1], identity_out[1]))
    enabled_delta = max(max_abs(base_out[0], candidate_eval_out[0]),
                        max_abs(base_out[1], candidate_eval_out[1]))

    candidate.zero_grad(set_to_none=True)
    candidate_out = candidate(x_v.clone(), x_i.clone(), lens_z)
    loss = candidate_out[0].square().mean() + candidate_out[1].square().mean()
    loss.backward()
    pruner = candidate.bridge_token_pruner
    grad_norms = {
        name: float(param.grad.norm().item())
        for name, param in pruner.named_parameters()
        if param.grad is not None
    }

    result = {
        "seed": 42,
        "shape": [batch, tokens, dim],
        "lens_z": lens_z,
        "disabled_max_abs_diff": parity_disabled,
        "keep1_max_abs_diff": parity_identity,
        "enabled_max_abs_diff": enabled_delta,
        "pruner_gradient_norms": grad_norms,
        "pruner_parameter_count": sum(p.numel() for p in pruner.parameters()),
        "checks": {
            "disabled_parity": parity_disabled < 1e-6,
            "keep1_identity": parity_identity < 1e-6,
            "enabled_effect": enabled_delta > 1e-7,
            "nonzero_pruner_gradient": bool(grad_norms) and all(v > 0 for v in grad_norms.values()),
        },
    }
    result["status"] = "PASS" if all(result["checks"].values()) else "FAIL"

    out_dir = os.path.join(ROOT, "output", "experiments", "bridge_token_pruner_p0")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "p0_result.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
