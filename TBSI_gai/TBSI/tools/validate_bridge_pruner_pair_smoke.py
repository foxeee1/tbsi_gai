#!/usr/bin/env python
"""Deterministic 1000-pair smoke for BridgeTokenPruner contracts."""
import json
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lib.models.layers.bridge_token_pruner import BridgeTokenPruner


def main():
    torch.manual_seed(3407)
    pairs = 1000
    batch, tokens, dim, lens_z = 2, 320, 768, 64
    pruner = BridgeTokenPruner(dim=dim, template_keep=0.75, search_keep=0.75,
                               hidden_dim=192, temperature=0.15)
    max_keep_error = 0.0
    max_finite_error = 0.0
    nonzero_steps = 0
    for _ in range(pairs):
        rgb = torch.randn(batch, tokens, dim)
        tir = torch.randn(batch, tokens, dim)
        out_rgb, out_tir, aux = pruner(rgb, tir, lens_z)
        loss = out_rgb.square().mean() + out_tir.square().mean()
        pruner.zero_grad(set_to_none=True)
        loss.backward()
        if all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in pruner.parameters()):
            nonzero_steps += 1
        for key in ("rgb_template", "rgb_search", "tir_template", "tir_search"):
            gate = aux[key]["gate"]
            keep = gate.mean(dim=1)
            target = 0.75
            max_keep_error = max(max_keep_error, float((keep - target).abs().max()))
            max_finite_error = max(max_finite_error, float((~torch.isfinite(gate)).sum()))
    result = {
        "pairs": pairs,
        "shape": [batch, tokens, dim],
        "lens_z": lens_z,
        "max_keep_mean_error": max_keep_error,
        "nonfinite_gate_count": int(max_finite_error),
        "all_pairs_nonzero_grad": nonzero_steps == pairs,
        "status": "PASS" if nonzero_steps == pairs and max_finite_error == 0 else "FAIL",
    }
    out_dir = os.path.join(ROOT, "output", "experiments", "bridge_token_pruner_pair_smoke")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pair_smoke_result.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
