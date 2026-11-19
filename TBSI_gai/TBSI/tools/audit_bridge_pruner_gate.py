#!/usr/bin/env python
"""Audit trained pruner gate behavior on a deterministic tensor probe."""
import argparse, json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lib.models.layers.bridge_token_pruner import BridgeTokenPruner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--keep', type=float, required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu')['net']
    prefix = 'backbone.tbsi_layers.2.bridge_token_pruner.'
    local = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    pruner = BridgeTokenPruner(dim=768, template_keep=args.keep, search_keep=args.keep,
                               hidden_dim=192, temperature=0.15)
    pruner.load_state_dict(local, strict=True)
    pruner.eval()
    torch.manual_seed(20260927)
    rows = []
    with torch.no_grad():
        for _ in range(1000):
            x = torch.randn(2, 320, 768)
            _, _, aux = pruner(x, x.clone(), 64)
            for name in ('rgb_template', 'rgb_search', 'tir_template', 'tir_search'):
                gate = aux[name]['gate']
                p = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
                entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1)
                rows.append({
                    'name': name,
                    'mean': float(gate.mean()),
                    'std': float(gate.std()),
                    'min': float(gate.min()),
                    'max': float(gate.max()),
                    'zero_fraction': float((gate <= 1e-6).float().mean()),
                    'one_fraction': float((gate >= 1 - 1e-6).float().mean()),
                    'entropy': float(entropy.mean()),
                })
    result = {'checkpoint': os.path.abspath(args.checkpoint), 'keep': args.keep,
              'res_scale': float(pruner.res_scale), 'alpha': float(torch.sigmoid(pruner.res_scale)),
              'probe_batches': 1000, 'groups': {}}
    for name in ('rgb_template', 'rgb_search', 'tir_template', 'tir_search'):
        subset = [r for r in rows if r['name'] == name]
        result['groups'][name] = {k: float(np.mean([r[k] for r in subset]))
                                  for k in ('mean','std','min','max','zero_fraction','one_fraction','entropy')}
    collapse = any(v['zero_fraction'] > 0.95 or v['one_fraction'] > 0.95
                   for v in result['groups'].values())
    result['gate_collapse'] = collapse
    result['status'] = 'FAIL_GATE_COLLAPSE' if collapse else 'PASS_NO_COLLAPSE'
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f: json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__': main()
