#!/usr/bin/env python3
"""Low-cost contract checks for the cross-mass-preserving relation module."""
import json
from pathlib import Path
import torch

from lib.models.layers.cross_mass_preserving_relation import (
    CrossMassPreservingReceiverSourceRelation,
)


def main():
    torch.manual_seed(42)
    b, h, nq, ns, d, e = 2, 3, 5, 7, 8, 3
    q = torch.randn(b, h, nq, d, requires_grad=True)
    k = torch.randn(b, h, ns, d, requires_grad=True)
    attention = torch.softmax(torch.randn(b, h, nq, ns), dim=-1)
    evidence = torch.randn(b, ns, e)
    module = CrossMassPreservingReceiverSourceRelation(evidence_index=1, max_bias=0.1)
    disabled_like, _ = module(q, k, attention, evidence)
    identity_error = float((disabled_like - attention).abs().max())
    before = attention.sum(-1)
    after = disabled_like.sum(-1)
    mass_error = float((before - after).abs().max())
    loss = disabled_like.square().mean()
    loss.backward()
    grad_finite = bool(module.beta.grad is not None and torch.isfinite(module.beta.grad).all())
    old = module.beta.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
    optimizer.step()
    updated = bool(not torch.equal(old, module.beta.detach()))
    payload = {'identity_error_at_beta_zero': identity_error,
               'cross_mass_max_error': mass_error, 'finite_beta_gradient': grad_finite,
               'beta_updated': updated, 'status': 'PASS' if identity_error < 1e-6 and
               mass_error < 1e-6 and grad_finite and updated else 'FAIL'}
    print(json.dumps(payload, indent=2))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
