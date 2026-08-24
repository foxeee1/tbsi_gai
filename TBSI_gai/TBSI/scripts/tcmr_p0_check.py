#!/usr/bin/env python3
import sys
import torch

sys.path.insert(0, '/root/autodl-tmp/TBSI_gai/TBSI')
from lib.models.layers.tbsi_layer import TCMRRouter


def main():
    torch.manual_seed(42)
    router = TCMRRouter(dim=32)
    visible = torch.randn(2, 16, 32, requires_grad=True)
    infrared = torch.randn(2, 16, 32, requires_grad=True)
    fused, route = router(visible, infrared)
    assert fused.shape == visible.shape
    assert route.shape == (2, 16, 2)
    assert torch.isfinite(fused).all() and torch.isfinite(route).all()
    assert torch.allclose(route.sum(dim=-1), torch.ones(2, 16), atol=1e-6)
    loss = fused.square().mean() + route[..., 0].mean()
    loss.backward()
    gradients = [p.grad for p in router.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    entropy = -(route * route.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
    assert entropy > 0.5
    assert not torch.allclose(fused, visible)
    print('TCMR_P0_PASS')
    print('route_mean=%s route_entropy=%.6f' % (route.mean(dim=(0, 1)).tolist(), entropy))


if __name__ == '__main__':
    main()
