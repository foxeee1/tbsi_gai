"""Small, identity-initialized post-fusion router for M1 diagnostics.

The router keeps the trained M1 feature as the default expert.  RGB/TIR
experts only contribute a bounded residual and are therefore safe to enable
for contract checks before any training is attempted.
"""
import torch
from torch import nn
import torch.nn.functional as F


class AdaptivePostFusionRouter(nn.Module):
    def __init__(self, dim, hidden_dim=64, max_residual=0.10,
                 force_zero=False, disagreement_aware=False,
                 gate_bias=-4.0):
        super().__init__()
        self.dim = int(dim)
        self.max_residual = float(max_residual)
        self.force_zero = bool(force_zero)
        self.disagreement_aware = bool(disagreement_aware)
        hidden_dim = max(8, int(hidden_dim))

        context_dim = dim * (4 if self.disagreement_aware else 3)
        self.context = nn.Sequential(
            nn.Conv2d(context_dim, hidden_dim, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.route = nn.Conv2d(hidden_dim, 3, 1, bias=True)
        self.need = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.rgb_residual = nn.Conv2d(hidden_dim, dim, 1, bias=True)
        self.tir_residual = nn.Conv2d(hidden_dim, dim, 1, bias=True)

        # Exact residual identity at initialization. Routing remains observable
        # for diagnostics but cannot change the parent output before training.
        nn.init.zeros_(self.rgb_residual.weight)
        nn.init.zeros_(self.rgb_residual.bias)
        nn.init.zeros_(self.tir_residual.weight)
        nn.init.zeros_(self.tir_residual.bias)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)
        nn.init.zeros_(self.need[-1].weight)
        nn.init.constant_(self.need[-1].bias, float(gate_bias))

    def forward(self, fused, feat_rgb, feat_tir):
        if self.force_zero:
            return fused, {'need_mean': 0.0, 'route_rgb_mean': 1 / 3,
                           'route_tir_mean': 1 / 3,
                           'residual_norm': 0.0}
        inputs = (feat_rgb, feat_tir, fused)
        if self.disagreement_aware:
            inputs = inputs + (torch.abs(feat_rgb - feat_tir),)
        context = self.context(torch.cat(inputs, dim=1))
        route = torch.softmax(self.route(context), dim=1)
        need = torch.sigmoid(self.need(context))
        rgb_delta = torch.tanh(self.rgb_residual(context)) * self.max_residual
        tir_delta = torch.tanh(self.tir_residual(context)) * self.max_residual
        residual = need * (route[:, 1:2] * rgb_delta +
                           route[:, 2:3] * tir_delta)
        output = fused + residual
        stats = {
            'need_mean': float(need.detach().mean()),
            'route_rgb_mean': float(route[:, 1:2].detach().mean()),
            'route_tir_mean': float(route[:, 2:3].detach().mean()),
            'residual_norm': float(residual.detach().flatten(1).norm(dim=1).mean()),
            'disagreement_aware': self.disagreement_aware,
        }
        return output, stats
