"""Small frozen-M1 adapters for screening MSHA-side fusion hypotheses."""
import torch
from torch import nn
import torch.nn.functional as F


class MshaReliabilityAdapter(nn.Module):
    """Local second-order RGB/TIR agreement gate with identity initialization."""
    def __init__(self, dim=768, hidden_dim=32, max_residual=0.10, gate_bias=-4.0):
        super().__init__()
        self.max_residual = float(max_residual)
        hidden_dim = max(8, int(hidden_dim))
        # fused, disagreement, first-order agreement, second-order agreement
        self.body = nn.Sequential(
            nn.Conv2d(dim + dim + 2, hidden_dim, 1), nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))

    def forward(self, fused, feat_rgb, feat_tir):
        if fused.shape != feat_rgb.shape or fused.shape != feat_tir.shape:
            raise RuntimeError('MshaReliabilityAdapter input shape mismatch')
        eps = 1e-6
        rgb_n = F.normalize(feat_rgb, dim=1, eps=eps)
        tir_n = F.normalize(feat_tir, dim=1, eps=eps)
        first = (rgb_n * tir_n).mean(dim=1, keepdim=True)
        # Channel covariance proxy: agreement of centered channel responses.
        rgb_c = feat_rgb - feat_rgb.mean(dim=1, keepdim=True)
        tir_c = feat_tir - feat_tir.mean(dim=1, keepdim=True)
        second = F.normalize(rgb_c, dim=1, eps=eps) * F.normalize(tir_c, dim=1, eps=eps)
        second = second.mean(dim=1, keepdim=True)
        disagreement = feat_rgb - feat_tir
        context = torch.cat((fused, disagreement, first, second), dim=1)
        gate = torch.sigmoid(self.gate(torch.cat((first, second), dim=1)))
        residual = torch.tanh(self.body(context)) * self.max_residual
        return fused + gate * residual, {'gate_mean': float(gate.detach().mean()),
                                         'residual_norm': float(residual.detach().flatten(1).norm(dim=1).mean()),
                                         'first_agreement': float(first.detach().mean()),
                                         'second_agreement': float(second.detach().mean())}


class TargetDynamicLocalOperator(nn.Module):
    """Target-conditioned local depthwise operator with a zero-init output path."""
    def __init__(self, dim=768, hidden_dim=32, max_residual=0.10):
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.max_residual = float(max_residual)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.condition = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        self.project = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, fused, target_context):
        if target_context is None or target_context.ndim != 2 or target_context.shape[1] != fused.shape[1]:
            raise RuntimeError('TargetDynamicLocalOperator requires target_context with shape (B, C)')
        scale = torch.tanh(self.condition(target_context))[:, :, None, None]
        local = self.local(fused) * (1.0 + scale)
        residual = torch.tanh(self.project(local)) * self.max_residual
        return fused + residual, {'scale_norm': float(scale.detach().flatten(1).norm(dim=1).mean()),
                                  'residual_norm': float(residual.detach().flatten(1).norm(dim=1).mean())}


class TargetComplementaryAdaptiveFusion(nn.Module):
    """Target-conditioned complementary branch with an explicit noop gate.

    The original dynamic local operator modifies every frame. This variant
    keeps the M1 fused feature as the default and admits a bounded residual
    only when target similarity and RGB/TIR disagreement support it.
    """
    def __init__(self, dim=768, hidden_dim=32, max_residual=0.10, gate_bias=-4.0):
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.max_residual = float(max_residual)
        self.branch = nn.Sequential(
            nn.Conv2d(dim * 2 + 1, hidden_dim, 1), nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.admission = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)
        nn.init.zeros_(self.admission[-1].weight)
        nn.init.constant_(self.admission[-1].bias, float(gate_bias))

    def forward(self, fused, feat_rgb, feat_tir, target_context):
        if target_context is None or target_context.ndim != 2:
            raise RuntimeError('TargetComplementaryAdaptiveFusion requires target context')
        if fused.shape != feat_rgb.shape or fused.shape != feat_tir.shape:
            raise RuntimeError('TargetComplementaryAdaptiveFusion input shape mismatch')
        eps = 1e-6
        ctx = F.normalize(target_context, dim=1, eps=eps)[:, :, None, None]
        fused_n = F.normalize(fused, dim=1, eps=eps)
        target_sim = (fused_n * ctx).sum(dim=1, keepdim=True)
        disagreement = F.normalize(feat_rgb - feat_tir, dim=1, eps=eps)
        disagreement_mag = (feat_rgb - feat_tir).abs().mean(dim=1, keepdim=True)
        # Candidate branch is explicitly complementary to the base fusion.
        branch = torch.tanh(self.branch(torch.cat((fused, feat_rgb - feat_tir,
                                                    target_sim), dim=1)))
        gate = torch.sigmoid(self.admission(torch.cat((target_sim, disagreement_mag), dim=1)))
        residual = branch * self.max_residual
        out = fused + gate * residual
        return out, {
            'gate_mean': float(gate.detach().mean()),
            'gate_std': float(gate.detach().std()),
            'target_similarity': float(target_sim.detach().mean()),
            'disagreement': float(disagreement_mag.detach().mean()),
            'residual_norm': float(residual.detach().flatten(1).norm(dim=1).mean()),
        }
