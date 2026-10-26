"""Small identity-initialized pre-CENTER gated fusion residual."""
import torch
from torch import nn


class PreCenterGatedResidual(nn.Module):
    """Add a bounded residual from modality disagreement before CENTER.

    Contract: all inputs are (B, C, H, W), output has the same shape.
    The last projection is zero initialized, so enabled construction is an
    exact identity until this module is trained.
    """

    def __init__(self, dim=768, hidden_dim=64, max_residual=0.10,
                 gate_bias=-4.0, force_zero=False, target_conditioned=False):
        super().__init__()
        self.max_residual = float(max_residual)
        self.force_zero = bool(force_zero)
        self.target_conditioned = bool(target_conditioned)
        hidden_dim = max(8, int(hidden_dim))
        input_dim = dim * (3 if self.target_conditioned else 2)
        self.residual = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, hidden_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, bias=True),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))

    def forward(self, fused, feat_rgb, feat_tir, target_context=None):
        if fused.shape != feat_rgb.shape or fused.shape != feat_tir.shape:
            raise RuntimeError('PreCenterGatedResidual input shape mismatch')
        if self.force_zero:
            return fused, {'gate_mean': 0.0, 'residual_norm': 0.0}
        disagreement = feat_rgb - feat_tir
        context = torch.cat((fused, disagreement), dim=1)
        if self.target_conditioned:
            if target_context is None:
                raise RuntimeError('target-conditioned residual requires target_context')
            if target_context.ndim != 2 or target_context.shape[0] != fused.shape[0] or target_context.shape[1] != fused.shape[1]:
                raise RuntimeError('target_context must have shape (B, C)')
            target_map = target_context[:, :, None, None].expand(-1, -1, fused.shape[2], fused.shape[3])
            context = torch.cat((context, target_map), dim=1)
        gate = torch.sigmoid(self.gate(context))
        residual = torch.tanh(self.residual(context)) * self.max_residual
        output = fused + gate * residual
        return output, {
            'gate_mean': float(gate.detach().mean()),
            'residual_norm': float(residual.detach().flatten(1).norm(dim=1).mean()),
        }
