"""Minimal target-conditioned spatial gate for the CENTER input.

The gate is identity at initialization. Its score map is supervised by a
separate GT-vs-hard-background ranking loss in the training actor.
"""
import torch
import torch.nn.functional as F
from torch import nn


class TargetRankGate(nn.Module):
    def __init__(self, dim=768, hidden_dim=32, max_residual=0.10):
        super().__init__()
        self.dim = int(dim)
        self.max_residual = float(max_residual)
        self.score_head = nn.Sequential(
            nn.Conv2d(dim + 1, int(hidden_dim), 1), nn.GELU(),
            nn.Conv2d(int(hidden_dim), 1, 1))
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, fused, template_rgb, template_tir):
        if fused.ndim != 4 or fused.shape[1] != self.dim:
            raise ValueError('TargetRankGate fused input must be [B,C,H,W]')
        if template_rgb.ndim != 3 or template_tir.shape != template_rgb.shape:
            raise ValueError('TargetRankGate template input must be [B,N,C]')
        if template_rgb.shape[2] != self.dim:
            raise ValueError('TargetRankGate template channel mismatch')
        proto = F.normalize(0.5 * (template_rgb.mean(1) + template_tir.mean(1)), dim=1)
        sim = (F.normalize(fused, dim=1) * proto[:, :, None, None]).sum(1, keepdim=True)
        score = self.score_head(torch.cat((fused, sim), dim=1))
        score = score - score.mean(dim=(2, 3), keepdim=True)
        # The learned score is bounded and identity-preserving. alpha starts at 0.
        gate = torch.tanh(score)
        output = fused + self.alpha * self.max_residual * gate * fused
        return output, {'target_rank_score': score, 'target_rank_gate': gate,
                        'target_rank_alpha': self.alpha.detach()}

