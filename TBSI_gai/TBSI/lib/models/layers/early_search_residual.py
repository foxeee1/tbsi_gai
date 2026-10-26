import torch
from torch import nn

from lib.models.layers.scsu import SpatialChannelSelection


class EarlySearchResidual(nn.Module):
    """Per-modality early-search residual with identity-preserving scalar gates."""

    def __init__(self, dim=768, force_zero=False, use_scsu=False,
                 scsu_k1=3, scsu_k2=5):
        super().__init__()
        self.rgb_proj = nn.Linear(dim, dim, bias=False)
        self.tir_proj = nn.Linear(dim, dim, bias=False)
        self.gamma_rgb = nn.Parameter(torch.zeros(()))
        self.gamma_tir = nn.Parameter(torch.zeros(()))
        self.force_zero = bool(force_zero)
        self.use_scsu = bool(use_scsu)
        self.scsu = (SpatialChannelSelection(dim, k1=int(scsu_k1), k2=int(scsu_k2))
                     if self.use_scsu else nn.Identity())

    def forward(self, late_rgb, late_tir, early_rgb, early_tir):
        if late_rgb.shape != early_rgb.shape or late_tir.shape != early_tir.shape:
            raise ValueError("early residual requires matching RGB/TIR token shapes")
        if late_rgb.shape[-1] != self.rgb_proj.in_features:
            raise ValueError("early residual channel dimension mismatch")
        if self.force_zero:
            return late_rgb, late_tir
        if self.use_scsu:
            b, n, c = early_rgb.shape
            side = int(n ** 0.5)
            if side * side != n:
                raise ValueError("SCSU requires a square search token grid")
            def select(tokens):
                fmap = tokens.transpose(1, 2).reshape(b, c, side, side)
                return self.scsu(fmap).flatten(2).transpose(1, 2)
            early_rgb = select(early_rgb)
            early_tir = select(early_tir)
        return (late_rgb + self.gamma_rgb * self.rgb_proj(early_rgb),
                late_tir + self.gamma_tir * self.tir_proj(early_tir))
