"""Predict crop-level modality utility from early RGB/TIR features."""

import torch
from torch import nn
import torch.nn.functional as F


class EarlyUtilityRouter(nn.Module):
    """Shared Stage-2 router for MSHA cross-modal Value messages."""

    def __init__(self, dim=768, descriptor_dim=32, hidden_dim=64):
        super().__init__()
        self.dim = int(dim)
        self.descriptor_dim = int(descriptor_dim)
        self.descriptor_projection = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.descriptor_dim),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(6 * self.descriptor_dim + 4, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    @staticmethod
    def _masked_mean(tokens, weights):
        weights = weights.to(device=tokens.device, dtype=tokens.dtype).clamp(0.0, 1.0)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _cosine(a, b):
        return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-6)

    def forward(self, early_context, fg_masks):
        if early_context is None or len(early_context) != 4:
            raise ValueError('EarlyUtilityRouter requires four early feature tensors')
        if fg_masks is None or len(fg_masks) != 2:
            raise ValueError('EarlyUtilityRouter requires RGB/TIR template foreground masks')

        rgb_t, tir_t, rgb_s, tir_s = early_context
        if any(x.ndim != 3 for x in (rgb_t, tir_t, rgb_s, tir_s)):
            raise ValueError('EarlyUtilityRouter features must have shape [B,N,C]')
        if rgb_t.shape != tir_t.shape or rgb_s.shape != tir_s.shape:
            raise ValueError('EarlyUtilityRouter RGB/TIR feature shapes must match')
        if rgb_t.shape[0] != rgb_s.shape[0] or rgb_t.shape[2] != self.dim:
            raise ValueError('EarlyUtilityRouter batch or channel contract mismatch')
        if rgb_t.shape[1] != 64 or rgb_s.shape[1] != 256:
            raise ValueError('EarlyUtilityRouter expects 64 template and 256 search tokens')
        expected_mask = (rgb_t.shape[0], rgb_t.shape[1])
        if fg_masks[0].shape != expected_mask or fg_masks[1].shape != expected_mask:
            raise ValueError('EarlyUtilityRouter foreground masks must match template tokens')

        rgb_fg_w = fg_masks[0].float().clamp(0.0, 1.0)
        tir_fg_w = fg_masks[1].float().clamp(0.0, 1.0)
        rgb_bg_w, tir_bg_w = 1.0 - rgb_fg_w, 1.0 - tir_fg_w
        rgb_fg = self._masked_mean(rgb_t, rgb_fg_w)
        rgb_bg = self._masked_mean(rgb_t, rgb_bg_w)
        tir_fg = self._masked_mean(tir_t, tir_fg_w)
        tir_bg = self._masked_mean(tir_t, tir_bg_w)
        rgb_search = rgb_s.mean(dim=1)
        tir_search = tir_s.mean(dim=1)

        descriptors = [rgb_fg, rgb_bg, rgb_search, tir_fg, tir_bg, tir_search]
        projected = [self.descriptor_projection(x.float()) for x in descriptors]
        relations = torch.stack((
            self._cosine(rgb_fg, rgb_search),
            self._cosine(tir_fg, tir_search),
            self._cosine(rgb_fg, tir_fg),
            self._cosine(rgb_search, tir_search),
        ), dim=1)
        router_input = torch.cat(projected + [relations], dim=1)
        return self.predictor(router_input.to(dtype=rgb_t.dtype))
