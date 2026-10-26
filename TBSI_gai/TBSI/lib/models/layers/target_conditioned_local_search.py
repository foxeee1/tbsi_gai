"""Minimal target-conditioned local cross-modal search adapter.

The class is not wired into TBSITrack until G0 and G1 pass. C1/C2 share this
exact module; C1 receives zero similarity maps and C2 receives live maps.
"""
import torch
from torch import nn
import torch.nn.functional as F

from lib.models.layers.local_attender import LocalAttender


class TargetConditionedLocalSearchAdapter(nn.Module):
    def __init__(self, dim=768, use_similarity=True, gamma_init=0.0):
        super().__init__()
        self.dim = dim
        self.use_similarity = bool(use_similarity)
        self.guide_rgb = nn.Conv2d(dim * 2 + 2, dim, kernel_size=1)
        self.guide_tir = nn.Conv2d(dim * 2 + 2, dim, kernel_size=1)
        self.value_rgb = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_tir = nn.Conv2d(dim, dim, kernel_size=1)
        self.output_rgb = nn.Conv2d(dim, dim, kernel_size=1)
        self.output_tir = nn.Conv2d(dim, dim, kernel_size=1)
        self.local_rgb = LocalAttender(dim, num_connected=5, conv_res=False)
        self.local_tir = LocalAttender(dim, num_connected=5, conv_res=False)
        self.gamma_rgb = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_tir = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def similarity(template, search):
        template = F.normalize(template.mean(dim=1), dim=-1)
        search = F.normalize(search, dim=-1)
        return torch.einsum('bc,bnc->bn', template, search)

    @staticmethod
    def to_map(tokens):
        b, n, c = tokens.shape
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError('search token count must be a square')
        return tokens.transpose(1, 2).reshape(b, c, side, side)

    def forward(self, template_rgb, template_tir, search_rgb, search_tir,
                similarity_enabled=None):
        use_similarity = self.use_similarity if similarity_enabled is None else bool(similarity_enabled)
        b, n, c = search_rgb.shape
        if search_tir.shape != search_rgb.shape:
            raise ValueError('RGB/TIR search shapes differ')
        p_rgb = self.similarity(template_rgb, search_rgb).reshape(b, 1, int(n ** 0.5), int(n ** 0.5))
        p_tir = self.similarity(template_tir, search_tir).reshape(b, 1, int(n ** 0.5), int(n ** 0.5))
        if not use_similarity:
            p_rgb = torch.zeros_like(p_rgb)
            p_tir = torch.zeros_like(p_tir)
        rgb = self.to_map(search_rgb)
        tir = self.to_map(search_tir)
        guide_rgb = self.guide_rgb(torch.cat([rgb, tir, p_rgb, p_tir], dim=1))
        guide_tir = self.guide_tir(torch.cat([tir, rgb, p_tir, p_rgb], dim=1))
        msg_rgb = self.output_rgb(self.local_rgb(guide_rgb, self.value_rgb(tir)))
        msg_tir = self.output_tir(self.local_tir(guide_tir, self.value_tir(rgb)))
        out_rgb = rgb + self.gamma_rgb * msg_rgb
        out_tir = tir + self.gamma_tir * msg_tir
        return out_rgb.flatten(2).transpose(1, 2), out_tir.flatten(2).transpose(1, 2), {
            'p_rgb': p_rgb.detach(), 'p_tir': p_tir.detach(),
            'msg_rgb': msg_rgb.detach(), 'msg_tir': msg_tir.detach(),
        }


__all__ = ['TargetConditionedLocalSearchAdapter']
