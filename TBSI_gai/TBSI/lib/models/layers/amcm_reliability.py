"""AMCM E2 legacy reliability modulation retained for the research branch."""

import torch
from torch import nn


class AMCMLiteReliability(nn.Module):
    """Spatially modulate each search stream from local context and template match."""

    def __init__(self, dim=768, hidden_dim=32, max_delta=0.10, mode='legacy'):
        super().__init__()
        if mode != 'legacy':
            raise ValueError("Only the retained AMCM E2 mode 'legacy' is supported")

        self.dim = int(dim)
        self.max_delta = float(max_delta)
        self.mode = 'legacy'
        self.context3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.context5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.context7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.context_mix = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.GELU())
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 3 + 2, max(8, int(hidden_dim)), 1), nn.GELU(),
            nn.Conv2d(max(8, int(hidden_dim)), 2, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _template_token_match(search, template):
        batch, channels, height, width = search.shape
        query = torch.nn.functional.normalize(search, dim=1).flatten(2).transpose(1, 2)
        key = torch.nn.functional.normalize(template, dim=-1)
        similarity = torch.bmm(query, key.transpose(1, 2))
        attention = torch.softmax(similarity / 0.07, dim=-1)
        return (attention * similarity).sum(dim=-1).view(
            batch, 1, height, width)

    def forward(self, search_rgb, search_tir, template_rgb, template_tir,
                template_fg_masks=None):
        if search_rgb.ndim != 4 or search_tir.shape != search_rgb.shape:
            raise ValueError('AMCM-lite search features must match [B,C,H,W]')
        if template_rgb.ndim != 3 or template_tir.shape != template_rgb.shape:
            raise ValueError('AMCM-lite template tokens must match [B,N,C]')
        if search_rgb.shape[1] != self.dim or template_rgb.shape[2] != self.dim:
            raise ValueError('AMCM-lite channel contract violated')

        rgb_context = self.context_mix(
            self.context3(search_rgb) + self.context5(search_rgb) +
            self.context7(search_rgb))
        tir_context = self.context_mix(
            self.context3(search_tir) + self.context5(search_tir) +
            self.context7(search_tir))
        rgb_match = self._template_token_match(search_rgb, template_rgb)
        tir_match = self._template_token_match(search_tir, template_tir)
        evidence = torch.cat((rgb_context, tir_context,
                              torch.abs(rgb_context - tir_context),
                              rgb_match, tir_match), dim=1)
        delta = torch.tanh(self.gate(evidence)) * self.max_delta
        out_rgb = search_rgb * (1.0 + delta[:, 0:1])
        out_tir = search_tir * (1.0 + delta[:, 1:2])
        return out_rgb, out_tir, {
            'rgb_gate_mean': float(delta[:, 0:1].detach().mean()),
            'tir_gate_mean': float(delta[:, 1:2].detach().mean()),
            'gate_abs_mean': float(delta.detach().abs().mean()),
            'rgb_gate_map': delta[:, 0:1].detach(),
            'tir_gate_map': delta[:, 1:2].detach(),
            'rgb_gate_map_live': delta[:, 0:1],
            'tir_gate_map_live': delta[:, 1:2],
            'rgb_sim_map_live': rgb_match,
            'tir_sim_map_live': tir_match,
        }
