"""Target-conditioned five-role evidence for MR source tokens.

The module consumes the similarity matrices that the Early MSHA path already
computes.  It does not create another feature extractor or matching branch.
For each modality and template token it returns five bounded, interpretable
channels: foreground assignment, match strength, peak competition,
concentration, and target/background support.
"""
import math

import torch
from torch import nn

from .structured_spatial_evidence import box_to_patch_coverage


class FiveRoleSourceEvidence(nn.Module):
    """Build per-source-token five-role evidence for RGB and TIR.

    ``rgb_roles`` and ``tir_roles`` have shape ``[B, Nt, 5]``.  The channel
    order is ``foreground, strength, competition, concentration, support``.
    A detached robust normalization is deliberately avoided: normalization is
    part of the trainable computation and remains identical at train/test.
    """

    NUM_ROLES = 5

    def __init__(self, temperature=0.1, clamp=4.0):
        super().__init__()
        if temperature <= 0 or clamp <= 0:
            raise ValueError('temperature and clamp must be positive')
        self.temperature = float(temperature)
        self.clamp = float(clamp)

    @staticmethod
    def _normalize(x, dim=-1, clamp=4.0):
        mean = x.mean(dim=dim, keepdim=True)
        std = x.float().std(dim=dim, unbiased=False, keepdim=True).to(x.dtype)
        z = (x - mean) / std.clamp_min(1e-4)
        return z.clamp(-clamp, clamp)

    def _one_modality(self, similarity, coverage):
        b, nt, ns = similarity.shape
        sim = similarity.float()
        foreground = coverage.float().clamp(0.0, 1.0)
        background = (1.0 - foreground).clamp_min(0.0)
        fg_norm = foreground.sum(-1, keepdim=True).clamp_min(1e-6)
        bg_norm = background.sum(-1, keepdim=True).clamp_min(1e-6)

        values, _ = sim.sort(dim=-1, descending=True)
        peak = values[..., 0]
        second = values[..., 1]
        strength = self._normalize(peak, dim=1, clamp=self.clamp)
        competition = self._normalize(peak - second, dim=1, clamp=self.clamp)

        distribution = torch.softmax(sim / self.temperature, dim=-1)
        entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(-1)
        concentration = (1.0 - entropy / math.log(max(ns, 2))).clamp(-1.0, 1.0)

        fg_response = torch.einsum('bn,bns->bs', foreground, sim) / fg_norm
        bg_response = torch.einsum('bn,bns->bs', background, sim) / bg_norm
        targetness = fg_response - bg_response
        spatial = torch.softmax(targetness / self.temperature, dim=-1)
        token_support = torch.einsum('bs,bns->bn', spatial, sim)
        background_reference = torch.einsum('bs,bs->b', spatial, bg_response)
        support = self._normalize(token_support - background_reference[:, None],
                                  dim=1, clamp=self.clamp)

        roles = torch.stack((foreground, strength, competition, concentration, support), dim=-1)
        return roles.clamp(-self.clamp, self.clamp), spatial

    def forward(self, rgb_similarity, tir_similarity, template_boxes):
        if (rgb_similarity.ndim != 3 or tir_similarity.shape != rgb_similarity.shape or
                template_boxes is None or template_boxes.ndim != 3 or
                template_boxes.shape != (rgb_similarity.shape[0], 2, 4)):
            raise ValueError('expected matching similarities and template_boxes [B,2,4]')
        b, nt, ns = rgb_similarity.shape
        side = math.isqrt(nt)
        if side * side != nt or ns < 2:
            raise ValueError('template similarity axis must be a square grid and search >= 2')
        if template_boxes.device != rgb_similarity.device:
            raise ValueError('template_boxes and similarities must share device')
        coverage_rgb = box_to_patch_coverage(template_boxes[:, 0], side)
        coverage_tir = box_to_patch_coverage(template_boxes[:, 1], side)
        rgb_roles, rgb_spatial = self._one_modality(rgb_similarity, coverage_rgb)
        tir_roles, tir_spatial = self._one_modality(tir_similarity, coverage_tir)
        diagnostics = {
            'rgb_spatial': rgb_spatial.detach(),
            'tir_spatial': tir_spatial.detach(),
            'rgb_roles': rgb_roles.detach(),
            'tir_roles': tir_roles.detach(),
        }
        return rgb_roles, tir_roles, diagnostics
