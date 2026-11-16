"""Research prototype: measure where a source template token finds support.

This readout does not estimate the target region. ``search_support`` must be
supplied explicitly; a GT support map is suitable for an oracle diagnostic
only, never for deployment. No tracker imports or integration are provided.
"""
from typing import NamedTuple

import torch
from torch import nn


class SupportContrast(NamedTuple):
    target_affinity: torch.Tensor  # [B, Nsource], FP32
    background_affinity: torch.Tensor  # [B, Nsource], FP32
    margin: torch.Tensor  # target minus background; not calibrated reliability
    usable: torch.Tensor  # [B, Nsource], bool; numerical validity only


class TargetSupportContrast(nn.Module):
    """Parameter-free contrast over cosine similarities [B,Nsource,Nsearch].

    Search support [B,Nsearch] contains fractional region membership in [0,1].
    Foreground and its complement are normalized separately. An empty or full
    region produces an unusable readout, not an invented uniform target.
    """

    def forward(self, similarity, search_support):
        if similarity.ndim != 3 or min(similarity.shape) < 1:
            raise ValueError('similarity must be nonempty [B,Nsource,Nsearch]')
        if tuple(search_support.shape) != (similarity.shape[0], similarity.shape[2]):
            raise ValueError('search_support must be [B,Nsearch] on the search axis')
        if similarity.device != search_support.device:
            raise ValueError('similarity and support must share a device')
        sim, support = similarity.float(), search_support.float()
        if not torch.isfinite(sim).all() or not torch.isfinite(support).all():
            raise ValueError('similarity and support must be finite')
        if (sim.abs() > 1.00001).any() or ((support < 0) | (support > 1)).any():
            raise ValueError('expected cosine similarity and support membership in [0,1]')
        foreground_mass = support.sum(dim=-1, keepdim=True)
        background_mass = (1.0 - support).sum(dim=-1, keepdim=True)
        usable = (foreground_mass > 1.e-6) & (background_mass > 1.e-6)
        foreground_weights = support / foreground_mass.clamp_min(1.e-6)
        background_weights = (1.0 - support) / background_mass.clamp_min(1.e-6)
        foreground = (sim * foreground_weights[:, None]).sum(dim=-1)
        background = (sim * background_weights[:, None]).sum(dim=-1)
        usable = usable.expand_as(foreground)
        margin = torch.where(usable, foreground - background, torch.zeros_like(foreground))
        return SupportContrast(foreground, background, margin, usable)
