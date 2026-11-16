"""Default-off, parameter-free cross-source attenuation research prototype.

Use only on the cross-modal copy of V, before the parent's Early gate. The
caller owns attention and same-modal V. This module cannot ensure that its
input is the cross copy; the host wiring still needs a separate check.
"""
import math

import torch
from torch import nn


class CrossSourceAttenuation(nn.Module):
    """Suppress negative target-support contrast without changing V direction.

    No claim of aggregate message norm conservation or normal-scene safety is
    implied. A mistaken target support can suppress useful information.
    """

    def __init__(self, max_attenuation=0.25, enabled=False):
        super().__init__()
        if not math.isfinite(max_attenuation) or not 0 <= max_attenuation <= 1:
            raise ValueError('max_attenuation must be finite and in [0,1]')
        self.max_attenuation = float(max_attenuation)
        self.enabled = bool(enabled)

    def coefficients(self, margin, usable):
        if margin.ndim != 2 or usable.shape != margin.shape or usable.dtype != torch.bool:
            raise ValueError('margin and boolean usable must be [B,Nsource]')
        if margin.device != usable.device or not torch.isfinite(margin).all():
            raise ValueError('margin/usable must share device and margin must be finite')
        if not self.enabled or self.max_attenuation == 0:
            return torch.ones_like(margin, dtype=torch.float32)
        # New control evidence is a read-only measurement. The original Early
        # gate and backbone still share the task loss: this is NOT a claim of
        # independent optimization or elimination of gradient conflict.
        bad_support = (-margin.detach().float() / 2.0).clamp(0.0, 1.0)
        bad_support = torch.where(usable, bad_support, torch.zeros_like(bad_support))
        return 1.0 - self.max_attenuation * bad_support

    def forward(self, cross_values, margin, usable):
        if not self.enabled or self.max_attenuation == 0:
            return cross_values
        if cross_values.ndim != 4 or tuple(margin.shape) != (cross_values.shape[0], cross_values.shape[2]):
            raise ValueError('cross Values must be [B,H,Nsource,D], margin [B,Nsource]')
        if margin.device != cross_values.device:
            raise ValueError('Values and margin must share a device')
        factor = self.coefficients(margin, usable).to(cross_values.dtype)
        return cross_values * factor[:, None, :, None]
