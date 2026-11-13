"""Optional gradient isolation for Early gate evidence."""
import torch
from torch import nn


class EarlyGateEvidenceGradientIsolation(nn.Module):
    """Keep evidence values while stopping their gradient to upstream features."""

    def forward(self, evidence):
        if not torch.is_tensor(evidence) or evidence.ndim != 3:
            raise ValueError('early gate evidence must be a rank-3 tensor')
        return evidence.detach()
