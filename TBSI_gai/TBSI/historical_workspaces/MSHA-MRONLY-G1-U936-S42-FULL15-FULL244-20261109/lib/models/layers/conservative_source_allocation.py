"""Reciprocal source allocation for the existing Early gate logits."""
import torch
from torch import nn


class ConservativeSourceAllocation(nn.Module):
    """Center two source logits before the parent's bounded exponential action.

    With two sources, centered logits are opposites. Applying the same odd
    ``tanh`` and exponent therefore gives reciprocal gains per token, keeping
    source allocation competitive instead of allowing simultaneous amplification
    or suppression. The predictor and evidence remain unchanged.
    """

    def forward(self, gate_logits, gate_scale):
        if gate_logits.ndim != 3 or gate_logits.shape[-1] != 2:
            raise ValueError('gate_logits must have shape [B,N,2]')
        centered = gate_logits - gate_logits.mean(dim=-1, keepdim=True)
        action = torch.tanh(gate_scale) * torch.tanh(centered)
        return torch.exp(action).clamp(0.25, 4.0)
