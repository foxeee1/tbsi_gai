"""Evidence-independent source amplitude calibration for joint template attention."""
import torch
import torch.nn as nn


class FixedSourceValueGate(nn.Module):
    """Learn one static gain per source modality, shared across its tokens."""

    def __init__(self):
        super().__init__()
        self.source_logits = nn.Parameter(torch.zeros(2))

    def forward(self, batch_size, token_count, reference):
        gain_logits = self.source_logits.to(dtype=reference.dtype)
        return gain_logits.view(1, 1, 2).expand(batch_size, token_count, 2)
