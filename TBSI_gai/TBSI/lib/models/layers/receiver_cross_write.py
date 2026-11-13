"""Bounded receiver-side acceptance for an existing cross-modal message."""

import torch
from torch import nn
from torch.nn import functional as F


class ReceiverCrossWrite(nn.Module):
    """Scale an aggregated cross message using receiver compatibility."""

    def __init__(self, max_delta=0.10):
        super().__init__()
        if not 0.0 < float(max_delta) < 1.0:
            raise ValueError('receiver write max_delta must be in (0, 1)')
        self.max_delta = float(max_delta)
        self.action = nn.Parameter(torch.zeros(()))

    def reset_identity(self):
        with torch.no_grad():
            self.action.zero_()

    def forward(self, self_message, cross_message):
        if self_message.ndim != 4 or cross_message.ndim != 4:
            raise ValueError('messages must have shape [B,H,N,D]')
        if self_message.shape != cross_message.shape:
            raise ValueError('self and cross message shapes must match')
        self_token = self_message.float().transpose(1, 2).reshape(
            self_message.shape[0], self_message.shape[2], -1)
        cross_token = cross_message.float().transpose(1, 2).reshape(
            cross_message.shape[0], cross_message.shape[2], -1)
        compatibility = F.cosine_similarity(self_token, cross_token, dim=-1,
                                            eps=1e-6).unsqueeze(-1)
        scale = 1.0 + self.max_delta * torch.tanh(self.action) * compatibility
        return cross_message * scale.to(cross_message.dtype).unsqueeze(1), {
            'compatibility': compatibility.detach(),
            'scale': scale.detach(),
        }


def reset_receiver_cross_write(model):
    count = 0
    for module in model.modules():
        if isinstance(module, ReceiverCrossWrite):
            module.reset_identity()
            count += 1
    return count
