"""Receiver-conditioned residual correction for an already aggregated MR message."""

import torch
from torch import nn
from torch.nn import functional as F


class ReceiverConditionedMessageResidual(nn.Module):
    """Apply a bounded bilinear correction after cross-Value aggregation.

    The module has an exact identity initialization: the final readout is zero,
    so enabling the branch cannot alter a checkpoint until the new parameters
    learn.  It consumes the receiver template token and the actual cross
    message; it does not alter attention weights or cross mass.
    """

    def __init__(self, dim=768, rank=16, max_delta=0.1):
        super().__init__()
        if int(dim) <= 0 or int(rank) <= 0:
            raise ValueError("dim and rank must be positive")
        if not 0.0 < float(max_delta) <= 1.0:
            raise ValueError("max_delta must be in (0, 1]")
        self.dim = int(dim)
        self.rank = int(rank)
        self.max_delta = float(max_delta)
        self.receiver_norm = nn.LayerNorm(self.dim)
        self.message_norm = nn.LayerNorm(self.dim)
        self.receiver_proj = nn.Linear(self.dim, self.rank, bias=False)
        self.message_proj = nn.Linear(self.dim, self.rank, bias=False)
        self.readout = nn.Linear(self.rank, self.dim, bias=True)
        nn.init.xavier_uniform_(self.receiver_proj.weight)
        nn.init.xavier_uniform_(self.message_proj.weight)
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, receiver_tokens, cross_message):
        if receiver_tokens.ndim != 3:
            raise ValueError("receiver_tokens must be [B,N,D]")
        if cross_message.ndim != 4:
            raise ValueError("cross_message must be [B,H,N,Dh]")
        b, heads, tokens, head_dim = cross_message.shape
        if receiver_tokens.shape != (b, tokens, self.dim):
            raise ValueError("receiver/message token axes or width mismatch")
        if heads * head_dim != self.dim:
            raise ValueError("message head width does not match dim")

        message = cross_message.transpose(1, 2).reshape(b, tokens, self.dim)
        receiver_latent = self.receiver_proj(self.receiver_norm(receiver_tokens.float()))
        message_latent = self.message_proj(self.message_norm(message.float()))
        interaction = receiver_latent * message_latent
        correction = self.readout(F.gelu(interaction))
        scale = message.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        correction = self.max_delta * torch.tanh(correction) * scale
        corrected = (message.float() + correction).to(cross_message.dtype)
        return corrected.reshape(b, tokens, heads, head_dim).transpose(1, 2)


def reset_receiver_conditioned_message_residual(model):
    count = 0
    for module in model.modules():
        if isinstance(module, ReceiverConditionedMessageResidual):
            module.reset_identity()
            count += 1
    return count
