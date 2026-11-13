"""Value-aware source reallocation with a conserved cross-attention budget."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class StableLinear(nn.Module):
    """Linear projection kept out of the host ViT's generic Linear initializer."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class ReceiverSourceMassReallocator(nn.Module):
    def __init__(self, dim=768, context_dim=64, rank=32, max_delta=0.25):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.rank = int(rank)
        self.max_delta = float(max_delta)
        self.receiver_norm = nn.LayerNorm(dim)
        self.receiver = StableLinear(dim, rank)
        self.source_norm = nn.LayerNorm(dim)
        self.source = StableLinear(dim, rank)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context = StableLinear(context_dim, rank)
        self.evidence = StableLinear(3, rank)
        self.readout = nn.Sequential(
            StableLinear(3 * rank, rank, bias=True), nn.GELU(),
            StableLinear(rank, 1, bias=True))
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, gated_attention, receiver_self, source_value, source_context,
                source_evidence):
        if gated_attention.ndim != 4:
            raise ValueError('gated_attention must be [B,H,Nq,Ns]')
        b, h, nq, ns = gated_attention.shape
        if receiver_self.shape != (b, nq, self.dim):
            raise ValueError('receiver_self token axes or width mismatch')
        if source_value.shape != (b, h, ns, self.dim // h):
            raise ValueError('source_value must be [B,H,Ns,D_head]')
        if source_context.shape != (b, ns, self.context_dim):
            raise ValueError('source_context shape mismatch')
        if source_evidence.shape != (b, ns, 3):
            raise ValueError('source_evidence shape mismatch')

        source_merged = source_value.transpose(1, 2).reshape(b, ns, self.dim)
        receiver = self.receiver(self.receiver_norm(receiver_self.float()))
        source = self.source(self.source_norm(source_merged.float()))
        context = self.context(self.context_norm(source_context.float()))
        context = context + self.evidence(source_evidence.float())
        interaction = (receiver[:, :, None, :] * source[:, None, :, :]) / math.sqrt(self.rank)
        pair = torch.cat((interaction, context[:, None].expand(-1, nq, -1, -1)), dim=-1)
        pair = torch.cat((pair, receiver[:, :, None].expand(-1, -1, ns, -1)), dim=-1)
        delta = self.max_delta * torch.tanh(self.readout(pair).squeeze(-1))

        mass = gated_attention.sum(dim=-1, keepdim=True)
        eps = torch.finfo(gated_attention.dtype).tiny
        active = gated_attention > 0
        logits = torch.log(gated_attention.float().clamp_min(eps)) + delta[:, None]
        logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        any_active = active.any(dim=-1, keepdim=True)
        conditional = torch.softmax(logits, dim=-1)
        conditional = torch.where(any_active, conditional, torch.zeros_like(conditional))
        return (mass.float() * conditional).to(gated_attention.dtype)


def reset_receiver_source_mass_reallocator(model):
    count = 0
    for module in model.modules():
        if isinstance(module, ReceiverSourceMassReallocator):
            module.reset_identity()
            count += 1
    return count
