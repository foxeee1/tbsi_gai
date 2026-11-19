"""Attention-aligned receiver conditioning for the Early-gate Value repair.

This is the single-variable EVDR-v2 intervention.  EVDR-v1 conditions a
source Value with the receiver token at the same index.  Here the receiver
representation is transported through the already computed cross-attention
map, so every source token receives the receiver context that actually
contributes to its message.  The Value geometry and parameterization are
otherwise unchanged.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class AttentionAlignedValueDirection(nn.Module):
    def __init__(self, dim, num_heads, rank=8, max_step=0.10):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError('dim must be divisible by a positive head count')
        if not 0 < rank <= dim // num_heads:
            raise ValueError('rank must be in [1, head_dim]')
        if not math.isfinite(max_step) or not 0 < max_step <= 0.25:
            raise ValueError('max_step must be finite and in (0,0.25]')
        self.dim, self.num_heads = int(dim), int(num_heads)
        self.rank, self.max_step = int(rank), float(max_step)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.source = nn.Linear(dim, num_heads * rank, bias=False)
        self.receiver = nn.Linear(dim, num_heads * rank, bias=False)
        self.condition = nn.Linear(3, num_heads * rank, bias=True)
        self.up = nn.Parameter(torch.zeros(num_heads, rank, dim // num_heads))

    def reset_identity(self):
        with torch.no_grad():
            self.up.zero_()

    def forward(self, values, source_tokens, receiver_tokens, evidence,
                receiver_attention, mode='direction', enabled=True):
        if not enabled:
            return values
        if mode != 'direction':
            raise ValueError('aligned receiver operator only supports direction mode')
        if values.ndim != 4 or source_tokens.ndim != 3 or receiver_tokens.ndim != 3:
            raise ValueError('expected Values [B,H,Ns,D] and tokens [B,N,C]')
        bsz, heads, source_len, width = values.shape
        receiver_len = receiver_tokens.shape[1]
        expected_source = (bsz, source_len, self.dim)
        if heads != self.num_heads or width * heads != self.dim:
            raise ValueError('Value head contract differs')
        if tuple(source_tokens.shape) != expected_source or receiver_tokens.shape[0] != bsz or receiver_tokens.shape[2] != self.dim:
            raise ValueError('source/receiver token contract differs')
        if tuple(evidence.shape) != (bsz, source_len, 3):
            raise ValueError('evidence must be [B,Nsource,3]')
        if tuple(receiver_attention.shape) != (bsz, heads, receiver_len, source_len):
            raise ValueError('receiver attention must be [B,H,Nreceiver,Nsource]')
        if any(t.device != values.device for t in
               (source_tokens, receiver_tokens, evidence, receiver_attention)):
            raise ValueError('all operator inputs must share device')

        src = self.source(self.norm(source_tokens)).reshape(bsz, source_len, heads, self.rank)
        receiver_code = self.receiver(self.norm(receiver_tokens)).reshape(
            bsz, receiver_len, heads, self.rank).permute(0, 2, 1, 3)
        # Transport receiver features over the exact QK cross-attention map.
        # Normalization is over receiver rows for each source column; this is
        # a context representation and does not alter the attention weights.
        with torch.cuda.amp.autocast(enabled=False):
            weights = receiver_attention.float()
            weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(1.e-12)
            aligned = torch.einsum('bhns,bhnr->bhsr', weights, receiver_code.float())
        aligned = aligned.permute(0, 2, 1, 3)
        cond = self.condition(evidence.to(src.dtype)).reshape(
            bsz, source_len, heads, self.rank)
        code = F.gelu(src) * (1.0 + torch.tanh(aligned.to(src.dtype) + cond))
        raw = torch.einsum('bnhr,hrd->bhnd', code, self.up)
        v, raw = values.float(), raw.float()
        norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        unit = v / safe_norm
        tangent = raw - (raw * unit).sum(dim=-1, keepdim=True) * unit
        tangent_sq = (tangent * tangent).sum(dim=-1, keepdim=True)
        step = self.max_step * norm * tangent / torch.sqrt(1.0 + tangent_sq)
        moved = v + step
        moved_norm = torch.linalg.vector_norm(moved, dim=-1, keepdim=True)
        safe_moved = torch.where(moved_norm > 0, moved_norm, torch.ones_like(moved_norm))
        return (moved * (norm / safe_moved)).to(values.dtype)


def install_aligned_value_direction(model, rank=8, max_step=0.10):
    targets = [m for m in model.modules()
               if m.__class__.__name__ == 'V2TemplateFusion']
    if not targets or any(not m.early_conditioned_cross_gate for m in targets):
        raise ValueError('Aligned Value direction requires an explicit Early gate parent')
    if any(hasattr(m, 'value_direction') for m in targets):
        raise ValueError('Value direction is already installed')
    for module in targets:
        dim, heads = module.qkv.in_features, module.num_heads
        if dim % heads or not 0 < rank <= dim // heads:
            raise ValueError('invalid rank/head contract')
    devices = sorted({m.qkv.weight.device.index for m in targets if m.qkv.weight.is_cuda})
    with torch.random.fork_rng(devices=devices):
        for module in targets:
            operator = AttentionAlignedValueDirection(
                module.qkv.in_features, module.num_heads, rank, max_step)
            operator.to(device=module.qkv.weight.device, dtype=module.qkv.weight.dtype)
            operator.train(module.training)
            module.add_module('value_direction', operator)
            module.value_direction_enabled = True
            module.value_direction_aligned_enabled = True
            module.value_direction_mode = 'direction'
    return len(targets)
