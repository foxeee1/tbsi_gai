"""Current-layer search-conditioned cross-modal Value direction repair.

The operator keeps the parent Q/K attention and Early-gate magnitude path
unchanged.  It uses the receiver modality's current search tokens to build a
local condition for each template source token, then applies a bounded,
norm-preserving residual to the cross-source Value.  The module is installed
only through the explicit ``VALUE_DIRECTION.RECEIVER_MODE=search`` switch.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SearchConditionedValueDirection(nn.Module):
    """Condition cross-source Value content on the current receiver crop."""

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
        self.search_query = nn.Linear(dim, num_heads * rank, bias=False)
        self.search_key = nn.Linear(dim, num_heads * rank, bias=False)
        self.search_value = nn.Linear(dim, num_heads * rank, bias=False)
        self.condition = nn.Linear(3, num_heads * rank, bias=True)
        # A zero output transform gives exact identity at construction.
        self.up = nn.Parameter(torch.zeros(num_heads, rank, dim // num_heads))

    def reset_identity(self):
        with torch.no_grad():
            self.up.zero_()

    @staticmethod
    def _transport_template(receiver_attention, receiver_code):
        # receiver_attention: [B,H,Nreceiver,Nsource]
        weights = receiver_attention.float()
        weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(1.e-12)
        return torch.einsum('bhns,bhnr->bhsr', weights, receiver_code.float())

    def forward(self, values, source_tokens, receiver_tokens, evidence,
                receiver_attention, receiver_search_tokens, mode='direction',
                enabled=True):
        if not enabled:
            return values
        if mode != 'direction':
            raise ValueError('search-conditioned Value repair only supports direction mode')
        if values.ndim != 4 or source_tokens.ndim != 3 or receiver_tokens.ndim != 3:
            raise ValueError('expected Values [B,H,Nsource,D] and tokens [B,N,C]')
        bsz, heads, source_len, width = values.shape
        if heads != self.num_heads or width * heads != self.dim:
            raise ValueError('Value head contract differs')
        if tuple(source_tokens.shape) != (bsz, source_len, self.dim):
            raise ValueError('source token contract differs')
        if (receiver_tokens.shape[0] != bsz or receiver_tokens.shape[1] != source_len or
                receiver_tokens.shape[2] != self.dim):
            raise ValueError('receiver template token contract differs')
        if (receiver_search_tokens.ndim != 3 or receiver_search_tokens.shape[0] != bsz or
                receiver_search_tokens.shape[2] != self.dim):
            raise ValueError('receiver search token contract differs')
        if tuple(evidence.shape) != (bsz, source_len, 3):
            raise ValueError('evidence must be [B,Nsource,3]')
        if tuple(receiver_attention.shape) != (bsz, heads, source_len, source_len):
            raise ValueError('receiver attention must be [B,H,Nreceiver,Nsource]')
        inputs = (source_tokens, receiver_tokens, receiver_search_tokens, evidence,
                  receiver_attention)
        if any(t.device != values.device for t in inputs):
            raise ValueError('all operator inputs must share device')

        src = self.source(self.norm(source_tokens)).reshape(
            bsz, source_len, heads, self.rank)
        receiver_code = self.receiver(self.norm(receiver_tokens)).reshape(
            bsz, source_len, heads, self.rank).permute(0, 2, 1, 3)

        # The existing template cross-attention transports receiver template
        # evidence to the exact source token that consumes the message.
        aligned_template = self._transport_template(receiver_attention, receiver_code)

        # A second, current-layer relation reads the receiver search crop.
        # It is a condition only: it does not replace or edit Q/K attention.
        search = self.norm(receiver_search_tokens)
        search_key = self.search_key(search).reshape(
            bsz, -1, heads, self.rank).permute(0, 2, 1, 3)
        search_value = self.search_value(search).reshape(
            bsz, -1, heads, self.rank).permute(0, 2, 1, 3)
        search_query = self.search_query(self.norm(source_tokens)).reshape(
            bsz, source_len, heads, self.rank).permute(0, 2, 1, 3)
        with torch.cuda.amp.autocast(enabled=False):
            search_logits = torch.einsum(
                'bhqr,bhkr->bhqk', search_query.float(), search_key.float())
            search_weights = (search_logits / math.sqrt(self.rank)).softmax(dim=-1)
            aligned_search = torch.einsum(
                'bhqk,bhkr->bhqr', search_weights, search_value.float())
        aligned_search = aligned_search.permute(0, 2, 1, 3)
        cond = self.condition(evidence.to(src.dtype)).reshape(
            bsz, source_len, heads, self.rank)
        code = F.gelu(src) * (1.0 + torch.tanh(
            aligned_template.permute(0, 2, 1, 3).to(src.dtype) +
            aligned_search.to(src.dtype) + cond))
        raw = torch.einsum('bnhr,hrd->bhnd', code, self.up)

        # Compute the geometry in FP32.  The tangent update preserves each
        # source Value norm and has a bounded relative step.
        v, raw = values.float(), raw.float()
        norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        unit = v / safe_norm
        tangent = raw - (raw * unit).sum(dim=-1, keepdim=True) * unit
        tangent_sq = (tangent * tangent).sum(dim=-1, keepdim=True)
        step = self.max_step * norm * tangent / torch.sqrt(1.0 + tangent_sq)
        moved = v + step
        moved_norm = torch.linalg.vector_norm(moved, dim=-1, keepdim=True)
        safe_moved = torch.where(moved_norm > 0, moved_norm,
                                 torch.ones_like(moved_norm))
        return (moved * (norm / safe_moved)).to(values.dtype)


def install_search_conditioned_value_direction(model, rank=8, max_step=0.10):
    """Attach the operator after parent initialization and before optimizer creation."""
    targets = [m for m in model.modules()
               if m.__class__.__name__ == 'V2TemplateFusion']
    if not targets or any(not m.early_conditioned_cross_gate for m in targets):
        raise ValueError('Search-conditioned Value repair requires an Early gate parent')
    if any(hasattr(m, 'value_direction') for m in targets):
        raise ValueError('Value direction is already installed')
    for module in targets:
        dim, heads = module.qkv.in_features, module.num_heads
        if dim % heads or not 0 < rank <= dim // heads:
            raise ValueError('invalid rank/head contract')
    devices = sorted({m.qkv.weight.device.index for m in targets
                      if m.qkv.weight.is_cuda})
    with torch.random.fork_rng(devices=devices):
        for module in targets:
            operator = SearchConditionedValueDirection(
                module.qkv.in_features, module.num_heads, rank, max_step)
            operator.to(device=module.qkv.weight.device,
                        dtype=module.qkv.weight.dtype)
            operator.train(module.training)
            module.add_module('value_direction', operator)
            module.value_direction_enabled = True
            module.value_direction_search_enabled = True
            module.value_direction_mode = 'direction'
    return len(targets)
