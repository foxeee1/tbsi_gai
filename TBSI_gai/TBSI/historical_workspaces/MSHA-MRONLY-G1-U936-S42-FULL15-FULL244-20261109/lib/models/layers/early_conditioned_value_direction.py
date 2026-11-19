"""Experimental cross-only Value direction repair; never enabled implicitly.

The early gate controls magnitude. This operator tests a separate hypothesis:
recipient-conditioned content directions help at a fixed per-head Value norm.
No search writeback, Q/K change, new loss, foreground-mask fallback or BN state.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class EarlyConditionedValueDirection(nn.Module):
    """Shared source/recipient low-rank bilinear correction.

    tokens: [B,N,C], evidence: [B,N,3] in RGB/TIR/agreement order,
    values: [B,H,N,D]. RGB/TIR roles are explicit at the call site.
    Early evidence is reused from the parent; no GT is consumed.
    """

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
        """Use only for a fresh candidate after outer init, before optimization.

        Never call after loading a trained candidate checkpoint.
        The up transform is a Parameter, so a generic nn.Linear initializer
        cannot overwrite its zero initialization.
        """
        with torch.no_grad():
            self.up.zero_()

    def forward(self, values, source_tokens, receiver_tokens, evidence,
                mode='direction', enabled=True):
        if not enabled:
            return values
        if mode not in ('direction', 'radial'):
            raise ValueError('mode must be direction or radial')
        if values.ndim != 4 or source_tokens.ndim != 3:
            raise ValueError('expected values [B,H,N,D] and tokens [B,N,C]')
        bsz, heads, length, width = values.shape
        expected = (bsz, length, self.dim)
        if (heads != self.num_heads or width * heads != self.dim or
                tuple(source_tokens.shape) != expected or
                tuple(receiver_tokens.shape) != expected or
                tuple(evidence.shape) != (bsz, length, 3)):
            raise ValueError('source, receiver, evidence and Value contracts differ')
        if any(t.device != values.device for t in
               (source_tokens, receiver_tokens, evidence)):
            raise ValueError('all operator inputs must be on the same device')
        src = self.source(self.norm(source_tokens))
        rec = self.receiver(self.norm(receiver_tokens))
        cond = self.condition(evidence.to(src.dtype))
        # Multiplication after separate projections is a real bilinear
        # interaction; concatenating [s,t,s-t] before one linear is not.
        code = F.gelu(src) * (1.0 + torch.tanh(rec + cond))
        code = code.reshape(bsz, length, heads, self.rank)
        raw = torch.einsum('bnhr,hrd->bhnd', code, self.up)
        # Geometric arithmetic is FP32 even under autocast. The only output
        # cast is back to the original Value dtype.
        v, raw = values.float(), raw.float()
        norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        unit = v / safe_norm
        if mode == 'radial':
            # Diagnostic control: identical predictor/parameter count, but
            # only a bounded scalar rescaling of the original vector.
            amount = torch.tanh((raw * unit).sum(dim=-1, keepdim=True))
            return (v * (1.0 + self.max_step * amount)).to(values.dtype)
        tangent = raw - (raw * unit).sum(dim=-1, keepdim=True) * unit
        tangent_sq = (tangent * tangent).sum(dim=-1, keepdim=True)
        step = self.max_step * norm * tangent / torch.sqrt(1.0 + tangent_sq)
        moved = v + step
        moved_norm = torch.linalg.vector_norm(moved, dim=-1, keepdim=True)
        safe_moved = torch.where(moved_norm > 0, moved_norm,
                                 torch.ones_like(moved_norm))
        # At zero up: moved==v and the multiplier is exactly one for nonzero
        # vectors. Zero vectors stay zero; gradients remain finite.
        return (moved * (norm / safe_moved)).to(values.dtype)


def install_value_direction(model, rank=8, max_step=0.10, mode='direction'):
    """Attach after parent construction/loading and before optimizer creation.

    Disabled configurations never call this. fork_rng preserves the parent's
    subsequent training RNG stream. A shared parent state is still required
    for separately constructed training arms.
    """
    if mode not in ('direction', 'radial'):
        raise ValueError('mode must be direction or radial')
    targets = [m for m in model.modules()
               if m.__class__.__name__ == 'V2TemplateFusion']
    if not targets or any(not m.early_conditioned_cross_gate for m in targets):
        raise ValueError('Value direction requires an explicit Early gate parent')
    if any(hasattr(m, 'value_direction') for m in targets):
        raise ValueError('Value direction is already installed')
    # Validate every target before mutating any module.
    for module in targets:
        dim, heads = module.qkv.in_features, module.num_heads
        if dim % heads or not 0 < rank <= dim // heads:
            raise ValueError('invalid rank/head contract')
    if not math.isfinite(max_step) or not 0 < max_step <= 0.25:
        raise ValueError('invalid max_step')
    # Host construction is CPU in the production builder. Preserve CUDA RNG
    # too if a diagnostic explicitly installs on an already moved model.
    devices = sorted({m.qkv.weight.device.index for m in targets
                      if m.qkv.weight.is_cuda})
    with torch.random.fork_rng(devices=devices):
        for module in targets:
            operator = EarlyConditionedValueDirection(
                module.qkv.in_features, module.num_heads, rank, max_step)
            operator.to(device=module.qkv.weight.device,
                        dtype=module.qkv.weight.dtype)
            operator.train(module.training)
            module.add_module('value_direction', operator)
            module.value_direction_enabled = True
            module.value_direction_mode = mode
    return len(targets)
