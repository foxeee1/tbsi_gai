"""Low-rank MR with a shared score for source allocation and block acceptance.

This is a research prototype, not an implementation of a published method.
The cross score changes both within-cross proportions and its competition with
the original self block. No extra acceptance network or new target labels.
"""
import math

import torch
from torch import nn

from .receiver_source_mass_reallocator import StableLinear


def fusion_conflict(source_mode, source_evidence, structured, dual_mode, fu,
                    receiver_write, normalization, spatial_mode, conservative,
                    evidence_isolation, target_mode, center_log_gain):
    """Refuse combinations whose action semantics have not been defined."""
    return (source_mode != "dynamic" or source_evidence or structured or
            dual_mode != "off" or fu or receiver_write or normalization != "off" or
            spatial_mode != "off" or conservative or evidence_isolation or
            target_mode != "max" or center_log_gain)


class LowRankJointMassReallocator(nn.Module):
    """Keep all token pairs, avoiding a [B,Nq,Ns,3*rank] pair MLP.

    Inputs: cross probabilities [B,H,Nq,Ns], self probability mass
    [B,H,Nq,1], receiver self message [B,Nq,D], source Value
    [B,H,Ns,D/H], source context [B,Ns,C], evidence [B,Ns,3].
    Returns cross weights (with the original dropout mask, if supplied) and
    a [B,H,Nq,1] scale for the original self message.
    """

    def __init__(self, dim=768, context_dim=64, rank=32, allocation="joint", max_delta=4.0):
        super().__init__()
        if rank <= 0 or allocation not in ("joint", "conserved"):
            raise ValueError("invalid low-rank MR dimensions/allocation")
        self.dim, self.context_dim, self.rank = dim, context_dim, rank
        if not 0 < max_delta <= 8:
            raise ValueError("max_delta must give a bounded finite exponential")
        self.max_delta = float(max_delta)
        self.allocation = allocation
        self.receiver_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.receiver = StableLinear(dim, rank)
        self.source = StableLinear(dim, rank)
        self.context = StableLinear(context_dim, rank)
        self.evidence = StableLinear(3, rank)
        self.reset_identity()

    def reset_identity(self):
        # StableLinear survives the host ViT's generic Linear initializer.
        # The source side receives gradients after the query leaves zero.
        nn.init.zeros_(self.receiver.weight)

    def score(self, receiver_self, source_value, source_context, source_evidence):
        b, h, ns, dh = source_value.shape
        if (h * dh != self.dim or receiver_self.ndim != 3 or
                receiver_self.shape[0] != b or receiver_self.shape[-1] != self.dim or
                source_context.shape != (b, ns, self.context_dim) or
                source_evidence.shape != (b, ns, 3)):
            raise ValueError("low-rank MR token axes/width mismatch")
        source = source_value.transpose(1, 2).reshape(b, ns, self.dim)
        q = self.receiver(self.receiver_norm(receiver_self.float()))
        k = (self.source(self.source_norm(source.float())) +
             self.context(self.context_norm(source_context.float())) +
             self.evidence(source_evidence.float()))
        # A head-shared correction, as in the parent MR. The head-specific
        # attention prior and self/cross budget remain separate.
        raw = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.rank)
        return self.max_delta * torch.tanh(raw / self.max_delta)

    def allocate(self, cross_probability, self_mass, delta, cross_attention=None):
        """Normalize before dropout; apply the parent's dropout mask after it.

        Scores are bounded by score(), so exp needs no pairwise log/probability
        reconstruction. An all-zero cross row stays exactly zero.
        `conserved` is the same-score control, not another candidate family.
        """
        if (cross_probability.ndim != 4 or
                self_mass.shape != cross_probability.shape[:-1] + (1,) or
                delta.shape != (cross_probability.shape[0],
                                cross_probability.shape[2], cross_probability.shape[3])):
            raise ValueError("invalid low-rank MR probability/score axes")
        p = cross_probability.float()
        a = p if cross_attention is None else cross_attention.float()
        if a.shape != p.shape:
            raise ValueError("dropout attention shape mismatch")
        delta = delta.float()[:, None]
        cross_factor = torch.exp(delta)
        cross_total = (p * cross_factor).sum(dim=-1, keepdim=True)
        tiny = torch.finfo(p.dtype).tiny
        if self.allocation == "conserved":
            cross_scale = p.sum(dim=-1, keepdim=True) / cross_total.clamp_min(tiny)
            return (a * cross_factor * cross_scale).to(cross_probability.dtype), torch.ones_like(self_mass)
        denominator = (self_mass.float() + cross_total).clamp_min(tiny)
        return ((a * cross_factor / denominator).to(cross_probability.dtype),
                denominator.reciprocal().to(self_mass.dtype))

    def forward(self, cross_probability, self_mass, receiver_self, source_value,
                source_context, source_evidence, cross_attention=None):
        delta = self.score(receiver_self, source_value, source_context, source_evidence)
        return self.allocate(cross_probability, self_mass, delta, cross_attention)
