"""Separate relative source selection from the mean signed acceptance score.

Parameter-identical refinement of low-rank joint MR. This removes the automatic
log-partition/dispersion uplift of its cross budget; it does not establish that
the learned scores measure reliability or that self attention is correct.
"""
import torch

from .low_rank_joint_mass_reallocator import LowRankJointMassReallocator


class ExpectedScoreMassReallocator(LowRankJointMassReallocator):
    """Use E_prior[delta] instead of log E_prior[exp(delta)] for the budget.

For cross mass m and self mass s, the new cross mass is
    m * exp(mu) / (s + m * exp(mu)), mu = sum(p * delta) / m.
Within-cross relative weights remain proportional to p * exp(delta).
Inputs, score function, initialization, dropout semantics and parameter shapes
are identical to LowRankJointMassReallocator(allocation='joint').
"""

    def __init__(self, dim=768, context_dim=64, rank=32, max_delta=4.0):
        super().__init__(dim, context_dim, rank, allocation="joint", max_delta=max_delta)

    def allocate(self, cross_probability, self_mass, delta, cross_attention=None):
        if (cross_probability.ndim != 4 or
                self_mass.shape != cross_probability.shape[:-1] + (1,) or
                delta.shape != (cross_probability.shape[0],
                                cross_probability.shape[2], cross_probability.shape[3])):
            raise ValueError("invalid expected-score MR probability/score axes")
        p = cross_probability.float()
        a = p if cross_attention is None else cross_attention.float()
        if a.shape != p.shape:
            raise ValueError("dropout attention shape mismatch")
        d = delta.float()[:, None]
        tiny = torch.finfo(p.dtype).tiny
        mass = p.sum(-1, keepdim=True)
        mean_score = (p * d).sum(-1, keepdim=True) / mass.clamp_min(tiny)
        factor = d.exp()
        tilted_mass = (p * factor).sum(-1, keepdim=True)
        accepted_mass = mass * mean_score.exp()
        denominator = (self_mass.float() + accepted_mass).clamp_min(tiny)
        cross_scale = (accepted_mass / denominator) / tilted_mass.clamp_min(tiny)
        return ((a * factor * cross_scale).to(cross_probability.dtype),
                denominator.reciprocal().to(self_mass.dtype))
