"""Receiver/source relation for cross-modal MHSA.

The operator keeps the observed cross mass of every receiver query and only
changes the conditional distribution over source tokens.  It is deliberately
small: the existing Q/K relation and Early evidence are the only inputs; no
Value scaling, auxiliary loss, or second reliability estimator is introduced.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CrossMassPreservingReceiverSourceRelation(nn.Module):
    """Reallocate cross attention without changing its per-query budget.

    Inputs use explicit attention axes:
      query: [B, H, Nq, D], source: [B, H, Ns, D]
      cross_attention: [B, H, Nq, Ns], rows are native attention mass
      source_evidence: [B, Ns, E], where one channel is selected by caller

    The returned attention has the same shape and row sums as
    ``cross_attention`` (up to floating point error).  ``beta`` is zero at
    construction, so enabled mode is identity-preserving before learning.
    """

    def __init__(self, evidence_index=0, max_bias=0.1):
        super().__init__()
        if int(evidence_index) < 0:
            raise ValueError("evidence_index must be non-negative")
        if float(max_bias) <= 0:
            raise ValueError("max_bias must be positive")
        self.evidence_index = int(evidence_index)
        self.max_bias = float(max_bias)
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, query, source, cross_attention, source_evidence):
        if query.ndim != 4 or source.ndim != 4:
            raise ValueError("query/source must be [B,H,N,D]")
        if cross_attention.ndim != 4:
            raise ValueError("cross_attention must be [B,H,Nq,Ns]")
        if source_evidence.ndim != 3:
            raise ValueError("source_evidence must be [B,Ns,E]")
        b, h, nq, d = query.shape
        if source.shape[:2] != (b, h) or source.shape[2] != cross_attention.shape[-1]:
            raise ValueError("source and cross attention source axes differ")
        if cross_attention.shape[:3] != (b, h, nq):
            raise ValueError("cross attention receiver axes differ")
        if source_evidence.shape[0] != b or source_evidence.shape[1] != source.shape[2]:
            raise ValueError("source evidence axis differs from source tokens")
        if self.evidence_index >= source_evidence.shape[-1]:
            raise ValueError("evidence_index outside source evidence channels")

        eps = torch.finfo(cross_attention.dtype).tiny
        mass = cross_attention.sum(dim=-1, keepdim=True)
        # Conditional distribution is defined only inside the cross block.
        conditional = cross_attention / mass.clamp_min(eps)
        qn = F.normalize(query.float(), dim=-1, eps=1e-6)
        sn = F.normalize(source.float(), dim=-1, eps=1e-6)
        cosine = torch.matmul(qn, sn.transpose(-2, -1))
        evidence = source_evidence[..., self.evidence_index].float()
        evidence = evidence - evidence.mean(dim=-1, keepdim=True)
        evidence = evidence / evidence.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        relation = torch.tanh(cosine) + evidence[:, None, None, :]
        bias = self.max_bias * torch.tanh(self.beta) * torch.tanh(relation)
        # log native conditional mass retains the original Q/K preference;
        # the bounded relation bias only reorders source tokens.
        reordered = F.softmax(torch.log(conditional.clamp_min(eps)) + bias, dim=-1)
        adjusted = (mass * reordered).to(dtype=cross_attention.dtype)
        return adjusted, {
            "cross_mass_before": mass.detach(),
            "cross_mass_after": adjusted.sum(dim=-1, keepdim=True).detach(),
            "relation_bias": bias.detach(),
            "beta": self.beta.detach(),
        }


def reset_cross_mass_preserving_relation(model):
    """Restore identity initialization after generic model initialization."""
    count = 0
    for module in model.modules():
        if isinstance(module, CrossMassPreservingReceiverSourceRelation):
            with torch.no_grad():
                module.beta.zero_()
            count += 1
    return count
