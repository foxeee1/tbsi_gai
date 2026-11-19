"""Direct source-logit modulation using bounded source evidence."""
import torch
from torch import nn


class DirectEvidenceModulation(nn.Module):
    """Convert source evidence into a row-mass-preserving logit correction.

    The correction is shared across attention heads and receivers.  Its final
    projection is zero initialized, so the enabled branch starts as the native
    cross-attention relation and can be audited independently from MR pair
    scoring.
    """

    def __init__(self, evidence_dim=5, max_delta=0.25):
        super().__init__()
        if evidence_dim <= 0 or max_delta <= 0:
            raise ValueError('invalid direct evidence dimensions')
        self.evidence_dim = int(evidence_dim)
        self.max_delta = float(max_delta)
        self.readout = nn.Linear(self.evidence_dim, 1)
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, cross_attention, source_evidence):
        if cross_attention.ndim != 4:
            raise ValueError('cross_attention must be [B,H,Nq,Ns]')
        b, _, _, ns = cross_attention.shape
        if source_evidence.shape != (b, ns, self.evidence_dim):
            raise ValueError('source_evidence shape mismatch')
        delta = self.max_delta * torch.tanh(
            self.readout(source_evidence.float()).squeeze(-1))
        logits = torch.log(cross_attention.float().clamp_min(
            torch.finfo(cross_attention.dtype).tiny)) + delta[:, None, None, :]
        active = cross_attention > 0
        logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        conditional = torch.softmax(logits, dim=-1)
        any_active = active.any(dim=-1, keepdim=True)
        conditional = torch.where(any_active, conditional,
                                  torch.zeros_like(conditional))
        mass = cross_attention.float().sum(dim=-1, keepdim=True)
        return (mass * conditional).to(cross_attention.dtype), delta


def reset_direct_evidence_modulation(model):
    count = 0
    for module in model.modules():
        if isinstance(module, DirectEvidenceModulation):
            module.reset_identity()
            count += 1
    return count
