"""One evidence-only experiment; no search GT, auxiliary loss or Value action.

Uniform/target arms differ ONLY in the prior of the same weighted log-pool.
The zero scalar makes the added evidence path an identity at initialization.
"""
import math
import torch
from torch import nn
from .structured_spatial_evidence import box_to_patch_coverage


class EarlyTargetSupportEvidence(nn.Module):
    def __init__(self, mode, temperature=0.1):
        super().__init__()
        if mode not in ('uniform', 'target') or temperature <= 0:
            raise ValueError('expected uniform/target and positive temperature')
        self.mode = mode
        self.temperature = float(temperature)
        self.strength = nn.Parameter(torch.zeros(()))
        self.last_diagnostics = None

    def forward(self, evidence, rgb_similarity, tir_similarity, template_boxes):
        if rgb_similarity.ndim != 3 or tir_similarity.shape != rgb_similarity.shape:
            raise ValueError('expected matching [B,Nsource,Nsearch] similarities')
        b, nt, ns = rgb_similarity.shape
        side = math.isqrt(nt)
        if side * side != nt or evidence.shape != (b, nt, 3):
            raise ValueError('expected square template grid and [B,Nsource,3] evidence')
        if template_boxes is None or template_boxes.shape != (b, 2, 4):
            raise ValueError('normalized template xywh boxes [B,2,4] are required')
        if template_boxes.device != rgb_similarity.device:
            raise ValueError('boxes and similarities must share device')
        pooled, diagnostics = [], []
        tau = self.temperature
        for modality, similarity in enumerate((rgb_similarity, tir_similarity)):
            sim = similarity.float()
            if not torch.isfinite(sim).all():
                raise ValueError('nonfinite similarity')
            coverage = box_to_patch_coverage(template_boxes[:, modality], side)
            # The query source cannot vote for its own search support.
            numerator = torch.einsum('bn,bnu->bu', coverage, sim)[:, None, :]
            remaining = coverage.sum(-1, keepdim=True) - coverage
            usable = remaining > 1e-6
            support = (numerator - coverage[..., None] * sim) / remaining[..., None].clamp_min(1e-6)
            support = torch.where(usable[..., None], support, torch.zeros_like(support))
            # Both arms build identical support; only its contribution differs.
            prior_logits = support / tau * (1.0 if self.mode == 'target' else 0.0)
            log_prior = prior_logits.log_softmax(-1)
            readout = tau * torch.logsumexp(sim / tau + log_prior, dim=-1)
            pooled.append(readout)
            diagnostics.append({'coverage': coverage.detach(), 'usable': usable.detach(),
                                'prior': log_prior.detach().exp(),
                                'pooled': readout.detach(),
                                'peak_index': sim.detach().argmax(-1)})
        proposed = torch.stack((pooled[0], pooled[1], evidence[..., 2].float()), -1)
        result = evidence.float() + self.strength.tanh() * (proposed - evidence.float())
        self.last_diagnostics = diagnostics
        return result
