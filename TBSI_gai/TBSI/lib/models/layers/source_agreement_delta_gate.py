"""Conservatively gate learned MR source-score corrections by local agreement."""
import torch
from torch import nn


class SourceAgreementDeltaGate(nn.Module):
    """Reduce a source's learned reallocation correction when modalities disagree.

    Agreement is the normalized same-position RGB/TIR cosine in evidence[..., 2].
    The gate has no learned parameters and does not alter the original attention
    prior or cross-row mass.
    """

    def forward(self, delta, source_evidence):
        if delta.ndim != 3 or source_evidence.ndim != 3:
            raise ValueError('delta/evidence must be [B,Nreceiver,Nsource] and [B,Nsource,3]')
        if (delta.shape[0] != source_evidence.shape[0] or
                delta.shape[-1] != source_evidence.shape[1] or
                source_evidence.shape[-1] != 3):
            raise ValueError('source agreement axes differ from MR score')
        agreement = source_evidence[..., 2].float().detach().clamp(0.0, 1.0)
        return delta.float() * agreement[:, None, :]
