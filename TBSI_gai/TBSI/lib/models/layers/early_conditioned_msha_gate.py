"""Early-conditioned cross-Value gate used by the retained M1 experiment."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyConditionedMSHAGate(nn.Module):
    """3-to-2 gate head with Sequential-compatible checkpoint parameter names."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        # Numeric child names preserve legacy keys such as early_gate.0.weight.
        self.add_module('0', nn.Linear(3, hidden_dim))
        self.add_module('1', nn.GELU())
        self.add_module('2', nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self[2].weight)
        nn.init.zeros_(self[2].bias)

    def __getitem__(self, index):
        return self._modules[str(index)]

    def forward(self, evidence):
        if evidence.ndim != 3 or evidence.shape[-1] != 3:
            raise ValueError('early MSHA gate evidence must have shape [B,N,3]')
        return self[2](self[1](self[0](evidence)))


def early_msha_gate_evidence(early_context, expected_rgb_template,
                             expected_tir_template):
    """Build per-template-token RGB peak, TIR peak, and modality agreement."""
    if early_context is None or len(early_context) != 4:
        raise ValueError('early conditioned gate requires four early token tensors')
    early_rgb_t, early_tir_t, early_rgb_s, early_tir_s = early_context
    if (early_rgb_t.shape != expected_rgb_template.shape or
            early_tir_t.shape != expected_tir_template.shape or
            early_rgb_s.ndim != 3 or early_tir_s.ndim != 3 or
            early_rgb_s.shape[0] != expected_rgb_template.shape[0] or
            early_tir_s.shape[0] != expected_tir_template.shape[0]):
        raise ValueError('early conditioned gate token shape mismatch')
    rgb_t = F.normalize(early_rgb_t.float(), dim=-1, eps=1e-6)
    tir_t = F.normalize(early_tir_t.float(), dim=-1, eps=1e-6)
    rgb_s = F.normalize(early_rgb_s.float(), dim=-1, eps=1e-6)
    tir_s = F.normalize(early_tir_s.float(), dim=-1, eps=1e-6)
    rgb_similarity = torch.matmul(rgb_t, rgb_s.transpose(1, 2))
    tir_similarity = torch.matmul(tir_t, tir_s.transpose(1, 2))
    rgb_query = torch.amax(rgb_similarity, dim=-1)
    tir_query = torch.amax(tir_similarity, dim=-1)
    agreement = (rgb_t * tir_t).sum(dim=-1)
    evidence = torch.stack((rgb_query, tir_query, agreement), dim=-1)
    return evidence, rgb_similarity, tir_similarity
