"""Bounded query-source conditional weighting for MSHA cross messages."""

import torch
from torch import nn


class QuerySourceConditionalWrite(nn.Module):
    """Predict a bounded write correction for each receiver/source pair."""

    def __init__(self, head_dim, hidden_dim=32, max_delta=0.15):
        super().__init__()
        if head_dim <= 0 or hidden_dim <= 0 or not 0.0 < float(max_delta) < 1.0:
            raise ValueError("invalid query-source write dimensions or bound")
        self.head_dim = int(head_dim)
        self.max_delta = float(max_delta)
        self.predictor = nn.Sequential(
            nn.Linear(3 * self.head_dim, int(hidden_dim)), nn.GELU(),
            nn.Linear(int(hidden_dim), 1))
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, query, source_key, source_value):
        if query.ndim != 4 or source_key.ndim != 4 or source_value.ndim != 4:
            raise ValueError("expected [B,H,N,D] query/key/value tensors")
        if source_key.shape != source_value.shape:
            raise ValueError("source key/value shapes must match")
        if query.shape[:2] != source_key.shape[:2] or query.shape[-1] != self.head_dim:
            raise ValueError("query/source axes differ")
        q = query.unsqueeze(3).expand(-1, -1, -1, source_key.shape[2], -1)
        k = source_key.unsqueeze(2).expand(-1, -1, query.shape[2], -1, -1)
        v = source_value.unsqueeze(2).expand_as(k)
        correction = self.predictor(torch.cat((q, k, v), dim=-1)).squeeze(-1)
        return torch.exp(self.max_delta * torch.tanh(correction))


def reset_query_source_conditional_write(model):
    count = 0
    for module in model.modules():
        if isinstance(module, QuerySourceConditionalWrite):
            module.reset_identity()
            count += 1
    return count
