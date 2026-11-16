"""Receiver-conditioned local residual constrained to message direction."""

import torch
from torch import nn
from torch.nn import functional as F


class OrthogonalLocalResidualMessage(nn.Module):
    """Bounded direction-only correction for a native cross message."""

    def __init__(self, head_dim, rank=8, hidden_dim=16, max_delta=0.1):
        super().__init__()
        if head_dim <= 0 or rank <= 0 or hidden_dim <= 0:
            raise ValueError('invalid orthogonal residual dimensions')
        if not 0.0 < float(max_delta) <= 1.0:
            raise ValueError('orthogonal residual max_delta must be in (0, 1]')
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.query_weight = nn.Parameter(torch.empty(self.rank, self.head_dim))
        self.key_weight = nn.Parameter(torch.empty(self.rank, self.head_dim))
        self.hidden_weight = nn.Parameter(torch.empty(self.hidden_dim, self.rank))
        self.hidden_bias = nn.Parameter(torch.zeros(self.hidden_dim))
        self.readout_weight = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.readout_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.query_weight)
        nn.init.xavier_uniform_(self.key_weight)
        nn.init.xavier_uniform_(self.hidden_weight)

    def reset_identity(self):
        with torch.no_grad():
            self.readout_weight.zero_()
            self.readout_bias.zero_()

    def forward(self, query, source_key, source_value, base_attention):
        if query.ndim != 4 or source_key.ndim != 4 or source_value.ndim != 4:
            raise ValueError('orthogonal residual tensors must be [B,H,N,D]')
        if base_attention.shape != query.shape[:3] + (source_key.shape[2],):
            raise ValueError('orthogonal residual attention shape mismatch')
        if source_key.shape != source_value.shape:
            raise ValueError('source key/value shape mismatch')
        if query.shape[:2] != source_key.shape[:2] or query.shape[-1] != self.head_dim:
            raise ValueError('query/source head axes mismatch')

        q_proj = F.linear(query.float(), self.query_weight.float())
        k_proj = F.linear(source_key.float(), self.key_weight.float())
        pair = q_proj.unsqueeze(3) * k_proj.unsqueeze(2)
        pair = pair * (self.rank ** -0.5)
        hidden = F.gelu(F.linear(pair, self.hidden_weight.float(), self.hidden_bias.float()))
        delta = self.max_delta * torch.tanh(
            F.linear(hidden, self.readout_weight.float(), self.readout_bias.float())
        ).squeeze(-1)
        residual = (base_attention.float() * delta).matmul(source_value.float())
        native = base_attention.float().matmul(source_value.float())
        denom = native.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        residual = residual - (residual * native).sum(dim=-1, keepdim=True) / denom * native
        ratio = residual.norm(dim=-1, keepdim=True) / native.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = self.max_delta * torch.tanh(ratio / self.max_delta) / ratio.clamp_min(1e-6)
        return (residual * scale).to(source_value.dtype)


def reset_orthogonal_local_residual_message(model):
    count = 0
    for module in model.modules():
        if isinstance(module, OrthogonalLocalResidualMessage):
            module.reset_identity()
            count += 1
    return count
