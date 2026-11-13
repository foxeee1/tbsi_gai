"""Learned directional template-to-search context for Early-gate source gains."""
import math

import torch
from torch import nn


class EarlyGateSearchContext(nn.Module):
    """Predict one residual gain logit per source template token.

    The summary mode attends only to tokens made from the parent's three
    evidence scalars. Cross mode adds the other modality's search tokens to
    the same memory. Both modes have identical parameter layouts.
    """

    def __init__(self, dim=768, rank=64, heads=4, mode="cross"):
        super().__init__()
        if dim % heads or rank % heads or mode not in ("off", "summary", "cross", "global_mean"):
            raise ValueError("invalid search-context attention dimensions or mode")
        self.dim = int(dim)
        self.rank = int(rank)
        self.heads = int(heads)
        self.head_dim = self.rank // self.heads
        self.mode = mode
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, rank, bias=False)
        self.key = nn.Linear(dim, rank, bias=False)
        self.value = nn.Linear(dim, rank, bias=False)
        self.summary_basis = nn.Parameter(torch.empty(3, dim))
        self.readout = nn.Sequential(
            nn.Linear(2 * rank + 3, rank),
            nn.GELU(),
            nn.Linear(rank, 1),
        )
        nn.init.normal_(self.summary_basis, std=dim ** -0.5)
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def _heads(self, tensor):
        b, n, _, _ = tensor.shape
        return tensor.reshape(b, n, -1, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)

    def _project_memory(self, memory):
        shape = memory.shape
        normalized = self.memory_norm(memory.reshape(-1, self.dim)).reshape(shape)
        return self._heads(self.key(normalized)), self._heads(self.value(normalized))

    def _one_direction(self, source_template, recipient_search, evidence, context_only=False):
        b, n, _ = source_template.shape
        q = self.query(self.query_norm(source_template))
        q = q.reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3).unsqueeze(3)

        summary = evidence.unsqueeze(-1) * self.summary_basis.view(1, 1, 3, self.dim)
        k_summary, v_summary = self._project_memory(summary)
        summary_logits = (q * k_summary).sum(-1) / math.sqrt(self.head_dim)
        logits = [summary_logits]
        memories = [(k_summary, v_summary)]
        if self.mode in ("cross", "global_mean"):
            if self.mode == "global_mean":
                recipient_search = recipient_search.mean(dim=1, keepdim=True).expand_as(recipient_search)
            k_search, v_search = self._project_memory(recipient_search.unsqueeze(1))
            search_logits = torch.matmul(
                q.squeeze(3), k_search.squeeze(2).transpose(-2, -1)) / math.sqrt(self.head_dim)
            logits.append(search_logits)
            memories.append((k_search, v_search))

        attention = torch.cat(logits, dim=-1).float().softmax(dim=-1)
        outputs = []
        start = 0
        for key, value in memories:
            width = key.shape[-2]
            weights = attention[..., start:start + width].to(value.dtype).unsqueeze(-2)
            outputs.append(torch.matmul(weights, value).squeeze(-2))
            start += width
        context = outputs[0]
        if len(outputs) == 2:
            context = context + outputs[1]
        context = context.permute(0, 2, 1, 3).reshape(b, n, self.rank)
        if context_only:
            return context
        query = q.squeeze(3).permute(0, 2, 1, 3).reshape(b, n, self.rank)
        return self.readout(torch.cat((query, context, evidence), dim=-1)).squeeze(-1), context

    def context_only(self, evidence, early_context, condition_context=None):
        """Same context math, without the disconnected g=1 gain readout."""
        context = early_context if condition_context is None else condition_context
        if len(context) != 4 or evidence.ndim != 3 or evidence.shape[-1] != 3:
            raise ValueError('invalid context-only MR input')
        rgb_t, tir_t, rgb_s, tir_s = context
        if rgb_t.shape != tir_t.shape or rgb_s.shape != tir_s.shape:
            raise ValueError('batched MR context requires matched modality shapes')
        evidence = evidence.to(dtype=rgb_t.dtype)
        b = rgb_t.shape[0]
        # The two directions share every context parameter. Batch them to
        # reduce small CUDA launches; never share parameters across layers.
        both = self._one_direction(
            torch.cat((rgb_t, tir_t), dim=0),
            torch.cat((tir_s, rgb_s), dim=0),
            torch.cat((evidence, evidence), dim=0), context_only=True)
        return torch.stack((both[:b], both[b:]), dim=-1)

    def forward(self, evidence, early_context, condition_context=None, return_context=False):
        if evidence.ndim != 3 or evidence.shape[-1] != 3 or len(early_context) != 4:
            raise ValueError("expected [B,N,3] evidence and four early token streams")
        context = early_context if condition_context is None else condition_context
        if len(context) != 4:
            raise ValueError("expected four token streams for search-context conditioning")
        rgb_t, tir_t, rgb_s, tir_s = context
        b, n, _ = rgb_t.shape
        if any(x.ndim != 3 or x.shape[:2] != (b, n) or x.shape[-1] != self.dim
               for x in (rgb_t, tir_t)):
            raise ValueError("template tokens must share the evidence token axes")
        if rgb_s.ndim != 3 or tir_s.ndim != 3 or rgb_s.shape[0] != b or tir_s.shape[0] != b:
            raise ValueError("search tokens must have shape [B,Nsearch,C]")
        if rgb_s.shape[-1] != self.dim or tir_s.shape[-1] != self.dim:
            raise ValueError("search token width differs from the configured embedding")
        evidence = evidence.to(dtype=rgb_t.dtype)
        rgb_to_tir, rgb_context = self._one_direction(rgb_t, tir_s, evidence)
        tir_to_rgb, tir_context = self._one_direction(tir_t, rgb_s, evidence)
        logits = torch.stack((rgb_to_tir, tir_to_rgb), dim=-1)
        if return_context:
            return logits, torch.stack((rgb_context, tir_context), dim=-1)
        return logits


def reset_early_gate_search_context(model):
    count = 0
    for module in model.modules():
        if isinstance(module, EarlyGateSearchContext):
            module.reset_identity()
            count += 1
    return count
