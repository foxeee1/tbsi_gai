import torch
from torch import nn
from torch.nn import functional as F


class _LocalDirection(nn.Module):
    def __init__(self, dim=768, hidden_dim=96, relevance_dim=64):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.query = nn.Linear(dim, hidden_dim, bias=False)
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(dim, hidden_dim, bias=False)
        self.relevance = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, relevance_dim),
            nn.GELU(),
            nn.Linear(relevance_dim, 1),
        )
        self.output = nn.Linear(hidden_dim, dim, bias=False)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)

    def forward(self, query_tokens, source_tokens, query_template,
                source_template, max_delta=0.05):
        bsz, count, dim = query_tokens.shape
        side = int(count ** 0.5)
        if side * side != count or dim != self.dim or source_tokens.shape != query_tokens.shape:
            raise ValueError('local correspondence expects matching square [B,N,C] streams')

        query = self.query(F.layer_norm(query_tokens, (dim,)))
        source = self.key(F.layer_norm(source_tokens, (dim,)))
        values = self.value(source_tokens)
        source_map = source.transpose(1, 2).reshape(bsz, self.hidden_dim, side, side)
        value_map = values.transpose(1, 2).reshape(bsz, self.hidden_dim, side, side)
        key_windows = F.unfold(source_map, kernel_size=3, padding=1)
        value_windows = F.unfold(value_map, kernel_size=3, padding=1)
        key_windows = key_windows.reshape(bsz, self.hidden_dim, 9, count).permute(0, 3, 2, 1)
        value_windows = value_windows.reshape(bsz, self.hidden_dim, 9, count).permute(0, 3, 2, 1)

        valid_map = F.unfold(source_map.new_ones((bsz, 1, side, side)),
                             kernel_size=3, padding=1)
        valid = valid_map.reshape(bsz, 9, count).transpose(1, 2) > 0
        logits = (query.unsqueeze(2) * key_windows).sum(-1) * (self.hidden_dim ** -0.5)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        correspondence = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        context = (correspondence.unsqueeze(-1) * value_windows).sum(dim=2)

        query_similarity = F.cosine_similarity(
            F.normalize(query, dim=-1),
            F.normalize(self.value(query_template), dim=-1)[:, None, :], dim=-1)
        context_similarity = F.cosine_similarity(
            F.normalize(context, dim=-1),
            F.normalize(self.value(source_template), dim=-1)[:, None, :], dim=-1)
        relevance_input = torch.cat((query, context, query_similarity.unsqueeze(-1),
                                     context_similarity.unsqueeze(-1)), dim=-1)
        relevance = torch.sigmoid(self.relevance(relevance_input))

        residual = self.output(context) * relevance
        source_norm = query_tokens.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        residual_norm = residual.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (float(max_delta) * source_norm / residual_norm).clamp(max=1.0)
        return query_tokens + residual * scale.to(dtype=residual.dtype)


class LocalCrossModalCorrespondence(nn.Module):
    """Bidirectional 3x3 search-token exchange with template-conditioned relevance."""
    def __init__(self, dim=768, hidden_dim=96, relevance_dim=64, max_delta=0.05):
        super().__init__()
        self.max_delta = float(max_delta)
        self.rgb_from_tir = _LocalDirection(dim, hidden_dim, relevance_dim)
        self.tir_from_rgb = _LocalDirection(dim, hidden_dim, relevance_dim)

    def reset_output(self):
        self.rgb_from_tir.reset_output()
        self.tir_from_rgb.reset_output()

    def forward(self, rgb_search, tir_search, rgb_template, tir_template, fg_masks):
        if fg_masks is None or len(fg_masks) != 2:
            raise ValueError('local correspondence requires RGB/TIR template foreground masks')
        if rgb_template.shape != tir_template.shape or rgb_template.shape[-1] != rgb_search.shape[-1]:
            raise ValueError('local correspondence template shape mismatch')
        if fg_masks[0].shape != rgb_template.shape[:2] or fg_masks[1].shape != tir_template.shape[:2]:
            raise ValueError('local correspondence foreground masks must match template tokens')

        def pool(tokens, weights):
            weights = weights.to(device=tokens.device, dtype=tokens.dtype).clamp_min(0)
            return (tokens * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-6)

        rgb_target = pool(rgb_template, fg_masks[0])
        tir_target = pool(tir_template, fg_masks[1])
        rgb_out = self.rgb_from_tir(rgb_search, tir_search, rgb_target, tir_target,
                                    self.max_delta)
        tir_out = self.tir_from_rgb(tir_search, rgb_search, tir_target, rgb_target,
                                    self.max_delta)
        return rgb_out, tir_out
