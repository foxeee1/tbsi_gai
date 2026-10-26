"""Sparse, spatially corresponding cross-modal search-token messages."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTokenBridge(nn.Module):
    def __init__(self, dim, rank=32, topk=8, max_delta=0.10,
                 temperature=0.10, spatial_sigma=2.0, grid_size=16):
        super().__init__()
        self.dim = int(dim)
        self.rank = max(1, int(rank))
        self.topk = max(1, int(topk))
        self.max_delta = float(max_delta)
        self.temperature = max(float(temperature), 1e-4)
        self.spatial_sigma = max(float(spatial_sigma), 1e-4)
        self.grid_size = int(grid_size)
        if self.grid_size * self.grid_size != 256:
            raise ValueError('SpatialTokenBridge requires a 16x16 search grid')
        in_dim = dim * 3 + 3
        self.rgb_down = nn.Linear(in_dim, self.rank, bias=False)
        self.rgb_up = nn.Linear(self.rank, dim, bias=False)
        self.tir_down = nn.Linear(in_dim, self.rank, bias=False)
        self.tir_up = nn.Linear(self.rank, dim, bias=False)
        nn.init.zeros_(self.rgb_up.weight)
        nn.init.zeros_(self.tir_up.weight)

        coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid_size), torch.arange(self.grid_size)),
            dim=-1).reshape(-1, 2).float()
        self.register_buffer('grid_coords', coords, persistent=False)

    def _select(self, search, template, fg_mask, mode):
        if fg_mask is None:
            raise ValueError('SpatialTokenBridge requires template foreground masks')
        weights = fg_mask.to(device=template.device, dtype=torch.float32).clamp(0, 1)
        if weights.ndim == 3:
            weights = weights.reshape(weights.shape[0], -1)
        if weights.shape != template.shape[:2]:
            raise ValueError('template foreground mask must match template token count')
        proto = (template.float() * weights.unsqueeze(-1)).sum(1)
        proto = F.normalize(proto / weights.sum(1, keepdim=True).clamp_min(1e-6), dim=-1)
        score = (F.normalize(search.float(), dim=-1) * proto[:, None, :]).sum(-1)
        indices = score.topk(min(self.topk, search.shape[1]), dim=1).indices
        if mode == 'shifted':
            shift = self.grid_size // 2
            y = torch.div(indices, self.grid_size, rounding_mode='floor')
            x = indices.remainder(self.grid_size)
            indices = ((y + shift).remainder(self.grid_size) * self.grid_size +
                       (x + shift).remainder(self.grid_size))
        return score, indices

    def _direction(self, source, peer, source_score, peer_score,
                   source_idx, peer_idx, down, up):
        batch, count, channels = source.shape
        source_tokens = source.gather(1, source_idx.unsqueeze(-1).expand(-1, -1, channels))
        peer_tokens = peer.gather(1, peer_idx.unsqueeze(-1).expand(-1, -1, channels))
        source_norm = F.normalize(source_tokens.float(), dim=-1)
        peer_norm = F.normalize(peer_tokens.float(), dim=-1)
        affinity = torch.matmul(source_norm, peer_norm.transpose(1, 2))
        coords = self.grid_coords.to(device=source.device)
        source_xy = coords[source_idx]
        peer_xy = coords[peer_idx]
        delta_xy = source_xy[:, :, None, :] - peer_xy[:, None, :, :]
        distance2 = delta_xy.square().sum(-1)
        affinity = affinity - distance2 / (2.0 * self.spatial_sigma ** 2)
        weights = torch.softmax(affinity / self.temperature, dim=-1)
        aligned_peer = torch.matmul(weights, peer_tokens.float())
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / torch.log(torch.tensor(
            float(max(peer_idx.shape[1], 2)), device=source.device))
        own_evidence = source_score.gather(1, source_idx)
        peer_evidence = torch.matmul(weights, peer_score.gather(1, peer_idx).unsqueeze(-1)).squeeze(-1)
        evidence = torch.stack((own_evidence, peer_evidence, entropy), dim=-1)
        inp = torch.cat((source_tokens.float(), aligned_peer,
                         source_tokens.float() - aligned_peer, evidence), dim=-1)
        delta = up(F.gelu(down(inp)))
        delta = delta.float()
        max_norm = self.max_delta * source_tokens.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (max_norm / delta_norm).clamp(max=1.0)
        updated = source.clone()
        updated.scatter_add_(1, source_idx.unsqueeze(-1).expand(-1, -1, channels),
                             delta.to(dtype=source.dtype))
        return updated, delta, weights

    def forward(self, rgb, tir, rgb_template, tir_template,
                rgb_mask, tir_mask, mode='target'):
        if rgb.shape != tir.shape or rgb.ndim != 3 or rgb.shape[-1] != self.dim:
            raise ValueError('search modality tokens must share [B,256,C] shape')
        if rgb.shape[1] != self.grid_size * self.grid_size:
            raise ValueError('search token count must match the configured 16x16 grid')
        if rgb_template.shape != tir_template.shape or rgb_template.shape[-1] != self.dim:
            raise ValueError('template modality tokens must share [B,N,C] shape')
        if mode == 'off':
            return rgb, tir, {'mode': 'off', 'candidate_fraction': 0.0,
                              'delta_rgb_norm': 0.0, 'delta_tir_norm': 0.0}
        if mode not in ('target', 'shifted'):
            raise ValueError('bridge mode must be off, target, or shifted')
        score_rgb, idx_rgb = self._select(rgb, rgb_template, rgb_mask, mode)
        score_tir, idx_tir = self._select(tir, tir_template, tir_mask, mode)
        out_rgb, delta_rgb, corr_rgb = self._direction(
            rgb, tir, score_rgb, score_tir, idx_rgb, idx_tir,
            self.rgb_down, self.rgb_up)
        out_tir, delta_tir, corr_tir = self._direction(
            tir, rgb, score_tir, score_rgb, idx_tir, idx_rgb,
            self.tir_down, self.tir_up)
        with torch.no_grad():
            stats = {
                'mode': mode,
                'candidate_fraction': float((idx_rgb.shape[1] + idx_tir.shape[1]) /
                                            (2.0 * rgb.shape[1])),
                'delta_rgb_norm': float(delta_rgb.norm(dim=-1).mean()),
                'delta_tir_norm': float(delta_tir.norm(dim=-1).mean()),
                'rgb_corr_entropy': float(-(corr_rgb.clamp_min(1e-8) *
                                            corr_rgb.clamp_min(1e-8).log()).sum(-1).mean()),
                'tir_corr_entropy': float(-(corr_tir.clamp_min(1e-8) *
                                            corr_tir.clamp_min(1e-8).log()).sum(-1).mean()),
            }
        return out_rgb, out_tir, stats
