"""Causal temporal residual for the TBSI search representation.

This is a tracker-specific adaptation, not a reproduction of the TAM paper.
The module consumes a previous head input, aligns it using the two actual
image-space crops, and writes a zero-gated local attention residual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalFusionResidual(nn.Module):
    def __init__(self, dim=768, hidden=192, heads=4, grid_size=16,
                 local_radius=2, max_message_ratio=0.10,
                 force_zero=False):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        if grid_size < 1 or local_radius < 0:
            raise ValueError("invalid grid_size or local_radius")
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.grid_size = int(grid_size)
        self.local_radius = int(local_radius)
        self.max_message_ratio = float(max_message_ratio)
        self.force_zero = bool(force_zero)
        self.q = nn.Linear(dim, hidden)
        self.k = nn.Linear(dim, hidden)
        self.v = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _as_rect(rect, batch, device, dtype):
        if rect is None:
            return None
        rect = torch.as_tensor(rect, device=device, dtype=dtype)
        if rect.dim() == 1:
            rect = rect.unsqueeze(0)
        if rect.shape != (batch, 4):
            raise ValueError("crop rectangles must have shape [B, 4] (x, y, w, h)")
        if not torch.isfinite(rect).all() or (rect[:, 2:] <= 0).any():
            raise ValueError("crop rectangles must be finite with positive size")
        return rect

    def _align_history(self, history, current_rect, history_rect):
        b, n, c = history.shape
        s = self.grid_size
        if n != s * s:
            raise ValueError("token count must equal grid_size squared")
        current_rect = self._as_rect(current_rect, b, history.device, history.dtype)
        history_rect = self._as_rect(history_rect, b, history.device, history.dtype)
        # Current token centers in image coordinates, then normalized into the
        # previous crop for grid_sample. align_corners=False is explicit so
        # the geometry is stable across supported torch versions.
        yy, xx = torch.meshgrid(
            torch.arange(s, device=history.device, dtype=history.dtype),
            torch.arange(s, device=history.device, dtype=history.dtype))
        px = current_rect[:, 0, None] + (xx.reshape(1, -1) + .5) * current_rect[:, 2, None] / s
        py = current_rect[:, 1, None] + (yy.reshape(1, -1) + .5) * current_rect[:, 3, None] / s
        nx = 2.0 * ((px - history_rect[:, 0, None]) / history_rect[:, 2, None]) - 1.0
        ny = 2.0 * ((py - history_rect[:, 1, None]) / history_rect[:, 3, None]) - 1.0
        grid = torch.stack((nx, ny), dim=-1).view(b, s, s, 2)
        limit = 1.0 - 1.0 / float(s)
        valid = ((grid[..., 0].abs() <= limit + 1e-6) &
                 (grid[..., 1].abs() <= limit + 1e-6))
        image = history.transpose(1, 2).reshape(b, c, s, s)
        aligned = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros',
                                align_corners=False)
        return aligned.flatten(2).transpose(1, 2), valid.flatten(1)

    def _local_mask(self, device):
        s = self.grid_size
        yy, xx = torch.meshgrid(torch.arange(s, device=device), torch.arange(s, device=device))
        pos = torch.stack((yy.flatten(), xx.flatten()), dim=1)
        dist = (pos[:, None, :] - pos[None, :, :]).abs().max(dim=-1).values
        return dist <= self.local_radius

    def forward(self, current, history=None, current_rect=None, history_rect=None,
                current_frame_id=None, history_frame_id=None,
                current_sequence_id=None, history_sequence_id=None,
                return_stats=True):
        if current.dim() != 3 or current.shape[-1] != self.dim:
            raise ValueError("current must have shape [B, grid_size^2, dim]")
        b, n, _ = current.shape
        if n != self.grid_size * self.grid_size:
            raise ValueError("token count must equal grid_size squared")
        empty = {'coverage': 0.0, 'message_ratio': 0.0,
                 'history_used': False, 'gamma': float(self.gamma.detach().item())}
        if history is None:
            return (current, empty) if return_stats else current
        if history.shape != current.shape:
            raise ValueError("history and current shapes must match")
        if current_frame_id is None or history_frame_id is None:
            raise ValueError("causal frame ids are required when history is used")
        cur_id = torch.as_tensor(current_frame_id, device=current.device).reshape(-1)
        old_id = torch.as_tensor(history_frame_id, device=current.device).reshape(-1)
        if cur_id.numel() != b or old_id.numel() != b or not (old_id < cur_id).all():
            raise ValueError("history frame must be strictly earlier than current frame")
        if current_sequence_id is not None or history_sequence_id is not None:
            if current_sequence_id is None or history_sequence_id is None:
                raise ValueError("both sequence ids are required when sequence checking is enabled")
            if list(current_sequence_id) != list(history_sequence_id):
                raise ValueError("history must belong to the same sequence")
        aligned, valid = self._align_history(history, current_rect, history_rect)
        coverage = float(valid.float().mean().detach().item())
        empty.update({'coverage': coverage, 'history_used': True})
        if self.force_zero or not valid.any():
            return (current, empty) if return_stats else current
        q = self.q(current).view(b, n, self.heads, -1).transpose(1, 2)
        k = self.k(aligned).view(b, n, self.heads, -1).transpose(1, 2)
        v = self.v(aligned).view(b, n, self.heads, -1).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        local = self._local_mask(current.device).view(1, 1, n, n)
        allowed = local & valid[:, None, None, :]
        has_local_key = allowed.any(dim=-1)
        query_ok = valid[:, None, :] & has_local_key
        logits = logits.masked_fill(~allowed, -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * allowed.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        msg = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, n, self.hidden)
        msg = self.out(msg)
        # Clear invalid queries after the biased output projection.
        msg = msg * query_ok.squeeze(1).unsqueeze(-1).to(msg.dtype)
        delta = torch.tanh(self.gamma) * msg
        cur_norm = current.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta_norm = delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cap = self.max_message_ratio * cur_norm
        scale = torch.clamp(cap / delta_norm, max=1.0).to(delta.dtype)
        delta = delta * scale
        out = current + delta
        actual_ratio = delta.float().norm(dim=-1) / current.float().norm(dim=-1).clamp_min(1e-6)
        empty.update({
            'query_usable_fraction': float(query_ok.float().mean().detach().item()),
            'actual_ratio_mean': float(actual_ratio.mean().detach().item()),
            'actual_ratio_max': float(actual_ratio.max().detach().item()),
            'message_ratio': float(actual_ratio.mean().detach().item()),
            'history_used': bool(query_ok.any().detach().item()),
        })
        return (out, empty) if return_stats else out
