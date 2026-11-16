"""Cycle-Laplacian repair for cross-modal template messages.

The operator is deliberately parameter-free.  It uses the two cross-attention
relations only as a graph, and edits background receiver tokens while leaving
foreground and boundary tokens unchanged.
"""

import math

import torch
from torch import nn


def _box_coverage(boxes, side):
    """Return exact patch coverage for normalized xywh boxes."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B,4]")
    boxes = boxes.float()
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("boxes must be finite with positive width and height")
    x0 = boxes[:, 0:1].clamp(0, 1)
    y0 = boxes[:, 1:2].clamp(0, 1)
    x1 = (boxes[:, 0:1] + boxes[:, 2:3]).clamp(0, 1)
    y1 = (boxes[:, 1:2] + boxes[:, 3:4]).clamp(0, 1)
    coord = torch.arange(side, device=boxes.device, dtype=boxes.dtype) / side
    cell_x0 = coord.view(1, 1, -1)
    cell_y0 = coord.view(1, -1, 1)
    cell_x1 = cell_x0 + 1.0 / side
    cell_y1 = cell_y0 + 1.0 / side
    overlap_x = (torch.minimum(x1[:, :, None], cell_x1) -
                 torch.maximum(x0[:, :, None], cell_x0)).clamp_min(0)
    overlap_y = (torch.minimum(y1[:, :, None], cell_y1) -
                 torch.maximum(y0[:, :, None], cell_y0)).clamp_min(0)
    return (overlap_y.unsqueeze(-1) * overlap_x.unsqueeze(-2) * (side ** 2)).flatten(1)


class CycleLaplacianMessageRepair(nn.Module):
    """Filter unsupported background cross messages using a cycle graph.

    ``msg_rt`` is received by RGB template tokens from TIR and ``msg_tr`` is
    received by TIR template tokens from RGB.  The attention inputs retain
    their original cross mass; row normalization is used only to build the
    graph.  Boundary tokens are excluded by requiring exact zero box overlap.
    """

    def __init__(self, strength=0.25, eps=1e-6, background_threshold=1e-6,
                 patch_side=8):
        super().__init__()
        strength = float(strength)
        if not 0.0 <= strength <= 0.5:
            raise ValueError("CBMR strength must be in [0, 0.5]")
        self.strength = strength
        self.eps = float(eps)
        self.background_threshold = float(background_threshold)
        self.patch_side = int(patch_side)
        self.last_stats = None

    def _repair_one(self, message, relation_forward, relation_backward,
                    background):
        # [B,H,Nr,Ns] -> a row-stochastic graph used only for topology.
        forward = relation_forward.float()
        backward = relation_backward.float()
        forward = forward / forward.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        backward = backward / backward.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        cycle = torch.matmul(forward, backward)
        weight = 0.5 * (cycle + cycle.transpose(-1, -2))
        bg = background[:, None, :].to(weight.dtype)
        weight = weight * bg.unsqueeze(-1) * bg.unsqueeze(-2)
        degree = weight.sum(dim=-1)
        supported = degree > self.eps
        inv_sqrt = degree.clamp_min(self.eps).rsqrt()
        normalized = weight * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
        smooth = torch.matmul(normalized, message.float())
        delta = message.float() - smooth
        # Isolated background nodes have no reliable neighborhood and are left
        # untouched; foreground and boundary nodes are identity by construction.
        edited = message.float() - self.strength * delta * supported.unsqueeze(-1)
        edited = torch.where(background[:, None, :, None], edited, message.float())
        return edited.to(dtype=message.dtype), degree, supported

    def forward(self, msg_rt, msg_tr, attn_rt, attn_tr, template_boxes):
        if template_boxes is None:
            raise ValueError("CBMR requires normalized template_boxes")
        if msg_rt.ndim != 4 or msg_tr.ndim != 4:
            raise ValueError("messages must have shape [B,H,N,D]")
        bsz, _, n_rgb, _ = msg_rt.shape
        _, _, n_tir, _ = msg_tr.shape
        if attn_rt.shape[:3] != (bsz, attn_rt.shape[1], n_rgb) or \
                attn_rt.shape[-1] != n_tir or attn_tr.shape[-2:] != (n_tir, n_rgb):
            raise ValueError("attention/message axes are inconsistent")
        if template_boxes.shape != (bsz, 2, 4):
            raise ValueError("template_boxes must have shape [B,2,4]")
        side_rgb = int(math.sqrt(n_rgb))
        side_tir = int(math.sqrt(n_tir))
        if side_rgb * side_rgb != n_rgb or side_tir * side_tir != n_tir:
            raise ValueError("CBMR expects square template token grids")
        cov_rgb = _box_coverage(template_boxes[:, 0], side_rgb)
        cov_tir = _box_coverage(template_boxes[:, 1], side_tir)
        # Only fully background tokens are edited; mixed boundary patches stay
        # on the native MR path so the mask cannot erase object edges.
        bg_rgb = cov_rgb <= self.background_threshold
        bg_tir = cov_tir <= self.background_threshold
        out_rt, deg_rgb, support_rgb = self._repair_one(
            msg_rt, attn_rt, attn_tr, bg_rgb)
        out_tr, deg_tir, support_tir = self._repair_one(
            msg_tr, attn_tr, attn_rt, bg_tir)
        with torch.no_grad():
            self.last_stats = {
                'background_tokens_rgb': float(bg_rgb.float().sum(dim=-1).mean()),
                'background_tokens_tir': float(bg_tir.float().sum(dim=-1).mean()),
                'supported_background_rgb': float(support_rgb.float().sum(dim=-1).mean()),
                'supported_background_tir': float(support_tir.float().sum(dim=-1).mean()),
                'message_delta_rgb': float((out_rt.float() - msg_rt.float()).norm(dim=-1).mean()),
                'message_delta_tir': float((out_tr.float() - msg_tr.float()).norm(dim=-1).mean()),
            }
        return out_rt, out_tr

