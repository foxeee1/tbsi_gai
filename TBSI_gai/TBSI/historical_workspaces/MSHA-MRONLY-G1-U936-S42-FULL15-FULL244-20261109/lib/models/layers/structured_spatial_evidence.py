"""Target-grounded spatial evidence for the Early-conditioned MSHA."""
import torch
from torch import nn
from torch.nn import functional as F


def box_to_patch_coverage(boxes, grid_size=8):
    """Convert normalized xywh boxes into exact box/patch area fractions."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('template boxes must have shape [B,4] normalized xywh')
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError('template boxes must be finite with positive width/height')
    boxes = boxes.float()
    x0 = boxes[:, 0:1].clamp(0, 1)
    y0 = boxes[:, 1:2].clamp(0, 1)
    x1 = (boxes[:, 0:1] + boxes[:, 2:3]).clamp(0, 1)
    y1 = (boxes[:, 1:2] + boxes[:, 3:4]).clamp(0, 1)
    coords = torch.arange(grid_size, device=boxes.device, dtype=boxes.dtype) / grid_size
    cell0x, cell0y = coords.view(1, 1, -1), coords.view(1, -1, 1)
    cell1x, cell1y = cell0x + 1.0 / grid_size, cell0y + 1.0 / grid_size
    overlap_x = (torch.minimum(x1[:, :, None], cell1x) -
                 torch.maximum(x0[:, :, None], cell0x)).clamp_min(0)
    overlap_y = (torch.minimum(y1[:, :, None], cell1y) -
                 torch.maximum(y0[:, :, None], cell0y)).clamp_min(0)
    coverage = (overlap_y.unsqueeze(-1) * overlap_x.unsqueeze(-2)) * (grid_size ** 2)
    coverage = coverage.flatten(1).clamp(0, 1)
    if (coverage.sum(dim=-1) <= 1e-8).any():
        raise ValueError('template box has no overlap with the template crop')
    return coverage


class StructuredSpatialEvidence(nn.Module):
    """Keep each template token's search map and encode it with shared weights."""

    def __init__(self, hidden_dim=8, grid_size=8, search_side=16, pooled_side=4):
        super().__init__()
        self.grid_size = int(grid_size)
        self.search_side = int(search_side)
        self.pooled_side = int(pooled_side)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU())
        self.to_token = nn.Linear(hidden_dim * pooled_side * pooled_side, 32)
        self.support = nn.Conv2d(hidden_dim, 1, 1)

    def _encode(self, similarity, target_response, coverage, agreement):
        b, nt, ns = similarity.shape
        if ns != self.search_side * self.search_side:
            raise ValueError('search similarity must be a square configured token grid')
        if coverage.shape != (b, nt) or agreement.shape != (b, nt):
            raise ValueError('template coverage/agreement axes must match similarity tokens')
        coord = torch.linspace(-1, 1, self.search_side, device=similarity.device,
                               dtype=torch.float32)
        yy, xx = torch.meshgrid(coord, coord, indexing='ij')
        maps = torch.stack((
            similarity.float().reshape(b, nt, 1, self.search_side, self.search_side),
            target_response.float().reshape(b, 1, 1, self.search_side, self.search_side)
            .expand(-1, nt, -1, -1, -1),
            xx.view(1, 1, 1, self.search_side, self.search_side).expand(b, nt, -1, -1, -1),
            yy.view(1, 1, 1, self.search_side, self.search_side).expand(b, nt, -1, -1, -1),
            coverage.float()[:, :, None, None, None].expand(-1, -1, 1, self.search_side, self.search_side),
            agreement.float()[:, :, None, None, None].expand(-1, -1, 1, self.search_side, self.search_side),
        ), dim=2).reshape(b * nt, 6, self.search_side, self.search_side)
        spatial = self.encoder(maps)
        pooled = F.adaptive_avg_pool2d(spatial, (self.pooled_side, self.pooled_side))
        state = self.to_token(pooled.flatten(1)).reshape(b, nt, 32)
        spatial = spatial.reshape(b, nt, spatial.shape[1], self.search_side, self.search_side)
        return state, spatial

    def forward(self, rgb_similarity, tir_similarity, template_boxes, old_evidence,
                action_mode='structured'):
        if rgb_similarity.ndim != 3 or tir_similarity.shape != rgb_similarity.shape:
            raise ValueError('similarity inputs must share [B,Ntemplate,Nsearch] shape')
        if template_boxes.shape != (rgb_similarity.shape[0], 2, 4):
            raise ValueError('template boxes must have shape [B,2,4] for RGB and TIR')
        if old_evidence.shape != (*rgb_similarity.shape[:2], 3):
            raise ValueError('old evidence must have shape [B,Ntemplate,3]')
        if action_mode not in ('summary', 'structured'):
            raise ValueError('action_mode must be summary or structured')

        coverage = torch.stack([
            box_to_patch_coverage(template_boxes[:, i], self.grid_size)
            for i in range(2)], dim=1)
        coverage = coverage.to(rgb_similarity.device)
        weights = coverage / coverage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        similarities = (rgb_similarity, tir_similarity)
        states, spatial_maps, support_logits = [], [], []
        for modality in range(2):
            sim = similarities[modality].float()
            target_response = torch.einsum('bn,bns->bs', weights[:, modality], sim)
            agreement = old_evidence[..., 2].float()
            real_state, real_spatial = self._encode(
                sim, target_response, coverage[:, modality], agreement)
            pooled_target = torch.einsum('bn,bnchw->bchw', weights[:, modality], real_spatial)
            support_logits.append(self.support(pooled_target))

            if action_mode == 'structured':
                state = real_state
            else:
                old_peak = old_evidence[..., modality].float()
                summary_map = old_peak[:, :, None].expand(-1, -1, sim.shape[-1])
                summary_response = torch.einsum('bn,bns->bs', weights[:, modality], summary_map)
                state, _ = self._encode(
                    summary_map, summary_response, coverage[:, modality], agreement)
            states.append(state)
            spatial_maps.append(real_spatial)
        return states[0], states[1], torch.cat(support_logits, dim=1), coverage, spatial_maps


def reset_structured_spatial_evidence(model):
    count = sum(1 for module in model.modules() if isinstance(module, StructuredSpatialEvidence))
    return count
