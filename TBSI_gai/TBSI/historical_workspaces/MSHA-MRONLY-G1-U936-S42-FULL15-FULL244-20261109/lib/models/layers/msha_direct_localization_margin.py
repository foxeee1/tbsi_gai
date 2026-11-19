"""Training-only Hann-weighted target-versus-competitor margin loss."""
import torch
import torch.nn.functional as F

from lib.test.utils.hann import hann2d


class MSHADirectLocalizationMarginLoss:
    def __init__(self, margin=0.1, eps=1e-8):
        if margin < 0 or eps <= 0:
            raise ValueError('margin must be nonnegative and eps must be positive')
        self.margin = float(margin)
        self.eps = float(eps)

    def __call__(self, score_map, gt_bbox):
        if score_map.ndim != 4 or score_map.shape[1] != 1:
            raise ValueError('CENTER score map must have shape [B,1,H,W]')
        if gt_bbox.shape != (score_map.shape[0], 4):
            raise ValueError('search GT boxes must have shape [B,4] in xywh format')
        batch, _, height, width = score_map.shape
        device = score_map.device
        window = hann2d(torch.tensor([height, width], dtype=torch.long), centered=True)
        window = window.to(device=device, dtype=torch.float32).reshape(1, height, width)
        response = (score_map[:, 0].float() * window).clamp_min(self.eps)
        log_response = response.log()

        boxes = gt_bbox.float()
        x, y, box_w, box_h = boxes.unbind(-1)
        center_x, center_y = x + 0.5 * box_w, y + 0.5 * box_h
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
            indexing='ij')

        positive = ((xx[None] - center_x[:, None, None]).abs() <= 1.0 / width) & (
            (yy[None] - center_y[:, None, None]).abs() <= 1.0 / height)
        expand_x, expand_y = 1.0 / width, 1.0 / height
        inside_dilated_box = ((xx[None] >= x[:, None, None] - expand_x) &
                              (xx[None] <= x[:, None, None] + box_w[:, None, None] + expand_x) &
                              (yy[None] >= y[:, None, None] - expand_y) &
                              (yy[None] <= y[:, None, None] + box_h[:, None, None] + expand_y))
        negative = ~inside_dilated_box
        valid = (torch.isfinite(boxes).all(-1) & (box_w > 0) & (box_h > 0) &
                 (center_x >= 0) & (center_x <= 1) & (center_y >= 0) & (center_y <= 1) &
                 positive.flatten(1).any(-1) & negative.flatten(1).any(-1))

        positive_peak = log_response.masked_fill(~positive, -torch.inf).flatten(1).amax(-1)
        negative_peak = log_response.masked_fill(~negative, -torch.inf).flatten(1).amax(-1)
        q = positive_peak - negative_peak
        per_sample = F.softplus(self.margin - q)
        loss = per_sample[valid].mean() if valid.any() else score_map.sum() * 0.0
        stats = {
            'MSHA/direct_margin_valid_fraction': float(valid.float().mean()),
            'MSHA/direct_margin_q_mean': float(q[valid].detach().mean()) if valid.any() else 0.0,
            'MSHA/direct_margin_loss_raw': float(loss.detach()),
        }
        return loss, stats
