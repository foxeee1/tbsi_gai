"""Training-only local utility ranking for the existing Early MSHA source gate."""
import torch
import torch.nn.functional as F

from lib.models.layers.tbsi_layer import V2TemplateFusion
from lib.test.utils.hann import hann2d


class MSHASourceUtilityRankLoss:
    def __init__(self, model, margin=0.01, mode='single'):
        if margin <= 0:
            raise ValueError('utility rank margin must be positive')
        if mode not in ('single', 'stable_region_pair'):
            raise ValueError('unknown source utility ranking mode: ' + str(mode))
        self.margin = float(margin)
        self.mode = mode
        self.owners = [module for module in model.modules()
                       if isinstance(module, V2TemplateFusion)
                       and module.early_gate_value_modulation
                       and module.early_gate_source_mode == 'dynamic']
        if len(self.owners) != 3:
            raise ValueError('expected exactly three dynamic Early gate fusion layers')
        for module in self.owners:
            module.capture_gate_utility = True
            module.last_gate_log_gain = None

    def _target_margins(self, score_map, gt_bbox):
        if score_map.ndim != 4 or score_map.shape[1] != 1:
            raise ValueError('CENTER score map must be [B,1,H,W]')
        if gt_bbox.shape != (score_map.shape[0], 4):
            raise ValueError('search GT boxes must be [B,4] xywh')
        bsz, _, height, width = score_map.shape
        device = score_map.device
        window = hann2d(torch.tensor([height, width], dtype=torch.long), centered=True)
        window = window.to(device=device, dtype=torch.float32).reshape(1, height, width)
        score = score_map[:, 0].float()
        log_response = torch.log((score * window).clamp_min(1e-8))

        boxes = gt_bbox.float()
        x, y, box_w, box_h = boxes.unbind(-1)
        cx, cy = x + 0.5 * box_w, y + 0.5 * box_h
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
            indexing='ij')
        positive_center = ((xx[None] - cx[:, None, None]).abs() <= 1.0 / width) & (
            (yy[None] - cy[:, None, None]).abs() <= 1.0 / height)
        positive_box = ((xx[None] >= x[:, None, None]) &
                        (xx[None] <= x[:, None, None] + box_w[:, None, None]) &
                        (yy[None] >= y[:, None, None]) &
                        (yy[None] <= y[:, None, None] + box_h[:, None, None]))
        expand_x, expand_y = 1.0 / width, 1.0 / height
        inside_dilated = ((xx[None] >= x[:, None, None] - expand_x) &
                          (xx[None] <= x[:, None, None] + box_w[:, None, None] + expand_x) &
                          (yy[None] >= y[:, None, None] - expand_y) &
                          (yy[None] <= y[:, None, None] + box_h[:, None, None] + expand_y))
        negative = ~inside_dilated
        in_crop = ((cx >= 0) & (cx <= 1) & (cy >= 0) & (cy <= 1) &
                   torch.isfinite(boxes).all(dim=-1) & (box_w > 0) & (box_h > 0))
        valid = (positive_center.flatten(1).any(-1) & positive_box.flatten(1).any(-1) &
                 negative.flatten(1).any(-1) & in_crop)
        center_peak = log_response.masked_fill(~positive_center, -torch.inf).flatten(1).amax(-1)
        box_peak = log_response.masked_fill(~positive_box, -torch.inf).flatten(1).amax(-1)
        neg_peak = log_response.masked_fill(~negative, -torch.inf).flatten(1).amax(-1)
        center_margin = torch.where(valid, center_peak - neg_peak, torch.zeros_like(center_peak))
        box_margin = torch.where(valid, box_peak - neg_peak, torch.zeros_like(box_peak))
        return center_margin, box_margin, valid

    def compute(self, tracking_loss, score_map, gt_bbox, calibration=False):
        gains = [module.last_gate_log_gain for module in self.owners]
        if any(value is None for value in gains):
            raise RuntimeError('one or more Early gate log-gains were not captured')
        if any(value.ndim != 3 or value.shape[-2:] != (64, 2) for value in gains):
            raise ValueError('expected log-gain tensors [B,64,2] at all three layers')

        q_center, q_box, valid = self._target_margins(score_map, gt_bbox)
        utility_center = torch.autograd.grad(q_center.sum(), tuple(gains), retain_graph=True,
                                             create_graph=False, allow_unused=True)
        if self.mode == 'stable_region_pair':
            utility_box = torch.autograd.grad(q_box.sum(), tuple(gains), retain_graph=True,
                                              create_graph=False, allow_unused=True)
        else:
            utility_box = utility_center
        layer_losses, valid_rows, utility_abs = [], 0, []
        stable_pairs, candidate_pairs = 0, 0
        for gain, raw_center, raw_box in zip(gains, utility_center, utility_box):
            if raw_center is None or raw_box is None:
                continue
            u_center = raw_center.detach().float()
            u_box = raw_box.detach().float()
            a = gain.float()
            for modality in range(2):
                center_u = u_center[:, :, modality]
                box_u = u_box[:, :, modality]
                finite = torch.isfinite(center_u) & torch.isfinite(box_u)
                safe_center = torch.where(finite, center_u, torch.zeros_like(center_u))
                safe_box = torch.where(finite, box_u, torch.zeros_like(box_u))
                score_u = (safe_center + safe_box) * 0.5
                score_a = a[:, :, modality]
                ordered, indices = score_u.sort(dim=-1)
                k = max(1, score_u.shape[-1] // 4)
                low_idx, high_idx = indices[:, :k], indices[:, -k:]
                low_u = ordered[:, :k].mean(-1)
                high_u = ordered[:, -k:].mean(-1)
                usable = (valid & (finite.sum(-1) == score_u.shape[-1]) &
                          torch.isfinite(high_u) & torch.isfinite(low_u) &
                          ((high_u - low_u) > 1e-10))
                low_gain = score_a.gather(1, low_idx).mean(-1)
                high_gain = score_a.gather(1, high_idx).mean(-1)
                if self.mode == 'stable_region_pair':
                    high_center = center_u.gather(1, high_idx)
                    low_center = center_u.gather(1, low_idx)
                    high_box = box_u.gather(1, high_idx)
                    low_box = box_u.gather(1, low_idx)
                    center_delta = high_center[:, :, None] - low_center[:, None, :]
                    box_delta = high_box[:, :, None] - low_box[:, None, :]
                    stable = (torch.isfinite(center_delta) & torch.isfinite(box_delta) &
                              (center_delta > 1e-10) & (box_delta > 1e-10))
                    candidate_pairs += int(stable.numel())
                    stable_pairs += int(stable.sum().item())
                    gain_delta = (score_a.gather(1, high_idx)[:, :, None] -
                                  score_a.gather(1, low_idx)[:, None, :])
                    pair_loss = F.relu(self.margin - gain_delta)
                    pair_count = stable.sum(dim=(1, 2))
                    row_loss = (pair_loss * stable.to(pair_loss.dtype)).sum(dim=(1, 2)) / pair_count.clamp_min(1)
                    usable = usable & (pair_count > 0)
                else:
                    row_loss = F.relu(self.margin - (high_gain - low_gain))
                    stable_pairs += int(usable.sum().item())
                    candidate_pairs += int(usable.numel())
                if usable.any():
                    layer_losses.append(row_loss[usable].mean())
                    valid_rows += int(usable.sum().item())
                    utility_abs.append((high_u[usable] - low_u[usable]).mean())

        if layer_losses:
            rank_loss = torch.stack(layer_losses).mean()
        else:
            rank_loss = tracking_loss.new_zeros(())

        stats = {
            'Loss/source_utility_rank_raw': float(rank_loss.detach()),
            'MSHA/source_utility_valid_rows': float(valid_rows),
            'MSHA/source_utility_valid_gt_fraction': float(valid.float().mean()),
            'MSHA/source_utility_stable_pair_fraction': (
                float(stable_pairs / candidate_pairs) if candidate_pairs else 0.0),
            'MSHA/source_utility_mean_top_bottom_gap': (
                float(torch.stack(utility_abs).mean()) if utility_abs else 0.0),
        }
        if calibration:
            task_grad = torch.autograd.grad(tracking_loss, tuple(gains), retain_graph=True,
                                            create_graph=False, allow_unused=True)
            rank_grad = (torch.autograd.grad(rank_loss, tuple(gains), retain_graph=True,
                                             create_graph=False, allow_unused=True)
                         if rank_loss.requires_grad else tuple(None for _ in gains))
            task_norm = self._global_norm(task_grad)
            rank_norm = self._global_norm(rank_grad)
            stats['MSHA/source_utility_task_gate_grad_norm'] = task_norm
            stats['MSHA/source_utility_rank_gate_grad_norm'] = rank_norm
        for module in self.owners:
            module.last_gate_log_gain = None
        return rank_loss, stats

    @staticmethod
    def _global_norm(gradients):
        values = [g.detach().float().square().sum() for g in gradients if g is not None]
        return float(torch.stack(values).sum().sqrt()) if values else 0.0
