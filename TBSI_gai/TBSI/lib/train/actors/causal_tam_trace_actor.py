"""Frozen-head causal TAM actor for trace-backed optimization audits."""
import torch
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap


def image_box_to_normalized_crop(boxes_image, transforms, crop_size=256):
    """Map image xywh boxes through per-sample image-to-crop affine transforms."""
    boxes = []
    for box, affine in zip(boxes_image, transforms):
        x, y, w, h = box.unbind(0)
        corners = torch.stack((
            torch.stack((x, y, x.new_tensor(1.0))),
            torch.stack((x + w, y, x.new_tensor(1.0))),
            torch.stack((x, y + h, x.new_tensor(1.0))),
            torch.stack((x + w, y + h, x.new_tensor(1.0))),
        ))
        points = corners @ affine.transpose(0, 1)
        x0, y0 = points[:, 0].min(), points[:, 1].min()
        x1, y1 = points[:, 0].max(), points[:, 1].max()
        crop_box = torch.stack((x0, y0, x1 - x0, y1 - y0)) / float(crop_size)
        boxes.append(crop_box)
    return torch.stack(boxes).clamp(0.0, 1.0)


class CausalTAMTraceActor:
    def __init__(self, net, optimizer, loss_weights=None, crop_size=256, stride=16):
        self.net = net
        self.optimizer = optimizer
        self.loss_weights = loss_weights or {'giou': 2.0, 'l1': 5.0, 'focal': 1.0}
        self.crop_size = int(crop_size)
        self.stride = int(stride)
        self.focal = FocalLoss()
        self.net.eval()
        self.net.causal_tam.train()
        for name, parameter in self.net.named_parameters():
            parameter.requires_grad_(name.startswith('causal_tam.'))

    def step(self, batch):
        device = next(self.net.causal_tam.parameters()).device
        current = batch['current_raw'].to(device=device, dtype=torch.float32)
        history = batch['history_raw'].to(device=device, dtype=torch.float32)
        current_rect = batch['current_crop_xywh'].to(device)
        history_rect = batch['history_crop_xywh'].to(device)
        frame_ids = batch['frame_id'].to(device)
        history_ids = batch['history_frame_id'].to(device)
        sequence_ids = list(batch['sequence_id'])
        if (history_ids >= frame_ids).any():
            raise ValueError('history frame must precede current frame')
        if len(sequence_ids) != len(history_ids):
            raise ValueError('sequence id batch length mismatch')

        self.optimizer.zero_grad(set_to_none=True)
        out, stats = self.net.causal_tam(
            current=current,
            history=history.detach(),
            current_rect=current_rect,
            history_rect=history_rect,
            current_frame_id=frame_ids,
            history_frame_id=history_ids,
            current_sequence_id=sequence_ids,
            history_sequence_id=sequence_ids,
            return_stats=True,
        )
        token_ratio = ((out.float() - current.float()).norm(dim=-1) /
                       current.float().norm(dim=-1).clamp_min(1e-6))
        feature_map = out.transpose(1, 2).reshape(-1, 768, 16, 16)
        gt = image_box_to_normalized_crop(
            batch['gt_box_image'].to(device),
            batch['current_crop_transform'].to(device),
            crop_size=self.crop_size,
        )
        if (gt[:, 2:] <= 0).any() or not torch.isfinite(gt).all():
            raise ValueError('invalid GT after crop-coordinate transform')

        heatmap = generate_heatmap(gt.unsqueeze(0), self.crop_size, self.stride)[0]
        score_map, pred_boxes, _, _ = self.net.box_head(feature_map, heatmap)
        pred_boxes = pred_boxes.reshape(-1, 4)
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_xywh_to_xyxy(gt).clamp(0.0, 1.0)
        giou, iou = giou_loss(pred_xyxy, target_xyxy)
        l1 = torch.nn.functional.l1_loss(pred_xyxy, target_xyxy)
        focal = self.focal(score_map, heatmap.unsqueeze(1))
        loss = (self.loss_weights['giou'] * giou +
                self.loss_weights['l1'] * l1 +
                self.loss_weights['focal'] * focal)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite trace actor loss')

        gamma_before = float(self.net.causal_tam.gamma.detach().item())
        loss.backward()
        grad_norms = {}
        for key, module in (('q', self.net.causal_tam.q),
                            ('k', self.net.causal_tam.k),
                            ('v', self.net.causal_tam.v),
                            ('out', self.net.causal_tam.out)):
            grads = [p.grad for p in module.parameters()]
            grad_norms[key] = float(sum(g.detach().float().norm().item()
                                        for g in grads if g is not None))
        gamma_grad = self.net.causal_tam.gamma.grad
        gamma_grad_value = float(gamma_grad.detach().item()) if gamma_grad is not None else None
        self.optimizer.step()
        gamma_after = float(self.net.causal_tam.gamma.detach().item())
        return {
            'loss': float(loss.detach().item()),
            'loss_giou': float(giou.detach().item()),
            'loss_l1': float(l1.detach().item()),
            'loss_focal': float(focal.detach().item()),
            'mean_iou': float(iou.detach().mean().item()),
            'gamma_before': gamma_before,
            'gamma_after': gamma_after,
            'gamma_grad': gamma_grad_value,
            'q_grad': grad_norms['q'],
            'k_grad': grad_norms['k'],
            'v_grad': grad_norms['v'],
            'out_grad': grad_norms['out'],
            'tam_stats': stats,
            'actual_ratio_mean': stats.get('actual_ratio_mean', 0.0),
            'actual_ratio_p95': float(torch.quantile(token_ratio.detach().flatten(), 0.95).item()),
            'actual_ratio_max': float(token_ratio.detach().max().item()),
            'finite': bool(torch.isfinite(out).all().item()),
        }
