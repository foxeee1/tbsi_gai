from . import BaseActor
import copy
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from ...utils.heapmap_utils import generate_heatmap


class TBSITrackActor(BaseActor):
    """ Actor for training TBSI_Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg
        self.reference_backbone = None
        self._reference_attention = None
        tsms = getattr(getattr(cfg, 'MODEL', None), 'TSMS', None)
        if tsms is not None and getattr(tsms, 'ENABLE', False) and getattr(tsms, 'PRESERVE_ENABLE', False):
            self.reference_backbone = copy.deepcopy(self.net.backbone).eval()
            for parameter in self.reference_backbone.parameters():
                parameter.requires_grad_(False)

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data['visible'])
        return loss, status

    def forward_pass(self, data):
        template_img_v = data['visible']['template_images'][0].view(-1, *data['visible']['template_images'].shape[2:])
        template_img_i = data['infrared']['template_images'][0].view(-1, *data['infrared']['template_images'].shape[2:])

        num_search = data['visible']['search_images'].shape[0]
        use_temporal = getattr(self.cfg.MODEL, "TEMPORAL_TOKENS", False)

        # Two-frame training: prev frame updates tokens, current frame uses them
        if use_temporal and num_search == 2:
            prev_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])
            prev_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])
            curr_v = data['visible']['search_images'][1].view(-1, *data['visible']['search_images'].shape[2:])
            curr_i = data['infrared']['search_images'][1].view(-1, *data['infrared']['search_images'].shape[2:])

            out_dict = self.net(template=[template_img_v, template_img_i],
                                search=[curr_v, curr_i],
                                prev_search=[prev_v, prev_i],
                                return_last_attn=False)
        else:
            search_img_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])
            search_img_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])

            out_dict = self.net(template=[template_img_v, template_img_i],
                                search=[search_img_v, search_img_i],
                                return_last_attn=False)

        self._reference_attention = None
        if self.reference_backbone is not None and not use_temporal:
            with torch.no_grad():
                self.reference_backbone.forward_features(
                    [template_img_v.detach(), template_img_i.detach()],
                    [search_img_v.detach(), search_img_i.detach()])
                getter = getattr(self.reference_backbone, 'get_smsa_reference_attention', None)
                if getter is not None:
                    self._reference_attention = getter()

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)

        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        tsms_settings = getattr(self.cfg.MODEL, 'TSMS', None)
        if tsms_settings is not None and getattr(tsms_settings, 'UTILITY_ENABLE', False):
            tsms_loss, tsms_status = self._compute_tsms_utility_loss(
                loss, iou, num_queries, loss.device)
        else:
            tsms_loss, tsms_status = self._compute_tsms_loss(gt_bbox, loss.device)
        loss = loss + tsms_loss
        # The attention tensors are retained by the backbone only so this
        # method can add L_cov to the current graph. Clear that reference as
        # soon as backward reaches the final loss to avoid graph accumulation.
        if tsms_loss.requires_grad:
            clear_fn = getattr(self.net.backbone, 'clear_tsms_attention', None)
            if clear_fn is not None:
                loss.register_hook(lambda _grad: clear_fn())

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            status.update(tsms_status)
            smsa_stats = getattr(self.net.backbone, 'get_smsa_stats', lambda: None)()
            if smsa_stats is not None:
                status.update({
                    'SMSA/entropy': smsa_stats['entropy'],
                    'SMSA/zero_ratio': smsa_stats['zero_ratio'],
                    'SMSA/effective_support': smsa_stats['effective_support'],
                    'SMSA/norm_error': smsa_stats['norm_error'],
                    'SMSA/nan_inf': smsa_stats['nan_inf'],
                })
            return loss, status
        else:
            return loss

    def _compute_tsms_utility_loss(self, tracking_loss, iou, num_queries, device):
        """Rank attention by first-order tracking utility on difficult samples.

        The utility target is detached, so the auxiliary loss does not create
        a second-order graph. Attention remains the only student-side graph.
        """
        settings = getattr(self.cfg.MODEL, 'TSMS', None)
        get_attention = getattr(self.net.backbone, 'get_tsms_attention', None)
        if settings is None or get_attention is None:
            return torch.zeros((), device=device), {'TSMS/utility_available': 0.0}
        records = get_attention()
        if not records or not tracking_loss.requires_grad:
            return torch.zeros((), device=device), {'TSMS/utility_available': 0.0}

        iou_per_sample = iou.detach().reshape(-1, num_queries).mean(dim=1)
        hard = iou_per_sample < float(getattr(settings, 'UTILITY_IOU_THRESHOLD', 0.50))
        hard_count = int(hard.sum().item())
        if hard_count == 0:
            return tracking_loss.new_zeros(()), {
                'TSMS/utility_available': 0.0,
                'TSMS/utility_hard_fraction': 0.0,
            }

        attention_tensors = [attention for _, _, attention in records]
        gradients = torch.autograd.grad(
            tracking_loss, tuple(attention_tensors), retain_graph=True,
            create_graph=False, allow_unused=True)
        margin = float(getattr(settings, 'UTILITY_MARGIN', 0.01))
        topk = int(getattr(settings, 'UTILITY_TOPK', 8))
        losses = []
        valid_maps = 0
        for (_, _, attention), gradient in zip(records, gradients):
            # The cross-attention maps used by TSMS must index the 16x16
            # search tokens on their last dimension.
            if gradient is None or attention.dim() != 3 or attention.shape[-1] != 256:
                continue
            k = min(topk, max(1, (attention.shape[-1] - 1) // 2))
            utility = -gradient.detach().float()
            utility = torch.where(torch.isfinite(utility), utility, torch.zeros_like(utility))
            student = attention.float()
            positive = utility.topk(k, dim=-1, largest=True).indices
            negative = utility.topk(k, dim=-1, largest=False).indices
            positive_attention = student.gather(-1, positive)
            negative_attention = student.gather(-1, negative)
            pair_loss = F.relu(margin - positive_attention + negative_attention).mean(dim=-1)
            hard_mask = hard.to(device=pair_loss.device, dtype=pair_loss.dtype).view(-1, 1)
            losses.append((pair_loss * hard_mask).sum() / hard_mask.sum().clamp_min(1.0))
            valid_maps += 1

        if not losses:
            return tracking_loss.new_zeros(()), {
                'TSMS/utility_available': 0.0,
                'TSMS/utility_hard_fraction': float(hard.float().mean().item()),
            }
        utility_raw = torch.stack(losses).mean()
        utility_loss = float(getattr(settings, 'UTILITY_LAMBDA', 0.005)) * utility_raw
        return utility_loss, {
            'Loss/tsms_utility': utility_loss.detach().item(),
            'TSMS/Lutility_raw': utility_raw.detach().item(),
            'TSMS/utility_available': float(valid_maps),
            'TSMS/utility_hard_fraction': float(hard.float().mean().item()),
        }

    def _build_tsms_mask(self, boxes, device, height=16, width=16):
        """Build a differentiable-free soft GT box mask on search tokens."""
        settings = getattr(self.cfg.MODEL, 'TSMS', None)
        dilation = float(getattr(settings, 'DILATION_TOKENS', 1.0))
        softness = max(float(getattr(settings, 'SOFTNESS_TOKENS', 0.5)), 1e-3)
        softness = softness / float(width)
        dilation = dilation / float(width)
        ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
        xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
        yy, xx = torch.meshgrid(ys, xs)
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)
        x, y, w, h = boxes.float().unbind(-1)
        left = (x - dilation).clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
        top = (y - dilation).clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
        right = (x + w + dilation).clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
        bottom = (y + h + dilation).clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
        mask_x = torch.sigmoid((xx - left) / softness) * torch.sigmoid((right - xx) / softness)
        mask_y = torch.sigmoid((yy - top) / softness) * torch.sigmoid((bottom - yy) / softness)
        return (mask_x * mask_y).flatten(1)

    def _compute_tsms_loss(self, gt_bbox, device):
        settings = getattr(self.cfg.MODEL, 'TSMS', None)
        if settings is None or not getattr(settings, 'ENABLE', False):
            clear_fn = getattr(self.net.backbone, 'clear_tsms_attention', None)
            if clear_fn is not None:
                clear_fn()
            return torch.zeros((), device=device), {}
        get_attention = getattr(self.net.backbone, 'get_tsms_attention', None)
        if get_attention is None:
            return torch.zeros((), device=device), {'TSMS/available': 0.0}
        records = get_attention()
        if not records:
            return torch.zeros((), device=device), {'TSMS/available': 0.0}

        mask = self._build_tsms_mask(gt_bbox, device)
        rho = float(getattr(settings, 'RHO', 0.20))
        losses = []
        js_losses = []
        metrics = {}
        reference_by_key = {}
        for location, direction, attention in (self._reference_attention or []):
            reference_by_key[(location, direction)] = attention.to(device=device)
        for location, direction, attention in records:
            # Attention_st keeps the native [B, query_tokens, search_tokens]
            # layout; average only over search queries to obtain one spatial
            # distribution per sample.
            probs = attention.float().mean(dim=1)
            reference_probs = reference_by_key.get((location, direction))
            if reference_probs is not None:
                reference_probs = reference_probs.float().mean(dim=1)
                reference_probs = reference_probs / reference_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            support = (probs > 1e-8).float()
            gt_mass = (probs * mask).sum(dim=-1)
            bg_mass = (probs * (1.0 - mask)).sum(dim=-1)
            gt_precision = (support * mask).sum(dim=-1) / (support.sum(dim=-1) + 1e-6)
            gt_recall = (support * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-6)
            if getattr(settings, 'RELATIVE_COVERAGE', False) and reference_probs is not None:
                reference_gt_mass = (reference_probs * mask).sum(dim=-1)
                rho_value = reference_gt_mass + float(getattr(settings, 'RELATIVE_DELTA', 0.02))
            else:
                rho_value = torch.full_like(gt_mass, rho)
            coverage = F.relu(rho_value - gt_mass)
            losses.append(coverage.mean())

            if reference_probs is not None and getattr(settings, 'PRESERVE_ENABLE', False):
                p = probs.clamp_min(1e-8)
                q = reference_probs.clamp_min(1e-8)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                midpoint = 0.5 * (p + q)
                js_losses.append(0.5 * (
                    (p * (p.log() - midpoint.log())).sum(dim=-1) +
                    (q * (q.log() - midpoint.log())).sum(dim=-1)
                ).mean())

            p = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            entropy = -(p * p.log()).sum(dim=-1)
            prefix = 'TSMS/L%d/%s/' % (location, direction)
            metrics[prefix + 'GTMass'] = gt_mass.detach().mean().item()
            metrics[prefix + 'GTPrecision'] = gt_precision.detach().mean().item()
            metrics[prefix + 'GTRecall'] = gt_recall.detach().mean().item()
            metrics[prefix + 'BGMass'] = bg_mass.detach().mean().item()
            metrics[prefix + 'effective_support'] = support.sum(dim=-1).detach().mean().item()
            metrics[prefix + 'entropy'] = entropy.detach().mean().item()
            metrics[prefix + 'zero_ratio'] = (support == 0).float().mean().item()

        cov_loss = torch.stack(losses).mean()
        preserve_loss = torch.stack(js_losses).mean() if js_losses else cov_loss.new_zeros(())
        tsms_loss = (float(getattr(settings, 'LAMBDA_COV', 0.05)) * cov_loss +
                     float(getattr(settings, 'LAMBDA_PRESERVE', 0.01)) * preserve_loss)
        metrics['Loss/tsms_cov'] = tsms_loss.detach().item()
        metrics['TSMS/Lcov_raw'] = cov_loss.detach().item()
        metrics['TSMS/Lpreserve_js'] = preserve_loss.detach().item()
        metrics['TSMS/available'] = float(len(records))
        return tsms_loss, metrics
