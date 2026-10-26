from . import BaseActor
import copy
import math
import os
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from ...utils.heapmap_utils import generate_heatmap
from lib.models.layers.tbsi_layer import template_box_to_token_weights
from lib.test.utils.hann import hann2d


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
        self._forward_step = 0
        self._current_epoch = 0
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
        self._current_epoch = int(data.get('epoch', self._current_epoch))
        template_img_v = data['visible']['template_images'][0].view(-1, *data['visible']['template_images'].shape[2:])
        template_img_i = data['infrared']['template_images'][0].view(-1, *data['infrared']['template_images'].shape[2:])
        expert_mode = getattr(self.cfg.MODEL.BACKBONE, 'TEMPLATE_EXPERT_MODE', 'off')
        template_fg_masks = None
        freq_mode = getattr(self.cfg.MODEL.BACKBONE, 'FREQ_EXCHANGE_MODE', 'off')
        der_enabled = getattr(self.cfg.MODEL.BACKBONE,
                              'DISCRIMINATIVE_EVIDENCE_ROUTER', False)
        utility_router_enabled = getattr(
            self.cfg.MODEL.BACKBONE, 'EARLY_UTILITY_ROUTER_ENABLE', False)
        role_pair_bias_enabled = getattr(
            self.cfg.MODEL.BACKBONE, 'MSHA_ROLE_PAIR_BIAS_ENABLE', False)
        spatial_token_bridge_enabled = getattr(
            getattr(self.cfg.MODEL, 'SPATIAL_TOKEN_BRIDGE', None), 'ENABLE', False)
        local_correspondence_enabled = getattr(
            self.cfg.MODEL.BACKBONE, 'LOCAL_CROSS_MODAL_CORRESPONDENCE_ENABLE', False)
        sparse_route_enabled = os.environ.get(
            'TBSI_SPARSE_CROSS_MODE', 'off').lower() != 'off'
        content_stser_enabled = getattr(
            self.cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_MODE', 'off') == 'stser_competition_content'
        if (expert_mode in ('target', 'target_disp_only', 'target_soft') or
                freq_mode != 'off' or der_enabled or utility_router_enabled or
                role_pair_bias_enabled or
                sparse_route_enabled or spatial_token_bridge_enabled or
                local_correspondence_enabled or content_stser_enabled):
            boxes = data['visible']['template_anno']
            boxes = boxes.reshape(-1, 4).to(template_img_v.device)
            weights = template_box_to_token_weights(boxes, grid_size=8)
            if weights.shape[0] != template_img_v.shape[0]:
                raise ValueError('template annotation batch does not match template image batch')
            template_fg_masks = (weights, weights)

        self._forward_step += 1
        for module in self.net.modules():
            setter = getattr(module, 'set_diagnostics_context', None)
            if setter is not None:
                setter(self._current_epoch, self._forward_step, 'train')

        num_search = data['visible']['search_images'].shape[0]
        use_temporal = getattr(self.cfg.MODEL, "TEMPORAL_TOKENS", False)
        use_tam = getattr(self.net, 'use_causal_tam', False)
        if use_tam and num_search < 2:
            raise RuntimeError('CAUSAL_TAM training requires explicit two-frame search pairs')

        # Two-frame training: prev frame updates tokens, current frame uses them
        if (use_temporal or use_tam) and num_search == 2:
            prev_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])
            prev_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])
            curr_v = data['visible']['search_images'][1].view(-1, *data['visible']['search_images'].shape[2:])
            curr_i = data['infrared']['search_images'][1].view(-1, *data['infrared']['search_images'].shape[2:])

            out_dict = self.net(template=[template_img_v, template_img_i],
                                search=[curr_v, curr_i],
                                prev_search=[prev_v, prev_i],
                                current_crop_xywh=data['visible'].get('current_crop_xywh'),
                                history_crop_xywh=data['visible'].get('history_crop_xywh'),
                                current_frame_id=data['visible'].get('frame_id'),
                                history_frame_id=data['visible'].get('history_frame_id'),
                                current_sequence_id=data['visible'].get('sequence_id'),
                                history_sequence_id=data['visible'].get('history_sequence_id'),
                                return_last_attn=False,
                                template_fg_masks=template_fg_masks)
        else:
            search_img_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])
            search_img_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])

            out_dict = self.net(template=[template_img_v, template_img_i],
                                search=[search_img_v, search_img_i],
                                return_last_attn=False,
                                template_fg_masks=template_fg_masks)

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
        utility_router_enabled = getattr(
            self.cfg.MODEL.BACKBONE, 'EARLY_UTILITY_ROUTER_ENABLE', False)
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
            score_for_location = pred_dict['score_map']
            if getattr(getattr(self.cfg, 'TRAIN', None), 'HANN_AWARE_CENTER_LOSS', False):
                # Match the probability map used by the tracker before argmax.
                size = torch.tensor(score_for_location.shape[-2:], device=score_for_location.device)
                window = hann2d(size, centered=True).to(
                    device=score_for_location.device, dtype=score_for_location.dtype)
                score_for_location = score_for_location * window
            location_loss = self.objective['focal'](score_for_location, gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        center_rank_loss = torch.zeros((), device=l1_loss.device)
        if (getattr(getattr(self.cfg, 'TRAIN', None), 'CENTER_RANK_LOSS_ENABLE', False)
                and 'score_map' in pred_dict):
            # Train the discrete center contract used by CENTER argmax. The
            # optional Hann/failure switches make the auxiliary target match
            # deployment and focus updates on actual background-peak errors.
            rank_score = pred_dict['score_map']
            rank_cfg = getattr(self.cfg, 'TRAIN', None)
            if getattr(rank_cfg, 'CENTER_RANK_LOSS_HANN_AWARE', False):
                size = torch.tensor(rank_score.shape[-2:], device=rank_score.device)
                rank_score = rank_score * hann2d(size, centered=True).to(
                    device=rank_score.device, dtype=rank_score.dtype)
            flat_score = rank_score.flatten(1)
            flat_target = gt_gaussian_maps.flatten(1)
            target_threshold = float(getattr(
                rank_cfg, 'CENTER_RANK_LOSS_TARGET_THRESHOLD', 0.5))
            target_mask = flat_target >= target_threshold
            # A region maximum is less brittle than supervising one rounded
            # cell, while the negative is exactly the highest competing peak.
            pos_score = flat_score.masked_fill(~target_mask, -1.0).max(dim=1, keepdim=True).values
            neg_mask = ~target_mask
            neg_score = flat_score.masked_fill(~neg_mask, -1.0)
            topk = min(int(getattr(self.cfg.TRAIN, 'CENTER_RANK_LOSS_TOPK', 8)), neg_score.shape[1])
            hard_neg = neg_score.topk(topk, dim=1, largest=True).values
            if getattr(self.cfg.TRAIN, 'CENTER_RANK_LOSS_PAIRWISE', False):
                # Use only the current highest background peak. This keeps the
                # auxiliary target aligned with the deployment argmax instead
                # of averaging gradients over many low-scoring cells.
                hard_neg = hard_neg[:, :1]
            margin = float(getattr(self.cfg.TRAIN, 'CENTER_RANK_LOSS_MARGIN', 0.05))
            pairwise = F.relu(margin - pos_score + hard_neg)
            if getattr(rank_cfg, 'CENTER_RANK_LOSS_FAILURE_ONLY', False):
                # Gate from the current discrete decision, detached so the
                # condition cannot itself become an optimization shortcut.
                top1 = flat_score.detach().argmax(dim=1, keepdim=True)
                failed = (~target_mask.gather(1, top1)).to(pairwise.dtype)
                center_rank_raw = (pairwise.mean(dim=1) * failed).sum() / failed.sum().clamp_min(1.0)
            else:
                center_rank_raw = pairwise.mean()
            center_rank_loss = float(getattr(self.cfg.TRAIN, 'CENTER_RANK_LOSS_WEIGHT', 0.10)) * center_rank_raw

        der_rank_loss = torch.zeros((), device=l1_loss.device)
        der_maps = pred_dict.get('der_score_maps')
        if der_maps:
            rank_cfg = getattr(self.cfg, 'TRAIN', None)
            der_margin = float(getattr(rank_cfg, 'DER_RANK_LOSS_MARGIN', 0.10))
            der_topk = max(1, int(getattr(rank_cfg, 'DER_RANK_LOSS_TOPK', 8)))
            gt = gt_bbox.float().clamp(0, 1)
            terms = []
            for rgb_score, tir_score in der_maps:
                for score in (rgb_score, tir_score):
                    side = int(round(score.shape[-1] ** 0.5))
                    if score.shape[-1] != side * side:
                        raise ValueError('DER score map must have square grid')
                    x, y, w, h = gt.unbind(-1)
                    coords = (torch.arange(side, device=score.device,
                                            dtype=score.dtype) + 0.5) / side
                    yy, xx = torch.meshgrid(coords, coords)
                    xx, yy = xx.flatten()[None], yy.flatten()[None]
                    positive = ((xx >= x[:, None]) & (xx <= (x + w)[:, None]) &
                                (yy >= y[:, None]) & (yy <= (y + h)[:, None]))
                    center_x = ((x + 0.5 * w) * side).long().clamp(0, side - 1)
                    center_y = ((y + 0.5 * h) * side).long().clamp(0, side - 1)
                    center_index = center_y * side + center_x
                    positive.scatter_(1, center_index[:, None], True)
                    positive_score = score.masked_fill(~positive, float('-inf'))
                    positive_score = torch.logsumexp(positive_score, dim=-1) - torch.log(
                        positive.float().sum(-1).clamp_min(1))
                    dilated = F.max_pool2d(positive.reshape(-1, 1, side, side).float(),
                                           kernel_size=3, stride=1, padding=1)
                    negative = dilated.flatten(1) < 0.5
                    negative_score = score.masked_fill(~negative, float('-inf'))
                    k = min(der_topk, max(1, score.shape[-1] - 1))
                    hard = negative_score.topk(k, dim=-1).values
                    hard_score = torch.logsumexp(hard, dim=-1) - math.log(k)
                    terms.append(F.softplus(der_margin - positive_score + hard_score).mean())
                der_rank_loss = float(getattr(rank_cfg, 'DER_RANK_LOSS_WEIGHT', 0.05)) * \
                    torch.stack(terms).mean()

        early_utility_loss = torch.zeros((), device=l1_loss.device)
        utility_logits = pred_dict.get('early_utility_logits')
        utility_maps = pred_dict.get('early_utility_teacher_score_maps')
        if utility_router_enabled and utility_logits is not None and utility_maps is not None:
            if utility_logits.ndim != 2 or utility_logits.shape[1] != 2:
                raise ValueError('early utility router logits must have shape [B,2]')
            rgb_map, tir_map = utility_maps
            if rgb_map.shape != tir_map.shape or rgb_map.shape != gt_gaussian_maps.shape:
                raise ValueError('early utility teacher maps must match CENTER target shape')
            positive = gt_gaussian_maps >= 0.5
            if not positive.flatten(1).any(dim=1).all():
                raise ValueError('early utility CENTER target region is empty for a sample')
            def utility(score_map):
                flat_score = score_map.flatten(1)
                flat_positive = positive.flatten(1)
                pos_score = flat_score.masked_fill(~flat_positive, float('-inf')).max(dim=1).values
                bg_score = flat_score.masked_fill(flat_positive, float('-inf')).max(dim=1).values
                return pos_score - bg_score
            utilities = torch.stack((utility(rgb_map), utility(tir_map)), dim=1).detach()
            valid = (utilities[:, 0] - utilities[:, 1]).abs() > 1e-6
            if valid.any():
                labels = utilities.argmax(dim=1)
                raw = F.cross_entropy(utility_logits[valid].float(), labels[valid])
                early_utility_loss = float(getattr(
                    self.cfg.TRAIN, 'EARLY_UTILITY_ROUTER_LOSS_WEIGHT', 0.10)) * raw

        amcm_residual_rank_loss = torch.zeros((), device=l1_loss.device)
        residual_rank_cfg = getattr(getattr(self.cfg, 'TRAIN', None),
                                    'AMCM_RESIDUAL_RANK_LOSS_ENABLE', False)
        if residual_rank_cfg and 'amcm_residual_energy' in pred_dict:
            # Directly rank AMCM residual energy at GT cells above the hardest
            # background cell. This does not alter the score-map loss.
            energy = pred_dict['amcm_residual_energy'][:, 0]
            _, h, w = energy.shape
            yy, xx = torch.meshgrid(
                torch.arange(h, device=energy.device),
                torch.arange(w, device=energy.device))
            bx = gt_bbox[:, 0:1] * w
            by = gt_bbox[:, 1:2] * h
            bw = gt_bbox[:, 2:3] * w
            bh = gt_bbox[:, 3:4] * h
            cell_x0, cell_y0 = xx.float()[None], yy.float()[None]
            target_mask = ((cell_x0 + 1.0 > bx[:, :, None, None]) &
                           (cell_x0 < (bx + bw)[:, :, None, None]) &
                           (cell_y0 + 1.0 > by[:, :, None, None]) &
                           (cell_y0 < (by + bh)[:, :, None, None])).squeeze(1)
            flat_energy = energy.flatten(1)
            flat_mask = target_mask.flatten(1)
            pos = flat_energy.masked_fill(~flat_mask, -1.0).max(dim=1).values
            bg = flat_energy.masked_fill(flat_mask, -1.0)
            topk = min(int(getattr(self.cfg.TRAIN,
                                  'AMCM_RESIDUAL_RANK_LOSS_TOPK', 1)), bg.shape[1])
            hard_bg = bg.topk(topk, dim=1).values[:, 0]
            margin = float(getattr(self.cfg.TRAIN,
                                   'AMCM_RESIDUAL_RANK_LOSS_MARGIN', 0.05))
            raw = F.relu(margin - pos + hard_bg).mean()
            amcm_residual_rank_loss = float(getattr(
                self.cfg.TRAIN, 'AMCM_RESIDUAL_RANK_LOSS_WEIGHT', 0.0)) * raw

        target_rank_loss = torch.zeros((), device=l1_loss.device)
        if (getattr(getattr(self.cfg, 'TRAIN', None), 'TARGET_RANK_LOSS_ENABLE', False)
                and 'target_rank_score' in pred_dict):
            rank_score = pred_dict['target_rank_score'][:, 0]
            _, h, w = rank_score.shape
            yy, xx = torch.meshgrid(
                torch.arange(h, device=rank_score.device),
                torch.arange(w, device=rank_score.device))
            bx = gt_bbox[:, 0:1] * w
            by = gt_bbox[:, 1:2] * h
            bw = gt_bbox[:, 2:3] * w
            bh = gt_bbox[:, 3:4] * h
            target_mask = (((xx.float()[None] + 1.0 > bx[:, :, None, None]) &
                            (xx.float()[None] < (bx + bw)[:, :, None, None]) &
                            (yy.float()[None] + 1.0 > by[:, :, None, None]) &
                            (yy.float()[None] < (by + bh)[:, :, None, None])).squeeze(1))
            flat = rank_score.flatten(1)
            mask = target_mask.flatten(1)
            pos = flat.masked_fill(~mask, -1e4).max(1).values
            bg = flat.masked_fill(mask, -1e4)
            topk = min(int(getattr(self.cfg.TRAIN, 'TARGET_RANK_LOSS_TOPK', 1)), bg.shape[1])
            hard_bg = bg.topk(topk, dim=1).values[:, 0]
            margin = float(getattr(self.cfg.TRAIN, 'TARGET_RANK_LOSS_MARGIN', 0.05))
            raw = F.relu(margin - pos + hard_bg).mean()
            target_rank_loss = float(getattr(
                self.cfg.TRAIN, 'TARGET_RANK_LOSS_WEIGHT', 0.50)) * raw

        amcm_repr_loss = torch.zeros((), device=l1_loss.device)
        repr_cfg = getattr(getattr(self.cfg, 'TRAIN', None), 'AMCM_REPR_LOSS_ENABLE', False)
        if repr_cfg and all(key in pred_dict for key in (
                'amcm_repr_rgb', 'amcm_repr_tir', 'amcm_repr_template_rgb',
                'amcm_repr_template_tir')):
            # Feature-level target/background separation. This is deliberately
            # the only new objective in the representation experiment; the
            # AMCM path, CENTER head and tracker state remain unchanged.
            rgb = F.normalize(pred_dict['amcm_repr_rgb'], dim=1)
            tir = F.normalize(pred_dict['amcm_repr_tir'], dim=1)
            fused = F.normalize(0.5 * (rgb + tir), dim=1)
            proto = F.normalize(0.5 * (
                pred_dict['amcm_repr_template_rgb'].mean(dim=1) +
                pred_dict['amcm_repr_template_tir'].mean(dim=1)), dim=1)
            sim = (fused * proto[:, :, None, None]).sum(dim=1)
            h, w = sim.shape[-2:]
            # PyTorch 1.9 compatibility: the two-argument meshgrid default is
            # row-major (ij), which is the same ordering requested explicitly
            # by newer PyTorch versions.
            yy, xx = torch.meshgrid(
                torch.arange(h, device=sim.device), torch.arange(w, device=sim.device))
            bx = gt_bbox[:, 0:1] * w
            by = gt_bbox[:, 1:2] * h
            bw = gt_bbox[:, 2:3] * w
            bh = gt_bbox[:, 3:4] * h
            cell_x0, cell_y0 = xx.float()[None] , yy.float()[None]
            cell_x1, cell_y1 = cell_x0 + 1.0, cell_y0 + 1.0
            target_mask = ((cell_x1 > bx[:, :, None, None]) &
                           (cell_x0 < (bx + bw)[:, :, None, None]) &
                           (cell_y1 > by[:, :, None, None]) &
                           (cell_y0 < (by + bh)[:, :, None, None])).squeeze(1)
            flat_sim = sim.flatten(1)
            flat_mask = target_mask.flatten(1)
            pos = flat_sim.masked_fill(~flat_mask, -1.0).max(dim=1).values
            bg = flat_sim.masked_fill(flat_mask, -1.0)
            topk = min(int(getattr(self.cfg.TRAIN, 'AMCM_REPR_LOSS_TOPK', 8)), bg.shape[1])
            hard_bg = bg.topk(topk, dim=1).values[:, 0]
            margin = float(getattr(self.cfg.TRAIN, 'AMCM_REPR_LOSS_MARGIN', 0.05))
            raw = F.relu(margin - pos + hard_bg).mean()
            amcm_repr_loss = float(getattr(self.cfg.TRAIN, 'AMCM_REPR_LOSS_WEIGHT', 0.0)) * raw

        amcm_supervision_loss = torch.zeros((), device=l1_loss.device)
        amcm_cfg = getattr(getattr(self.cfg, 'MODEL', None), 'AMCM_RELIABILITY', None)
        amcm_mode = getattr(amcm_cfg, 'MODE', 'legacy') if amcm_cfg is not None else 'legacy'
        if (amcm_cfg is not None and amcm_mode == 'directed_residual'
                and getattr(amcm_cfg, 'SUPERVISION', False)
                and all(key in pred_dict for key in ('amcm_prob_rgb', 'amcm_prob_tir',
                                                       'amcm_sim_rgb', 'amcm_sim_tir'))):
            # Distill the reliability distribution that is actually used to
            # route the cross-modal residual. Similarity is a frozen target;
            # gradients update only the reliability predictor.
            grid = pred_dict['amcm_prob_rgb'].shape[-1]
            centers = ((gt_bbox[:, :2] + 0.5 * gt_bbox[:, 2:]) * grid).round().long()
            centers[:, 0].clamp_(0, grid - 1)
            centers[:, 1].clamp_(0, grid - 1)
            batch = torch.arange(gt_bbox.shape[0], device=l1_loss.device)
            prob = torch.cat((
                pred_dict['amcm_prob_rgb'][batch, 0, centers[:, 1], centers[:, 0]].unsqueeze(1),
                pred_dict['amcm_prob_tir'][batch, 0, centers[:, 1], centers[:, 0]].unsqueeze(1)), dim=1)
            sims = torch.cat((
                pred_dict['amcm_sim_rgb'][batch, 0, centers[:, 1], centers[:, 0]].detach().unsqueeze(1),
                pred_dict['amcm_sim_tir'][batch, 0, centers[:, 1], centers[:, 0]].detach().unsqueeze(1)), dim=1)
            temperature = float(getattr(amcm_cfg, 'SUPERVISION_TEMPERATURE', 4.0))
            target_prob = torch.softmax(temperature * sims, dim=1)
            amcm_supervision_loss = F.kl_div(prob.clamp_min(1e-6).log(), target_prob, reduction='batchmean')
            amcm_supervision_loss = float(getattr(amcm_cfg, 'SUPERVISION_WEIGHT', 0.0)) * amcm_supervision_loss
        if (amcm_cfg is not None and getattr(amcm_cfg, 'SUPERVISION', False)
                and amcm_mode != 'directed_residual'
                and all(key in pred_dict for key in ('amcm_gate_rgb', 'amcm_gate_tir',
                                                       'amcm_sim_rgb', 'amcm_sim_tir'))):
            # Explicit target-location modality preference proxy. It supervises
            # only the relative gate at the GT cell; no tracker state is used.
            grid = pred_dict['amcm_gate_rgb'].shape[-1]
            centers = ((gt_bbox[:, :2] + 0.5 * gt_bbox[:, 2:]) * grid).round().long()
            centers[:, 0].clamp_(0, grid - 1)
            centers[:, 1].clamp_(0, grid - 1)
            batch = torch.arange(gt_bbox.shape[0], device=l1_loss.device)
            rgb_gate = pred_dict['amcm_gate_rgb'][batch, 0, centers[:, 1], centers[:, 0]]
            tir_gate = pred_dict['amcm_gate_tir'][batch, 0, centers[:, 1], centers[:, 0]]
            rgb_sim = pred_dict['amcm_sim_rgb'][batch, 0, centers[:, 1], centers[:, 0]].detach()
            tir_sim = pred_dict['amcm_sim_tir'][batch, 0, centers[:, 1], centers[:, 0]].detach()
            temperature = float(getattr(amcm_cfg, 'SUPERVISION_TEMPERATURE', 4.0))
            target_relative_gate = float(getattr(amcm_cfg, 'MAX_DELTA', 0.10)) * torch.tanh(
                temperature * (tir_sim - rgb_sim))
            amcm_supervision_loss = F.mse_loss(tir_gate - rgb_gate, target_relative_gate)
            weight = float(getattr(amcm_cfg, 'SUPERVISION_WEIGHT', 0.0))
            amcm_supervision_loss = weight * amcm_supervision_loss

        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss + center_rank_loss + der_rank_loss +
                early_utility_loss +
                amcm_supervision_loss + amcm_repr_loss + amcm_residual_rank_loss +
                target_rank_loss)

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
                      "Loss/center_rank": center_rank_loss.item(),
                      "Loss/der_rank": der_rank_loss.item(),
                      "Loss/early_utility_router": early_utility_loss.item(),
                      "Loss/amcm_residual_rank": amcm_residual_rank_loss.item(),
                      "Loss/amcm_supervision": amcm_supervision_loss.item(),
                      "Loss/amcm_repr": amcm_repr_loss.item(),
                      "Loss/target_rank": target_rank_loss.item(),
                      "IoU": mean_iou.item()}
            status.update(tsms_status)
            if getattr(getattr(self.cfg, 'TRAIN', None), 'HANN_AWARE_CENTER_LOSS', False):
                status['CENTER/hann_aware_loss'] = 1.0
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
