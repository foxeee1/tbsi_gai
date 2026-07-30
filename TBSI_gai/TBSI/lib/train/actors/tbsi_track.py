from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
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

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        loss_terms = self.compute_tracking_terms(pred_dict, gt_bbox, gt_gaussian_maps)
        giou_loss = loss_terms['giou_loss']
        iou = loss_terms['iou']
        l1_loss = loss_terms['l1_loss']
        location_loss = loss_terms['location_loss']

        fcc_aux_loss = self.compute_fcc_aux_loss(pred_dict, gt_bbox, l1_loss.device)
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        fcc_aux_weight = getattr(self.cfg.TRAIN, "FCC_AUX_WEIGHT", 0.0)
        if fcc_aux_weight > 0:
            loss = loss + fcc_aux_weight * fcc_aux_loss

        utility_loss = self.compute_utility_loss(pred_dict, gt_bbox, gt_gaussian_maps, l1_loss.device)
        utility_weight = getattr(self.cfg.TRAIN, "UTILITY_WEIGHT", 0.0)
        if utility_weight > 0:
            loss = loss + utility_weight * utility_loss

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/utility": utility_loss.item(),
                      "Loss/fcc_aux": fcc_aux_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

    def compute_tracking_terms(self, pred_dict, gt_bbox, gt_gaussian_maps):
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)

        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss = torch.tensor(0.0, device=pred_boxes.device)
            iou = torch.tensor(0.0, device=pred_boxes.device)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        return {
            'giou_loss': giou_loss,
            'iou': iou,
            'l1_loss': l1_loss,
            'location_loss': location_loss,
        }

    def compute_tracking_terms_per_sample(self, pred_dict, gt_bbox, gt_gaussian_maps):
        pred_boxes = pred_dict['pred_boxes']
        bsz, num_queries, _ = pred_boxes.shape
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_boxes_xyxy = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).clamp(min=0.0, max=1.0)

        giou, _ = generalized_box_iou(pred_boxes_xyxy.reshape(-1, 4), gt_boxes_xyxy.reshape(-1, 4))
        giou_loss = (1.0 - giou).view(bsz, num_queries).mean(dim=1)
        l1_loss = F.l1_loss(pred_boxes_xyxy, gt_boxes_xyxy, reduction='none').mean(dim=(1, 2))

        if 'score_map' in pred_dict:
            location_loss = self.compute_focal_loss_per_sample(pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.zeros(bsz, device=pred_boxes.device)
        return {
            'giou_loss': giou_loss,
            'l1_loss': l1_loss,
            'location_loss': location_loss,
        }

    def compute_focal_loss_per_sample(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()
        negative_weights = torch.pow(1 - target, self.objective['focal'].beta)
        prediction = torch.clamp(prediction, 1e-12, 1 - 1e-6)

        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.objective['focal'].alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction, self.objective['focal'].alpha) * negative_weights * negative_index

        reduce_dims = tuple(range(1, prediction.dim()))
        num_positive = positive_index.sum(dim=reduce_dims)
        positive_loss = positive_loss.sum(dim=reduce_dims)
        negative_loss = negative_loss.sum(dim=reduce_dims)

        loss = torch.where(
            num_positive > 0,
            -(positive_loss + negative_loss) / num_positive.clamp_min(1.0),
            -negative_loss,
        )
        return loss

    def compute_utility_loss(self, pred_dict, gt_bbox, gt_gaussian_maps, device):
        utility_logits = pred_dict.get('utility_logits', None)
        utility_candidates = pred_dict.get('utility_candidates', None)
        if utility_logits is None or utility_candidates is None:
            return torch.tensor(0.0, device=device)

        per_action_losses = []
        action_order = ['keep', 'rgb', 'tir']
        for action_name in action_order:
            cand_terms = self.compute_tracking_terms_per_sample(utility_candidates[action_name], gt_bbox, gt_gaussian_maps)
            per_action_losses.append(
                self.loss_weight['giou'] * cand_terms['giou_loss']
                + self.loss_weight['l1'] * cand_terms['l1_loss']
                + self.loss_weight['focal'] * cand_terms['location_loss']
            )

        action_losses = torch.stack(per_action_losses, dim=1)
        tau = getattr(self.cfg.MODEL, "CUTR_LITE_TAU", 0.5)
        pi_star = torch.softmax(-action_losses / max(tau, 1e-6), dim=1).detach()
        log_pi = F.log_softmax(utility_logits, dim=1)
        return F.kl_div(log_pi, pi_star, reduction='batchmean')

    def compute_fcc_aux_loss(self, pred_dict, gt_bbox, device):
        penalty = pred_dict.get("fcc_aux_penalty", None)
        if penalty is None:
            return torch.tensor(0.0, device=device)
        if penalty.dim() != 3:
            return torch.tensor(0.0, device=device)
        if penalty.shape[1] != 1 and penalty.shape[-1] == 1:
            penalty = penalty.transpose(1, 2)
        bsz, _, num_tokens = penalty.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            return torch.tensor(0.0, device=penalty.device)

        gt_xyxy = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0).to(device=penalty.device)
        coord = (torch.arange(side, device=penalty.device, dtype=penalty.dtype) + 0.5) / side
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        centers_x = xx.reshape(1, num_tokens)
        centers_y = yy.reshape(1, num_tokens)
        x1, y1, x2, y2 = [gt_xyxy[:, i:i + 1].to(dtype=penalty.dtype) for i in range(4)]
        target_mask = ((centers_x >= x1) & (centers_x <= x2) &
                       (centers_y >= y1) & (centers_y <= y2)).to(dtype=penalty.dtype).unsqueeze(1)
        denom = target_mask.sum().clamp_min(1.0)
        return (penalty * target_mask).sum() / denom
