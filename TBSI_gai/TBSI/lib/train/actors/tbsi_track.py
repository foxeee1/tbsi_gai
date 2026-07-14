from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
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

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
