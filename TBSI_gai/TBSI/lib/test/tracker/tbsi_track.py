import math

from lib.models.tbsi_track import build_tbsi_track
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import json

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.utils.box_ops import box_xywh_to_xyxy, box_iou

# Module-level cache: avoid loading 2.3GB checkpoint from disk for every sequence
_checkpoint_cache = {}


class TBSITrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TBSITrack, self).__init__(params)
        network = build_tbsi_track(params.cfg, training=False)
        ckpt_key = self.params.checkpoint
        if ckpt_key not in _checkpoint_cache:
            _checkpoint_cache[ckpt_key] = torch.load(self.params.checkpoint, map_location='cpu')['net']
        network.load_state_dict(_checkpoint_cache[ckpt_key], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda().half()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda().half()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.oracle_mode = os.environ.get("TBSI_ORACLE_MODE", "0") == "1"
        self.oracle_eta = float(os.environ.get("TBSI_ORACLE_ETA", "0.10"))
        self.oracle_gamma = float(os.environ.get("TBSI_ORACLE_GAMMA", "0.08"))
        self.oracle_prior_sigma = float(os.environ.get("TBSI_ORACLE_PRIOR_SIGMA", "0.35"))
        self.oracle_log_root = os.environ.get("TBSI_ORACLE_LOG_ROOT", "").strip()
        self.oracle_action_names = ["keep", "rgb", "tir", "temp"]
        self.oracle_log_path = None

    def initialize(self, image, info: dict):
        # forward the template once (FP16)
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr, half=True)
        self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.oracle_log_path = None
        if self.oracle_mode and self.oracle_log_root:
            seq_name = os.environ.get("TBSI_CURRENT_SEQUENCE", "unknown_sequence")
            os.makedirs(self.oracle_log_root, exist_ok=True)
            self.oracle_log_path = os.path.join(self.oracle_log_root, f"{seq_name}.jsonl")
            if os.path.exists(self.oracle_log_path):
                os.remove(self.oracle_log_path)
        # Reset temporal tokens for new sequence (Section 3.1)
        if hasattr(self.network, 'reset_temporal_tokens'):
            self.network.reset_temporal_tokens()
        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    @torch.inference_mode()
    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr, half=True)

        x_dict = search
        # merge the template and the search
        # run the transformer (all FP16)
        out_dict = self.network.forward(
            template=[self.z_dict1.tensors[:,:3,:,:],self.z_dict1.tensors[:,3:,:,:]],
            search=[x_dict.tensors[:,:3,:,:], x_dict.tensors[:,3:,:,:]],
            ce_template_mask=self.box_mask_z)

        chosen = self._select_prediction(out_dict, resize_factor, info)
        pred_score_map = chosen['score_map']
        pred_boxes = chosen['pred_boxes']
        pred_box = chosen['pred_box']
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(0, 2, 3, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def _select_prediction(self, out_dict, resize_factor, info):
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map']).view(-1, 4)
        pred_box = pred_boxes.mean(dim=0) * self.params.search_size / resize_factor

        chosen = {
            'score_map': pred_score_map,
            'pred_boxes': pred_boxes,
            'pred_box': pred_box,
        }
        if not self.oracle_mode or info is None or info.get('gt_bbox', None) is None:
            return chosen

        cat_feature = out_dict['backbone_feat']
        if isinstance(cat_feature, list):
            cat_feature = cat_feature[-1]
        quality_hint = out_dict.get('quality_signal', None)
        fusion_pack = self.network.get_fusion_pack(cat_feature, quality_hint=quality_hint)
        fused_feat = fusion_pack['fused_feat']
        diff_feat = self._layer_norm_2d(fusion_pack['feat_rgb']) - self._layer_norm_2d(fusion_pack['feat_tir'])
        temp_feat = (self._build_temporal_prior(fused_feat) - 0.5) * fused_feat

        candidates = [
            fused_feat,
            fused_feat + self.oracle_eta * diff_feat,
            fused_feat - self.oracle_eta * diff_feat,
            fused_feat + self.oracle_gamma * temp_feat,
        ]

        gt_bbox = torch.tensor(info['gt_bbox'], device=fused_feat.device, dtype=fused_feat.dtype).view(1, 4)
        gt_bbox_xyxy = box_xywh_to_xyxy(gt_bbox)
        best_idx = 0
        best_iou = -1.0
        action_ious = []

        for idx, fused_candidate in enumerate(candidates):
            cand_out = self.network.forward_head_from_fused(fused_candidate)
            cand_response = self.output_window * cand_out['score_map']
            cand_boxes = self.network.box_head.cal_bbox(
                cand_response, cand_out['size_map'], cand_out['offset_map']
            ).view(-1, 4)
            cand_box = cand_boxes.mean(dim=0) * self.params.search_size / resize_factor
            mapped_box = torch.tensor(
                self.map_box_back(cand_box, resize_factor),
                device=fused_feat.device,
                dtype=fused_feat.dtype
            ).view(1, 4)
            cand_iou = float(box_iou(box_xywh_to_xyxy(mapped_box), gt_bbox_xyxy)[0].item())
            action_ious.append(cand_iou)
            if cand_iou > best_iou:
                best_iou = cand_iou
                best_idx = idx
                chosen = {
                    'score_map': cand_out['score_map'],
                    'pred_boxes': cand_boxes,
                    'pred_box': cand_box,
                }

        self._append_oracle_log(info, best_idx, action_ious, best_iou)
        return chosen

    def _layer_norm_2d(self, feat):
        feat_ln = F.layer_norm(feat.permute(0, 2, 3, 1), (feat.shape[1],))
        return feat_ln.permute(0, 3, 1, 2)

    def _build_temporal_prior(self, fused_feat):
        _, _, h, w = fused_feat.shape
        yy = torch.linspace(-1.0, 1.0, steps=h, device=fused_feat.device, dtype=fused_feat.dtype)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=fused_feat.device, dtype=fused_feat.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        sigma = max(self.oracle_prior_sigma, 1e-3)
        prior = torch.exp(-(grid_x.pow(2) + grid_y.pow(2)) / (2.0 * sigma * sigma))
        return prior.unsqueeze(0).unsqueeze(0)

    def _append_oracle_log(self, info, best_idx, action_ious, best_iou):
        if not self.oracle_log_path:
            return
        payload = {
            'frame_id': int(self.frame_id),
            'selected_action': self.oracle_action_names[best_idx],
            'selected_index': int(best_idx),
            'best_iou': float(best_iou),
            'action_ious': {name: float(iou) for name, iou in zip(self.oracle_action_names, action_ious)},
            'gt_bbox': [float(x) for x in info['gt_bbox']],
        }
        with open(self.oracle_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=True) + '\n')

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        out = torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h])
        # Keep on GPU as tensor; convert to list only at final save
        return out.tolist()

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return TBSITrack
