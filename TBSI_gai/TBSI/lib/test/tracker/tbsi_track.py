import math
import json

from lib.models.tbsi_track import build_tbsi_track
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

# Module-level cache: avoid loading 2.3GB checkpoint from disk for every sequence
_checkpoint_cache = {}


class TBSITrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TBSITrack, self).__init__(params)
        network = build_tbsi_track(params.cfg, training=False)
        ckpt_key = self.params.checkpoint
        if ckpt_key not in _checkpoint_cache:
            _checkpoint_cache[ckpt_key] = torch.load(self.params.checkpoint, map_location='cpu')['net']
        network.load_state_dict(_checkpoint_cache[ckpt_key], strict=True)
        self.cfg = params.cfg
        self.use_fp16 = os.environ.get('TBSI_INFER_FP16', '1') != '0'
        self.network = network.cuda()
        if self.use_fp16:
            self.network = self.network.half()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        if self.use_fp16:
            self.output_window = self.output_window.half()

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
        self.attr_stats_path = os.environ.get('TBSI_ATTR_STATS_PATH', '')

    def initialize(self, image, info: dict):
        # forward the template once (FP16)
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr, half=self.use_fp16)
        self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
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
        state_before = list(self.state)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr, half=self.use_fp16)

        x_dict = search
        # merge the template and the search
        # run the transformer (all FP16)
        out_dict = self.network.forward(
            template=[self.z_dict1.tensors[:,:3,:,:],self.z_dict1.tensors[:,3:,:,:]],
            search=[x_dict.tensors[:,:3,:,:], x_dict.tensors[:,3:,:,:]],
            ce_template_mask=self.box_mask_z)

        if self.attr_stats_path and os.environ.get('TBSI_ATTR_STATS', '0') == '1' and info is not None:
            self._record_smsa_support_stats(
                info, state_before, float(resize_factor),
                sequence_name=info.get('sequence_name', 'unknown'),
                frame_num=info.get('frame_num', self.frame_id))

        # add hann windows (all GPU FP16, no CPU sync)
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor)  # (cx, cy, w, h) [0,1] GPU
        # get the final box result — still on GPU, convert to list only at save time
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

    def _record_smsa_support_stats(self, info, state_before, resize_factor,
                                   sequence_name, frame_num):
        """Write per-frame SMSA support diagnostics without changing inference."""
        gt = info.get('gt_bbox')
        if gt is None:
            return
        search_size = float(self.params.search_size)
        half_side = 0.5 * search_size / resize_factor
        cx = state_before[0] + 0.5 * state_before[2]
        cy = state_before[1] + 0.5 * state_before[3]
        crop_x = cx - half_side
        crop_y = cy - half_side
        gx, gy, gw, gh = [float(v) for v in gt]
        gx = (gx - crop_x) * resize_factor
        gy = (gy - crop_y) * resize_factor
        gw *= resize_factor
        gh *= resize_factor
        rows = []
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16))
        token_x0, token_y0 = xx.flatten().float() * 16.0, yy.flatten().float() * 16.0
        token_x1, token_y1 = token_x0 + 16.0, token_y0 + 16.0
        gt_mask = (token_x1 > gx) & (token_x0 < gx + gw) & (token_y1 > gy) & (token_y0 < gy + gh)
        gt_count = max(int(gt_mask.sum().item()), 1)
        layer_ids = getattr(self.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9])
        for layer_id, layer in zip(layer_ids, self.network.backbone.tbsi_layers):
            for direction, block in (('TIR2F', layer.ca_s2t_i2f), ('RGB2F', layer.ca_s2t_v2f)):
                attn = getattr(block.attn_reshape, 'last_smsa_attention', None)
                if attn is None:
                    continue
                p = attn.detach().float().cpu()[0]  # [template_query, search_token]
                active = p > 1e-8
                mask = gt_mask.view(1, -1)
                keff = active.sum(dim=-1).float().mean().item()
                gt_mass = (p * mask).sum(dim=-1).mean().item()
                active_gt = (active & mask).sum(dim=-1).float()
                gt_precision = (active_gt / active.sum(dim=-1).clamp_min(1)).mean().item()
                gt_recall = (active_gt / float(gt_count)).mean().item()
                rows.append({
                    'sequence': sequence_name,
                    'frame': int(frame_num),
                    'layer': int(layer_id),
                    'direction': direction,
                    'Keff': keff,
                    'GTMass': gt_mass,
                    'GTPrecision': gt_precision,
                    'GTRecall': gt_recall,
                    'BGMass': 1.0 - gt_mass,
                })
        if rows:
            with open(self.attr_stats_path, 'a') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')

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
