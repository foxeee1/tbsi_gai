import math
import json
import hashlib
import numpy as np

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
from lib.models.layers.tbsi_layer import template_box_to_token_weights
from lib.test.tracker.input_perturbation import apply_input_perturbation

# Module-level cache: avoid loading 2.3GB checkpoint from disk for every sequence
_checkpoint_cache = {}


class TBSITrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TBSITrack, self).__init__(params)
        network = build_tbsi_track(params.cfg, training=False)
        ckpt_key = self.params.checkpoint
        if ckpt_key not in _checkpoint_cache:
            _checkpoint_cache[ckpt_key] = torch.load(self.params.checkpoint, map_location='cpu')['net']
        checkpoint_state = _checkpoint_cache[ckpt_key]
        if getattr(network, 'use_causal_tam', False):
            missing, unexpected = network.load_state_dict(checkpoint_state, strict=False)
            allowed = set()
            if getattr(network, 'use_causal_tam', False):
                allowed.update('causal_tam.' + name
                               for name, _ in network.causal_tam.named_parameters())
            unexpected = list(unexpected)
            invalid_missing = [name for name in missing if name not in allowed]
            if invalid_missing or unexpected:
                raise RuntimeError('TAM checkpoint contract mismatch: missing={}, unexpected={}'.format(
                    invalid_missing, unexpected))
        else:
            if os.environ.get('TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE', '0') == '1':
                # Diagnostic-only extension: the protected M1 checkpoint does
                # not contain the newly added zero-initialized probe weights.
                # Permit only those exact missing keys; keep all other
                # checkpoint mismatches fatal.
                missing, unexpected = network.load_state_dict(checkpoint_state, strict=False)
                allowed_missing = {
                    name for name, _ in network.named_parameters()
                    if 'query_conditioned_' in name
                }
                invalid_missing = [name for name in missing if name not in allowed_missing]
                if invalid_missing or unexpected:
                    raise RuntimeError('query-conditioned diagnostic checkpoint contract mismatch: '
                                       'missing={}, unexpected={}'.format(invalid_missing, unexpected))
            else:
                network.load_state_dict(checkpoint_state, strict=True)
        signal_mode = os.environ.get('TBSI_SINGLE_SIGNAL_MODE', 'off')
        if signal_mode != 'off':
            from lib.models.layers.single_signal_prefusion import SingleSignalPrefusion
            artifact_dir = os.environ.get('TBSI_SINGLE_SIGNAL_ARTIFACT_DIR')
            if not artifact_dir and signal_mode not in ('fixed_rgb', 'fixed_tir'):
                raise RuntimeError('TBSI_SINGLE_SIGNAL_ARTIFACT_DIR is required')
            network.single_signal_gate = SingleSignalPrefusion(
                signal_mode, artifact_dir,
                alpha=float(os.environ.get('TBSI_SINGLE_SIGNAL_ALPHA', '0.15')))
        local_mode = os.environ.get('TBSI_LOCAL_SIGNAL_MODE', 'off')
        if local_mode != 'off':
            from lib.models.layers.local_signal_post_fusion import LocalSignalPostFusion
            artifact_dir = os.environ.get('TBSI_LOCAL_SIGNAL_ARTIFACT_DIR')
            if not artifact_dir:
                raise RuntimeError('TBSI_LOCAL_SIGNAL_ARTIFACT_DIR is required')
            network.local_signal_post_fusion = LocalSignalPostFusion(
                local_mode, artifact_dir,
                rho=float(os.environ.get('TBSI_LOCAL_SIGNAL_RHO', '0.05')))
        self.cfg = params.cfg
        self.use_fp16 = os.environ.get('TBSI_INFER_FP16', '1') != '0'
        self.network = network.cuda()
        if self.use_fp16:
            self.network = self.network.half()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        # Optional inference-only template EMA. Disabled by default so the
        # protected M1 path keeps its original template contract.
        self.template_update_mode = os.environ.get('TBSI_TEMPLATE_UPDATE_MODE', 'off')
        if self.template_update_mode not in ('off', 'ema_confident'):
            raise RuntimeError('unknown TBSI_TEMPLATE_UPDATE_MODE: {}'.format(
                self.template_update_mode))
        self.template_update_alpha = float(os.environ.get('TBSI_TEMPLATE_UPDATE_ALPHA', '0.10'))
        self.template_update_min_margin = float(os.environ.get('TBSI_TEMPLATE_UPDATE_MIN_MARGIN', '0.08'))
        self.template_update_max_shift = float(os.environ.get('TBSI_TEMPLATE_UPDATE_MAX_SHIFT', '0.50'))
        self.template_update_max_scale = float(os.environ.get('TBSI_TEMPLATE_UPDATE_MAX_SCALE', '0.30'))
        if not (0.0 < self.template_update_alpha <= 1.0):
            raise RuntimeError('TBSI_TEMPLATE_UPDATE_ALPHA must be in (0, 1]')
        self.template_update_audit_dir = os.environ.get('TBSI_TEMPLATE_UPDATE_AUDIT_DIR', '')
        self.template_update_audit_file = None

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
        # Internal diagnostics are opt-in so protected M1 inference remains
        # byte-for-byte equivalent when the environment flag is absent.
        self.diagnostic = os.environ.get('TBSI_INTERNAL_DIAGNOSTIC', '0') == '1'
        self.diagnostic_early_block = 2
        self.diagnostic_late_block = 9
        self.diagnostic_tbsi_index = 0
        self.diagnostic_post_block = 5
        self.readout_predictor = None
        self.center_ranker = None
        self.center_ranker_state_mode = os.environ.get('TBSI_CENTER_RANKER_STATE_MODE', 'coupled')
        center_ranker_path = os.environ.get('TBSI_CENTER_RANKER_JSON', '')
        if center_ranker_path:
            with open(center_ranker_path) as stream:
                ranker_payload = json.load(stream)
            self.center_ranker = ranker_payload.get('model', ranker_payload)
            self.center_ranker_features = ranker_payload.get('features', [
                'rank', 'x', 'y', 'score', 'margin', 'center_dist',
                'size_w', 'size_h', 'offset_x', 'offset_y'])
            if self.center_ranker_state_mode not in ('coupled', 'output_only'):
                raise RuntimeError('unknown TBSI_CENTER_RANKER_STATE_MODE: {}'.format(self.center_ranker_state_mode))
        self.readout_state_mode = os.environ.get('TBSI_READOUT_STATE_MODE', 'coupled')
        if self.readout_state_mode not in ('coupled', 'output_only', 'state_only', 'selective_commit'):
            raise RuntimeError('unknown TBSI_READOUT_STATE_MODE: {}'.format(self.readout_state_mode))
        predictor_path = os.environ.get('TBSI_BOUNDED_READOUT_PREDICTOR', '')
        self.readout_policy = None
        self.state_risk_policy = None
        self.readout_audit_dir = os.environ.get('TBSI_READOUT_AUDIT_DIR', '')
        if predictor_path:
            with open(predictor_path, 'r') as stream:
                self.readout_predictor = json.load(stream)
            if self.readout_predictor.get('feature') != 'peak_margin':
                raise RuntimeError('bounded readout predictor must use peak_margin')
            self.readout_prev_state_center = None
            self.readout_last_action = 'noop'
            self.readout_pending_action = 'noop'
            self.readout_pending_count = 0
            self.readout_inactive_count = 0
            policy_path = os.environ.get('TBSI_READOUT_POLICY', '')
            if policy_path:
                with open(policy_path, 'r') as stream:
                    self.readout_policy = json.load(stream)
        state_risk_path = os.environ.get('TBSI_STATE_RISK_POLICY', '')
        if state_risk_path:
            with open(state_risk_path, 'r') as stream:
                self.state_risk_policy = json.load(stream)
        if self.readout_state_mode in ('output_only', 'selective_commit') and self.readout_predictor is None:
            raise RuntimeError('{} state mode requires a readout predictor'.format(self.readout_state_mode))
        if self.readout_state_mode == 'selective_commit' and self.state_risk_policy is None:
            raise RuntimeError('selective_commit state mode requires a separate frozen state-risk policy')
        self.feature_state_mode = os.environ.get('TBSI_FEATURE_STATE_MODE', 'coupled')
        if self.feature_state_mode not in ('coupled', 'output_only'):
            raise RuntimeError('unknown TBSI_FEATURE_STATE_MODE: {}'.format(self.feature_state_mode))
        self.partial_state_commit = os.environ.get('TBSI_PARTIAL_STATE_COMMIT', '0') == '1'
        self.partial_alpha_center = float(os.environ.get('TBSI_PARTIAL_ALPHA_CENTER', '0.0'))
        self.partial_alpha_size = float(os.environ.get('TBSI_PARTIAL_ALPHA_SIZE', '0.0'))
        if self.partial_state_commit and self.readout_predictor is None:
            raise RuntimeError('partial state commit requires a bounded readout predictor')
        if not (0.0 <= self.partial_alpha_center <= 1.0 and 0.0 <= self.partial_alpha_size <= 1.0):
            raise RuntimeError('partial state commit alphas must be in [0, 1]')

    def initialize(self, image, info: dict):
        image = apply_input_perturbation(image, os.environ.get('TBSI_PERTURBATION', ''))
        # forward the template once (FP16)
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr, half=self.use_fp16)
        self.z_dict1 = template
        # Oracle template-token mask for C1 compatibility analysis.
        crop_side = float(self.params.template_size) / float(resize_factor)
        init = [float(v) for v in info['init_bbox']]
        cx, cy = init[0] + init[2] / 2.0, init[1] + init[3] / 2.0
        x0, y0 = cx - crop_side / 2.0, cy - crop_side / 2.0
        bx, by = (init[0] - x0) * resize_factor, (init[1] - y0) * resize_factor
        bw, bh = init[2] * resize_factor, init[3] * resize_factor
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8))
        self.c1_template_mask = ((xx.flatten() + 0.5) * 16.0 >= bx) & ((xx.flatten() + 0.5) * 16.0 <= bx + bw)
        self.c1_template_mask &= ((yy.flatten() + 0.5) * 16.0 >= by) & ((yy.flatten() + 0.5) * 16.0 <= by + bh)
        template_box = torch.tensor([[bx / float(self.params.template_size),
                                      by / float(self.params.template_size),
                                      bw / float(self.params.template_size),
                                      bh / float(self.params.template_size)]],
                                    device=template.tensors.device, dtype=torch.float32)
        self.template_fg_mask = template_box_to_token_weights(template_box, grid_size=8)
        signal_gate = getattr(self.network, 'single_signal_gate', None)
        if signal_gate is not None:
            signal_gate.template_fg = self.template_fg_mask
        local_signal = getattr(self.network, 'local_signal_post_fusion', None)
        if local_signal is not None:
            local_signal.template_fg = self.template_fg_mask
            local_signal.random_counter = 0

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.readout_prev_state_center = None
        # Reset temporal tokens for new sequence (Section 3.1)
        if hasattr(self.network, 'reset_temporal_tokens'):
            self.network.reset_temporal_tokens()
        self.prev_search = None
        self.prev_temporal_tokens = None
        self.prev_head_feature = None
        self.prev_crop_xywh = None
        self.prev_crop_transform = None
        self.prev_head_frame_id = None
        self.prev_head_sequence_id = None
        self.trace_dir = os.environ.get('TBSI_M1_TRACE_DIR', '')
        self.trace_save_raw = os.environ.get('TBSI_TRACE_SAVE_RAW', '0') == '1'
        self.trace_raw_dir = None
        self.trace_file = None
        self.sequence_id = info.get('sequence_name', info.get('sequence_id', 'unknown'))
        if self.template_update_mode != 'off' and self.template_update_audit_dir:
            os.makedirs(self.template_update_audit_dir, exist_ok=True)
            self.template_update_audit_file = os.path.join(
                self.template_update_audit_dir, '{}.jsonl'.format(self.sequence_id))
            if os.path.exists(self.template_update_audit_file):
                os.remove(self.template_update_audit_file)
        self.readout_audit_file = None
        if self.readout_predictor is not None and self.readout_audit_dir:
            audit_seq_dir = os.path.join(self.readout_audit_dir, self.sequence_id)
            os.makedirs(audit_seq_dir, exist_ok=True)
            self.readout_audit_file = os.path.join(audit_seq_dir, 'actions.jsonl')
            if os.path.exists(self.readout_audit_file):
                os.remove(self.readout_audit_file)
            self.readout_last_action = 'noop'
            self.readout_pending_action = 'noop'
            self.readout_pending_count = 0
            self.readout_inactive_count = 0
        if self.trace_dir:
            seq_dir = os.path.join(self.trace_dir, self.sequence_id)
            os.makedirs(seq_dir, exist_ok=True)
            if self.trace_save_raw:
                self.trace_raw_dir = os.path.join(seq_dir, 'raw_head_tokens')
                os.makedirs(self.trace_raw_dir, exist_ok=True)
            self.trace_file = os.path.join(seq_dir, 'frames.jsonl')
            if os.path.exists(self.trace_file):
                os.remove(self.trace_file)
        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    @torch.inference_mode()
    def track(self, image, info: dict = None):
        image = apply_input_perturbation(image, os.environ.get('TBSI_PERTURBATION', ''))
        H, W, _ = image.shape
        self.frame_id += 1
        state_before = list(self.state)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        signal_gate = getattr(self.network, 'single_signal_gate', None)
        if signal_gate is not None and signal_gate.mode in ('s3', 's1s2s3'):
            from lib.models.layers.single_signal_prefusion import image_features_from_patch
            signal_gate.runtime_s3_features = image_features_from_patch(x_patch_arr, state_before)
        local_signal = getattr(self.network, 'local_signal_post_fusion', None)
        if local_signal is not None and local_signal.mode == 's1s2s3':
            from lib.models.layers.single_signal_prefusion import image_features_from_patch
            local_signal.runtime_s3_features = image_features_from_patch(x_patch_arr, state_before)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr, half=self.use_fp16)
        half_side = 0.5 * float(self.params.search_size) / float(resize_factor)
        state_cx = float(state_before[0]) + 0.5 * float(state_before[2])
        state_cy = float(state_before[1]) + 0.5 * float(state_before[3])
        current_crop_xywh = [state_cx - half_side, state_cy - half_side,
                             2.0 * half_side, 2.0 * half_side]
        crop_sz = int(math.ceil(math.sqrt(float(state_before[2]) * float(state_before[3])) *
                                float(self.params.search_factor)))
        crop_x = int(round(state_cx - 0.5 * crop_sz))
        crop_y = int(round(state_cy - 0.5 * crop_sz))
        pad_left = max(0, -crop_x)
        pad_top = max(0, -crop_y)
        pad_right = max(crop_x + crop_sz - W + 1, 0)
        pad_bottom = max(crop_y + crop_sz - H + 1, 0)
        actual_crop_xywh = [float(crop_x), float(crop_y), float(crop_sz), float(crop_sz)]
        resize = float(self.params.search_size) / float(crop_sz)
        crop_transform = [[resize, 0.0, -resize * crop_x],
                          [0.0, resize, -resize * crop_y]]
        current_crop_xywh = actual_crop_xywh
        history_crop_for_row = self.prev_crop_xywh
        history_transform_for_row = self.prev_crop_transform
        history_frame_for_row = self.prev_head_frame_id
        history_sequence_for_row = self.prev_head_sequence_id

        x_dict = search
        # merge the template and the search
        # run the transformer (all FP16)
        out_dict = self.network.forward(
            template=[self.z_dict1.tensors[:,:3,:,:],self.z_dict1.tensors[:,3:,:,:]],
            search=[x_dict.tensors[:,:3,:,:], x_dict.tensors[:,3:,:,:]],
                prev_search=self.prev_search,
                prev_tokens=self.prev_temporal_tokens,
                diagnostic=self.diagnostic,
                diagnostic_early_block=self.diagnostic_early_block,
                diagnostic_late_block=self.diagnostic_late_block,
                diagnostic_tbsi_index=self.diagnostic_tbsi_index,
                diagnostic_post_block=self.diagnostic_post_block,
                prev_head_feature=self.prev_head_feature,
                current_crop_xywh=current_crop_xywh,
                history_crop_xywh=self.prev_crop_xywh,
                current_frame_id=self.frame_id,
                history_frame_id=self.prev_head_frame_id,
                current_sequence_id=[self.sequence_id],
                history_sequence_id=([self.prev_head_sequence_id]
                                     if self.prev_head_sequence_id is not None else None),
                ce_template_mask=self.box_mask_z,
                # The target expert consumes one foreground mask per modality.
                # The initial RGB/TIR crop uses the same annotated target box.
                template_fg_masks=(self.template_fg_mask, self.template_fg_mask))
        if self.diagnostic:
            self._last_diagnostic_features = out_dict.get('diagnostic_features')
        if getattr(self.cfg.MODEL, 'TEMPORAL_TOKENS', False):
            self.prev_search = [x_dict.tensors[:,:3,:,:].detach(),
                                x_dict.tensors[:,3:,:,:].detach()]
            self.prev_temporal_tokens = out_dict.get('temporal_tokens')

        if getattr(self.network, 'use_causal_tam', False):
            raw_head = out_dict.get('tam_raw_head_feature')
            if raw_head is None:
                raise RuntimeError('CAUSAL_TAM enabled but raw head feature was not returned')
            self.prev_head_feature = raw_head.detach()
            self.prev_crop_xywh = current_crop_xywh
            self.prev_head_frame_id = self.frame_id
            self.prev_head_sequence_id = self.sequence_id

        if self.attr_stats_path and os.environ.get('TBSI_ATTR_STATS', '0') == '1' and info is not None:
            self._record_smsa_support_stats(
                info, state_before, float(resize_factor),
                sequence_name=info.get('sequence_name', 'unknown'),
                frame_num=info.get('frame_num', self.frame_id))

        # add hann windows (all GPU FP16, no CPU sync)
        pred_score_map = out_dict['score_map']
        risk_probability = None
        state_risk_probability = None
        state_commit_corrected = None
        response = self.output_window * pred_score_map
        size_map = out_dict['size_map']
        offset_map = out_dict['offset_map']
        baseline_response = response.clone()
        baseline_size_map = size_map
        baseline_offset_map = offset_map
        if self.feature_state_mode == 'output_only':
            if 'baseline_score_map' not in out_dict:
                raise RuntimeError('TBSI_FEATURE_STATE_MODE=output_only requires B2 baseline head outputs')
            baseline_response = self.output_window * out_dict['baseline_score_map']
            baseline_size_map = out_dict['baseline_size_map']
            baseline_offset_map = out_dict['baseline_offset_map']
        center_ranker_selected = None
        if self.center_ranker is not None:
            # Candidate utility is predicted from deployment-visible CENTER values.
            flat_center = response.flatten(1)[0]
            order_center = torch.argsort(flat_center, descending=True)[0:8]
            margin_center = float(flat_center[order_center[0]] - flat_center[order_center[1]])
            values = []
            ranker_mean = np.asarray(self.center_ranker['mean'], dtype=np.float64)
            ranker_scale = np.asarray(self.center_ranker['scale'], dtype=np.float64)
            ranker_weight = np.asarray(self.center_ranker['weight'], dtype=np.float64)
            for rank_idx, idx_tensor in enumerate(order_center.tolist(), 1):
                px_center, py_center = idx_tensor % self.feat_sz, idx_tensor // self.feat_sz
                feat = np.asarray([
                    rank_idx, px_center, py_center, float(flat_center[idx_tensor]), margin_center,
                    math.hypot(px_center - 7.5, py_center - 7.5) / 10.6066,
                    float(size_map[0, 0, py_center, px_center]), float(size_map[0, 1, py_center, px_center]),
                    float(offset_map[0, 0, py_center, px_center]), float(offset_map[0, 1, py_center, px_center])], dtype=np.float64)
                z = np.clip((feat - ranker_mean) / ranker_scale, -8.0, 8.0)
                utility = float(ranker_weight[0] + np.dot(z, ranker_weight[1:]))
                values.append((utility, idx_tensor))
            best_utility, best_idx = max(values, key=lambda item: item[0])
            if best_utility > 0.0:
                py_center, px_center = divmod(int(best_idx), self.feat_sz)
                response[:, :, py_center, px_center] = flat_center.max() + 1e-4
                center_ranker_selected = {'index': int(best_idx), 'utility': float(best_utility), 'rank': int([v[1] for v in values].index(best_idx) + 1)}
            else:
                center_ranker_selected = {'index': None, 'utility': float(best_utility), 'rank': None}
        if os.environ.get('TBSI_FIXED_STATE_DIAG', '0') == '1':
            # Opt-in fixed-state audit hook; disabled for normal tracking.
            self._last_fixed_state_diag = {
                'score_map': pred_score_map.detach().float().cpu(),
                'size_map': size_map.detach().float().cpu(),
                'offset_map': offset_map.detach().float().cpu(),
                'resize_factor': float(resize_factor),
                'state_before': [float(v) for v in state_before],
                'image_shape': [int(H), int(W)],
            }
        if self.readout_predictor is not None:
            flat = response.flatten(1)[0]
            order = torch.argsort(flat, descending=True)
            peak = int(order[0].item())
            px, py = peak % self.feat_sz, peak // self.feat_sz
            peak_margin = flat[order[0]] - flat[order[1]]
            top5_margin = flat[order[0]] - flat[order[min(4, flat.numel() - 1)]]
            model = self.readout_predictor['models']
            z = torch.clamp((peak_margin.float() - float(model['offset_utility']['mean'][0])) /
                            float(model['offset_utility']['scale'][0]), -8.0, 8.0)
            po = float(model['offset_utility']['bias']) + z * float(model['offset_utility']['weight'][0])
            pz = float(model['size_utility']['bias']) + z * float(model['size_utility']['weight'][0])
            threshold = float(self.readout_predictor.get('threshold', 0.08))
            policy = self.readout_policy or {'mode': 'argmax'}
            mode = policy.get('mode', 'argmax')
            admission = policy.get('admission', '')
            offset_ok, size_ok = po > threshold, pz > threshold
            entropy = torch.softmax(flat.float(), dim=0)
            entropy = -(entropy * torch.log(entropy.clamp_min(1e-12))).sum() / math.log(max(int(flat.numel()), 2))
            if self.readout_state_mode == 'selective_commit':
                risk_model = self.state_risk_policy['risk_model']
                edge_distance = min(px, py, self.feat_sz - 1 - px, self.feat_sz - 1 - py) / max(self.feat_sz / 2.0, 1.0)
                signal = torch.stack((peak_margin.float(), entropy.float(), top5_margin.float(),
                                      torch.tensor(edge_distance, device=flat.device))).detach().cpu().numpy()
                z_risk = (signal - risk_model['mean']) / risk_model['scale']
                risk_logit = float(risk_model['bias']) + float((z_risk * risk_model['weight']).sum())
                state_risk_probability = 1.0 / (1.0 + math.exp(-max(min(risk_logit, 30.0), -30.0)))
            risk_probability = None
            motion_innovation = None
            if admission == 'selective_risk':
                risk_model = self.readout_policy['risk_model']
                edge_distance = min(px, py, self.feat_sz - 1 - px, self.feat_sz - 1 - py) / max(self.feat_sz / 2.0, 1.0)
                signal = torch.stack((peak_margin.float(), entropy.float(), top5_margin.float(),
                                      torch.tensor(edge_distance, device=flat.device))).detach().cpu().numpy()
                z_risk = (signal - risk_model['mean']) / risk_model['scale']
                risk_logit = float(risk_model['bias']) + float((z_risk * risk_model['weight']).sum())
                risk_probability = 1.0 / (1.0 + math.exp(-max(min(risk_logit, 30.0), -30.0)))
                if self.readout_state_mode != 'selective_commit':
                    offset_ok = offset_ok and risk_probability <= float(self.readout_policy['max_harm_probability'])
            if admission == 'motion_innovation':
                raw_boxes = self.network.box_head.cal_bbox(response, size_map, offset_map).view(-1, 4)
                raw_box = raw_boxes.mean(dim=0) * self.params.search_size / resize_factor
                raw_xywh = self.map_box_back(raw_box, resize_factor)
                current_center = (float(state_before[0]) + 0.5 * float(state_before[2]),
                                  float(state_before[1]) + 0.5 * float(state_before[3]))
                predicted_center = current_center
                if self.readout_prev_state_center is not None:
                    predicted_center = (2.0 * current_center[0] - self.readout_prev_state_center[0],
                                        2.0 * current_center[1] - self.readout_prev_state_center[1])
                    raw_center = (float(raw_xywh[0]) + 0.5 * float(raw_xywh[2]),
                                  float(raw_xywh[1]) + 0.5 * float(raw_xywh[3]))
                    scale = max(math.sqrt(max(float(state_before[2]) * float(state_before[3]), 1.0)), 1.0)
                    motion_innovation = math.hypot(raw_center[0] - predicted_center[0],
                                                   raw_center[1] - predicted_center[1]) / scale
                self.readout_prev_state_center = current_center
                max_innovation = float(self.readout_policy['max_normalized_innovation'])
                if motion_innovation is not None:
                    offset_ok = offset_ok and motion_innovation <= max_innovation
            if mode in ('reliability', 'hysteresis'):
                offset_ok = offset_ok and float(peak_margin) >= float(policy.get('min_peak_margin', threshold)) and float(entropy) <= float(policy.get('max_entropy', 0.92))
                size_ok = size_ok and float(peak_margin) >= float(policy.get('min_peak_margin', threshold)) and float(entropy) <= float(policy.get('max_entropy', 0.92))
            if mode == 'independent':
                action = ('joint' if offset_ok and size_ok else
                          'offset' if offset_ok else 'size' if size_ok else 'noop')
            else:
                action = 'offset' if po >= pz and offset_ok else ('size' if size_ok else 'noop')
            if mode == 'hysteresis':
                required = int(policy.get('confirm_frames', 2))
                if action == 'noop':
                    self.readout_inactive_count += 1
                    self.readout_pending_action = 'noop'
                    self.readout_pending_count = 0
                    if self.readout_inactive_count >= int(policy.get('release_frames', 2)):
                        self.readout_last_action = 'noop'
                    action = self.readout_last_action
                else:
                    self.readout_inactive_count = 0
                    if action == self.readout_pending_action:
                        self.readout_pending_count += 1
                    else:
                        self.readout_pending_action = action
                        self.readout_pending_count = 1
                    if self.readout_pending_count >= required:
                        self.readout_last_action = action
                    action = self.readout_last_action
            self.readout_last_action = action
            if self.readout_audit_file:
                audit_row = {'frame_id': int(self.frame_id), 'action': action,
                             'offset_utility': float(po), 'size_utility': float(pz),
                             'peak_margin': float(peak_margin), 'normalized_entropy': float(entropy),
                             'top5_margin': float(top5_margin),
                             'risk_probability': risk_probability,
                             'normalized_motion_innovation': motion_innovation,
                             'offset_gate': bool(action in ('offset', 'joint')),
                             'size_gate': bool(action in ('size', 'joint'))}
                with open(self.readout_audit_file, 'a') as audit_stream:
                    audit_stream.write(json.dumps(audit_row, separators=(',', ':')) + '\n')
            if action != 'noop':
                if action in ('offset', 'joint'):
                    delta = torch.as_tensor(model['offset_delta']['bias'], device=offset_map.device,
                                            dtype=offset_map.dtype)
                    weights = torch.as_tensor(model['offset_delta']['weight'], device=offset_map.device,
                                              dtype=offset_map.dtype)
                    delta = delta + torch.clamp((peak_margin.float() - float(model['offset_delta']['mean'][0])) /
                                                float(model['offset_delta']['scale'][0]), -8.0, 8.0) * weights
                    changed = offset_map.clone()
                    changed[:, :, py, px] = offset_map[:, :, py, px] + float(self.readout_predictor.get('rho', 0.5)) * delta
                    offset_map = changed
                if action in ('size', 'joint'):
                    delta = torch.as_tensor(model['size_delta']['bias'], device=size_map.device,
                                            dtype=size_map.dtype)
                    weights = torch.as_tensor(model['size_delta']['weight'], device=size_map.device,
                                              dtype=size_map.dtype)
                    delta = delta + torch.clamp((peak_margin.float() - float(model['size_delta']['mean'][0])) /
                                                float(model['size_delta']['scale'][0]), -8.0, 8.0) * weights
                    changed = size_map.clone()
                    changed[:, :, py, px] = torch.clamp(
                        size_map[:, :, py, px] + float(self.readout_predictor.get('rho', 0.5)) * delta, min=0.01)
                    size_map = changed
        pred_boxes = self.network.box_head.cal_bbox(response, size_map, offset_map)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor)  # (cx, cy, w, h) [0,1] GPU
        reported_state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        if self.center_ranker is not None:
            baseline_boxes = self.network.box_head.cal_bbox(
                baseline_response,
                baseline_size_map, baseline_offset_map).view(-1, 4)
            baseline_box = (baseline_boxes.mean(dim=0) * self.params.search_size / resize_factor)
            m1_state = clip_box(self.map_box_back(baseline_box, resize_factor), H, W, margin=10)
            if self.center_ranker_state_mode == 'output_only':
                self.state = m1_state
            else:
                self.state = reported_state
            returned_state = reported_state
        elif self.partial_state_commit:
            baseline_boxes = self.network.box_head.cal_bbox(
                baseline_response, baseline_size_map, baseline_offset_map).view(-1, 4)
            baseline_box = baseline_boxes.mean(dim=0) * self.params.search_size / resize_factor
            m1_state = clip_box(self.map_box_back(baseline_box, resize_factor), H, W, margin=10)
            self.state = [
                m1_state[0] + self.partial_alpha_center * (reported_state[0] - m1_state[0]),
                m1_state[1] + self.partial_alpha_center * (reported_state[1] - m1_state[1]),
                m1_state[2] + self.partial_alpha_size * (reported_state[2] - m1_state[2]),
                m1_state[3] + self.partial_alpha_size * (reported_state[3] - m1_state[3]),
            ]
            self.state = clip_box(self.state, H, W, margin=10)
            returned_state = reported_state
        elif self.feature_state_mode == 'output_only':
            baseline_boxes = self.network.box_head.cal_bbox(
                baseline_response, baseline_size_map, baseline_offset_map).view(-1, 4)
            baseline_box = baseline_boxes.mean(dim=0) * self.params.search_size / resize_factor
            m1_state = clip_box(self.map_box_back(baseline_box, resize_factor), H, W, margin=10)
            self.state = m1_state
            returned_state = reported_state
        elif self.readout_state_mode in ('output_only', 'state_only', 'selective_commit'):
            baseline_boxes = self.network.box_head.cal_bbox(
                response, baseline_size_map, baseline_offset_map).view(-1, 4)
            baseline_box = baseline_boxes.mean(dim=0) * self.params.search_size / resize_factor
            m1_state = clip_box(self.map_box_back(baseline_box, resize_factor), H, W, margin=10)
            if self.readout_state_mode == 'state_only':
                self.state = reported_state
                returned_state = m1_state
            elif self.readout_state_mode == 'selective_commit':
                threshold = float(self.state_risk_policy['max_harm_probability'])
                state_commit_corrected = bool(state_risk_probability is not None and state_risk_probability <= threshold)
                self.state = reported_state if state_commit_corrected else m1_state
            else:
                self.state = m1_state
        else:
            self.state = reported_state

        if self.template_update_mode == 'ema_confident':
            flat_response = response.detach().float().flatten()
            order = torch.argsort(flat_response, descending=True)
            peak_margin = float((flat_response[order[0]] - flat_response[order[1]]).item())
            old_center = (float(state_before[0]) + 0.5 * float(state_before[2]),
                          float(state_before[1]) + 0.5 * float(state_before[3]))
            new_center = (float(self.state[0]) + 0.5 * float(self.state[2]),
                          float(self.state[1]) + 0.5 * float(self.state[3]))
            norm_size = max(math.sqrt(max(float(state_before[2]) * float(state_before[3]), 1.0)), 1.0)
            center_shift = math.hypot(new_center[0] - old_center[0],
                                      new_center[1] - old_center[1]) / norm_size
            scale_change = max(float(self.state[2]) / max(float(state_before[2]), 1e-6),
                               float(state_before[2]) / max(float(self.state[2]), 1e-6),
                               float(self.state[3]) / max(float(state_before[3]), 1e-6),
                               float(state_before[3]) / max(float(self.state[3]), 1e-6)) - 1.0
            accepted = (self.frame_id > 1 and peak_margin >= self.template_update_min_margin and
                        center_shift <= self.template_update_max_shift and
                        scale_change <= self.template_update_max_scale)
            if accepted:
                update_patch, _, update_mask = sample_target(
                    image, self.state, self.params.template_factor,
                    output_sz=self.params.template_size)
                update_template = self.preprocessor.process(
                    update_patch, update_mask, half=self.use_fp16)
                alpha = self.template_update_alpha
                self.z_dict1.tensors = ((1.0 - alpha) * self.z_dict1.tensors +
                                        alpha * update_template.tensors)
                self.z_dict1.mask = update_template.mask
                crop_side = max(math.sqrt(max(float(self.state[2]) * float(self.state[3]), 1.0)) *
                                float(self.params.template_factor), 1.0)
                bx = 0.5 * float(self.params.template_size) - 0.5 * float(self.state[2]) * float(self.params.template_size) / crop_side
                by = 0.5 * float(self.params.template_size) - 0.5 * float(self.state[3]) * float(self.params.template_size) / crop_side
                bw = float(self.state[2]) * float(self.params.template_size) / crop_side
                bh = float(self.state[3]) * float(self.params.template_size) / crop_side
                template_box = torch.tensor([[bx / float(self.params.template_size),
                                              by / float(self.params.template_size),
                                              bw / float(self.params.template_size),
                                              bh / float(self.params.template_size)]],
                                             device=self.z_dict1.tensors.device,
                                             dtype=torch.float32)
                self.template_fg_mask = template_box_to_token_weights(template_box, grid_size=8)
                signal_gate = getattr(self.network, 'single_signal_gate', None)
                if signal_gate is not None:
                    signal_gate.template_fg = self.template_fg_mask
                local_signal = getattr(self.network, 'local_signal_post_fusion', None)
                if local_signal is not None:
                    local_signal.template_fg = self.template_fg_mask
            if self.template_update_audit_file:
                with open(self.template_update_audit_file, 'a') as audit_stream:
                    audit_stream.write(json.dumps({
                        'frame_id': int(self.frame_id), 'accepted': bool(accepted),
                        'peak_margin': peak_margin, 'center_shift': center_shift,
                        'scale_change': scale_change,
                        'alpha': self.template_update_alpha}, separators=(',', ':')) + '\n')

        if self.readout_state_mode != 'state_only':
            returned_state = reported_state

        if self.trace_file:
            raw = out_dict.get('tam_raw_head_feature')
            raw_digest = None
            raw_shape = None
            raw_file = None
            if raw is not None:
                raw_cpu = raw.detach().float().cpu().contiguous()
                raw_shape = list(raw_cpu.shape)
                raw_digest = hashlib.sha256(raw_cpu.numpy().tobytes()).hexdigest()
                if self.trace_raw_dir is not None:
                    raw_file = 'raw_head_tokens/frame_%06d.npy' % int(self.frame_id)
                    np.save(os.path.join(self.trace_dir, self.sequence_id, raw_file),
                            raw_cpu.numpy(), allow_pickle=False)
            gt = info.get('gt_bbox') if info else None
            # Peak attribution is computed in the same 16x16 search grid used
            # by CENTER.  The GT score is the best cell overlapping the GT
            # box, which avoids penalizing a valid peak on a neighboring cell.
            peak_diag = {}
            if gt is not None:
                gx, gy, gw, gh = [float(v) for v in gt]
                cx0, cy0, cw, ch = actual_crop_xywh
                sx0, sy0 = (gx - cx0) * float(resize), (gy - cy0) * float(resize)
                sx1, sy1 = (gx + gw - cx0) * float(resize), (gy + gh - cy0) * float(resize)
                yy, xx = torch.meshgrid(torch.arange(self.feat_sz), torch.arange(self.feat_sz))
                tx0, ty0 = xx.float() * (self.params.search_size / self.feat_sz), yy.float() * (self.params.search_size / self.feat_sz)
                tx1, ty1 = tx0 + self.params.search_size / self.feat_sz, ty0 + self.params.search_size / self.feat_sz
                gt_mask = (tx1 > sx0) & (tx0 < sx1) & (ty1 > sy0) & (ty0 < sy1)
                crop_area = max(0.0, min(cx0 + cw, gx + gw) - max(cx0, gx)) * max(0.0, min(cy0 + ch, gy + gh) - max(cy0, gy))
                raw_flat = pred_score_map.detach().float().flatten()
                hann_flat = response.detach().float().flatten()
                valid = gt_mask.flatten().to(raw_flat.device)
                if bool(valid.any()):
                    raw_gt = raw_flat[valid].max()
                    hann_gt = hann_flat[valid].max()
                    raw_order = torch.argsort(raw_flat, descending=True)
                    hann_order = torch.argsort(hann_flat, descending=True)
                    raw_peak = int(raw_order[0]); hann_peak = int(hann_order[0])
                    background_valid = ~valid
                    raw_bg = raw_flat.masked_fill(~background_valid, -float('inf'))
                    hann_bg = hann_flat.masked_fill(~background_valid, -float('inf'))
                    raw_bg_peak = int(raw_bg.argmax())
                    hann_bg_peak = int(hann_bg.argmax())
                    raw_bg_score = raw_bg[raw_bg_peak]
                    hann_bg_score = hann_bg[hann_bg_peak]
                    gt_center_x = ((sx0 + sx1) / 2.0) / (self.params.search_size / self.feat_sz)
                    gt_center_y = ((sy0 + sy1) / 2.0) / (self.params.search_size / self.feat_sz)
                    raw_rank = int((raw_flat > raw_gt).sum().item() + 1)
                    hann_rank = int((hann_flat > hann_gt).sum().item() + 1)
                    peak_diag = {
                        'crop_contains_gt': bool(crop_area > 0.0),
                        'gt_cell_count': int(valid.sum().item()),
                        'gt_score_raw': float(raw_gt), 'gt_score_hann': float(hann_gt),
                        'background_score_raw': float(raw_bg_score),
                        'background_score_hann': float(hann_bg_score),
                        'target_background_margin_raw': float(raw_gt - raw_bg_score),
                        'target_background_margin_hann': float(hann_gt - hann_bg_score),
                        'background_peak_raw_xy': [int(raw_bg_peak % self.feat_sz), int(raw_bg_peak // self.feat_sz)],
                        'background_peak_hann_xy': [int(hann_bg_peak % self.feat_sz), int(hann_bg_peak // self.feat_sz)],
                        'background_peak_hann_distance': float(math.hypot(
                            hann_bg_peak % self.feat_sz - gt_center_x,
                            hann_bg_peak // self.feat_sz - gt_center_y)),
                        'gt_rank_raw': raw_rank, 'gt_rank_hann': hann_rank,
                        'raw_top1_xy': [int(raw_peak % self.feat_sz), int(raw_peak // self.feat_sz)],
                        'hann_top1_xy': [int(hann_peak % self.feat_sz), int(hann_peak // self.feat_sz)],
                        'raw_peak_gt_margin': float(raw_flat[raw_peak] - raw_gt),
                        'hann_peak_gt_margin': float(hann_flat[hann_peak] - hann_gt),
                        'raw_peak_gt_distance': float(math.hypot(raw_peak % self.feat_sz - ((sx0 + sx1) / 2.0) / (self.params.search_size / self.feat_sz), raw_peak // self.feat_sz - ((sy0 + sy1) / 2.0) / (self.params.search_size / self.feat_sz))),
                    }
                    # TargetRankGate attribution is trace-only. It records
                    # whether the learned spatial score separates the same
                    # GT cells from the hard-background cells used above.
                    rank_score = out_dict.get('target_rank_score')
                    if torch.is_tensor(rank_score):
                        rank_flat = rank_score.detach().float().flatten()
                        rank_gt = rank_flat[valid].max()
                        rank_bg_values = rank_flat.masked_fill(valid, -float('inf'))
                        rank_bg = rank_bg_values.max()
                        rank_order = torch.argsort(rank_flat, descending=True)
                        rank_gt_rank = int((rank_flat > rank_gt).sum().item() + 1)
                        peak_diag['target_rank_score_gt'] = float(rank_gt)
                        peak_diag['target_rank_score_background'] = float(rank_bg)
                        peak_diag['target_rank_score_margin'] = float(rank_gt - rank_bg)
                        peak_diag['target_rank_score_gt_rank'] = rank_gt_rank
                        peak_diag['target_rank_score_top1_xy'] = [
                            int(rank_order[0] % self.feat_sz), int(rank_order[0] // self.feat_sz)]
                        peak_diag['target_rank_score_mean'] = float(rank_flat.mean())
                        peak_diag['target_rank_score_std'] = float(rank_flat.std(unbiased=False))
            amcm_diag = {}
            for name in ('prob_rgb', 'prob_tir', 'residual_rgb', 'residual_tir',
                         'message_norm_rgb', 'message_norm_tir', 'input_rgb',
                         'input_tir', 'output_rgb', 'output_tir'):
                value = out_dict.get('amcm_' + name)
                if torch.is_tensor(value):
                    amcm_diag[name + '_mean'] = float(value.detach().float().mean())
                    amcm_diag[name + '_norm'] = float(value.detach().float().norm())
            row = {
                'schema_version': 1,
                'sequence_id': self.sequence_id,
                'frame_id': int(self.frame_id),
                'history_frame_id': (int(history_frame_for_row)
                                     if history_frame_for_row is not None else None),
                'has_history': bool(history_frame_for_row is not None),
                'history_sequence_id': history_sequence_for_row,
                'current_crop_xywh': actual_crop_xywh,
                'history_crop_xywh': history_crop_for_row,
                'current_crop_transform': crop_transform,
                'history_crop_transform': history_transform_for_row,
                'crop_size': [crop_sz, crop_sz],
                'padding': [pad_left, pad_top, pad_right, pad_bottom],
                'tracker_state_before': [float(v) for v in state_before],
                'tracker_state_after': [float(v) for v in self.state],
                'pred_box': [float(v) for v in self.state],
                'reported_box': [float(v) for v in returned_state],
                'readout_state_mode': self.readout_state_mode,
                'state_commit_corrected': state_commit_corrected,
                'state_commit_risk_probability': state_risk_probability,
                'state_commit_risk_threshold': (float(self.state_risk_policy['max_harm_probability'])
                                                if self.readout_state_mode == 'selective_commit' else None),
                'gt_box': ([float(v) for v in gt] if gt is not None else None),
                'gt_valid': bool(gt is not None),
                'peak_diagnostics': peak_diag,
                'early_gate_stats': out_dict.get('early_gate_stats', []),
                'amcm_diagnostics': amcm_diag,
                'center_ranker': center_ranker_selected,
                'raw_head_shape': raw_shape,
                'raw_head_sha256': raw_digest,
                'raw_head_file': raw_file,
            }
            with open(self.trace_file, 'a') as handle:
                handle.write(json.dumps(row, separators=(',', ':')) + '\n')
            self.prev_crop_transform = crop_transform

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
            return {"target_bbox": returned_state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": returned_state}

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

    def _record_fusion_diagnostics(self, sequence_name, frame_num):
        """Write V2 attention directionality and redundancy diagnostics."""
        if os.environ.get('TBSI_FUSION_DIAGNOSTICS', '0') != '1':
            return
        layer_ids = getattr(self.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9])
        rows = []
        for layer_id, layer in zip(layer_ids, self.network.backbone.tbsi_layers):
            fusion = getattr(layer, 'v2_template_fusion', None)
            stats = getattr(fusion, 'last_fusion_stats', None)
            if stats is None:
                continue
            row = {'sequence': sequence_name, 'frame': int(frame_num),
                   'layer': int(layer_id)}
            for key, value in stats.items():
                # Snapshot metadata may be categorical (e.g. fusion_pattern);
                # keep it intact while normalizing tensor/scalar metrics.
                if isinstance(value, str):
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
        if rows:
            with open(self.attr_stats_path, 'a') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')

    def _record_c1_pair_diagnostics(self, sequence_name, frame_num):
        """Summarize oracle target-pair purity and compatibility for C1."""
        if os.environ.get('TBSI_FUSION_PAIR_STATS', '0') != '1' or not self.attr_stats_path:
            return
        mask = self.c1_template_mask
        coords = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8)), dim=-1).reshape(-1, 2).float()
        dist = ((coords[:, None] - coords[None, :]) ** 2).sum(-1).sqrt()
        target_pair = mask[:, None] & mask[None, :]
        rows = []
        active = set(getattr(self.cfg.MODEL.BACKBONE, 'TBSI_FUSION_ACTIVE', []) or
                     getattr(self.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9]))
        for layer_id, layer in zip(getattr(self.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9]),
                                   self.network.backbone.tbsi_layers):
            if layer_id not in active:
                continue
            fusion = getattr(layer, 'v2_template_fusion', None)
            attn = getattr(fusion, 'last_pair_attention', None)
            q = getattr(fusion, 'last_pair_q', None)
            k = getattr(fusion, 'last_pair_k', None)
            if attn is None or q is None or k is None:
                continue
            for direction, block, q_slice, k_slice in (
                    ('RGB2TIR', attn[:, :, :64, 64:128], q[:, :, :64], k[:, :, 64:128]),
                    ('TIR2RGB', attn[:, :, 64:128, :64], q[:, :, 64:128], k[:, :, :64])):
                weights = block[0].mean(0)
                q_mean = q_slice[0].mean(0)
                k_mean = k_slice[0].mean(0)
                sims = torch.nn.functional.normalize(q_mean, dim=-1) @ torch.nn.functional.normalize(k_mean, dim=-1).transpose(0, 1)
                topn = max(1, min(10, weights.numel()))
                top_idx = weights.flatten().topk(topn).indices
                target_flat = target_pair.flatten()
                rows.append({'sequence': sequence_name, 'frame': int(frame_num), 'layer': int(layer_id),
                             'direction': direction, 'target_pair_mass': float(weights[target_pair].sum()),
                             'cross_mass': float(weights.sum()),
                             'target_pair_purity': float(weights[target_pair].sum() / weights.sum().clamp_min(1e-12)),
                             'top10_target_purity': float(target_flat[top_idx].float().mean()),
                             'target_distance': float(dist[target_pair].mean()) if target_pair.any() else 0.0,
                             'target_similarity': float(sims[target_pair].mean()) if target_pair.any() else 0.0,
                             'non_target_similarity': float(sims[~target_pair].mean()) if (~target_pair).any() else 0.0})
        if rows:
            with open(self.attr_stats_path, 'a') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')

    def _record_token_gain_diagnostics(self, sequence_name, frame_num):
        """Record token-wise fusion gain for the routing hypothesis."""
        if os.environ.get('TBSI_FUSION_GAIN_STATS', '0') != '1' or not self.attr_stats_path:
            return
        mask = self.c1_template_mask
        rows = []
        for layer_id, layer in zip(getattr(self.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9]),
                                   self.network.backbone.tbsi_layers):
            fusion = getattr(layer, 'v2_template_fusion', None)
            gain = getattr(fusion, 'last_token_gain', None)
            if gain is None:
                continue
            gain = gain[0]
            for modality, values in (('RGB', gain[:64]), ('TIR', gain[64:128])):
                for region, selected in (('target', mask), ('background', ~mask)):
                    if selected.any():
                        vals = values[selected]
                        rows.append({'sequence': sequence_name, 'frame': int(frame_num),
                                     'layer': int(layer_id), 'modality': modality,
                                     'region': region, 'count': int(vals.numel()),
                                     'gain_mean': float(vals.mean()),
                                     'gain_median': float(vals.median()),
                                     'gain_p90': float(torch.quantile(vals, 0.90)),
                                     'gain_max': float(vals.max())})
        if rows:
            with open(self.attr_stats_path, 'a') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')

    def _record_correspondence_diagnostics(self, sequence_name, frame_num):
        """Compare independent cosine and Sinkhorn token correspondences."""
        if os.environ.get('TBSI_CORRESPONDENCE_STATS', '0') != '1' or not self.attr_stats_path:
            return
        mask = self.c1_template_mask
        coords = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8)), dim=-1).reshape(-1, 2).float()
        distance = ((coords[:, None] - coords[None, :]) ** 2).sum(-1).sqrt()
        target_pair = mask[:, None] & mask[None, :]

        def sinkhorn(log_scores, iterations=30):
            # Uniform 64x64 marginals; log-domain updates are stable in FP32.
            log_p = log_scores.float()
            log_marginal = -torch.log(torch.tensor(float(log_p.shape[-1])))
            for _ in range(iterations):
                log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True) + log_marginal
                log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True) + log_marginal
            return log_p.exp()

        rows = []
        for layer_id, layer in zip(getattr(self.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9]),
                                   self.network.backbone.tbsi_layers):
            fusion = getattr(layer, 'v2_template_fusion', None)
            tokens = getattr(fusion, 'last_template_tokens', None)
            if tokens is None:
                continue
            rgb, tir = tokens
            similarity = torch.nn.functional.normalize(rgb[0], dim=-1) @ torch.nn.functional.normalize(tir[0], dim=-1).t()
            for direction, scores in (('RGB2TIR', similarity), ('TIR2RGB', similarity.t())):
                pair_mask = target_pair if direction == 'RGB2TIR' else target_pair.t()
                dist = distance if direction == 'RGB2TIR' else distance.t()
                cosine_prob = scores.softmax(dim=-1)
                ot_prob = sinkhorn(scores / 0.07)
                for method, prob in (('cosine', cosine_prob), ('sinkhorn', ot_prob)):
                    top1 = prob.argmax(dim=-1)
                    top5 = prob.topk(5, dim=-1).indices
                    top10 = prob.topk(10, dim=-1).indices
                    row_ids = torch.arange(prob.shape[0])
                    top1_target = pair_mask[row_ids, top1]
                    top5_target = pair_mask[row_ids[:, None], top5].any(dim=-1)
                    top10_target = pair_mask[row_ids[:, None], top10].any(dim=-1)
                    target_rows = mask
                    bg_rows = ~mask
                    false_target_to_bg = (~pair_mask[target_rows, :].gather(1, top1[target_rows, None]).squeeze(1)).float().mean()
                    rows.append({'sequence': sequence_name, 'frame': int(frame_num), 'layer': int(layer_id),
                                 'direction': direction, 'method': method,
                                 'match_precision': float(top1_target.float().mean()),
                                 'target_top5_recall': float(top5_target[target_rows].float().mean()),
                                 'target_top10_recall': float(top10_target[target_rows].float().mean()),
                                 'target_to_background_false_rate': float(false_target_to_bg),
                                 'mean_match_distance': float(dist[row_ids, top1].mean()),
                                 'target_match_distance': float(dist[target_rows][top1_target[target_rows]].mean()) if top1_target[target_rows].any() else 0.0})
        if rows:
            with open(self.attr_stats_path, 'a') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')

    def _record_complement_diagnostics(self, sequence_name, frame_num):
        """Measure whether joint-fusion residuals align with the other modality."""
        if os.environ.get('TBSI_COMPLEMENT_STATS', '0') != '1' or not self.attr_stats_path:
            return
        mask = self.c1_template_mask
        rows = []
        for layer_id, layer in zip(getattr(self.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9]),
                                   self.network.backbone.tbsi_layers):
            fusion = getattr(layer, 'v2_template_fusion', None)
            before = getattr(fusion, 'last_complement_before', None)
            after = getattr(fusion, 'last_complement_after', None)
            if before is None or after is None:
                continue
            before = before[0]
            after = after[0]
            rgb, tir = before[:64], before[64:128]
            fused_rgb, fused_tir = after[:64], after[64:128]
            for direction, target, source, fused in (
                    ('TIR2RGB', rgb, tir, fused_rgb),
                    ('RGB2TIR', tir, rgb, fused_tir)):
                residual = fused - target
                residual_norm = residual.norm(dim=-1)
                residual_unit = torch.nn.functional.normalize(residual, dim=-1)
                target_unit = torch.nn.functional.normalize(target, dim=-1)
                source_unit = torch.nn.functional.normalize(source, dim=-1)
                source_cos = (residual_unit * source_unit).sum(-1)
                target_cos = (residual_unit * target_unit).sum(-1)
                rgb_preservation = (torch.nn.functional.normalize(rgb, dim=-1) *
                                    torch.nn.functional.normalize(fused_rgb, dim=-1)).sum(-1)
                tir_preservation = (torch.nn.functional.normalize(tir, dim=-1) *
                                    torch.nn.functional.normalize(fused_tir, dim=-1)).sum(-1)
                fused_rgb_unit = torch.nn.functional.normalize(fused_rgb, dim=-1)
                fused_tir_unit = torch.nn.functional.normalize(fused_tir, dim=-1)
                cross_after = (fused_rgb_unit * fused_tir_unit).sum(-1)
                cross_before = (torch.nn.functional.normalize(rgb, dim=-1) *
                                torch.nn.functional.normalize(tir, dim=-1)).sum(-1)
                for region, selected in (('target', mask), ('background', ~mask)):
                    if selected.any():
                        rows.append({'sequence': sequence_name, 'frame': int(frame_num),
                                     'layer': int(layer_id), 'direction': direction,
                                     'region': region, 'count': int(selected.sum()),
                                     'residual_norm': float(residual_norm[selected].mean()),
                                     'source_cosine': float(source_cos[selected].mean()),
                                     'target_cosine': float(target_cos[selected].mean()),
                                     'source_minus_target_cosine': float((source_cos[selected] - target_cos[selected]).mean()),
                                     'rgb_preservation': float(rgb_preservation[selected].mean()),
                                     'tir_preservation': float(tir_preservation[selected].mean()),
                                     'cross_modal_before_cosine': float(cross_before[selected].mean()),
                                     'cross_modal_after_cosine': float(cross_after[selected].mean())})
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
