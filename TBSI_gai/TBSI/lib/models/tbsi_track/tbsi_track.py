"""
TBSI_Track model. Developed on OSTrack.
"""
import math
from operator import ipow
import os
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from easydict import EasyDict as edict
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.tbsi_track.vit_tbsi_care import vit_base_patch16_224_tbsi
from lib.models.layers.temporal_token import TemporalTokenBlock
from lib.models.layers.causal_tam_adapter import CausalTemporalFusionResidual
from lib.models.layers.adaptive_post_fusion_router import AdaptivePostFusionRouter
from lib.models.layers.precenter_gated_residual import PreCenterGatedResidual
from lib.models.layers.msha_screen_adapters import (
    MshaReliabilityAdapter, TargetDynamicLocalOperator,
    TargetComplementaryAdaptiveFusion,
)
from lib.models.layers.amcm_reliability import AMCMLiteReliability
from lib.models.layers.pvl_adapter import PVLAdapter
from lib.models.layers.target_rank_gate import TargetRankGate
from lib.models.layers.spatial_token_bridge import SpatialTokenBridge
from lib.utils.box_ops import box_xyxy_to_cxcywh


class DegradationAwareFusion(nn.Module):
    """
    Quality-Aware Fusion Module.

    Core design:
      - quality_conv: global modality quality estimation → scalar per modality
      - spatial_gate: per-pixel spatial weight (B, 2, H, W)
      - channel_gate: per-channel modulation (B, 2C, 1, 1)
      - MADC: zero-param amplitude normalization on spatial weights
      - CSR: complementary-aware channel enhancement on fused output

    All sub-networks default-initialized. quality_conv and channel_gate
    last conv layers are zero-initialized so initial fusion ≈ uniform.
    """

    def __init__(self, dim, reduction=16, da_mode='spatial',
                 use_madc=False, use_csr=False):
        super().__init__()
        rdim = max(dim // reduction, 16)
        self.da_mode = da_mode
        self.use_madc = use_madc
        self.use_csr = use_csr

        # Global modality quality (scalar per modality)
        self.quality_conv = nn.Sequential(
            nn.Conv2d(dim * 2, rdim, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(rdim, 2, kernel_size=1, padding=0, bias=True),
        )
        # Zero-init last layer: initial output ≈ 0 → sigmoid(0) ≈ 0.5 → uniform quality
        nn.init.zeros_(self.quality_conv[-1].weight)
        nn.init.zeros_(self.quality_conv[-1].bias)

        # Spatial gate (B, 2, H, W)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2, rdim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(rdim),
            nn.ReLU(inplace=True),
            nn.Conv2d(rdim, 2, kernel_size=1, padding=0, bias=True),
        )

        # Channel gate (B, 2C, 1, 1) — only in channel mode
        if da_mode == 'channel':
            self.channel_gate = nn.Sequential(
                nn.Conv2d(dim * 2, rdim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(rdim, dim * 2, kernel_size=1, padding=0, bias=True),
            )
            # Zero-init last layer: initial channel_gate ≈ 0.5 → uniform per-channel weight
            nn.init.zeros_(self.channel_gate[-1].weight)
            nn.init.zeros_(self.channel_gate[-1].bias)

        # CSR: Complementary-aware channel enhancement on fused features
        if use_csr:
            rdim_csr = max(dim // reduction, 16)
            self.comp_reduce = nn.Conv2d(dim, rdim_csr, kernel_size=1, bias=True)
            self.comp_channel = nn.Conv2d(rdim_csr, dim, kernel_size=1, bias=True)
            # Zero-init: start from identity (channel_gate=0 → no modulation)
            nn.init.zeros_(self.comp_channel.weight)
            nn.init.zeros_(self.comp_channel.bias)

    def forward(self, feat_rgb: torch.Tensor, feat_tir: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_rgb.shape
        cat_feat = torch.cat([feat_rgb, feat_tir], dim=1)
        pooled = cat_feat.mean(dim=[2, 3], keepdim=True)

        # (1) Global quality scores
        q = torch.sigmoid(self.quality_conv(pooled))          # (B, 2, 1, 1)
        q_rgb, q_tir = q[:, 0:1], q[:, 1:2]

        # (2) Spatial gate
        s = torch.sigmoid(self.spatial_gate(cat_feat))         # (B, 2, H, W)
        s_rgb, s_tir = s[:, 0:1], s[:, 1:2]

        # (3) MADC: amplitude normalization (zero-param)
        if self.use_madc:
            mag_rgb = feat_rgb.abs().mean(dim=[1,2,3], keepdim=True) + 1e-6
            mag_tir = feat_tir.abs().mean(dim=[1,2,3], keepdim=True) + 1e-6
            alpha = 0.5
            s_rgb = s_rgb * (1.0 / mag_rgb).pow(alpha)
            s_tir = s_tir * (1.0 / mag_tir).pow(alpha)

        # (4) Fusion weights = quality * spatial gate
        w_rgb = s_rgb * q_rgb
        w_tir = s_tir * q_tir

        # (5) Channel-wise modulation
        if self.da_mode == 'channel':
            c = torch.sigmoid(self.channel_gate(pooled))
            w_rgb = w_rgb * c[:, :C]
            w_tir = w_tir * c[:, C:]

        # (6) Normalize and fuse
        total = w_rgb + w_tir + 1e-8
        fused = feat_rgb * (w_rgb / total) + feat_tir * (w_tir / total)

        # (7) CSR: complementary channel enhancement (residual)
        if self.use_csr:
            feat_diff = feat_rgb - feat_tir
            diff_feat = F.relu(self.comp_reduce(feat_diff))
            channel_gate = torch.tanh(self.comp_channel(diff_feat))
            channel_gate = torch.clamp(channel_gate, -0.5, 0.5)    # stability clamp
            fused = fused * (1.0 + 0.1 * channel_gate)

        return fused

    def forward_gate_only(self, feat_rgb, feat_tir):
        """Plan B: lightweight spatial gate modulating base_fused."""
        cat_feat = torch.cat([feat_rgb, feat_tir], dim=1)
        s = torch.sigmoid(self.spatial_gate(cat_feat))
        # Single modulation map: average RGB/TIR gates
        gate = (s[:, 0:1] + s[:, 1:2]) / 2.0  # (B, 1, H, W)
        return gate  # base_fused will multiply by (1 + gate)



class TemporalChannelCalibration(nn.Module):
    """Temporal-Conditioned Channel Calibration (TC3).
    
    Uses MDTA temporal tokens to predict channel-wise modulation weights
    for the fused features. Complements tbsi_fuse_search by adding
    temporal-context-aware channel adaptation.
    
    Forward: tokens → MLP → sigmoid → channel_weights → modulate fused_feat
    Params: ~10K (768→48→768), converges in 4ep sprint.
    """
    def __init__(self, dim=768, reduction=16):
        super().__init__()
        rdim = max(dim // reduction, 16)
        self.channel_predictor = nn.Sequential(
            nn.Linear(dim, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, dim),
            nn.Sigmoid(),
        )
        # Zero-init last layer: start from weight=1 (identity)
        nn.init.zeros_(self.channel_predictor[-2].weight)
        nn.init.zeros_(self.channel_predictor[-2].bias)
    
    def forward(self, fused_feat, temporal_tokens):
        """Args:
            fused_feat: (B, C, H, W) from tbsi_fuse_search
            temporal_tokens: (B, K, D) from MDTA
        Returns:
            calibrated_feat: (B, C, H, W)
        """
        # Aggregate temporal tokens into context vector
        token_context = temporal_tokens.mean(dim=1)  # (B, D)
        # Predict channel-wise weights
        channel_weight = self.channel_predictor(token_context)  # (B, C)
        # Modulate fused features
        return fused_feat * channel_weight.unsqueeze(-1).unsqueeze(-1)


class TBSITrack(nn.Module):
    """ TBSI with Temporal Token + Quality-Aware Fusion (spatial+channel+MADC+CSR) """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 use_temporal_tokens=False, num_temporal_tokens=4,
                 use_degradation_aware=False, da_mode='spatial',
                 use_madc=False, use_csr=False, causal_tam_cfg=None,
                 adaptive_post_fusion_cfg=None, precenter_gated_residual_cfg=None,
                 amcm_reliability_cfg=None, pvl_cfg=None, target_rank_gate_cfg=None,
                 spatial_token_bridge_cfg=None):
        super().__init__()
        self._amcm_reliability_only = False
        self._amcm_reliability_joint = False
        self._amcm_reliability_joint_head = False
        self._early_utility_router_only = False
        self._spatial_token_bridge_only = False
        self._last_amcm_stats = None
        self._last_amcm_repr = None
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.early_utility_router_enabled = bool(
            getattr(transformer, 'early_utility_router_enable', False))
        self.tbsi_fuse_search = conv(hidden_dim * 2, hidden_dim)
        self.spatial_token_bridge = None
        if (spatial_token_bridge_cfg is not None and
                getattr(spatial_token_bridge_cfg, 'ENABLE', False)):
            if use_degradation_aware:
                raise ValueError('SpatialTokenBridge cannot be combined with degradation-aware fusion')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261023)
                self.spatial_token_bridge = SpatialTokenBridge(
                    dim=hidden_dim,
                    rank=getattr(spatial_token_bridge_cfg, 'RANK', 32),
                    topk=getattr(spatial_token_bridge_cfg, 'TOPK', 8),
                    max_delta=getattr(spatial_token_bridge_cfg, 'MAX_DELTA', 0.10),
                    temperature=getattr(spatial_token_bridge_cfg, 'TEMPERATURE', 0.10),
                    spatial_sigma=getattr(spatial_token_bridge_cfg, 'SPATIAL_SIGMA', 2.0),
                    grid_size=int(box_head.feat_sz))
        self.pvl_neck = None
        if pvl_cfg is not None and getattr(pvl_cfg, 'ENABLE', False):
            pvl_position = getattr(pvl_cfg, 'POSITION', 'template_loc6')
            if pvl_position not in ('template_loc6', 'search_neck'):
                raise ValueError('MODEL.PVL.POSITION must be template_loc6 or search_neck')
            if pvl_position == 'search_neck':
                # 与模板位置相同，隔离新模块权重初始化对 parent 随机流的影响。
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(20261020)
                    self.pvl_neck = PVLAdapter(
                        dim=hidden_dim,
                        adapter_dim=getattr(pvl_cfg, 'ADAPTER_DIM', 64),
                        beta=getattr(pvl_cfg, 'BETA', 2.35))
        self.amcm_reliability = None
        self.amcm_reliability_position = 'pre_fusion'
        self.amcm_reliability_mode = 'legacy'
        if amcm_reliability_cfg is not None and getattr(amcm_reliability_cfg, 'ENABLE', False):
            self.amcm_reliability_position = getattr(
                amcm_reliability_cfg, 'POSITION', 'pre_fusion')
            self.amcm_reliability_mode = getattr(
                amcm_reliability_cfg, 'MODE', 'legacy')
            if self.amcm_reliability_position not in ('pre_fusion', 'post_fusion'):
                raise ValueError('AMCM_RELIABILITY.POSITION must be pre_fusion or post_fusion')
            self.amcm_reliability = AMCMLiteReliability(
                dim=hidden_dim,
                hidden_dim=getattr(amcm_reliability_cfg, 'HIDDEN_DIM', 32),
                max_delta=getattr(amcm_reliability_cfg, 'MAX_DELTA', 0.10),
                mode=self.amcm_reliability_mode)
        self.adaptive_post_fusion = None
        self.precenter_gated_residual = None
        self.msha_reliability_adapter = None
        self.target_dynamic_local_operator = None
        self.target_rank_gate = None
        self.target_rank_response_coupling = False
        self.target_rank_response_beta = 0.0
        if target_rank_gate_cfg is not None and getattr(target_rank_gate_cfg, 'ENABLE', False):
            self.target_rank_gate = TargetRankGate(
                dim=hidden_dim,
                hidden_dim=getattr(target_rank_gate_cfg, 'HIDDEN_DIM', 32),
                max_residual=getattr(target_rank_gate_cfg, 'MAX_RESIDUAL', 0.10))
            self.target_rank_response_coupling = bool(getattr(
                target_rank_gate_cfg, 'RESPONSE_COUPLING', False))
            self.target_rank_response_beta = float(getattr(
                target_rank_gate_cfg, 'RESPONSE_BETA', 0.0))
        self.target_complementary_adaptive_fusion = None
        if adaptive_post_fusion_cfg is not None and getattr(
                adaptive_post_fusion_cfg, 'ENABLE', False):
            self.adaptive_post_fusion = AdaptivePostFusionRouter(
                hidden_dim,
                hidden_dim=getattr(adaptive_post_fusion_cfg, 'HIDDEN_DIM', 64),
                max_residual=getattr(adaptive_post_fusion_cfg, 'MAX_RESIDUAL', 0.10),
                force_zero=getattr(adaptive_post_fusion_cfg, 'FORCE_ZERO', False),
                disagreement_aware=getattr(adaptive_post_fusion_cfg, 'DISAGREEMENT_AWARE', False),
                gate_bias=getattr(adaptive_post_fusion_cfg, 'GATE_BIAS', -4.0),
            )
        if precenter_gated_residual_cfg is not None and getattr(
                precenter_gated_residual_cfg, 'ENABLE', False):
            self.precenter_gated_residual = PreCenterGatedResidual(
                hidden_dim=hidden_dim,
                dim=hidden_dim,
                max_residual=getattr(precenter_gated_residual_cfg, 'MAX_RESIDUAL', 0.10),
                gate_bias=getattr(precenter_gated_residual_cfg, 'GATE_BIAS', -4.0),
                force_zero=getattr(precenter_gated_residual_cfg, 'FORCE_ZERO', False),
                target_conditioned=getattr(precenter_gated_residual_cfg, 'TARGET_CONDITIONED', False))
        reliability_cfg = getattr(getattr(transformer, '_screen_cfg', None),
                                  'MSHA_RELIABILITY_ADAPTER', None)
        dynamic_cfg = getattr(getattr(transformer, '_screen_cfg', None),
                              'TARGET_DYNAMIC_LOCAL_OPERATOR', None)
        if reliability_cfg is not None and getattr(reliability_cfg, 'ENABLE', False):
            self.msha_reliability_adapter = MshaReliabilityAdapter(
                dim=hidden_dim,
                hidden_dim=getattr(reliability_cfg, 'HIDDEN_DIM', 32),
                max_residual=getattr(reliability_cfg, 'MAX_RESIDUAL', 0.10),
                gate_bias=getattr(reliability_cfg, 'GATE_BIAS', -4.0))
        if dynamic_cfg is not None and getattr(dynamic_cfg, 'ENABLE', False):
            self.target_dynamic_local_operator = TargetDynamicLocalOperator(
                dim=hidden_dim,
                hidden_dim=getattr(dynamic_cfg, 'HIDDEN_DIM', 32),
                max_residual=getattr(dynamic_cfg, 'MAX_RESIDUAL', 0.10))
        complementary_cfg = getattr(getattr(transformer, '_screen_cfg', None),
                                    'TARGET_COMPLEMENTARY_ADAPTIVE_FUSION', None)
        if complementary_cfg is not None and getattr(complementary_cfg, 'ENABLE', False):
            self.target_complementary_adaptive_fusion = TargetComplementaryAdaptiveFusion(
                dim=hidden_dim,
                hidden_dim=getattr(complementary_cfg, 'HIDDEN_DIM', 32),
                max_residual=getattr(complementary_cfg, 'MAX_RESIDUAL', 0.10),
                gate_bias=getattr(complementary_cfg, 'GATE_BIAS', -4.0))
        # The active model uses the original M1 fusion path.
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        # Temporal Token Module (pre-fusion)
        self.use_temporal_tokens = use_temporal_tokens
        if use_temporal_tokens:
            self.temporal_token_block = TemporalTokenBlock(
                dim=hidden_dim, num_tokens=num_temporal_tokens,
            )
            self.temporal_token_state = None
            # TC3: Temporal-Conditioned Channel Calibration
            self.tc3 = TemporalChannelCalibration(dim=hidden_dim)

        # Quality-Aware Fusion Module (post temporal token)
        self.use_degradation_aware = use_degradation_aware
        if use_degradation_aware:
            self.da_fusion = DegradationAwareFusion(dim=hidden_dim, da_mode=da_mode,
                                                    use_madc=use_madc, use_csr=use_csr)
            self.da_fusion_mode = 'residual'  # 'residual' | 'gate'
            self.da_fusion_scale = 0.5        # residual scale

        self._cplain_only = False
        self.use_causal_tam = bool(causal_tam_cfg is not None and
                                   getattr(causal_tam_cfg, 'ENABLE', False))
        if self.use_causal_tam:
            self.causal_tam = CausalTemporalFusionResidual(
                dim=hidden_dim,
                hidden=getattr(causal_tam_cfg, 'HIDDEN', 192),
                heads=getattr(causal_tam_cfg, 'HEADS', 4),
                grid_size=getattr(causal_tam_cfg, 'GRID_SIZE', self.feat_sz_s),
                local_radius=getattr(causal_tam_cfg, 'LOCAL_RADIUS', 2),
                max_message_ratio=getattr(causal_tam_cfg, 'MAX_MESSAGE_RATIO', 0.10),
                force_zero=getattr(causal_tam_cfg, 'FORCE_ZERO', False))

        # Print summary
        parts = []
        if use_temporal_tokens: parts.append(f'TemporalTokens(K={num_temporal_tokens})')
        if use_degradation_aware:
            da_str = 'QualityAwareFusion'
            if use_madc: da_str += '+MADC'
            if use_csr: da_str += '+CSR'
            parts.append(da_str)
        if parts: print(f'TBSITrack with: {", ".join(parts)}')

    def set_cplain_only(self, enabled=True):
        self._cplain_only = bool(enabled)

    def train(self, mode=True):
        super().train(mode)
        if mode and self._amcm_reliability_only and self.amcm_reliability is not None:
            # Frozen M1 modules must retain eval-mode buffers while AMCM trains.
            super().train(False)
            self.amcm_reliability.train(True)
        if mode and self._amcm_reliability_joint and self.amcm_reliability is not None:
            # Keep the frozen backbone in eval mode, while AMCM and the
            # fusion projection (including its BN) remain trainable.
            super().train(False)
            self.amcm_reliability.train(True)
            self.tbsi_fuse_search.train(True)
        if mode and self._amcm_reliability_joint_head and self.amcm_reliability is not None:
            super().train(False)
            self.amcm_reliability.train(True)
            self.tbsi_fuse_search.train(True)
            self.box_head.train(True)
        if mode and self._early_utility_router_only:
            super().train(False)
            self.backbone.early_utility_router.train(True)
        if mode and self._spatial_token_bridge_only:
            super().train(False)
            self.spatial_token_bridge.train(True)
        if self._cplain_only:
            # Frozen M1 modules must retain eval-mode buffers while the
            # residual adapter remains trainable.
            for module in self.modules():
                parameters = list(module.parameters(recurse=False))
                if parameters and not any(parameter.requires_grad for parameter in parameters):
                    module.eval()
            for adapter_name in ('early_search_residual', 'post_tbsi_residual'):
                adapter = getattr(self.backbone, adapter_name, None)
                if adapter is not None:
                    adapter.train(mode)
        return self

    def reset_temporal_tokens(self):
        self.temporal_token_state = None

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                prev_search: torch.Tensor = None,
                prev_tokens: torch.Tensor = None,
                template_fg_masks=None,
                diagnostic=False,
                diagnostic_early_block=2,
                diagnostic_late_block=9,
                diagnostic_tbsi_index=0,
                diagnostic_post_block=5,
                prev_head_feature=None,
                current_crop_xywh=None,
                history_crop_xywh=None,
                current_frame_id=None,
                history_frame_id=None,
                current_sequence_id=None,
                history_sequence_id=None,
                ):
        """
        Args:
            template: (B, 6, 128, 128) — [RGB, TIR]
            search: (B, 6, 256, 256) — current frame
            prev_search: (B, 6, 256, 256) or None — previous frame (for temporal token update)
            prev_tokens: (B, K*2, D) or None — previous temporal tokens
        """
        # Two-pass: update token state with prev frame, then process current
        if self.use_temporal_tokens and prev_search is not None:
            with torch.no_grad():
                x_prev, _ = self.backbone(z=template, x=prev_search,
                                           ce_template_mask=ce_template_mask,
                                           ce_keep_rate=ce_keep_rate,
                                           template_fg_masks=template_fg_masks)
                feat_prev = x_prev[-1] if isinstance(x_prev, list) else x_prev
                # Split RGB/TIR search tokens (64 template + 256 search per modality)
                B, L, C = feat_prev.shape
                enc_rgb = feat_prev[:, 64:64+256, :]
                enc_tir = feat_prev[:, -256:, :]
                # Update tokens with prev frame (no_grad)
                _, _, tokens_updated = self.temporal_token_block(enc_rgb, enc_tir, prev_tokens)

            # Current frame forward with updated tokens (pass to backbone for DCMA)
            x_curr, aux_dict = self.backbone(z=template, x=search,
                                              ce_template_mask=ce_template_mask,
                                              ce_keep_rate=ce_keep_rate,
                                              return_last_attn=return_last_attn,
                                              temporal_tokens=tokens_updated,
                                              template_fg_masks=template_fg_masks,
                                              diagnostic=diagnostic,
                                              diagnostic_early_block=diagnostic_early_block,
                                              diagnostic_late_block=diagnostic_late_block,
                                              diagnostic_tbsi_index=diagnostic_tbsi_index,
                                              diagnostic_post_block=diagnostic_post_block)
            feat_curr = x_curr[-1] if isinstance(x_curr, list) else x_curr
            out = self.forward_head(feat_curr, None, temporal_tokens=tokens_updated,
                                    template_fg_masks=template_fg_masks,
                                    early_sparse_context=aux_dict.get('early_sparse_context'),
                                    prev_head_feature=prev_head_feature,
                                    current_crop_xywh=current_crop_xywh,
                                    history_crop_xywh=history_crop_xywh,
                                    current_frame_id=current_frame_id,
                                    history_frame_id=history_frame_id,
                                    current_sequence_id=current_sequence_id,
                                    history_sequence_id=history_sequence_id)
            out['temporal_tokens'] = tokens_updated
            out.update(aux_dict)
            out['backbone_feat'] = x_curr
            return out

        # Single-frame path
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    template_fg_masks=template_fg_masks,
                                    diagnostic=diagnostic,
                                    diagnostic_early_block=diagnostic_early_block,
                                    diagnostic_late_block=diagnostic_late_block,
                                    diagnostic_tbsi_index=diagnostic_tbsi_index,
                                    diagnostic_post_block=diagnostic_post_block)

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        # Explicit training/integration path: derive the previous raw head
        # token from the same frozen backbone, never from the TAM output.
        if (self.use_causal_tam and prev_head_feature is None and
                prev_search is not None):
            with torch.no_grad():
                x_prev, _ = self.backbone(
                    z=template, x=prev_search,
                    ce_template_mask=ce_template_mask,
                    ce_keep_rate=ce_keep_rate,
                    template_fg_masks=template_fg_masks)
                prev_last = x_prev[-1] if isinstance(x_prev, list) else x_prev
                prev_map = self.forward_fusion_only(prev_last)
                prev_head_feature = prev_map.flatten(2).transpose(1, 2).detach()
        out = self.forward_head(feat_last, None, temporal_tokens=prev_tokens,
                                template_fg_masks=template_fg_masks,
                                early_sparse_context=aux_dict.get('early_sparse_context'),
                                prev_head_feature=prev_head_feature,
                                current_crop_xywh=current_crop_xywh,
                                history_crop_xywh=history_crop_xywh,
                                current_frame_id=current_frame_id,
                                history_frame_id=history_frame_id,
                                current_sequence_id=current_sequence_id,
                                history_sequence_id=history_sequence_id)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_fusion_only(self, cat_feature, template_fg_masks=None,
                            early_sparse_context=None):
        """Extract search RGB/TIR + DA fusion. Returns (B, C, H, W) fused features.
        [Phase 1 Fix] Removed .detach() on base_fused so gradients flow through."""
        B = cat_feature.shape[0]
        C = cat_feature.shape[-1]
        num_search_token = 256
        num_template_token = 64
        expected_tokens = 2 * (num_template_token + num_search_token)
        if cat_feature.ndim != 3 or cat_feature.shape[1] != expected_tokens:
            raise ValueError(
                'post-fusion token layout expects [B, 640, C], got {}'.format(
                    tuple(cat_feature.shape)))
        if self.feat_sz_s ** 2 != num_search_token:
            raise ValueError(
                'post-fusion search token count {} does not match head grid {}'.format(
                    num_search_token, self.feat_sz_s))
        enc_rgb = cat_feature[:, 64:64 + num_search_token, :]
        enc_tir = cat_feature[:, -num_search_token:, :]
        if self.pvl_neck is not None:
            enc_rgb, enc_tir = self.pvl_neck(enc_rgb, enc_tir)
        signal_gate = getattr(self, 'single_signal_gate', None)
        if signal_gate is not None:
            enc_rgb, enc_tir = signal_gate(cat_feature, enc_rgb, enc_tir)

        enc_opt = torch.cat([enc_rgb, enc_tir], dim=2)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        _, _, C_, _ = opt.size()
        opt_feat = opt.view(-1, C_, self.feat_sz_s, self.feat_sz_s)

        amcm_stats = None
        amcm_post_gate = None
        if self.amcm_reliability is not None:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            template_rgb = cat_feature[:, :num_template_token, :]
            template_tir = cat_feature[:, 320:320 + num_template_token, :]
            feat_rgb, feat_tir, amcm_stats = self.amcm_reliability(
                feat_rgb, feat_tir, template_rgb, template_tir,
                template_fg_masks=template_fg_masks)
            self._last_amcm_stats = amcm_stats
            self._last_amcm_repr = {
                'rgb': feat_rgb, 'tir': feat_tir,
                'template_rgb': template_rgb, 'template_tir': template_tir,
            }
            if self.amcm_reliability_mode == 'directed_residual':
                # Residual message path: corrected modality tokens are the
                # actual inputs to TBSI fusion. This follows the same
                # identity-preserving message-passing contract as the native
                # TBSI cross-modal residual blocks.
                opt_feat = torch.cat((feat_rgb, feat_tir), dim=1)
                amcm_stats['injection_path'] = 'prefusion_corrected_modalities'
            elif self.amcm_reliability_position == 'pre_fusion':
                opt_feat = torch.cat((feat_rgb, feat_tir), dim=1)
                amcm_stats['injection_path'] = 'prefusion_corrected_modalities'
            else:
                amcm_post_gate = 0.5 * (
                    amcm_stats['rgb_gate_map_live'] + amcm_stats['tir_gate_map_live'])
                amcm_stats['injection_path'] = 'postfusion_average_gate'

        if self.use_degradation_aware:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            base_fused = self.tbsi_fuse_search(opt_feat)

            if getattr(self, 'da_fusion_mode', 'residual') == 'gate':
                gate = self.da_fusion.forward_gate_only(feat_rgb, feat_tir)
                return base_fused * (1.0 + gate)
            else:
                da_fused = self.da_fusion(feat_rgb, feat_tir)
                scale = getattr(self, 'da_fusion_scale', 0.5)
                fused = base_fused + scale * da_fused  # [Fix] no .detach()
                return fused
        else:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            bridge_mode = os.environ.get(
                'TBSI_SPATIAL_TOKEN_BRIDGE',
                'target' if self.spatial_token_bridge is not None else 'off').lower()
            if self.spatial_token_bridge is not None and bridge_mode != 'off':
                if template_fg_masks is None:
                    raise RuntimeError('SpatialTokenBridge requires template foreground masks')
                template_rgb = cat_feature[:, :num_template_token, :]
                template_tir = cat_feature[:, 320:320 + num_template_token, :]
                enc_rgb, enc_tir, self._last_spatial_token_bridge_stats = self.spatial_token_bridge(
                    enc_rgb, enc_tir, template_rgb, template_tir,
                    template_fg_masks[0], template_fg_masks[1], mode=bridge_mode)
                feat_rgb = enc_rgb.transpose(1, 2).reshape(
                    B, C, self.feat_sz_s, self.feat_sz_s)
                feat_tir = enc_tir.transpose(1, 2).reshape(
                    B, C, self.feat_sz_s, self.feat_sz_s)
                opt_feat = torch.cat((feat_rgb, feat_tir), dim=1)
            else:
                self._last_spatial_token_bridge_stats = {'mode': 'off'}
                if bridge_mode not in ('off', ''):
                    raise ValueError('spatial bridge mode requested but module is disabled')
            fused = self.tbsi_fuse_search(opt_feat)
            sparse_mode = os.environ.get('TBSI_SPARSE_CROSS_MODE', 'off').lower()
            if sparse_mode not in ('off', 'neck_union'):
                raise ValueError('unsupported sparse cross mode: {}'.format(sparse_mode))
            if sparse_mode == 'neck_union':
                if early_sparse_context is None:
                    early_sparse_context = getattr(
                        self.backbone, '_last_early_sparse_context', None)
                if early_sparse_context is None:
                    raise RuntimeError('neck_union requires block02 context from the current backbone forward')
                if template_fg_masks is None:
                    raise RuntimeError('neck_union requires template foreground masks')
                rgb_t, tir_t, rgb_s, tir_s = early_sparse_context
                masks = []
                for template, search, fg in ((rgb_t, rgb_s, template_fg_masks[0]),
                                             (tir_t, tir_s, template_fg_masks[1])):
                    weight = fg.to(device=template.device, dtype=template.dtype).clamp(0, 1)
                    proto = (template * weight.unsqueeze(-1)).sum(1) / weight.sum(
                        1, keepdim=True).clamp_min(1e-6)
                    proto = F.normalize(proto.float(), dim=-1, eps=1e-6)
                    search_n = F.normalize(search.float(), dim=-1, eps=1e-6)
                    score = (search_n * proto.unsqueeze(1)).sum(-1)
                    k = min(max(1, int(os.environ.get('TBSI_SPARSE_TOPK', '8'))), score.shape[1])
                    idx = score.topk(k, dim=-1).indices
                    mask = torch.zeros_like(score).scatter(1, idx, 1.0)
                    mask = F.max_pool2d(mask.reshape(B, 1, self.feat_sz_s, self.feat_sz_s),
                                        kernel_size=3, stride=1, padding=1).reshape(B, -1)
                    masks.append(mask.clamp(0, 1))
                candidate = torch.maximum(masks[0], masks[1]).to(dtype=fused.dtype)
                if os.environ.get('TBSI_SPARSE_MASK_CONTROL', 'none').lower() == 'shifted':
                    shifted = candidate.reshape(B, 1, self.feat_sz_s, self.feat_sz_s)
                    shift = max(1, self.feat_sz_s // 2)
                    candidate = torch.roll(shifted, shifts=(shift, shift), dims=(2, 3)).reshape(B, -1)
                zeros = torch.zeros_like(feat_rgb)
                rgb_only = self.tbsi_fuse_search(torch.cat((feat_rgb, zeros), dim=1))
                tir_only = self.tbsi_fuse_search(torch.cat((zeros, feat_tir), dim=1))
                fallback = 0.5 * (rgb_only + tir_only)
                fused = fused * candidate.reshape(B, 1, self.feat_sz_s, self.feat_sz_s) + \
                        fallback * (1.0 - candidate.reshape(B, 1, self.feat_sz_s, self.feat_sz_s))
                with torch.no_grad():
                    self._last_sparse_stats = {
                        'mode': sparse_mode,
                        'candidate_fraction': float(candidate.mean().detach()),
                        'rgb_candidate_fraction': float(masks[0].mean().detach()),
                        'tir_candidate_fraction': float(masks[1].mean().detach()),
                    }
            else:
                self._last_sparse_stats = {'mode': 'off'}
            if amcm_post_gate is not None and self.amcm_reliability_mode != 'directed_residual':
                fused = fused * (1.0 + amcm_post_gate)
            return fused

    def forward_head(self, cat_feature, gt_score_map=None, temporal_tokens=None,
                     template_fg_masks=None,
                     early_sparse_context=None,
                     prev_head_feature=None, current_crop_xywh=None,
                     history_crop_xywh=None, current_frame_id=None,
                     history_frame_id=None, current_sequence_id=None,
                     history_sequence_id=None):
        B, L, C = cat_feature.shape
        num_search_token = 256

        # Extract per-modality search tokens
        enc_rgb = cat_feature[:, 64:64 + num_search_token, :]    # (B, 256, C)
        enc_tir = cat_feature[:, -num_search_token:, :]           # (B, 256, C)

        # Temporal Token: enhance RGB/TIR features before fusion
        if self.use_temporal_tokens:
            enc_rgb, enc_tir, tok_out = self.temporal_token_block(enc_rgb, enc_tir, temporal_tokens)
            # Reconstruct cat_feature with enhanced search tokens
            cat_feature = torch.cat([
                cat_feature[:, :64, :],                  # RGB template
                enc_rgb,
                cat_feature[:, 320:384, :],               # TIR template
                enc_tir,
            ], dim=1)

        # Fusion
        fused_feat = self.forward_fusion_only(cat_feature, template_fg_masks,
                                              early_sparse_context=early_sparse_context)

        local_signal = getattr(self, 'local_signal_post_fusion', None)
        if local_signal is not None:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            zeros = torch.zeros_like(feat_rgb)
            rgb_only = self.tbsi_fuse_search(torch.cat((feat_rgb, zeros), dim=1))
            tir_only = self.tbsi_fuse_search(torch.cat((zeros, feat_tir), dim=1))
            fused_feat = local_signal(fused_feat, feat_rgb, feat_tir,
                                      rgb_only, tir_only, cat_feature)

        adaptive_stats = None
        if self.adaptive_post_fusion is not None:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            fused_feat, adaptive_stats = self.adaptive_post_fusion(
                fused_feat, feat_rgb, feat_tir)

        precenter_stats = None
        if self.precenter_gated_residual is not None:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            target_context = None
            if getattr(self.precenter_gated_residual, 'target_conditioned', False):
                template_rgb = cat_feature[:, :64, :].mean(dim=1)
                template_tir = cat_feature[:, 320:384, :].mean(dim=1)
                target_context = 0.5 * (template_rgb + template_tir)
            fused_feat, precenter_stats = self.precenter_gated_residual(
                fused_feat, feat_rgb, feat_tir, target_context=target_context)

        target_rank_stats = None
        if self.target_rank_gate is not None:
            target_rank_stats = None
            fused_feat, target_rank_stats = self.target_rank_gate(
                fused_feat, cat_feature[:, :64, :], cat_feature[:, 320:384, :])

        # Optional B2 attribution path: retain the pre-complementary feature so
        # the tracker can return the corrected box while committing M1 state.
        pre_complementary_feat = None
        if os.environ.get('TBSI_RETURN_B2_BASELINE', '0') == '1':
            pre_complementary_feat = fused_feat

        screen_stats = None
        if self.msha_reliability_adapter is not None:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            fused_feat, screen_stats = self.msha_reliability_adapter(
                fused_feat, feat_rgb, feat_tir)
        elif self.target_dynamic_local_operator is not None:
            template_rgb = cat_feature[:, :64, :].mean(dim=1)
            template_tir = cat_feature[:, 320:384, :].mean(dim=1)
            target_context = 0.5 * (template_rgb + template_tir)
            fused_feat, screen_stats = self.target_dynamic_local_operator(
                fused_feat, target_context)
        elif self.target_complementary_adaptive_fusion is not None:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            template_rgb = cat_feature[:, :64, :].mean(dim=1)
            template_tir = cat_feature[:, 320:384, :].mean(dim=1)
            target_context = 0.5 * (template_rgb + template_tir)
            fused_feat, screen_stats = self.target_complementary_adaptive_fusion(
                fused_feat, feat_rgb, feat_tir, target_context)

        # TC3: Temporal-Conditioned Channel Calibration
        if self.use_temporal_tokens and temporal_tokens is not None:
            fused_feat = self.tc3(fused_feat, temporal_tokens)

        tam_stats = None
        raw_head_feature = fused_feat
        raw_head_tokens = fused_feat.flatten(2).transpose(1, 2)
        if self.use_causal_tam:
            fused_tokens = raw_head_tokens
            fused_tokens, tam_stats = self.causal_tam(
                fused_tokens, prev_head_feature, current_crop_xywh,
                history_crop_xywh, current_frame_id, history_frame_id,
                current_sequence_id, history_sequence_id)
            fused_feat = fused_tokens.transpose(1, 2).reshape_as(fused_feat)

        # Head
        opt_feat = fused_feat
        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            if (self.target_rank_response_coupling and target_rank_stats is not None and
                    self.target_rank_response_beta != 0.0):
                # Diagnostic-only coupling: expose the target score to the
                # actual CENTER peak selector while preserving bbox/size/offset.
                score = target_rank_stats['target_rank_score']
                score = score / score.detach().flatten(1).std(dim=1).clamp_min(1e-6).view(B, 1, 1, 1)
                score = torch.tanh(score)
                score_map_ctr = score_map_ctr + self.target_rank_response_beta * score
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(B, -1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            if (self.early_utility_router_enabled and
                    (self.training or self._early_utility_router_only or
                     os.environ.get('TBSI_RETURN_UTILITY_TEACHER', '0') == '1')):
                # Counterfactual teachers share the existing neck/head weights but
                # must not alter their BatchNorm running state or the train graph.
                saved_modes = [(module, module.training) for root in
                               (self.tbsi_fuse_search, self.box_head)
                               for module in root.modules()]
                self.tbsi_fuse_search.eval()
                self.box_head.eval()
                try:
                    with torch.no_grad():
                        feat_rgb = enc_rgb.transpose(1, 2).reshape(
                            B, C, self.feat_sz_s, self.feat_sz_s)
                        feat_tir = enc_tir.transpose(1, 2).reshape(
                            B, C, self.feat_sz_s, self.feat_sz_s)
                        zeros = torch.zeros_like(feat_rgb)
                        rgb_only = self.tbsi_fuse_search(torch.cat((feat_rgb, zeros), dim=1))
                        tir_only = self.tbsi_fuse_search(torch.cat((zeros, feat_tir), dim=1))
                        rgb_teacher = self.box_head(rgb_only, None)[0]
                        tir_teacher = self.box_head(tir_only, None)[0]
                finally:
                    for module, mode in saved_modes:
                        module.training = mode
                out['early_utility_teacher_score_maps'] = (rgb_teacher, tir_teacher)
            if target_rank_stats is not None:
                out['target_rank_score'] = target_rank_stats['target_rank_score']
                out['target_rank_gate'] = target_rank_stats['target_rank_gate']
                out['target_rank_alpha'] = target_rank_stats['target_rank_alpha']
            if pre_complementary_feat is not None and self.target_complementary_adaptive_fusion is not None:
                baseline_score, _, baseline_size, baseline_offset = self.box_head(
                    pre_complementary_feat, gt_score_map)
                out['baseline_score_map'] = baseline_score
                out['baseline_size_map'] = baseline_size
                out['baseline_offset_map'] = baseline_offset
            if self.use_causal_tam:
                out['tam_raw_head_feature'] = raw_head_tokens
                out['tam_raw_head_map'] = raw_head_feature
                out['tam_head_feature'] = fused_feat
                out['tam_stats'] = tam_stats
            if adaptive_stats is not None:
                out['adaptive_post_fusion_stats'] = adaptive_stats
            if precenter_stats is not None:
                out['precenter_gated_residual_stats'] = precenter_stats
            if self._last_amcm_stats is not None:
                out['amcm_gate_rgb'] = self._last_amcm_stats.get('rgb_gate_map_live')
                out['amcm_gate_tir'] = self._last_amcm_stats.get('tir_gate_map_live')
                out['amcm_sim_rgb'] = self._last_amcm_stats.get('rgb_sim_map_live')
                out['amcm_sim_tir'] = self._last_amcm_stats.get('tir_sim_map_live')
                out['amcm_prob_rgb'] = self._last_amcm_stats.get('rgb_prob_live')
                out['amcm_prob_tir'] = self._last_amcm_stats.get('tir_prob_live')
                # Keep live tensors for test-time attribution. These are only
                # consumed by the opt-in tracker trace and do not affect the
                # prediction path.
                for key in ('rgb_message_norm_live', 'tir_message_norm_live',
                            'rgb_residual_live', 'tir_residual_live',
                            'rgb_input_live', 'tir_input_live',
                            'rgb_output_live', 'tir_output_live'):
                    out['amcm_' + key.replace('_live', '')] = self._last_amcm_stats.get(key)
                residual_rgb = self._last_amcm_stats.get('rgb_residual_live')
                residual_tir = self._last_amcm_stats.get('tir_residual_live')
                if residual_rgb is not None and residual_tir is not None:
                    # Energy map is the AMCM-internal signal supervised by the
                    # optional residual ranking loss. Keeping it as a tensor
                    # preserves gradient flow to the AMCM parameters.
                    out['amcm_residual_energy'] = 0.5 * (
                        residual_rgb.abs().mean(dim=1, keepdim=True) +
                        residual_tir.abs().mean(dim=1, keepdim=True))
            if self._last_amcm_repr is not None:
                out['amcm_repr_rgb'] = self._last_amcm_repr['rgb']
                out['amcm_repr_tir'] = self._last_amcm_repr['tir']
                out['amcm_repr_template_rgb'] = self._last_amcm_repr['template_rgb']
                out['amcm_repr_template_tir'] = self._last_amcm_repr['template_tir']
            if screen_stats is not None:
                out['screen_adapter_stats'] = screen_stats
            if self.use_temporal_tokens:
                out['temporal_tokens'] = tok_out
            return out
        else:
            raise NotImplementedError


def build_tbsi_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    pvl_cfg = getattr(cfg.MODEL, 'PVL', None)
    if cfg.MODEL.PRETRAIN_FILE and ('TBSITrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_tbsi':
        da_in_layer = getattr(cfg.MODEL, "DA_IN_LAYER", False)
        use_checkpoint = getattr(cfg.TRAIN, "USE_CHECKPOINT", False)
        post_tbsi_cfg = getattr(cfg.MODEL.BACKBONE, 'POST_TBSI_RESIDUAL', None)
        smsa_enabled = getattr(cfg.MODEL.RIMA, 'ENABLE', False) and getattr(cfg.MODEL.RIMA, 'SMSA_ENABLE', False)
        backbone = vit_base_patch16_224_tbsi(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            tbsi_loc=cfg.MODEL.BACKBONE.TBSI_LOC,
                                            tbsi_drop_path=cfg.TRAIN.TBSI_DROP_PATH,
                                            da_in_layer=da_in_layer,
                                            use_smsa=smsa_enabled,
                                            use_smsa_rgb=getattr(cfg.MODEL.RIMA, 'SMSA_RGB_ENABLE', smsa_enabled),
                                            use_smsa_tir=getattr(cfg.MODEL.RIMA, 'SMSA_TIR_ENABLE', smsa_enabled),
                                            smsa_alpha=getattr(cfg.MODEL.RIMA, 'ENTMAX_ALPHA', 1.3),
        smsa_fp32=getattr(cfg.MODEL.RIMA, 'FP32_CORE', True),
                                            use_checkpoint=use_checkpoint,
                                            fusion_mode=getattr(cfg.MODEL.BACKBONE, 'TBSI_FUSION', 'legacy'),
                                            template_expert_mode=getattr(cfg.MODEL.BACKBONE, 'TEMPLATE_EXPERT_MODE', 'off'),
                                            template_expert_loc=list(getattr(cfg.MODEL.BACKBONE, 'TEMPLATE_EXPERT_LOC', [])),
                                            template_expert_count=getattr(cfg.MODEL.BACKBONE, 'TEMPLATE_EXPERT_COUNT', 4),
                                            template_expert_slots=getattr(cfg.MODEL.BACKBONE, 'TEMPLATE_EXPERT_SLOTS', 4),
                                            template_expert_hidden=getattr(cfg.MODEL.BACKBONE, 'TEMPLATE_EXPERT_HIDDEN', 4),
                                            cfvi_mode=getattr(cfg.MODEL.BACKBONE, 'CFVI_MODE', 'off'),
                                            cfvi_lambda_init=getattr(cfg.MODEL.BACKBONE, 'CFVI_LAMBDA_INIT', 0.001),
                                            cfvi_gate_hidden=getattr(cfg.MODEL.BACKBONE, 'CFVI_GATE_HIDDEN', 64),
                                            cfvi_loc=list(getattr(cfg.MODEL.BACKBONE, 'CFVI_LOC', [])),
                                            cfvi_band_mask=list(getattr(cfg.MODEL.BACKBONE, 'CFVI_BAND_MASK', [1.0] * 4)),
                                            cfvi_band_renorm=getattr(cfg.MODEL.BACKBONE, 'CFVI_BAND_RENORM', True),
                                            freq_exchange_mode=getattr(cfg.MODEL.BACKBONE, 'FREQ_EXCHANGE_MODE', 'off'),
                                            freq_exchange_loc=list(getattr(cfg.MODEL.BACKBONE, 'FREQ_EXCHANGE_LOC', [])),
                                            freq_exchange_diagnostics_path=getattr(
                                                cfg.MODEL.BACKBONE,
                                                'FREQ_EXCHANGE_DIAGNOSTICS_PATH',
                                                ''),
                                            message_freq_refine_mode=getattr(
                                                cfg.MODEL.BACKBONE,
                                                'MESSAGE_FREQ_REFINE_MODE', 'off'),
                                            message_freq_refine_loc=list(getattr(
                                                cfg.MODEL.BACKBONE,
                                                'MESSAGE_FREQ_REFINE_LOC', [])),
                                            freq_relation_mode=getattr(
                                                cfg.MODEL.BACKBONE, 'FREQ_RELATION_MODE', 'off'),
                                            freq_relation_loc=list(getattr(
                                                cfg.MODEL.BACKBONE, 'FREQ_RELATION_LOC', [])),
                                            freq_relation_hidden=getattr(
                                                cfg.MODEL.BACKBONE, 'FREQ_RELATION_HIDDEN', 64),
                                            freq_relation_max_bias=getattr(
                                                cfg.MODEL.BACKBONE, 'FREQ_RELATION_MAX_BIAS', 0.5),
                                            freq_relation_calibration=getattr(
                                                cfg.MODEL.BACKBONE, 'FREQ_RELATION_CALIBRATION', 'off'),
                                            freq_relation_topk=getattr(
                                                cfg.MODEL.BACKBONE, 'FREQ_RELATION_TOPK', 8),
                                            freq_relation_clip=getattr(
                                                cfg.MODEL.BACKBONE, 'FREQ_RELATION_CLIP', 3.0),
                                            freq_relation_eps=getattr(
                                                cfg.MODEL.BACKBONE, 'FREQ_RELATION_CALIBRATION_EPS', 1e-6),
                                            early_search_residual_enable=getattr(
                                                cfg.MODEL.BACKBONE.EARLY_SEARCH_RESIDUAL, 'ENABLE', False),
                                            early_search_residual_early_block=getattr(
                                                cfg.MODEL.BACKBONE.EARLY_SEARCH_RESIDUAL, 'EARLY_BLOCK', 2),
                                            early_search_residual_late_block=getattr(
                                                cfg.MODEL.BACKBONE.EARLY_SEARCH_RESIDUAL, 'LATE_BLOCK', 9),
                                            early_search_residual_force_zero=getattr(
                                                cfg.MODEL.BACKBONE.EARLY_SEARCH_RESIDUAL, 'FORCE_ZERO', False),
                                            early_search_residual_use_scsu=getattr(
                                                cfg.MODEL.BACKBONE.EARLY_SEARCH_RESIDUAL, 'USE_SCSU', False),
                                            early_search_residual_scsu_k1=getattr(
                                                cfg.MODEL.BACKBONE.EARLY_SEARCH_RESIDUAL, 'SCSU_K1', 3),
                                            early_search_residual_scsu_k2=getattr(
                                                cfg.MODEL.BACKBONE.EARLY_SEARCH_RESIDUAL, 'SCSU_K2', 5),
                                            post_tbsi_residual_enable=getattr(
                                                post_tbsi_cfg, 'ENABLE', False),
                                            post_tbsi_residual_mode=getattr(
                                                post_tbsi_cfg, 'MODE', 'spatial'),
                                            post_tbsi_residual_block=getattr(
                                                post_tbsi_cfg, 'BLOCK', 5),
                                            post_tbsi_residual_hidden_dim=getattr(
                                                post_tbsi_cfg, 'HIDDEN_DIM', 192),
                                            post_tbsi_residual_heads=getattr(
                                                post_tbsi_cfg, 'HEADS', 4),
                                            post_tbsi_residual_force_zero=getattr(
                                                post_tbsi_cfg, 'FORCE_ZERO', False),
                                            pvl_enable=bool(pvl_cfg is not None and
                                                            getattr(pvl_cfg, 'ENABLE', False)),
                                            pvl_position=getattr(pvl_cfg, 'POSITION', 'template_loc6'),
                                            pvl_loc=getattr(pvl_cfg, 'LOC', 6),
                                            pvl_adapter_dim=getattr(pvl_cfg, 'ADAPTER_DIM', 64),
                                            pvl_beta=getattr(pvl_cfg, 'BETA', 2.35),
                                            early_conditioned_cross_gate=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_CONDITIONED_CROSS_GATE', False),
                                            early_gate_hidden=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_GATE_HIDDEN', 32),
                                            early_target_margin_gate=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_TARGET_MARGIN_GATE', False),
                                            early_margin_hidden=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_MARGIN_HIDDEN', 16),
                                            early_margin_hard_negative_k=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_MARGIN_HARD_NEGATIVE_K', 4),
                                            early_direct_margin_gate=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_DIRECT_MARGIN_GATE', False),
                                            early_direct_margin_rho=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_DIRECT_MARGIN_RHO', 0.0),
                                            early_direct_margin_temperature=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_DIRECT_MARGIN_TEMPERATURE', 0.25),
                                            early_direct_margin_trainable=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_DIRECT_MARGIN_TRAINABLE', False),
                                            early_template_value_reconstruction=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_TEMPLATE_VALUE_RECONSTRUCTION', False),
                                            early_template_value_rank=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_TEMPLATE_VALUE_RANK', 8),
                                            early_template_value_max_delta=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_TEMPLATE_VALUE_MAX_DELTA', 0.10),
                                            discriminative_evidence_router=getattr(
                                                cfg.MODEL.BACKBONE, 'DISCRIMINATIVE_EVIDENCE_ROUTER', False),
                                            der_score_dim=getattr(
                                                cfg.MODEL.BACKBONE, 'DER_SCORE_DIM', 64),
                                            der_hidden_dim=getattr(
                                                cfg.MODEL.BACKBONE, 'DER_HIDDEN_DIM', 8),
                                            msha_role_pair_bias_enable=getattr(
                                                cfg.MODEL.BACKBONE, 'MSHA_ROLE_PAIR_BIAS_ENABLE', False),
                                            msha_role_pair_bias_loc=getattr(
                                                cfg.MODEL.BACKBONE, 'MSHA_ROLE_PAIR_BIAS_LOC', 9),
                                            early_search_relation_bias_enable=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_SEARCH_RELATION_BIAS_ENABLE', False),
                                            early_search_relation_bias_loc=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_SEARCH_RELATION_BIAS_LOC', 9),
                                            early_search_relation_dim=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_SEARCH_RELATION_DIM', 32),
                                            early_search_relation_temperature=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_SEARCH_RELATION_TEMPERATURE', 0.07),
                                            early_search_relation_max_bias=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_SEARCH_RELATION_MAX_BIAS', 0.5),
                                            search_conditioned_t2s_mode=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_MODE', 'off'),
                                            search_conditioned_t2s_loc=list(getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_LOC', [])),
                                            search_conditioned_t2s_rank=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_RANK', 64),
                                            search_conditioned_t2s_max_delta=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_MAX_DELTA', 0.10),
                                            search_conditioned_t2s_temperature=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_TEMPERATURE', 0.07),
                                            search_conditioned_t2s_competition_topk=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_COMPETITION_TOPK', 4),
                                            search_conditioned_t2s_router_temperature=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_ROUTER_TEMPERATURE', None),
                                            search_conditioned_t2s_content_strength=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_CONTENT_STRENGTH', 0.0),
                                            search_conditioned_t2s_content_match_temperature=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_CONTENT_MATCH_TEMPERATURE', 0.2),
                                            search_conditioned_t2s_content_prior_floor=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_CONTENT_PRIOR_FLOOR', 0.05),
                                            search_conditioned_t2s_content_search_factor=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_CONTENT_SEARCH_FACTOR', 4.0),
                                            search_conditioned_t2s_content_neighborhood=getattr(
                                                cfg.MODEL.BACKBONE, 'SEARCH_CONDITIONED_T2S_CONTENT_NEIGHBORHOOD', 'object'),
                                            local_cross_modal_correspondence_enable=getattr(
                                                cfg.MODEL.BACKBONE, 'LOCAL_CROSS_MODAL_CORRESPONDENCE_ENABLE', False),
                                            local_cross_modal_correspondence_block=getattr(
                                                cfg.MODEL.BACKBONE, 'LOCAL_CROSS_MODAL_CORRESPONDENCE_BLOCK', 2),
                                            local_cross_modal_correspondence_hidden_dim=getattr(
                                                cfg.MODEL.BACKBONE, 'LOCAL_CROSS_MODAL_CORRESPONDENCE_HIDDEN_DIM', 96),
                                            local_cross_modal_correspondence_relevance_dim=getattr(
                                                cfg.MODEL.BACKBONE, 'LOCAL_CROSS_MODAL_CORRESPONDENCE_RELEVANCE_DIM', 64),
                                            local_cross_modal_correspondence_max_delta=getattr(
                                                cfg.MODEL.BACKBONE, 'LOCAL_CROSS_MODAL_CORRESPONDENCE_MAX_DELTA', 0.05),
                                            early_utility_router_enable=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_UTILITY_ROUTER_ENABLE', False),
                                            early_utility_descriptor_dim=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_UTILITY_DESCRIPTOR_DIM', 32),
                                            early_utility_hidden_dim=getattr(
                                                cfg.MODEL.BACKBONE, 'EARLY_UTILITY_HIDDEN_DIM', 64),
                                            bridge_token_pruner_enable=getattr(cfg.MODEL.BRIDGE_TOKEN_PRUNER, 'ENABLE', False),
                                            bridge_token_pruner_layer_ids=list(getattr(cfg.MODEL.BRIDGE_TOKEN_PRUNER, 'LAYER_IDS', [])),
                                            bridge_token_pruner_cfg={
                                                'dim': 768,
                                                'template_keep': getattr(cfg.MODEL.BRIDGE_TOKEN_PRUNER, 'TEMPLATE_KEEP', 0.75),
                                                'search_keep': getattr(cfg.MODEL.BRIDGE_TOKEN_PRUNER, 'SEARCH_KEEP', 0.75),
                                                'hidden_dim': getattr(cfg.MODEL.BRIDGE_TOKEN_PRUNER, 'HIDDEN_DIM', 192),
                                                'temperature': getattr(cfg.MODEL.BRIDGE_TOKEN_PRUNER, 'TEMPERATURE', 0.15),
                                                'hard_eval': getattr(cfg.MODEL.BRIDGE_TOKEN_PRUNER, 'HARD_EVAL', False),
                                            })
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    box_head = build_box_head(cfg, hidden_dim)

    use_temporal = getattr(cfg.MODEL, "TEMPORAL_TOKENS", False)
    use_da = getattr(cfg.MODEL, "DEGRADATION_AWARE", False)
    da_mode = getattr(cfg.MODEL, "DA_MODE", "spatial")
    use_madc = getattr(cfg.MODEL, "DA_MADC", False)
    use_csr = getattr(cfg.MODEL, "DA_CSR", False)
    causal_tam_cfg = getattr(cfg.MODEL, 'CAUSAL_TAM', None)
    adaptive_post_fusion_cfg = getattr(cfg.MODEL, 'ADAPTIVE_POST_FUSION', None)
    precenter_gated_residual_cfg = getattr(cfg.MODEL, 'PRE_CENTER_GATED_RESIDUAL', None)
    amcm_reliability_cfg = getattr(cfg.MODEL, 'AMCM_RELIABILITY', None)
    target_rank_gate_cfg = getattr(cfg.MODEL, 'TARGET_RANK_GATE', None)
    spatial_token_bridge_cfg = getattr(cfg.MODEL, 'SPATIAL_TOKEN_BRIDGE', None)
    num_temporal = getattr(cfg.MODEL, "NUM_TEMPORAL_TOKENS", 4)
    screen_cfg = edict()
    screen_cfg.MSHA_RELIABILITY_ADAPTER = getattr(cfg.MODEL, 'MSHA_RELIABILITY_ADAPTER', None)
    screen_cfg.TARGET_DYNAMIC_LOCAL_OPERATOR = getattr(cfg.MODEL, 'TARGET_DYNAMIC_LOCAL_OPERATOR', None)
    screen_cfg.TARGET_COMPLEMENTARY_ADAPTIVE_FUSION = getattr(
        cfg.MODEL, 'TARGET_COMPLEMENTARY_ADAPTIVE_FUSION', None)
    backbone._screen_cfg = screen_cfg
    if (screen_cfg.MSHA_RELIABILITY_ADAPTER is not None and
            getattr(screen_cfg.MSHA_RELIABILITY_ADAPTER, 'ENABLE', False) and
            screen_cfg.TARGET_DYNAMIC_LOCAL_OPERATOR is not None and
            getattr(screen_cfg.TARGET_DYNAMIC_LOCAL_OPERATOR, 'ENABLE', False)):
        raise ValueError('MSHA reliability and target dynamic adapters are mutually exclusive')
    enabled_screen = [getattr(screen_cfg, name, None) for name in (
        'MSHA_RELIABILITY_ADAPTER', 'TARGET_DYNAMIC_LOCAL_OPERATOR',
        'TARGET_COMPLEMENTARY_ADAPTIVE_FUSION')]
    if sum(bool(x is not None and getattr(x, 'ENABLE', False)) for x in enabled_screen) > 1:
        raise ValueError('Screen fusion adapters are mutually exclusive')
    if use_da:
        da_str = f'Building model with Quality-Aware Fusion ({da_mode}'
        if use_madc: da_str += '+MADC'
        if use_csr: da_str += '+CSR'
        print(da_str + ')')

    model = TBSITrack(
        backbone, box_head,
        aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE,
        use_temporal_tokens=use_temporal,
        num_temporal_tokens=num_temporal,
        use_degradation_aware=use_da,
        da_mode=da_mode,
        use_madc=use_madc,
        use_csr=use_csr,
        causal_tam_cfg=causal_tam_cfg,
        adaptive_post_fusion_cfg=adaptive_post_fusion_cfg,
        precenter_gated_residual_cfg=precenter_gated_residual_cfg,
        amcm_reliability_cfg=amcm_reliability_cfg,
        pvl_cfg=pvl_cfg,
        target_rank_gate_cfg=target_rank_gate_cfg,
        spatial_token_bridge_cfg=spatial_token_bridge_cfg,
    )

    init_checkpoint = getattr(cfg.MODEL, 'INIT_CHECKPOINT', '')
    if init_checkpoint and training:
        if getattr(cfg.MODEL, 'INIT_CHECKPOINT_SOT_PROVENANCE', False) and (
                cfg.MODEL.PRETRAIN_FILE != 'TBSITrack_SOT_Pretrained.pth.tar'):
            raise ValueError('paired initialized state must originate from SOT pretraining')
        init_path = init_checkpoint if os.path.isabs(init_checkpoint) else os.path.join(current_dir, '../../../', init_checkpoint)
        if not os.path.exists(init_path):
            raise FileNotFoundError('INIT_CHECKPOINT not found: {}'.format(init_path))
        checkpoint = torch.load(init_path, map_location='cpu')
        if getattr(cfg.MODEL, 'INIT_CHECKPOINT_SOT_PROVENANCE', False):
            if not isinstance(checkpoint, dict) or checkpoint.get('origin') != 'TBSITrack_SOT_Pretrained.pth.tar':
                raise ValueError('paired initial state must declare SOT pretraining provenance')
            if checkpoint.get('seed') != 42 or checkpoint.get('kind') != 'common_initialized_weights_no_optimizer':
                raise ValueError('paired initial state has incompatible seed or checkpoint kind')
            if cfg.MODEL.BACKBONE.SEARCH_CONDITIONED_T2S_MODE != 'stser_competition_content':
                raise ValueError('paired initial state requires the content host architecture')
            from pathlib import Path
            import json
            import hashlib
            manifest_path = Path(init_path).with_name('initialization_manifest.json')
            if not manifest_path.is_file():
                raise FileNotFoundError('paired initialization manifest missing: {}'.format(manifest_path))
            with manifest_path.open() as handle:
                manifest = json.load(handle)
            hasher = hashlib.sha256()
            with open(init_path, 'rb') as handle:
                for block in iter(lambda: handle.read(1 << 20), b''):
                    hasher.update(block)
            if manifest['shared_initial_sha256'] != hasher.hexdigest():
                raise ValueError('paired initial checkpoint hash differs from manifest')
            if checkpoint.get('scope') != manifest.get('scope'):
                raise ValueError('paired initial checkpoint scope differs from manifest')
            run_kind = 'quick3' if cfg.TRAIN.EPOCH == 3 else 'full15' if cfg.TRAIN.EPOCH == 15 else None
            if run_kind != manifest.get('scope'):
                raise ValueError('paired initial checkpoint scope does not match training schedule')
            # The wrapper checks the run config against this manifest.
        state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(
            state, strict=bool(getattr(cfg.MODEL, 'INIT_CHECKPOINT_STRICT', False)))
        print('Load initialization checkpoint from: {}'.format(init_path))
        print('  Missing keys: {}'.format(list(missing)))
        print('  Unexpected keys: {}'.format(list(unexpected)))

    if getattr(cfg.TRAIN, 'EARLY_UTILITY_ROUTER_ONLY', False) and training:
        router = getattr(model.backbone, 'early_utility_router', None)
        if router is None:
            raise ValueError('EARLY_UTILITY_ROUTER_ONLY requires the enabled utility router')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('backbone.early_utility_router.')
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError('EARLY_UTILITY_ROUTER_ONLY found no trainable router parameters')
        model._early_utility_router_only = True
        model.eval()
        router.train(True)
        print('Early utility router-only training; M1 weights and normalization state are frozen.')

    # Stage 2: load baseline checkpoint + freeze all except post_fusion_block
    stage2_baseline = getattr(cfg.MODEL, "STAGE2_BASELINE", "")
    if stage2_baseline and training:
        baseline_path = os.path.join(current_dir, '../../../', stage2_baseline)
        if os.path.exists(baseline_path):
            checkpoint = torch.load(baseline_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(checkpoint["net"], strict=False)
            print(f'Stage 2: Loaded baseline from {stage2_baseline}')
            print(f'  Missing keys (expected): {[k for k in missing if "da_fusion" in k]}')
            print(f'  Unexpected keys: {len(unexpected)}')
        else:
            print(f'WARNING: Stage 2 baseline not found: {baseline_path}')

        # Freeze everything EXCEPT post_fusion_block, da_fusion, box_head, and temporal_token
        for n, p in model.named_parameters():
            if any(k in n for k in ["post_fusion_block", "da_fusion", "box_head", "temporal_token"]):
                p.requires_grad = True
            else:
                p.requires_grad = False
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        print(f'Stage 2: Frozen {total_count - trainable_count:,}/{total_count:,} params. '
              f'Trainable: {trainable_count:,} ({100*trainable_count/total_count:.2f}%)')
    elif not init_checkpoint and 'TBSITrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    if getattr(cfg.TRAIN, 'CPLAIN_ONLY', False) and training:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ('early_search_residual' in name or
                                       'post_tbsi_residual' in name)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('TRAIN.CPLAIN_ONLY requires an enabled residual adapter')
        print('Cplain-only training; M1 frozen. Trainable parameters:')
        for name in trainable:
            print(name)
        model.set_cplain_only(True)

    if getattr(cfg.TRAIN, 'ADAPTIVE_POST_ROUTER_ONLY', False) and training:
        if model.adaptive_post_fusion is None:
            raise ValueError('TRAIN.ADAPTIVE_POST_ROUTER_ONLY requires MODEL.ADAPTIVE_POST_FUSION.ENABLE')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('adaptive_post_fusion.')
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('TRAIN.ADAPTIVE_POST_ROUTER_ONLY found no router parameters')
        print('Adaptive post-fusion router-only training; frozen M1. Trainable parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'PRE_CENTER_GATED_RESIDUAL_ONLY', False) and training:
        if model.precenter_gated_residual is None:
            raise ValueError('TRAIN.PRE_CENTER_GATED_RESIDUAL_ONLY requires the module enabled')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('precenter_gated_residual.')
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('TRAIN.PRE_CENTER_GATED_RESIDUAL_ONLY found no trainable parameters')
        print('Pre-CENTER gated residual-only training; frozen M1 parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'AMCM_RELIABILITY_ONLY', False) and training:
        if model.amcm_reliability is None:
            raise ValueError('TRAIN.AMCM_RELIABILITY_ONLY requires MODEL.AMCM_RELIABILITY.ENABLE')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('amcm_reliability.')
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('TRAIN.AMCM_RELIABILITY_ONLY found no trainable parameters')
        print('AMCM-lite reliability-only training; frozen M1 parameters:')
        for name in trainable:
            print(name)
        model._amcm_reliability_only = True
        # Keep frozen BatchNorm statistics identical to the M1 checkpoint.
        # ``model.train()`` is called by the trainer, so explicitly restore
        # eval mode for the frozen parent and re-enable training only for AMCM.
        model.eval()
        model.amcm_reliability.train()

    if getattr(cfg.TRAIN, 'TARGET_RANK_GATE_ONLY', False) and training:
        if model.target_rank_gate is None:
            raise ValueError('TRAIN.TARGET_RANK_GATE_ONLY requires MODEL.TARGET_RANK_GATE.ENABLE')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('target_rank_gate.')
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('TARGET_RANK_GATE_ONLY found no trainable parameters')
        model.eval()
        model.target_rank_gate.train()
        print('TargetRankGate-only training; frozen M1 parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'AMCM_RELIABILITY_JOINT', False) and training:
        if model.amcm_reliability is None:
            raise ValueError('TRAIN.AMCM_RELIABILITY_JOINT requires MODEL.AMCM_RELIABILITY.ENABLE')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (name.startswith('amcm_reliability.') or
                                       name.startswith('tbsi_fuse_search.'))
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('TRAIN.AMCM_RELIABILITY_JOINT found no trainable parameters')
        model._amcm_reliability_joint = True
        model.eval()
        model.amcm_reliability.train()
        model.tbsi_fuse_search.train()
        print('AMCM-lite + tbsi_fuse_search joint training; frozen parent parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'AMCM_RELIABILITY_JOINT_HEAD', False) and training:
        if model.amcm_reliability is None:
            raise ValueError('TRAIN.AMCM_RELIABILITY_JOINT_HEAD requires MODEL.AMCM_RELIABILITY.ENABLE')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (name.startswith('amcm_reliability.') or
                                       name.startswith('tbsi_fuse_search.') or
                                       name.startswith('box_head.'))
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('TRAIN.AMCM_RELIABILITY_JOINT_HEAD found no trainable parameters')
        model._amcm_reliability_joint_head = True
        model.eval()
        model.amcm_reliability.train()
        model.tbsi_fuse_search.train()
        model.box_head.train()
        print('AMCM-lite + fusion + CENTER head joint training; frozen parent parameters:')
        for name in trainable:
            print(name)

    screen_only_flags = [
        bool(getattr(cfg.TRAIN, 'MSHA_RELIABILITY_ONLY', False)),
        bool(getattr(cfg.TRAIN, 'TARGET_DYNAMIC_LOCAL_ONLY', False)),
        bool(getattr(cfg.TRAIN, 'TARGET_COMPLEMENTARY_ADAPTIVE_ONLY', False)),
    ]
    if sum(screen_only_flags) > 1:
        raise ValueError('Only one screen adapter may be trained at a time')
    if any(screen_only_flags) and training:
        if screen_only_flags[0]:
            if model.msha_reliability_adapter is None:
                raise ValueError('TRAIN.MSHA_RELIABILITY_ONLY requires the adapter enabled')
            prefix = 'msha_reliability_adapter.'
        elif screen_only_flags[1]:
            if model.target_dynamic_local_operator is None:
                raise ValueError('TRAIN.TARGET_DYNAMIC_LOCAL_ONLY requires the operator enabled')
            prefix = 'target_dynamic_local_operator.'
        else:
            if model.target_complementary_adaptive_fusion is None:
                raise ValueError('TRAIN.TARGET_COMPLEMENTARY_ADAPTIVE_ONLY requires the module enabled')
            prefix = 'target_complementary_adaptive_fusion.'
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(prefix)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('Screen adapter-only training found no trainable parameters')
        print('Screen adapter-only training; frozen M1 parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'EARLY_CONDITIONED_MSHA_GATE_ONLY', False) and training:
        prefix = 'backbone.tbsi_layers.'
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith(prefix) and
                '.v2_template_fusion.early_gate.' in name or
                name.startswith(prefix) and
                '.v2_template_fusion.early_gate_scale' in name)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('EARLY_CONDITIONED_MSHA_GATE_ONLY found no trainable gate parameters')
        model.eval()
        for layer in model.backbone.tbsi_layers:
            if getattr(layer, 'v2_template_fusion', None) is not None:
                layer.v2_template_fusion.train()
        print('Early-conditioned MSHA gate-only training; frozen M1 parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'MSHA_ROLE_PAIR_BIAS_ONLY', False) and training:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.endswith('.role_pair_bias_gain')
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('MSHA_ROLE_PAIR_BIAS_ONLY found no trainable role-pair parameters')
        expected = sum(
            layer.v2_template_fusion.num_heads
            for layer in model.backbone.tbsi_layers
            if getattr(layer, 'v2_template_fusion', None) is not None and
            getattr(layer.v2_template_fusion, 'role_pair_bias', False))
        if len(trainable) != 1 or sum(dict(model.named_parameters())[name].numel()
                                      for name in trainable) != expected:
            raise ValueError('MSHA role-pair training parameter contract mismatch')
        model.eval()
        print('MSHA role-pair bias-only training; frozen M1 parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'EARLY_SEARCH_RELATION_BIAS_ONLY', False) and training:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = '.v2_template_fusion.early_search_relation_' in name
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('EARLY_SEARCH_RELATION_BIAS_ONLY found no relation-bias parameters')
        model.eval()
        for layer in model.backbone.tbsi_layers:
            if getattr(layer, 'v2_template_fusion', None) is not None:
                layer.v2_template_fusion.train()
        print('Early search relation bias-only training; frozen parent parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'EARLY_GATE_MARGIN_ONLY', False) and training:
        prefix = 'backbone.tbsi_layers.'
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith(prefix) and
                '.v2_template_fusion.early_margin_gate.' in name)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('EARLY_GATE_MARGIN_ONLY found no trainable margin-gate parameters')
        model.eval()
        for layer in model.backbone.tbsi_layers:
            if getattr(layer, 'v2_template_fusion', None) is not None:
                layer.v2_template_fusion.train()
        print('Early target-margin gate-only training; frozen parent and early gate parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'EARLY_TEMPLATE_VALUE_RECONSTRUCTION_ONLY', False) and training:
        prefix = 'backbone.tbsi_layers.'
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith(prefix) and
                '.v2_template_fusion.early_value_reconstruction.' in name)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('EARLY_TEMPLATE_VALUE_RECONSTRUCTION_ONLY found no reconstruction parameters')
        model.eval()
        for layer in model.backbone.tbsi_layers:
            if getattr(layer, 'v2_template_fusion', None) is not None:
                layer.v2_template_fusion.train()
        print('Early template Value reconstruction-only training; frozen parent and gate parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'QUERY_CONDITIONED_CROSS_VALUE_ONLY', False) and training:
        prefix = 'backbone.tbsi_layers.'
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith(prefix) and
                '.v2_template_fusion.query_conditioned_' in name)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('QUERY_CONDITIONED_CROSS_VALUE_ONLY found no query-conditioned parameters')
        model.eval()
        for layer in model.backbone.tbsi_layers:
            if getattr(layer, 'v2_template_fusion', None) is not None:
                layer.v2_template_fusion.train()
        print('Query-conditioned cross-Value-only training; frozen M1 parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'SPATIAL_TOKEN_BRIDGE_ONLY', False) and training:
        if model.spatial_token_bridge is None:
            raise ValueError('TRAIN.SPATIAL_TOKEN_BRIDGE_ONLY requires MODEL.SPATIAL_TOKEN_BRIDGE.ENABLE')
        prefix = 'spatial_token_bridge.'
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(prefix)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('SPATIAL_TOKEN_BRIDGE_ONLY found no trainable bridge parameters')
        model.eval()
        model.spatial_token_bridge.train(True)
        model._spatial_token_bridge_only = True
        print('Spatial token bridge-only training; frozen M1 parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'DER_ONLY', False) and training:
        prefix = 'backbone.tbsi_layers.'
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (name.startswith(prefix) and
                                       '.v2_template_fusion.der.' in name)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('TRAIN.DER_ONLY requires enabled DER modules')
        model.eval()
        print('Discriminative Evidence Router-only training; frozen parent parameters:')
        for name in trainable:
            print(name)

    if getattr(cfg.TRAIN, 'CAUSAL_TAM_ONLY', False) and training:
        if not getattr(model, 'use_causal_tam', False):
            raise ValueError('TRAIN.CAUSAL_TAM_ONLY requires MODEL.CAUSAL_TAM.ENABLE')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ('causal_tam' in name)
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError('TRAIN.CAUSAL_TAM_ONLY found no trainable TAM parameters')
        model.set_cplain_only(True)
        print('Causal-TAM-only training enabled; M1 frozen.')

    # === Token warm-start: 从backbone cls_token 初始化 temporal tokens ===
    if use_temporal and training and hasattr(model.backbone, 'cls_token'):
        cls_w = model.backbone.cls_token.data  # (1, 1, D)
        tl = model.temporal_token_block.temporal_layer
        if hasattr(tl, 'rgb_tokens') and hasattr(tl, 'tir_tokens'):
            with torch.no_grad():
                # 用cls_token的均值初始化token, 保留标准差缩放
                tl.rgb_tokens.data.copy_(cls_w.expand(-1, tl.num_tokens, -1) + 0.02 * torch.randn_like(tl.rgb_tokens))
                tl.tir_tokens.data.copy_(cls_w.expand(-1, tl.num_tokens, -1) + 0.02 * torch.randn_like(tl.tir_tokens))
            print(f'Token warm-start: initialized from backbone.cls_token (+N(0,0.02))')

    # === QAF warm-start: 从 tbsi_fuse_search 初始化 quality_conv 第一层 ===
    if use_da and training and hasattr(model, 'da_fusion') and hasattr(model, 'tbsi_fuse_search'):
        da = model.da_fusion
        fs = model.tbsi_fuse_search  # conv(2C, C, 3x3) = Conv2d+BN+ReLU
        if hasattr(da, 'quality_conv') and hasattr(fs, '0'):
            with torch.no_grad():
                # fs[0] = Conv2d(2C, C, 3x3), quality_conv[0] = Conv2d(2C, rdim, 1x1)
                # 用fs[0].weight的通道维度均值初始化quality_conv[0]
                fs_w = fs[0].weight.data  # (C, 2C, 3, 3)
                qc_w = da.quality_conv[0].weight.data  # (rdim, 2C, 1, 1)
                # 对fs_w做spatial mean → (C, 2C) → 用前rdim个channel
                fs_w_mean = fs_w.mean(dim=[2, 3])  # (C, 2C)
                k = min(qc_w.shape[0], fs_w_mean.shape[0])
                qc_w[:k] = fs_w_mean[:k].unsqueeze(-1).unsqueeze(-1)
                print(f'QAF warm-start: quality_conv[0] initialized from tbsi_fuse_search')

    # Set DA fusion mode from config
    if use_da:
        model.da_fusion_mode = getattr(cfg.MODEL, 'DA_FUSION_MODE', 'residual')
        model.da_fusion_scale = getattr(cfg.MODEL, 'DA_FUSION_SCALE', 0.5)
        if model.da_fusion_mode == 'gate':
            print(f'DA mode: gate modulation')
        else:
            print(f'DA mode: residual (detach_base, scale={model.da_fusion_scale})')

    return model
