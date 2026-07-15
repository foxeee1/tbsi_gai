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
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.tbsi_track.vit_tbsi_care import vit_base_patch16_224_tbsi
from lib.models.layers.temporal_token import TemporalTokenBlock
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

    def forward(self, feat_rgb: torch.Tensor, feat_tir: torch.Tensor,
                quality_hint: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            quality_hint: (B, 2) optional — per-modality confidence from TBSILayer's DM.
                          Used as prior to modulate global quality scores.
        """
        B, C, H, W = feat_rgb.shape
        cat_feat = torch.cat([feat_rgb, feat_tir], dim=1)
        pooled = cat_feat.mean(dim=[2, 3], keepdim=True)

        # (1) Global quality scores
        q = torch.sigmoid(self.quality_conv(pooled))          # (B, 2, 1, 1)
        # Modulate with TBSILayer quality_hint if available (attention-fusion joint gating)
        if quality_hint is not None:
            q = q * quality_hint.unsqueeze(-1).unsqueeze(-1)  # (B, 2, 1, 1) * (B, 2, 1, 1)
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
                 use_madc=False, use_csr=False):
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.tbsi_fuse_search = conv(hidden_dim * 2, hidden_dim)
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

        # Print summary
        parts = []
        if use_temporal_tokens: parts.append(f'TemporalTokens(K={num_temporal_tokens})')
        if use_degradation_aware:
            da_str = 'QualityAwareFusion'
            if use_madc: da_str += '+MADC'
            if use_csr: da_str += '+CSR'
            parts.append(da_str)
        if parts: print(f'TBSITrack with: {", ".join(parts)}')

    def reset_temporal_tokens(self):
        self.temporal_token_state = None

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                prev_search: torch.Tensor = None,
                prev_tokens: torch.Tensor = None,
                ):
        """
        Args:
            template: (B, 6, 128, 128) — [RGB, TIR]
            search: (B, 6, 256, 256) — current frame
            prev_search: (B, 6, 256, 256) or None — previous frame (for temporal token update)
            prev_tokens: (B, K*2, D) or None — previous temporal tokens
        """
        # Two-pass: update token state with prev frame, then process current
        if self.training and self.use_temporal_tokens and prev_search is not None:
            with torch.no_grad():
                x_prev, _ = self.backbone(z=template, x=prev_search,
                                           ce_template_mask=ce_template_mask,
                                           ce_keep_rate=ce_keep_rate)
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
                                              temporal_tokens=tokens_updated)
            feat_curr = x_curr[-1] if isinstance(x_curr, list) else x_curr
            quality_hint = aux_dict.get("quality_signal", None)
            out = self.forward_head(feat_curr, None, temporal_tokens=tokens_updated,
                                    quality_hint=quality_hint)
            out['temporal_tokens'] = tokens_updated
            out.update(aux_dict)
            out['backbone_feat'] = x_curr
            return out

        # Single-frame path
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        quality_hint = aux_dict.get("quality_signal", None)
        out = self.forward_head(feat_last, None, temporal_tokens=prev_tokens,
                                quality_hint=quality_hint)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_fusion_only(self, cat_feature, quality_hint=None):
        """Extract search RGB/TIR + DA fusion. Returns (B, C, H, W) fused features.
        quality_hint: (B, 2) from TBSILayer average per-modality confidence, or None."""
        B = cat_feature.shape[0]
        C = cat_feature.shape[-1]
        num_search_token = 256
        enc_rgb = cat_feature[:, 64:64 + num_search_token, :]
        enc_tir = cat_feature[:, -num_search_token:, :]

        enc_opt = torch.cat([enc_rgb, enc_tir], dim=2)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C_, HW_ = opt.size()
        HW_ = int(HW_ / 2)
        opt_feat = opt.view(-1, C_, self.feat_sz_s, self.feat_sz_s)

        if self.use_degradation_aware:
            feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
            base_fused = self.tbsi_fuse_search(opt_feat)

            if getattr(self, 'da_fusion_mode', 'residual') == 'gate':
                gate = self.da_fusion.forward_gate_only(feat_rgb, feat_tir)
                return base_fused * (1.0 + gate)
            else:
                # Pass TBSILayer quality_hint to DaFusion for attention-fusion joint gating
                da_fused = self.da_fusion(feat_rgb, feat_tir, quality_hint=quality_hint)
                scale = getattr(self, 'da_fusion_scale', 0.5)
                return base_fused + scale * da_fused
        else:
            return self.tbsi_fuse_search(opt_feat)

    def forward_head(self, cat_feature, gt_score_map=None, temporal_tokens=None, quality_hint=None):
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
                cat_feature[:, :64, :],                  # keep template
                enc_rgb, enc_tir
            ], dim=1)

        # Fusion (with quality hint from TBSILayer for path-level gating)
        fused_feat = self.forward_fusion_only(cat_feature, quality_hint=quality_hint)

        # TC3: Temporal-Conditioned Channel Calibration
        if self.use_temporal_tokens and temporal_tokens is not None:
            fused_feat = self.tc3(fused_feat, temporal_tokens)

        # Head
        opt_feat = fused_feat
        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(B, -1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_tbsi_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('TBSITrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_tbsi':
        da_in_layer = getattr(cfg.MODEL, "DA_IN_LAYER", False)
        use_checkpoint = getattr(cfg.TRAIN, "USE_CHECKPOINT", False)
        backbone = vit_base_patch16_224_tbsi(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            tbsi_loc=cfg.MODEL.BACKBONE.TBSI_LOC,
                                            tbsi_drop_path=cfg.TRAIN.TBSI_DROP_PATH,
                                            da_in_layer=da_in_layer,
                                            use_checkpoint=use_checkpoint)
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
    num_temporal = getattr(cfg.MODEL, "NUM_TEMPORAL_TOKENS", 4)

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
    )

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
    elif 'TBSITrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

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
