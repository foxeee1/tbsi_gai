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
from lib.models.layers.dafusion_v2 import DaFusionV2
from lib.utils.box_ops import box_xyxy_to_cxcywh


def _stage2_trainable_keywords(cfg):
    """Return parameter-name fragments that should stay trainable in Stage-2."""
    model_cfg = cfg.MODEL
    flag_to_keys = {
        "TEMPORAL_TOKENS": ["temporal_token", "tc3"],
        "DEGRADATION_AWARE": ["da_fusion"],
        "DGSFUSION": ["dgs"],
        "SIGNAL_DECOUPLE": ["signal_decoupler"],
        "SOFT_SEARCH_RELIABILITY": ["soft_search_reliability"],
        "TEMPLATE_SEARCH_COMPETITION": ["template_search_competition"],
        "OUTPUT_RESIDUAL_GATE": ["output_residual_gate"],
        "TEMPLATE_CONDITIONED_BRIDGE": ["template_conditioned_bridge"],
        "CFS_RELIABILITY_BRIDGE": ["cfs_reliability_bridge"],
        "COMPETITIVE_BRIDGE": ["competitive_bridge"],
        "FREQ_GATE": ["freq_gate"],
        "FREQ_CONSISTENCY": ["freq_consistency"],
        "CUTR_LITE": ["utility_router"],
        "RTM": ["rtm"],
        "EGIR": ["egir"],
        "RSM": ["rs_mamba"],
    }
    keys = []
    for flag, fragments in flag_to_keys.items():
        if bool(getattr(model_cfg, flag, False)):
            keys.extend(fragments)
    if not keys:
        keys = ["post_fusion_block", "da_fusion", "box_head", "temporal_token", "tc3"]
    return sorted(set(keys))


def _active_stage2_module_keywords(cfg):
    keys = []
    for flag, fragments in {
        "TEMPORAL_TOKENS": ["temporal_token", "tc3"],
        "DEGRADATION_AWARE": ["da_fusion"],
        "DGSFUSION": ["dgs"],
        "SIGNAL_DECOUPLE": ["signal_decoupler"],
        "SOFT_SEARCH_RELIABILITY": ["soft_search_reliability"],
        "TEMPLATE_SEARCH_COMPETITION": ["template_search_competition"],
        "OUTPUT_RESIDUAL_GATE": ["output_residual_gate"],
        "TEMPLATE_CONDITIONED_BRIDGE": ["template_conditioned_bridge"],
        "CFS_RELIABILITY_BRIDGE": ["cfs_reliability_bridge"],
        "COMPETITIVE_BRIDGE": ["competitive_bridge"],
        "FREQ_GATE": ["freq_gate"],
        "FREQ_CONSISTENCY": ["freq_consistency"],
        "CUTR_LITE": ["utility_router"],
        "RTM": ["rtm"],
        "EGIR": ["egir"],
        "RSM": ["rs_mamba"],
    }.items():
        if bool(getattr(cfg.MODEL, flag, False)):
            keys.extend(fragments)
    return sorted(set(keys))


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


class EvidenceGuidedInteractionRouter(nn.Module):
    """Group-wise quality routing with an exact baseline-preserving start."""

    def __init__(self, dim, groups=16, hidden=32, scale=0.10):
        super().__init__()
        if dim % groups != 0:
            raise ValueError(f"EGIR groups={groups} must divide dim={dim}")
        self.groups = int(groups)
        self.group_dim = dim // groups
        self.scale = float(scale)
        evidence_dim = groups * 3
        self.evidence = nn.Sequential(
            nn.Conv2d(evidence_dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden)), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, groups * 2, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.evidence[-1].weight)
        nn.init.zeros_(self.evidence[-1].bias)
        with torch.no_grad():
            self.evidence[-1].bias[:groups].fill_(4.0)
            self.evidence[-1].bias[groups:].fill_(-4.0)

    def _group_mean(self, feat):
        b, c, h, w = feat.shape
        return feat.reshape(b, self.groups, self.group_dim, h, w).mean(dim=2)

    def forward(self, feat_rgb, feat_tir, base_fused):
        rgb_g = self._group_mean(feat_rgb)
        tir_g = self._group_mean(feat_tir)
        base_g = self._group_mean(base_fused)
        agreement = F.cosine_similarity(feat_rgb, feat_tir, dim=1, eps=1e-6)
        evidence = torch.cat([
            agreement.unsqueeze(1).expand(-1, self.groups, -1, -1),
            (rgb_g - base_g).abs(),
            (tir_g - base_g).abs(),
        ], dim=1)
        logits = self.evidence(evidence).reshape(
            feat_rgb.shape[0], 2, self.groups, feat_rgb.shape[2], feat_rgb.shape[3]
        )
        route = torch.softmax(logits, dim=1)
        exchange = 0.5 * (feat_rgb + feat_tir)
        delta = exchange - base_fused
        route_exchange = route[:, 1].repeat_interleave(self.group_dim, dim=1)
        out = base_fused + self.scale * route_exchange * delta
        return out, route, evidence



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


class UtilityRouter(nn.Module):
    """Tiny 3-action router for keep/RGB/TIR rectification."""

    def __init__(self, dim=768, hidden_dim=128):
        super().__init__()
        stats_dim = dim * 3 + 7
        self.norm = nn.LayerNorm(stats_dim)
        self.fc1 = nn.Linear(stats_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.bias[0] = 2.0

    def forward(self, rgb_roi, tir_roi, base_roi, diff_stats, score_stats):
        x = torch.cat([rgb_roi, tir_roi, base_roi, diff_stats, score_stats], dim=1)
        x = self.norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x


class RestrictedTargetMemory(nn.Module):
    """Identity-initialized FiLM adapter restricted to target-like locations."""

    def __init__(self, dim, hidden_dim=64, scale=0.10,
                 similarity_threshold=0.35, similarity_temperature=0.10):
        super().__init__()
        self.scale = float(scale)
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_temperature = float(similarity_temperature)
        self.norm = nn.LayerNorm(dim)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 2),
        )
        # Keep the baseline path exact at initialization while retaining a
        # small gradient signal for the memory branch.
        nn.init.normal_(self.adapter[-1].weight, std=1e-3)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, fused_feat, memory):
        if memory is None:
            return fused_feat
        memory = F.normalize(memory, dim=1)
        gamma, beta = self.adapter(self.norm(memory)).chunk(2, dim=1)
        gamma = self.scale * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = self.scale * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        feature_direction = F.normalize(fused_feat, dim=1)
        similarity = (feature_direction * memory[:, :, None, None]).sum(dim=1, keepdim=True)
        spatial_gate = torch.sigmoid(
            (similarity - self.similarity_threshold)
            / max(self.similarity_temperature, 1e-6)
        )
        return fused_feat * (1.0 + spatial_gate * gamma) + spatial_gate * beta


class ReliabilityStateMamba(nn.Module):
    """A tiny Mamba-inspired selective state updater for route prediction."""

    def __init__(self, input_dim=13, state_dim=64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.in_proj = nn.Linear(self.input_dim, self.state_dim * 3)
        self.state_proj = nn.Linear(self.state_dim, self.state_dim)
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.route_head = nn.Linear(self.state_dim, 3)
        self.reliability_head = nn.Linear(self.state_dim, 2)

        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)
        with torch.no_grad():
            self.route_head.bias[0] = 2.0
        nn.init.zeros_(self.reliability_head.weight)
        nn.init.zeros_(self.reliability_head.bias)

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)

    def predict(self, state):
        state = self.state_norm(state)
        route_logits = self.route_head(state)
        reliability_logits = self.reliability_head(state)
        reliability = torch.sigmoid(reliability_logits)
        return route_logits, reliability, reliability_logits

    def forward_step(self, obs, prev_state=None):
        state_dtype = self.input_norm.weight.dtype
        obs = obs.to(dtype=state_dtype)
        if prev_state is not None:
            prev_state = prev_state.to(dtype=state_dtype)
        obs = self.input_norm(obs)
        if prev_state is None:
            prev_state = self.init_state(obs.shape[0], obs.device, obs.dtype)
        x_gate, x_value, x_decay = self.in_proj(obs).chunk(3, dim=1)
        gate = torch.sigmoid(x_gate)
        value = torch.tanh(x_value + self.state_proj(prev_state))
        decay = torch.sigmoid(x_decay)
        next_state = (1.0 - gate * decay) * prev_state + (gate * decay) * value
        route_logits, reliability, reliability_logits = self.predict(next_state)
        return next_state, route_logits, reliability, reliability_logits


class TBSITrack(nn.Module):
    """ TBSI with Temporal Token + Quality-Aware Fusion (spatial+channel+MADC+CSR) """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 use_temporal_tokens=False, num_temporal_tokens=4,
                 use_degradation_aware=False, da_mode='spatial',
                 use_madc=False, use_csr=False, da_v2=False,
                 use_cutr_lite=False, cutr_hidden_dim=128, cutr_eta=0.10,
                 use_rtm=False, rtm_hidden_dim=64, rtm_scale=0.10,
                 rtm_similarity_threshold=0.35, rtm_similarity_temperature=0.10,
                 use_egir=False, egir_groups=16, egir_hidden=32, egir_scale=0.10,
                 use_rsm=False, rsm_state_dim=64, rsm_input_dim=13,
                 rsm_rho=0.10, rsm_reliability_scale=0.50):
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
        self.use_cutr_lite = use_cutr_lite
        self.cutr_eta = cutr_eta
        self.use_egir = bool(use_egir)
        self.use_rsm = bool(use_rsm)
        self.rsm_rho = float(rsm_rho)
        self.rsm_reliability_scale = float(rsm_reliability_scale)
        if self.use_egir:
            self.egir = EvidenceGuidedInteractionRouter(
                dim=hidden_dim, groups=egir_groups, hidden=egir_hidden,
                scale=egir_scale,
            )
        if self.use_rsm:
            self.rs_mamba = ReliabilityStateMamba(
                input_dim=rsm_input_dim, state_dim=rsm_state_dim
            )

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
            if da_v2:
                print('  Using DaFusionV2 — true degradation-aware fusion')
                self.da_fusion = DaFusionV2(dim=hidden_dim, da_mode=da_mode,
                                            use_madc=use_madc, use_csr=use_csr)
            else:
                self.da_fusion = DegradationAwareFusion(dim=hidden_dim, da_mode=da_mode,
                                                        use_madc=use_madc, use_csr=use_csr)
            self.da_fusion_mode = 'residual'  # 'residual' | 'gate'
            self.da_fusion_scale = 0.5        # residual scale

        if use_cutr_lite:
            self.utility_router = UtilityRouter(dim=hidden_dim, hidden_dim=cutr_hidden_dim)

        self.use_rtm = use_rtm
        if use_rtm:
            self.rtm = RestrictedTargetMemory(
                dim=hidden_dim, hidden_dim=rtm_hidden_dim, scale=rtm_scale,
                similarity_threshold=rtm_similarity_threshold,
                similarity_temperature=rtm_similarity_temperature,
            )

        # Print summary
        parts = []
        if use_temporal_tokens: parts.append(f'TemporalTokens(K={num_temporal_tokens})')
        if use_degradation_aware:
            da_str = 'DaFusionV2' if da_v2 else 'DaFusionV1'
            if use_madc: da_str += '+MADC'
            if use_csr: da_str += '+CSR'
            parts.append(da_str)
        if use_cutr_lite:
            parts.append(f'CUTR-Lite(eta={cutr_eta}, hidden={cutr_hidden_dim})')
        if use_rtm:
            parts.append(f'RTM(hidden={rtm_hidden_dim}, scale={rtm_scale})')
        if self.use_egir:
            parts.append(f'EGIR(groups={egir_groups}, scale={egir_scale})')
        if self.use_rsm:
            parts.append(f'RS-Mamba(state={rsm_state_dim}, rho={rsm_rho})')
        if parts: print(f'TBSITrack with: {", ".join(parts)}')

    def reset_temporal_tokens(self):
        self.temporal_token_state = None

    def reset_rsm_state(self):
        return None

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                prev_search: torch.Tensor = None,
                prev_tokens: torch.Tensor = None,
                rtm_memory: torch.Tensor = None,
                rsm_state: torch.Tensor = None,
                rsm_route_logits: torch.Tensor = None,
                rsm_reliability: torch.Tensor = None,
                rsm_reliability_logits: torch.Tensor = None,
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
                                    quality_hint=quality_hint,
                                    rtm_memory=rtm_memory,
                                    rsm_state=rsm_state,
                                    rsm_route_logits=rsm_route_logits,
                                    rsm_reliability=rsm_reliability,
                                    rsm_reliability_logits=rsm_reliability_logits)
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
                                quality_hint=quality_hint,
                                rtm_memory=rtm_memory,
                                rsm_state=rsm_state,
                                rsm_route_logits=rsm_route_logits,
                                rsm_reliability=rsm_reliability,
                                rsm_reliability_logits=rsm_reliability_logits)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def get_fusion_pack(self, cat_feature, quality_hint=None):
        """Return the modality feature pack used by the prediction head."""
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

        feat_rgb = enc_rgb.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
        feat_tir = enc_tir.transpose(1, 2).reshape(B, C, self.feat_sz_s, self.feat_sz_s)
        base_fused = self.tbsi_fuse_search(opt_feat)

        if self.use_egir:
            egir_fused, egir_route, egir_evidence = self.egir(
                feat_rgb, feat_tir, base_fused
            )
        else:
            egir_fused, egir_route, egir_evidence = base_fused, None, None

        if self.use_degradation_aware:
            if getattr(self, 'da_fusion_mode', 'residual') == 'gate':
                gate = self.da_fusion.forward_gate_only(feat_rgb, feat_tir)
                fused_feat = base_fused * (1.0 + gate)
            else:
                # Pass TBSILayer quality_hint to DaFusion for attention-fusion joint gating
                da_fused = self.da_fusion(feat_rgb, feat_tir, quality_hint=quality_hint)
                scale = getattr(self, 'da_fusion_scale', 0.5)
                fused_feat = base_fused + scale * da_fused
        else:
            fused_feat = egir_fused if self.use_egir else base_fused

        return {
            'enc_rgb': enc_rgb,
            'enc_tir': enc_tir,
            'feat_rgb': feat_rgb,
            'feat_tir': feat_tir,
            'base_fused': base_fused,
            'fused_feat': fused_feat,
            'egir_route': egir_route,
            'egir_evidence': egir_evidence,
        }

    def _layer_norm_2d(self, feat):
        feat_ln = F.layer_norm(feat.permute(0, 2, 3, 1), (feat.shape[1],))
        return feat_ln.permute(0, 3, 1, 2)

    def _global_pool(self, feat):
        return feat.mean(dim=(2, 3))

    def _score_stats(self, score_map):
        flat = score_map.flatten(1)
        peak = flat.max(dim=1).values
        mean = flat.mean(dim=1)
        topk = torch.topk(flat, k=min(2, flat.shape[1]), dim=1).values
        gap = topk[:, 0] - topk[:, -1]
        prob = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(prob * prob.clamp_min(1e-6).log()).sum(dim=1)
        return torch.stack([peak, mean, gap, entropy], dim=1)

    def _build_cutr_router_inputs(self, fusion_pack):
        rgb_roi = self._global_pool(fusion_pack['feat_rgb'])
        tir_roi = self._global_pool(fusion_pack['feat_tir'])
        base_roi = self._global_pool(fusion_pack['base_fused'])
        diff_roi = rgb_roi - tir_roi
        diff_stats = torch.stack([
            diff_roi.abs().mean(dim=1),
            diff_roi.pow(2).mean(dim=1).sqrt(),
            diff_roi.mean(dim=1),
        ], dim=1)
        return rgb_roi, tir_roi, base_roi, diff_stats

    def _apply_cutr_lite(self, fusion_pack, gt_score_map=None):
        base_head = self.forward_head_from_fused(fusion_pack['base_fused'], gt_score_map=gt_score_map)
        rgb_roi, tir_roi, base_roi, diff_stats = self._build_cutr_router_inputs(fusion_pack)
        score_stats = self._score_stats(base_head['score_map'])
        utility_logits = self.utility_router(rgb_roi, tir_roi, base_roi, diff_stats, score_stats)
        utility_probs = torch.softmax(utility_logits, dim=1)

        diff_feat = self._layer_norm_2d(fusion_pack['feat_rgb']) - self._layer_norm_2d(fusion_pack['feat_tir'])
        a_diff = self.cutr_eta * (utility_probs[:, 1] - utility_probs[:, 2]).view(-1, 1, 1, 1)
        rectified_feat = fusion_pack['base_fused'] + a_diff * diff_feat

        pred_dict = self.forward_head_from_fused(rectified_feat, gt_score_map=gt_score_map)
        pred_dict['utility_logits'] = utility_logits
        pred_dict['utility_probs'] = utility_probs
        pred_dict['router_score_stats'] = score_stats
        pred_dict['router_diff_stats'] = diff_stats

        if self.training:
            with torch.no_grad():
                pred_dict['utility_candidates'] = {
                    'keep': base_head,
                    'rgb': self.forward_head_from_fused(
                        fusion_pack['base_fused'] + self.cutr_eta * diff_feat,
                        gt_score_map=gt_score_map,
                    ),
                    'tir': self.forward_head_from_fused(
                        fusion_pack['base_fused'] - self.cutr_eta * diff_feat,
                        gt_score_map=gt_score_map,
                    ),
                }
        return pred_dict

    def _build_rsm_input(self, fusion_pack, base_head, quality_hint=None):
        rgb_roi, tir_roi, base_roi, diff_stats = self._build_cutr_router_inputs(fusion_pack)
        score_stats = self._score_stats(base_head['score_map'])
        pred_box = base_head['pred_boxes'].mean(dim=1)
        agreement = F.cosine_similarity(rgb_roi, tir_roi, dim=1, eps=1e-6).unsqueeze(1)
        if quality_hint is None:
            quality_vec = base_roi.new_full((base_roi.shape[0], 2), 0.5)
        else:
            quality_vec = quality_hint.to(dtype=base_roi.dtype)
        obs = torch.cat([score_stats, diff_stats, pred_box, agreement, quality_vec], dim=1)
        return obs

    def _apply_rsm_route(self, fusion_pack, base_head, route_logits, reliability, gt_score_map=None):
        if route_logits is None:
            pred_dict = dict(base_head)
            pred_dict['rsm_applied_action'] = base_head['pred_boxes'].new_zeros(base_head['pred_boxes'].shape[0], dtype=torch.long)
            return pred_dict

        fused_feat = fusion_pack['base_fused']
        rgb_direction = self._layer_norm_2d(fusion_pack['feat_rgb']) - self._layer_norm_2d(fused_feat)
        tir_direction = self._layer_norm_2d(fusion_pack['feat_tir']) - self._layer_norm_2d(fused_feat)
        route_bias = 0.0
        if reliability is not None:
            route_bias = self.rsm_reliability_scale * (reliability - 0.5)
        adjusted_logits = route_logits.clone()
        if not isinstance(route_bias, float):
            adjusted_logits[:, 1:] = adjusted_logits[:, 1:] + route_bias
        action = adjusted_logits.argmax(dim=1)

        candidates = {
            'keep': base_head,
            'rgb': self.forward_head_from_fused(fused_feat + self.rsm_rho * rgb_direction, gt_score_map=gt_score_map),
            'tir': self.forward_head_from_fused(fused_feat + self.rsm_rho * tir_direction, gt_score_map=gt_score_map),
        }
        pred_dict = {}
        keys = ['pred_boxes', 'score_map', 'size_map', 'offset_map']
        action_order = ['keep', 'rgb', 'tir']
        for key in keys:
            stacked = torch.stack([candidates[name][key] for name in action_order], dim=1)
            gather_index = action.view(-1, 1, *([1] * (stacked.dim() - 2))).expand(-1, 1, *stacked.shape[2:])
            pred_dict[key] = torch.gather(stacked, 1, gather_index).squeeze(1)
        pred_dict['rsm_applied_action'] = action
        pred_dict['utility_candidates'] = candidates
        pred_dict['utility_logits'] = route_logits
        pred_dict['rsm_reliability'] = reliability
        return pred_dict

    def forward_fusion_only(self, cat_feature, quality_hint=None):
        """Extract search RGB/TIR + DA fusion. Returns (B, C, H, W) fused features."""
        return self.get_fusion_pack(cat_feature, quality_hint=quality_hint)['fused_feat']

    def forward_head_from_fused(self, fused_feat, gt_score_map=None):
        """Run the prediction head from a prepared fused feature map."""
        B = fused_feat.shape[0]
        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(fused_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(B, -1, 4)
            return {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map,
            }
        raise NotImplementedError

    def _template_memory(self, cat_feature):
        """Build a stable prototype from the two 8x8 template token blocks."""
        if not self.use_rtm:
            return None
        num_template = 64
        rgb_template = cat_feature[:, :num_template, :]
        tir_template = cat_feature[:, 256 + num_template:256 + 2 * num_template, :]
        return 0.5 * (rgb_template.mean(dim=1) + tir_template.mean(dim=1))

    @staticmethod
    def extract_rtm_prototype(fused_feat, boxes):
        """Pool target prototypes from normalized xywh boxes on a feature map."""
        _, _, height, width = fused_feat.shape
        prototypes = []
        for feat, box in zip(fused_feat, boxes):
            x, y, box_w, box_h = [float(v) for v in box.detach().view(-1)]
            x1 = max(0, min(width - 1, int(x * width)))
            y1 = max(0, min(height - 1, int(y * height)))
            x2 = max(x1 + 1, min(width, int((x + box_w) * width + 0.999)))
            y2 = max(y1 + 1, min(height, int((y + box_h) * height + 0.999)))
            prototypes.append(feat[:, y1:y2, x1:x2].mean(dim=(1, 2)))
        return F.normalize(torch.stack(prototypes, dim=0), dim=1)

    def forward_head(self, cat_feature, gt_score_map=None, temporal_tokens=None,
                     quality_hint=None, rtm_memory=None, rsm_state=None,
                     rsm_route_logits=None, rsm_reliability=None,
                     rsm_reliability_logits=None):
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
        fusion_pack = self.get_fusion_pack(cat_feature, quality_hint=quality_hint)
        fused_feat = fusion_pack['fused_feat']
        base_head = self.forward_head_from_fused(fusion_pack['base_fused'], gt_score_map=gt_score_map)

        # No template proxy: training and inference both use a prototype pooled
        # from a preceding search frame. The first tracked frame stays baseline-exact.
        active_memory = rtm_memory
        if self.use_rtm:
            fused_feat = self.rtm(fused_feat, active_memory)

        if self.use_cutr_lite:
            return self._apply_cutr_lite(fusion_pack, gt_score_map=gt_score_map)

        if self.use_rsm:
            pred_dict = dict(base_head) if self.training else self._apply_rsm_route(
                fusion_pack, base_head, rsm_route_logits, rsm_reliability, gt_score_map=gt_score_map
            )
            obs = self._build_rsm_input(fusion_pack, base_head, quality_hint=quality_hint)
            next_rsm_state, next_route_logits, next_reliability, next_reliability_logits = self.rs_mamba.forward_step(
                obs, prev_state=rsm_state
            )
            if self.training:
                rgb_direction = self._layer_norm_2d(fusion_pack['feat_rgb']) - self._layer_norm_2d(fusion_pack['base_fused'])
                tir_direction = self._layer_norm_2d(fusion_pack['feat_tir']) - self._layer_norm_2d(fusion_pack['base_fused'])
                pred_dict['utility_candidates'] = {
                    'keep': base_head,
                    'rgb': self.forward_head_from_fused(
                        fusion_pack['base_fused'] + self.rsm_rho * rgb_direction,
                        gt_score_map=gt_score_map,
                    ),
                    'tir': self.forward_head_from_fused(
                        fusion_pack['base_fused'] + self.rsm_rho * tir_direction,
                        gt_score_map=gt_score_map,
                    ),
                }
                pred_dict['utility_logits'] = rsm_route_logits
                pred_dict['rsm_reliability_logits'] = rsm_reliability_logits
            pred_dict['rsm_state'] = next_rsm_state
            pred_dict['rsm_next_route_logits'] = next_route_logits
            pred_dict['rsm_next_reliability'] = next_reliability
            pred_dict['rsm_next_reliability_logits'] = next_reliability_logits
            pred_dict['rsm_observation'] = obs
            pred_dict['rsm_reliability'] = rsm_reliability
            return pred_dict

        # TC3: Temporal-Conditioned Channel Calibration
        if self.use_temporal_tokens and temporal_tokens is not None:
            fused_feat = self.tc3(fused_feat, temporal_tokens)
        pred_dict = self.forward_head_from_fused(fused_feat, gt_score_map=gt_score_map)
        if self.use_rtm:
            pred_dict['rtm_memory'] = active_memory
            pred_dict['rtm_source_feat'] = fusion_pack['fused_feat']
            pred_dict['rtm_fused_feat'] = fused_feat
        return pred_dict


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
        use_dgs = getattr(cfg.MODEL, "DGSFUSION", False)
        dgs_mode = getattr(cfg.MODEL, "DGS_MODE", "v1")
        use_checkpoint = getattr(cfg.TRAIN, "USE_CHECKPOINT", False)
        use_signal_decouple = getattr(cfg.MODEL, "SIGNAL_DECOUPLE", False)
        signal_decouple_mode = getattr(cfg.MODEL, "SIGNAL_DECOUPLE_MODE", "split")
        signal_decouple_layers = getattr(cfg.MODEL, "SIGNAL_DECOUPLE_LAYERS", [])
        signal_decouple_scale = getattr(cfg.MODEL, "SIGNAL_DECOUPLE_SCALE", 0.5)
        signal_decouple_layer_scales = getattr(cfg.MODEL, "SIGNAL_DECOUPLE_LAYER_SCALES", [])
        signal_decouple_alpha_init = getattr(cfg.MODEL, "SIGNAL_DECOUPLE_ALPHA_INIT", 0.1)
        use_soft_search_reliability = getattr(cfg.MODEL, "SOFT_SEARCH_RELIABILITY", False)
        soft_search_reliability_layers = getattr(cfg.MODEL, "SOFT_SEARCH_RELIABILITY_LAYERS", [])
        soft_search_reliability_scale = getattr(cfg.MODEL, "SOFT_SEARCH_RELIABILITY_SCALE", 0.25)
        soft_search_reliability_alpha_init = getattr(cfg.MODEL, "SOFT_SEARCH_RELIABILITY_ALPHA_INIT", 0.05)
        use_template_search_competition = getattr(cfg.MODEL, "TEMPLATE_SEARCH_COMPETITION", False)
        template_search_competition_layers = getattr(cfg.MODEL, "TEMPLATE_SEARCH_COMPETITION_LAYERS", [])
        template_search_competition_scale = getattr(cfg.MODEL, "TEMPLATE_SEARCH_COMPETITION_SCALE", 0.20)
        template_search_competition_alpha_init = getattr(cfg.MODEL, "TEMPLATE_SEARCH_COMPETITION_ALPHA_INIT", 0.10)
        template_search_competition_temperature = getattr(cfg.MODEL, "TEMPLATE_SEARCH_COMPETITION_TEMPERATURE", 4.0)
        use_output_residual_gate = getattr(cfg.MODEL, "OUTPUT_RESIDUAL_GATE", False)
        output_residual_gate_layers = getattr(cfg.MODEL, "OUTPUT_RESIDUAL_GATE_LAYERS", [])
        output_residual_gate_mode = getattr(cfg.MODEL, "OUTPUT_RESIDUAL_GATE_MODE", "naive")
        output_residual_gate_scale = getattr(cfg.MODEL, "OUTPUT_RESIDUAL_GATE_SCALE", 1.0)
        use_template_conditioned_bridge = getattr(cfg.MODEL, "TEMPLATE_CONDITIONED_BRIDGE", False)
        template_conditioned_bridge_layers = getattr(cfg.MODEL, "TEMPLATE_CONDITIONED_BRIDGE_LAYERS", [])
        template_conditioned_bridge_scale = getattr(cfg.MODEL, "TEMPLATE_CONDITIONED_BRIDGE_SCALE", 0.2)
        template_conditioned_bridge_temperature = getattr(
            cfg.MODEL, "TEMPLATE_CONDITIONED_BRIDGE_TEMPERATURE", 4.0)
        template_conditioned_bridge_bias_mode = getattr(
            cfg.MODEL, "TEMPLATE_CONDITIONED_BRIDGE_BIAS_MODE", "signed")
        template_conditioned_bridge_neg_floor = getattr(
            cfg.MODEL, "TEMPLATE_CONDITIONED_BRIDGE_NEG_FLOOR", -0.3)
        use_cfs_reliability_bridge = getattr(cfg.MODEL, "CFS_RELIABILITY_BRIDGE", False)
        cfs_reliability_bridge_layers = getattr(cfg.MODEL, "CFS_RELIABILITY_BRIDGE_LAYERS", [])
        cfs_reliability_bridge_scale = getattr(cfg.MODEL, "CFS_RELIABILITY_BRIDGE_SCALE", 0.2)
        cfs_reliability_bridge_hidden = getattr(cfg.MODEL, "CFS_RELIABILITY_BRIDGE_HIDDEN", 32)
        use_competitive_bridge = getattr(cfg.MODEL, "COMPETITIVE_BRIDGE", False)
        competitive_bridge_layers = getattr(cfg.MODEL, "COMPETITIVE_BRIDGE_LAYERS", [])
        competitive_bridge_temperature = getattr(cfg.MODEL, "COMPETITIVE_BRIDGE_TEMPERATURE", 1.0)
        competitive_bridge_residual_scale = getattr(cfg.MODEL, "COMPETITIVE_BRIDGE_RESIDUAL_SCALE", 1.0)
        use_freq_gate = getattr(cfg.MODEL, "FREQ_GATE", False)
        freq_gate_layers = getattr(cfg.MODEL, "FREQ_GATE_LAYERS", [])
        freq_gate_scale = getattr(cfg.MODEL, "FREQ_GATE_SCALE", 0.1)
        freq_gate_cutoff = getattr(cfg.MODEL, "FREQ_GATE_CUTOFF", 0.25)
        use_freq_consistency = getattr(cfg.MODEL, "FREQ_CONSISTENCY", False)
        freq_consistency_layers = getattr(cfg.MODEL, "FREQ_CONSISTENCY_LAYERS", [])
        freq_consistency_scale = getattr(cfg.MODEL, "FREQ_CONSISTENCY_SCALE", 0.3)
        freq_consistency_cutoff = getattr(cfg.MODEL, "FREQ_CONSISTENCY_CUTOFF", 0.25)
        freq_consistency_mode = getattr(cfg.MODEL, "FREQ_CONSISTENCY_MODE", "global")
        freq_consistency_local_kernel = getattr(cfg.MODEL, "FREQ_CONSISTENCY_LOCAL_KERNEL", 3)
        freq_consistency_margin = getattr(cfg.MODEL, "FREQ_CONSISTENCY_MARGIN", 0.0)
        freq_consistency_learnable_scale = getattr(cfg.MODEL, "FREQ_CONSISTENCY_LEARNABLE_SCALE", False)
        freq_consistency_bands = getattr(cfg.MODEL, "FREQ_CONSISTENCY_BANDS", "low")
        freq_consistency_form = getattr(cfg.MODEL, "FREQ_CONSISTENCY_FORM", "ratio")
        freq_consistency_layer_forms = getattr(cfg.MODEL, "FREQ_CONSISTENCY_LAYER_FORMS", [])
        freq_consistency_keep_ratio = getattr(cfg.MODEL, "FREQ_CONSISTENCY_KEEP_RATIO", 0.5)
        freq_consistency_conflict_act = getattr(cfg.MODEL, "FREQ_CONSISTENCY_CONFLICT_ACT", False)
        freq_consistency_conflict_top_ratio = getattr(cfg.MODEL, "FREQ_CONSISTENCY_CONFLICT_TOP_RATIO", 0.1)
        freq_consistency_conflict_tau = getattr(cfg.MODEL, "FREQ_CONSISTENCY_CONFLICT_TAU", 0.02)
        freq_consistency_conflict_gamma = getattr(cfg.MODEL, "FREQ_CONSISTENCY_CONFLICT_GAMMA", 40.0)
        backbone = vit_base_patch16_224_tbsi(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            tbsi_loc=cfg.MODEL.BACKBONE.TBSI_LOC,
                                            tbsi_drop_path=cfg.TRAIN.TBSI_DROP_PATH,
                                            da_in_layer=da_in_layer,
                                            use_dgs=use_dgs, dgs_mode=dgs_mode,
                                            use_checkpoint=use_checkpoint,
                                            use_signal_decouple=use_signal_decouple,
                                            signal_decouple_mode=signal_decouple_mode,
                                            signal_decouple_layers=signal_decouple_layers,
                                            signal_decouple_scale=signal_decouple_scale,
                                            signal_decouple_layer_scales=signal_decouple_layer_scales,
                                            signal_decouple_alpha_init=signal_decouple_alpha_init,
                                            use_soft_search_reliability=use_soft_search_reliability,
                                            soft_search_reliability_layers=soft_search_reliability_layers,
                                            soft_search_reliability_scale=soft_search_reliability_scale,
                                            soft_search_reliability_alpha_init=soft_search_reliability_alpha_init,
                                            use_template_search_competition=use_template_search_competition,
                                            template_search_competition_layers=template_search_competition_layers,
                                            template_search_competition_scale=template_search_competition_scale,
                                            template_search_competition_alpha_init=template_search_competition_alpha_init,
                                            template_search_competition_temperature=template_search_competition_temperature,
                                            use_output_residual_gate=use_output_residual_gate,
                                            output_residual_gate_layers=output_residual_gate_layers,
                                            output_residual_gate_mode=output_residual_gate_mode,
                                            output_residual_gate_scale=output_residual_gate_scale,
                                            use_template_conditioned_bridge=use_template_conditioned_bridge,
                                            template_conditioned_bridge_layers=template_conditioned_bridge_layers,
                                            template_conditioned_bridge_scale=template_conditioned_bridge_scale,
                                            template_conditioned_bridge_temperature=template_conditioned_bridge_temperature,
                                            template_conditioned_bridge_bias_mode=template_conditioned_bridge_bias_mode,
                                            template_conditioned_bridge_neg_floor=template_conditioned_bridge_neg_floor,
                                            use_cfs_reliability_bridge=use_cfs_reliability_bridge,
                                            cfs_reliability_bridge_layers=cfs_reliability_bridge_layers,
                                            cfs_reliability_bridge_scale=cfs_reliability_bridge_scale,
                                            cfs_reliability_bridge_hidden=cfs_reliability_bridge_hidden,
                                            use_competitive_bridge=use_competitive_bridge,
                                            competitive_bridge_layers=competitive_bridge_layers,
                                            competitive_bridge_temperature=competitive_bridge_temperature,
                                            competitive_bridge_residual_scale=competitive_bridge_residual_scale,
                                            use_freq_gate=use_freq_gate,
                                            freq_gate_layers=freq_gate_layers,
                                            freq_gate_scale=freq_gate_scale,
                                            freq_gate_cutoff=freq_gate_cutoff,
                                            use_freq_consistency=use_freq_consistency,
                                            freq_consistency_layers=freq_consistency_layers,
                                            freq_consistency_scale=freq_consistency_scale,
                                            freq_consistency_cutoff=freq_consistency_cutoff,
                                            freq_consistency_mode=freq_consistency_mode,
                                            freq_consistency_local_kernel=freq_consistency_local_kernel,
                                            freq_consistency_margin=freq_consistency_margin,
                                            freq_consistency_learnable_scale=freq_consistency_learnable_scale,
                                            freq_consistency_bands=freq_consistency_bands,
                                            freq_consistency_form=freq_consistency_form,
                                            freq_consistency_layer_forms=freq_consistency_layer_forms,
                                            freq_consistency_keep_ratio=freq_consistency_keep_ratio,
                                            freq_consistency_conflict_act=freq_consistency_conflict_act,
                                            freq_consistency_conflict_top_ratio=freq_consistency_conflict_top_ratio,
                                            freq_consistency_conflict_tau=freq_consistency_conflict_tau,
                                            freq_consistency_conflict_gamma=freq_consistency_conflict_gamma)

        if use_signal_decouple:
            layer_msg = "all" if not signal_decouple_layers else list(signal_decouple_layers)
            print(f'  [SignalDecouple] Task-aware Feature Decoupling enabled '
                  f'(mode={signal_decouple_mode}, layers={layer_msg})')
        if use_soft_search_reliability:
            layer_msg = "all" if not soft_search_reliability_layers else list(soft_search_reliability_layers)
            print(f'  [SoftSearchReliability] Search token soft modulation enabled '
                  f'(layers={layer_msg}, scale={soft_search_reliability_scale}, '
                  f'alpha_init={soft_search_reliability_alpha_init})')
        if use_template_search_competition:
            layer_msg = "all" if not template_search_competition_layers else list(template_search_competition_layers)
            print(f'  [TemplateSearchCompetition] Template-guided search competition enabled '
                  f'(layers={layer_msg}, scale={template_search_competition_scale}, '
                  f'alpha_init={template_search_competition_alpha_init}, '
                  f'temperature={template_search_competition_temperature})')
        if use_output_residual_gate:
            layer_msg = "all" if not output_residual_gate_layers else list(output_residual_gate_layers)
            print(f'  [OutputResidualGate] Delta-level bridge output adapter enabled '
                  f'(layers={layer_msg}, mode={output_residual_gate_mode}, '
                  f'scale={output_residual_gate_scale})')
        if use_template_conditioned_bridge:
            layer_msg = ("all" if not template_conditioned_bridge_layers
                         else list(template_conditioned_bridge_layers))
            print(f'  [TemplateConditionedBridge] Pre-softmax bridge bias enabled '
                  f'(layers={layer_msg}, scale={template_conditioned_bridge_scale}, '
                  f'temperature={template_conditioned_bridge_temperature}, '
                  f'bias_mode={template_conditioned_bridge_bias_mode}, '
                  f'neg_floor={template_conditioned_bridge_neg_floor})')
        if use_cfs_reliability_bridge:
            layer_msg = "all" if not cfs_reliability_bridge_layers else list(cfs_reliability_bridge_layers)
            print(f'  [CFSReliabilityBridge] Cross-modal feature-structure bridge bias enabled '
                  f'(layers={layer_msg}, scale={cfs_reliability_bridge_scale}, '
                  f'hidden={cfs_reliability_bridge_hidden})')
        if use_competitive_bridge:
            layer_msg = "all" if not competitive_bridge_layers else list(competitive_bridge_layers)
            print(f'  [CompetitiveBridge] RGB/TIR s2t competition enabled '
                  f'(layers={layer_msg}, temperature={competitive_bridge_temperature}, '
                  f'residual_scale={competitive_bridge_residual_scale})')
        if use_freq_gate:
            layer_msg = "all" if not freq_gate_layers else list(freq_gate_layers)
            print(f'  [FreqGATE] Spatial-frequency bridge bias enabled '
                  f'(layers={layer_msg}, scale={freq_gate_scale}, cutoff={freq_gate_cutoff})')
        if use_freq_consistency:
            layer_msg = "all" if not freq_consistency_layers else list(freq_consistency_layers)
            print(f'  [FCC] Cross-modal frequency-consistency bridge bias enabled '
                  f'(layers={layer_msg}, scale={freq_consistency_scale}, '
                  f'cutoff={freq_consistency_cutoff}, mode={freq_consistency_mode}, '
                  f'kernel={freq_consistency_local_kernel}, margin={freq_consistency_margin}, '
                  f'learnable_scale={freq_consistency_learnable_scale}, '
                  f'bands={freq_consistency_bands}, form={freq_consistency_form}, '
                  f'layer_forms={list(freq_consistency_layer_forms)}, '
                  f'conflict_act={freq_consistency_conflict_act})')
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    box_head = build_box_head(cfg, hidden_dim)

    use_temporal = getattr(cfg.MODEL, "TEMPORAL_TOKENS", False)
    use_da = getattr(cfg.MODEL, "DEGRADATION_AWARE", False)
    da_v2 = getattr(cfg.MODEL, "DA_V2", False)
    da_mode = getattr(cfg.MODEL, "DA_MODE", "spatial")
    use_madc = getattr(cfg.MODEL, "DA_MADC", False)
    use_csr = getattr(cfg.MODEL, "DA_CSR", False)
    num_temporal = getattr(cfg.MODEL, "NUM_TEMPORAL_TOKENS", 4)
    use_cutr_lite = getattr(cfg.MODEL, "CUTR_LITE", False)
    cutr_hidden_dim = getattr(cfg.MODEL, "CUTR_LITE_HIDDEN", 128)
    cutr_eta = getattr(cfg.MODEL, "CUTR_LITE_ETA", 0.10)
    use_rtm = getattr(cfg.MODEL, "RTM", False)
    rtm_hidden_dim = getattr(cfg.MODEL, "RTM_HIDDEN", 64)
    rtm_scale = getattr(cfg.MODEL, "RTM_SCALE", 0.10)
    rtm_similarity_threshold = getattr(cfg.MODEL, "RTM_SIM_THRESHOLD", 0.35)
    rtm_similarity_temperature = getattr(cfg.MODEL, "RTM_SIM_TEMPERATURE", 0.10)
    use_egir = getattr(cfg.MODEL, "EGIR", False)
    egir_groups = getattr(cfg.MODEL, "EGIR_GROUPS", 16)
    egir_hidden = getattr(cfg.MODEL, "EGIR_HIDDEN", 32)
    egir_scale = getattr(cfg.MODEL, "EGIR_SCALE", 0.10)
    use_rsm = getattr(cfg.MODEL, "RSM", False)
    rsm_state_dim = getattr(cfg.MODEL, "RSM_STATE_DIM", 64)
    rsm_input_dim = getattr(cfg.MODEL, "RSM_INPUT_DIM", 13)
    rsm_rho = getattr(cfg.MODEL, "RSM_RHO", 0.10)
    rsm_reliability_scale = getattr(cfg.MODEL, "RSM_RELIABILITY_SCALE", 0.50)

    if use_da:
        da_str = 'DaFusionV2' if da_v2 else 'DaFusionV1'
        da_str += f' ({da_mode}'
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
        da_v2=da_v2,
        use_cutr_lite=use_cutr_lite,
        cutr_hidden_dim=cutr_hidden_dim,
        cutr_eta=cutr_eta,
        use_rtm=use_rtm,
        rtm_hidden_dim=rtm_hidden_dim,
        rtm_scale=rtm_scale,
        rtm_similarity_threshold=rtm_similarity_threshold,
        rtm_similarity_temperature=rtm_similarity_temperature,
        use_egir=use_egir,
        egir_groups=egir_groups,
        egir_hidden=egir_hidden,
        egir_scale=egir_scale,
        use_rsm=use_rsm,
        rsm_state_dim=rsm_state_dim,
        rsm_input_dim=rsm_input_dim,
        rsm_rho=rsm_rho,
        rsm_reliability_scale=rsm_reliability_scale,
    )

    # Stage 2: load baseline checkpoint + freeze all except post_fusion_block
    stage2_baseline = getattr(cfg.MODEL, "STAGE2_BASELINE", "")
    cutr_baseline = getattr(cfg.MODEL, "CUTR_LITE_BASELINE", "")
    router_only = getattr(cfg.TRAIN, "ROUTER_ONLY", False)
    if stage2_baseline and training:
        baseline_path = os.path.join(current_dir, '../../../', stage2_baseline)
        if os.path.exists(baseline_path):
            checkpoint = torch.load(baseline_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(checkpoint["net"], strict=False)
            print(f'Stage 2: Loaded baseline from {stage2_baseline}')
            expected_missing_fragments = _stage2_trainable_keywords(cfg)
            unexpected_missing = [
                k for k in missing
                if not any(fragment in k for fragment in expected_missing_fragments)
            ]
            print(f'  Missing keys: {len(missing)} (sample: {missing[:20]})')
            print(f'  Unexpected keys: {len(unexpected)} (sample: {unexpected[:20]})')
            if unexpected_missing or unexpected:
                raise RuntimeError(
                    'Stage-2 baseline/config mismatch. unexpected_missing={}, unexpected={}'.format(
                        unexpected_missing[:20], unexpected[:20]
                    )
                )
        else:
            print(f'WARNING: Stage 2 baseline not found: {baseline_path}')

        # Freeze everything EXCEPT the explicitly selected lightweight module.
        trainable_keys = _stage2_trainable_keywords(cfg)
        for n, p in model.named_parameters():
            if any(k in n for k in trainable_keys):
                p.requires_grad = True
            else:
                p.requires_grad = False
        active_module_keys = _active_stage2_module_keywords(cfg)
        missing_active = [
            k for k in active_module_keys
            if not any(k in n and p.requires_grad for n, p in model.named_parameters())
        ]
        if missing_active:
            raise RuntimeError(
                'Stage-2 enabled module has no trainable parameters: {}. '
                'Check module parameter names and freeze whitelist.'.format(missing_active)
            )
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        print(f'Stage 2: Frozen {total_count - trainable_count:,}/{total_count:,} params. '
              f'Trainable: {trainable_count:,} ({100*trainable_count/total_count:.2f}%)')
        print(f'Stage 2 trainable keywords: {trainable_keys}')
        print('Stage 2 trainable parameters:')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print('  ' + n)
    elif use_cutr_lite and cutr_baseline and training:
        baseline_path = os.path.join(current_dir, '../../../', cutr_baseline)
        if os.path.exists(baseline_path):
            checkpoint = torch.load(baseline_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(checkpoint["net"], strict=False)
            print(f'CUTR-Lite: Loaded baseline from {cutr_baseline}')
            expected_missing = [k for k in missing if "utility_router" in k]
            unexpected_missing = [k for k in missing if "utility_router" not in k]
            print(f'  Missing keys: {len(missing)} (sample: {missing[:20]})')
            print(f'  Unexpected keys: {len(unexpected)} (sample: {unexpected[:20]})')
            if unexpected_missing or unexpected:
                raise RuntimeError(
                    'CUTR-Lite baseline/config mismatch. unexpected_missing={}, unexpected={}'.format(
                        unexpected_missing[:20], unexpected[:20]
                    )
                )
            if not expected_missing:
                raise RuntimeError(
                    'CUTR-Lite baseline load found no missing utility_router parameters. '
                    'Check whether the baseline checkpoint already contains router weights.'
                )
        else:
            print(f'WARNING: CUTR-Lite baseline not found: {baseline_path}')

        if router_only:
            for n, p in model.named_parameters():
                p.requires_grad = ("utility_router" in n)
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_count = sum(p.numel() for p in model.parameters())
            if trainable_count == 0:
                raise RuntimeError(
                    'CUTR-Lite router-only enabled, but no utility_router parameters are trainable.'
                )
            print(f'CUTR-Lite router-only: Frozen {total_count - trainable_count:,}/{total_count:,} params. '
                  f'Trainable: {trainable_count:,} ({100*trainable_count/total_count:.4f}%)')
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
