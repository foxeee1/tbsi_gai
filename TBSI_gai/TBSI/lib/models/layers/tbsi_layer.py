"""
TBSILayer: Core cross-attention between RGB and TIR modalities.
Supports degradation-aware modulation at the per-layer level.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock


class DegradationModulator(nn.Module):
    """
    [Phase 1 Fix] Joint modality confidence estimator for TBSILayer.

    First-principles fix:
      - BEFORE: independent MLPs for RGB and TIR — violates "reliability is a
        relationship property" (a token is reliable RELATIVE to the other modality)
      - AFTER: single joint MLP on cat(rgb, tir) → [conf_v, conf_i]
        The joint representation captures cross-modal contrast, so a token with
        poor RGB but good TIR correctly gets low conf_v and high conf_i.

    Architecture:
      - Input: cat(x_v_search, x_i_search)  →  (B, N_s, 2*C)
      - Joint MLP: Linear(2C → rdim) → ReLU → Linear(rdim → 2)
      - Output split: [conf_v, conf_i], each (B, N_s, 1) in [0,1]
    """
    def __init__(self, dim, reduction=4, temporal_dim=None):
        super().__init__()
        rdim = max(dim // reduction, 16)
        input_dim = dim * 2  # always joint: cat(rgb, tir)
        self.use_temporal = temporal_dim is not None
        if self.use_temporal:
            input_dim = dim * 2 + temporal_dim  # cat(rgb, tir, temporal)

        self.conf_joint = nn.Sequential(
            nn.Linear(input_dim, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, 2),  # 2 outputs: [conf_v, conf_i]
            nn.Sigmoid(),
        )
        # Zero-init last layer: initial sigmoid(0) ≈ 0.5 → uniform confidence
        nn.init.zeros_(self.conf_joint[-2].weight)
        nn.init.zeros_(self.conf_joint[-2].bias)

    def forward(self, x_v_search, x_i_search, temporal_tokens=None):
        """
        Args:
            x_v_search: (B, N_s, C) visible search tokens
            x_i_search: (B, N_s, C) infrared search tokens
        Returns:
            conf_v, conf_i: each (B, N_s, 1) confidence in [0,1]
        """
        if self.use_temporal and temporal_tokens is not None:
            t = temporal_tokens.mean(dim=1, keepdim=True).expand(-1, x_v_search.shape[1], -1)
            joint_inp = torch.cat([x_v_search, x_i_search, t], dim=-1)
        else:
            joint_inp = torch.cat([x_v_search, x_i_search], dim=-1)

        conf_joint = self.conf_joint(joint_inp)  # (B, N_s, 2)
        conf_v = conf_joint[:, :, 0:1]           # (B, N_s, 1)
        conf_i = conf_joint[:, :, 1:2]           # (B, N_s, 1)
        return conf_v, conf_i


class TBSILayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_degradation=False,
                 use_attn_gate=False, use_temporal_tokens=False):
        super().__init__()

        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.ca_s2t_v2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )
        self.ca_t2s_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )
        self.ca_s2t_i2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )
        self.ca_t2s_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )
        self.ca_t2t_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )
        self.ca_t2t_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )

        self.use_degradation = use_degradation
        if use_degradation:
            # Only enable temporal context in DM when temporal tokens are actually running
            temporal_dim = dim if use_temporal_tokens else None
            self.degradation_mod = DegradationModulator(dim, temporal_dim=temporal_dim)

    def forward(self, x_v, x_i, lens_z, temporal_tokens=None):
        # x_v: [B, N, C], N = 320 (64 template + 256 search)
        # x_i: [B, N, C]
        fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2)
        fused_t = self.t_fusion(fused_t)  # [B, 64, C]

        if self.use_degradation:
            # Per-patch confidence for search regions (with optional temporal context)
            conf_v, conf_i = self.degradation_mod(x_v[:, lens_z:, :], x_i[:, lens_z:, :],
                                                  temporal_tokens=temporal_tokens)
            # conf_v: (B, N_s, 1) high where RGB reliable
            # conf_i: (B, N_s, 1) high where TIR reliable
            # qm_v/qm_i: quality masks for cross-attention modulation
            qm_v = conf_v  # visible quality → modulate visible-related CA
            qm_i = conf_i  # infrared quality → modulate infrared-related CA

        # Search-to-Template: infrared search → fused template (use ir quality mask)
        fused_t = self.ca_s2t_i2f(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1),
                                  quality_mask=qm_i if self.use_degradation else None)[:, :lens_z, :]

        # Template-to-Search: fused template → visible search (use vis quality mask)
        temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1),
                                   quality_mask=qm_v if self.use_degradation else None)[:, lens_z:, :]

        # Search-to-Template: visible search → fused template (use vis quality mask)
        fused_t = self.ca_s2t_v2f(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1),
                                  quality_mask=qm_v if self.use_degradation else None)[:, :lens_z, :]

        # Template-to-Search: fused template → infrared search (use ir quality mask)
        temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1),
                                   quality_mask=qm_i if self.use_degradation else None)[:, lens_z:, :]

        # Apply degradation-aware gating if enabled (实验1)
        if self.use_degradation:
            x_v = torch.cat([x_v[:, :lens_z, :], temp_x_v * (1 - conf_v) + x_v[:, lens_z:, :] * conf_v], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], temp_x_i * (1 - conf_i) + x_i[:, lens_z:, :] * conf_i], dim=1)
        else:
            x_v = torch.cat([x_v[:, :lens_z, :], temp_x_v], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], temp_x_i], dim=1)

        # Template self-attention (no quality_mask — operates on template tokens)
        x_v[:, :lens_z, :] = self.ca_t2t_f2v(torch.cat([x_v[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]
        x_i[:, :lens_z, :] = self.ca_t2t_f2i(torch.cat([x_i[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]

        # Return quality signal for downstream fusion (avg confidence per modality)
        if self.use_degradation:
            q_global = torch.cat([conf_v.mean(dim=1), conf_i.mean(dim=1)], dim=-1)  # (B, 2)
        else:
            q_global = None
        return x_v, x_i, q_global
