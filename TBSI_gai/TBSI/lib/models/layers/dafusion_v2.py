"""
DaFusion v2 — True Degradation-Aware Fusion Module

Design:
  1. DegradationDescriptor (6D → per-channel quality vector)
  2. Per-modality quality scalars (6D → 2: global modality confidence)
  3. Per-channel quality modulation (6D → 2*C: channel-level modality trust)
  4. Quality modulates both spatial & channel gates

Key differences from v2-Lite:
  - Quality outputs: (B,2) scalars + (B,2*C) per-channel weights
  - quality_proj(6D → 8 → 2*C) conditions channel_gate output
  - More expressive degradation→fusion pathway (~12K params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationDescriptor(nn.Module):
    """
    6D degradation fingerprint:
      [rgb_mean, rgb_std, tir_mean, tir_std, diff_mean, bias]

    Outputs:
      quality:      (B, 2)    — per-modality global quality scalars
      channel_mod:  (B, 2*C)  — per-channel modality trust weights

    Params: ~12K (6→8→2 + 6→8→2*C)
    """
    def __init__(self, dim=768):
        super().__init__()
        # Global quality scalars (proven in v2-Lite)
        self.quality_net = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2),
        )
        # Per-channel quality modulation (NEW)
        self.channel_net = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, dim * 2),
        )
        nn.init.zeros_(self.quality_net[-1].weight)
        nn.init.zeros_(self.quality_net[-1].bias)
        nn.init.zeros_(self.channel_net[-1].weight)
        nn.init.zeros_(self.channel_net[-1].bias)

    def compute_fingerprint(self, rgb_feat, tir_feat):
        """Extract 6D degradation fingerprint."""
        B, C, H, W = rgb_feat.shape
        rgb_mean = rgb_feat.mean(dim=[2, 3])
        rgb_std = rgb_feat.std(dim=[2, 3])
        tir_mean = tir_feat.mean(dim=[2, 3])
        tir_std = tir_feat.std(dim=[2, 3])
        diff = (rgb_feat - tir_feat).abs()
        diff_mean = diff.mean(dim=[2, 3])

        desc = torch.stack([
            rgb_mean.mean(dim=1),
            rgb_std.mean(dim=1),
            tir_mean.mean(dim=1),
            tir_std.mean(dim=1),
            diff_mean.mean(dim=1),
            (rgb_mean - tir_mean).abs().mean(dim=1),
        ], dim=1)
        return desc

    def forward(self, rgb_feat, tir_feat):
        desc = self.compute_fingerprint(rgb_feat, tir_feat)
        quality = torch.sigmoid(self.quality_net(desc))
        channel_mod = torch.tanh(self.channel_net(desc)) * 0.1  # [-0.1, 0.1]
        return quality, channel_mod


class DaFusionV2(nn.Module):
    """
    DaFusionV2 — Degradation-Gated Fusion (DGF).

    Key insight: degradation descriptor output is injected DIRECTLY into
    the spatial gate input, making per-pixel weights degradation-aware.

    Architecture:
      1. DegradationDescriptor extracts 6D fingerprint
      2. Spatial gate sees: cat_feat (2C) + deg_feat (1) → per-pixel weights
      3. Quality scalars modulate per-modality confidence
      4. Channel gate (SENet), MADC, CSR (from v1, proven effective)

    Degradation-augmented spatial gate:
      - concat(cat_feat, deg_feat_broadcast) as input
      - deg_feat = 6D → Linear(8) → ReLU → Linear(1) as spatial bias
      - GW + Degradation signal = per-pixel weights that know about global quality
    """

    def __init__(self, dim=768, reduction=16, da_mode='channel',
                 use_madc=False, use_csr=False):
        super().__init__()
        rdim = max(dim // reduction, 16)
        self.da_mode = da_mode
        self.use_madc = use_madc
        self.use_csr = use_csr

        # ===== 1. Degradation descriptor (6D → 2 quality scalars) =====
        self.deg_desc = DegradationDescriptor()

        # ===== 2. Degradation projection → spatial gate bias =====
        # Projects 6D fingerprint → 1D spatial bias → broadcast to (H,W)
        self.deg_spatial_proj = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(self.deg_spatial_proj[-1].weight)
        nn.init.zeros_(self.deg_spatial_proj[-1].bias)

        # ===== 3. Degradation-augmented spatial gate =====
        # Input: cat_feat (2C) + deg_feat (1 concat) = 2C+1
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2 + 1, rdim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(rdim),
            nn.ReLU(inplace=True),
            nn.Conv2d(rdim, 2, kernel_size=1, padding=0, bias=True),
        )

        # ===== 4. Channel gate =====
        if da_mode == 'channel':
            self.channel_gate = nn.Sequential(
                nn.Conv2d(dim * 2, rdim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(rdim, dim * 2, kernel_size=1, padding=0, bias=True),
            )
            nn.init.zeros_(self.channel_gate[-1].weight)
            nn.init.zeros_(self.channel_gate[-1].bias)

        # ===== 5. CSR =====
        if use_csr:
            rdim_csr = max(dim // reduction, 16)
            self.comp_reduce = nn.Conv2d(dim, rdim_csr, kernel_size=1, bias=True)
            self.comp_channel = nn.Conv2d(rdim_csr, dim, kernel_size=1, bias=True)
            nn.init.zeros_(self.comp_channel.weight)
            nn.init.zeros_(self.comp_channel.bias)

        total = sum(p.numel() for p in self.parameters())
        print(f"  [DaFusionV2-DGF] Deg-in-spatial, total={total/1e3:.1f}K")

    def forward(self, feat_rgb, feat_tir, quality_hint=None):
        B, C, H, W = feat_rgb.shape
        cat_feat = torch.cat([feat_rgb, feat_tir], dim=1)  # (B, 2C, H, W)
        pooled = cat_feat.mean(dim=[2, 3], keepdim=True)

        # (1) Degradation → quality scalars
        quality, _ = self.deg_desc(feat_rgb, feat_tir) if hasattr(self.deg_desc, 'channel_net') \
                     else (self.deg_desc(feat_rgb, feat_tir), None)
        if quality_hint is not None:
            quality = quality * quality_hint
        q_rgb = quality[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        q_tir = quality[:, 1:2].unsqueeze(-1).unsqueeze(-1)

        # (2) Degradation → spatial bias
        desc = self.deg_desc.compute_fingerprint(feat_rgb, feat_tir)  # (B, 6)
        deg_bias = self.deg_spatial_proj(desc)  # (B, 1)
        deg_bias = deg_bias.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        deg_bias = torch.tanh(deg_bias) * 0.5  # [-0.5, 0.5]

        # (3) Degradation-augmented spatial gate
        deg_map = deg_bias.expand(-1, -1, H, W)  # (B, 1, H, W)
        gate_input = torch.cat([cat_feat, deg_map], dim=1)  # (B, 2C+1, H, W)
        s = torch.sigmoid(self.spatial_gate(gate_input))
        s_rgb, s_tir = s[:, 0:1], s[:, 1:2]

        # (4) MADC
        if self.use_madc:
            mag_rgb = feat_rgb.abs().mean(dim=[1,2,3], keepdim=True) + 1e-6
            mag_tir = feat_tir.abs().mean(dim=[1,2,3], keepdim=True) + 1e-6
            s_rgb = s_rgb * (1.0 / mag_rgb).pow(0.5)
            s_tir = s_tir * (1.0 / mag_tir).pow(0.5)

        # (5) Fusion weights = quality × spatial
        w_rgb = s_rgb * q_rgb
        w_tir = s_tir * q_tir

        # (6) Channel modulation
        if self.da_mode == 'channel':
            c = torch.sigmoid(self.channel_gate(pooled))
            w_rgb = w_rgb * c[:, :C]
            w_tir = w_tir * c[:, C:]

        # (7) Normalize & fuse
        total = w_rgb + w_tir + 1e-8
        fused = feat_rgb * (w_rgb / total) + feat_tir * (w_tir / total)

        # (8) CSR
        if self.use_csr:
            feat_diff = feat_rgb - feat_tir
            diff_feat = F.relu(self.comp_reduce(feat_diff))
            cg = torch.tanh(self.comp_channel(diff_feat))
            cg = torch.clamp(cg, -0.5, 0.5)
            fused = fused * (1.0 + 0.1 * cg)

        return fused
