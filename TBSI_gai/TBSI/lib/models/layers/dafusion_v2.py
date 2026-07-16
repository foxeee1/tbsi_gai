"""
DaFusion v2 — True Degradation-Aware Fusion Module (Lightweight)

Design philosophy (from authoritative reference guide):
  1. Information bottleneck (SENet):  6D descriptor → 8D hidden → 2D quality
  2. Multi-dim decomposition (CBAM): spatial + channel + degradation dims
  3. Zero-init (Fixup/LayerScale): stable training from identity

v1 root cause fixed:
  quality_conv used cat_feat.mean(dim=[2,3]) — global avg pooling
  → Can't distinguish low-light from thermal-cross from motion blur
  → ViT LayerNorm normalizes activations, so mean pooling loses all signal

v2 approach: 6D degradation fingerprint from per-modality statistics
  - mean: overall intensity level
  - std: texture richness (low in blur/degradation)
  - diff_mean: modality disagreement (high in thermal-cross)
  - bias: channel-wise offset (high when modalities systematically differ)

  Low-light:  rgb_mean↓, rgb_std↓, diff_mean ≈ normal
  Thermal-X:  tir_std↓, diff_mean↑, bias↑
  Motion blur: both std↓, both mean ≈ normal
  Background clutter: diff_mean↑ (localized modality disagreement)

Params: 74 (vs v1 quality_conv 73.9K — 1000x reduction, more expressive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationDescriptor(nn.Module):
    """
    6D degradation fingerprint → 2D per-modality quality score.
    Input:  (B, 6) statistics: [rgb_mean, rgb_std, tir_mean, tir_std, diff_mean, bias]
    Output: (B, 2) quality:    [q_rgb, q_tir] in [0, 1]
    Params: 74
    """
    def __init__(self):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor):
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
        return torch.sigmoid(self.predictor(desc))


class DaFusionV2(nn.Module):
    """
    Degradation-Aware Fusion v2 — Lightweight.

    Replaces v1 quality_conv (73.9K, global pool + 2×Conv1x1)
      → DegradationDescriptor (74 params, 6D fingerprint + MLP)

    Keeps v1's proven components: spatial_gate, channel_gate, MADC, CSR.
    Removes over-engineered prototype routing (was 113K, overfit on 25 seqs).
    """

    def __init__(self, dim=768, reduction=16, da_mode='channel',
                 use_madc=False, use_csr=False):
        super().__init__()
        rdim = max(dim // reduction, 16)
        self.da_mode = da_mode
        self.use_madc = use_madc
        self.use_csr = use_csr

        # ===== 1. Lightweight degradation descriptor (74 params) =====
        self.quality_descriptor = DegradationDescriptor()

        # ===== 2. Base spatial gate (same as v1) =====
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2, rdim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(rdim),
            nn.ReLU(inplace=True),
            nn.Conv2d(rdim, 2, kernel_size=1, padding=0, bias=True),
        )

        # ===== 3. Base channel gate (same as v1) =====
        if da_mode == 'channel':
            self.channel_gate = nn.Sequential(
                nn.Conv2d(dim * 2, rdim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(rdim, dim * 2, kernel_size=1, padding=0, bias=True),
            )
            nn.init.zeros_(self.channel_gate[-1].weight)
            nn.init.zeros_(self.channel_gate[-1].bias)

        # ===== 4. CSR (same as v1) =====
        if use_csr:
            rdim_csr = max(dim // reduction, 16)
            self.comp_reduce = nn.Conv2d(dim, rdim_csr, kernel_size=1, bias=True)
            self.comp_channel = nn.Conv2d(rdim_csr, dim, kernel_size=1, bias=True)
            nn.init.zeros_(self.comp_channel.weight)
            nn.init.zeros_(self.comp_channel.bias)

        total = sum(p.numel() for p in self.parameters())
        desc_p = sum(p.numel() for p in self.quality_descriptor.parameters())
        print(f"  [DaFusionV2-Lite] Desc={desc_p/1e3:.2f}K, total={total/1e3:.1f}K")

    def forward(self, feat_rgb: torch.Tensor, feat_tir: torch.Tensor,
                quality_hint: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = feat_rgb.shape
        cat_feat = torch.cat([feat_rgb, feat_tir], dim=1)
        pooled = cat_feat.mean(dim=[2, 3], keepdim=True)

        # (1) Degradation-aware quality
        quality = self.quality_descriptor(feat_rgb, feat_tir)
        if quality_hint is not None:
            quality = quality * quality_hint

        q_rgb = quality[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        q_tir = quality[:, 1:2].unsqueeze(-1).unsqueeze(-1)

        # (2) Spatial gate
        s = torch.sigmoid(self.spatial_gate(cat_feat))
        s_rgb, s_tir = s[:, 0:1], s[:, 1:2]

        # (3) MADC
        if self.use_madc:
            mag_rgb = feat_rgb.abs().mean(dim=[1,2,3], keepdim=True) + 1e-6
            mag_tir = feat_tir.abs().mean(dim=[1,2,3], keepdim=True) + 1e-6
            s_rgb = s_rgb * (1.0 / mag_rgb).pow(0.5)
            s_tir = s_tir * (1.0 / mag_tir).pow(0.5)

        # (4) Fusion weights = quality × spatial
        w_rgb = s_rgb * q_rgb
        w_tir = s_tir * q_tir

        # (5) Channel modulation
        if self.da_mode == 'channel':
            c = torch.sigmoid(self.channel_gate(pooled))
            w_rgb = w_rgb * c[:, :C]
            w_tir = w_tir * c[:, C:]

        # (6) Normalize & fuse
        total = w_rgb + w_tir + 1e-8
        fused = feat_rgb * (w_rgb / total) + feat_tir * (w_tir / total)

        # (7) CSR
        if self.use_csr:
            feat_diff = feat_rgb - feat_tir
            diff_feat = F.relu(self.comp_reduce(feat_diff))
            cg = torch.tanh(self.comp_channel(diff_feat))
            cg = torch.clamp(cg, -0.5, 0.5)
            fused = fused * (1.0 + 0.1 * cg)

        return fused
