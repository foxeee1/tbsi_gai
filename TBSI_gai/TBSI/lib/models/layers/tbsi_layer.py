"""
TBSILayer: Core cross-attention between RGB and TIR modalities.
Supports DGSFusion — multiple router modes:
  v1: DivergenceRouter — 6D→1 + α=0.3+0.4*θ (original, 19 params)
  v2: DivergenceRouterV2 — 6D→2, free α ∈ (0,1), qm=α (v2-free, ~14 params)
  v3: CrossAttnConfidence — attention entropy, 0 params (v3-attnent)
  v4: DiffProjRouter — learnable diff projection Linear(768→16) (v4-diffproj, ~12K params)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock


# =============================================================================
# DGSFusion Router Variants
# =============================================================================

class DivergenceRouter(nn.Module):
    """
    DGSFusion v1: 6D→1 + α ∈ [0.3, 0.7] (硬编码范围).
    19 params: LayerNorm(6) + Linear(6→1).
    """
    def __init__(self, dim=768):
        super().__init__()
        self.norm = nn.LayerNorm(6)
        self.gate = nn.Linear(6, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @staticmethod
    def compute_divergence(x_v_search, x_i_search):
        rgb_mean = x_v_search.mean(dim=-1)
        rgb_std = x_v_search.std(dim=-1)
        tir_mean = x_i_search.mean(dim=-1)
        tir_std = x_i_search.std(dim=-1)
        diff = (x_v_search - x_i_search).abs()
        diff_mean = diff.mean(dim=-1)
        bias = (x_v_search.mean(dim=-1) - x_i_search.mean(dim=-1)).abs()
        return torch.stack([rgb_mean, rgb_std, tir_mean, tir_std, diff_mean, bias], dim=-1)

    def forward(self, x_v_search, x_i_search):
        d = self.compute_divergence(x_v_search, x_i_search)
        d = self.norm(d)
        θ = torch.sigmoid(self.gate(d))
        α = 0.3 + 0.4 * θ
        return α  # (B, N_s, 1)


class DivergenceRouterV2(nn.Module):
    """
    DGSFusion v2-free: 6D→2, free α ∈ (0,1), 无硬编码范围.
    核心改进:
      - Linear(6→2) 替代 Linear(6→1): RGB 和 TIR 独立置信度
      - 去掉 0.3+0.4*θ: sigmoid 自然输出 (0,1), 极端场景模型自己学
      - C1(双好): α_v=0.85, α_i=0.85; C4(双差): α_v=0.2, α_i=0.2
    ~14 params: LayerNorm(6) + Linear(6→2).
    """
    def __init__(self, dim=768):
        super().__init__()
        self.norm = nn.LayerNorm(6)
        self.gate = nn.Linear(6, 2)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @staticmethod
    def compute_divergence(x_v_search, x_i_search):
        rgb_mean = x_v_search.mean(dim=-1)
        rgb_std = x_v_search.std(dim=-1)
        tir_mean = x_i_search.mean(dim=-1)
        tir_std = x_i_search.std(dim=-1)
        diff = (x_v_search - x_i_search).abs()
        diff_mean = diff.mean(dim=-1)
        bias = (x_v_search.mean(dim=-1) - x_i_search.mean(dim=-1)).abs()
        return torch.stack([rgb_mean, rgb_std, tir_mean, tir_std, diff_mean, bias], dim=-1)

    def forward(self, x_v_search, x_i_search):
        d = self.compute_divergence(x_v_search, x_i_search)
        d = self.norm(d)
        α = torch.sigmoid(self.gate(d))  # (B, N_s, 2), α_v, α_i ∈ (0, 1)
        return α


class CrossAttnConfidence(nn.Module):
    """
    DGSFusion v3-attnent: 零参数, 交叉注意力熵 → 模态置信度.

    原理:
      Step 1: RGB query → TIR key 的交叉注意力 (B,256,256)
      Step 2: 计算注意力熵:
        熵高(≈log256) → 均匀关注 → 模态一致(处处都有对应) → 高置信度
        熵低(≈0)     → 集中在少量token → 模态差异大 → 低置信度
      Step 3: 归一化到 [0,1]: α = 1 - entropy / log(256)

    0 额外参数, 与 TBSI 架构天然契合 (cross-attention 为核心运算).
    """
    def forward(self, x_v_search, x_i_search):
        scale = x_v_search.shape[-1] ** 0.5  # √768
        # Cross-attention: RGB query → TIR key
        attn_v2i = F.softmax(x_v_search @ x_i_search.transpose(-2, -1) / scale, dim=-1)
        attn_i2v = F.softmax(x_i_search @ x_v_search.transpose(-2, -1) / scale, dim=-1)

        # Entropy: -sum(p * log(p)), 高 = 模态一致
        entropy_v = -(attn_v2i * (attn_v2i + 1e-8).log()).sum(dim=-1)  # (B, N_s)
        entropy_i = -(attn_i2v * (attn_i2v + 1e-8).log()).sum(dim=-1)

        # Normalize to [0,1]: 低熵→高置信度
        max_ent = math.log(x_v_search.shape[1])  # log(256) ≈ 5.55
        α_v = 1.0 - (entropy_v / max_ent).clamp(0, 1)  # (B, N_s)
        α_i = 1.0 - (entropy_i / max_ent).clamp(0, 1)

        return torch.stack([α_v, α_i], dim=-1)  # (B, N_s, 2)


class DiffProjRouter(nn.Module):
    """
    DGSFusion v4-diffproj: 可学习差异投影, 不假设差异可被6个统计量捕获.

    用 Linear(768→16) 替代 6D 手工统计量:
      - 输入: 原始跨模态差异 (x_v - x_i) ∈ ℝ^768
      - 输出: 低维差异表征 → Linear(16→2) → sigmoid → α_v, α_i

    ~12.3K params: Linear(768,16) + LayerNorm(16) + Linear(16,2).
    """
    def __init__(self, dim=768):
        super().__init__()
        self.diff_proj = nn.Sequential(
            nn.Linear(dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.gate = nn.Linear(16, 2)
        # Zero-init for safe start (α ≈ 0.5 at init)
        nn.init.zeros_(self.diff_proj[0].weight)
        nn.init.zeros_(self.diff_proj[0].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x_v_search, x_i_search):
        diff = x_v_search - x_i_search  # (B, N_s, 768)
        d = self.diff_proj(diff)        # (B, N_s, 16)
        α = torch.sigmoid(self.gate(d))  # (B, N_s, 2)
        return α


class DegradationModulator(nn.Module):
    """Joint modality confidence estimator (kept for backward compat)."""
    def __init__(self, dim, reduction=4, temporal_dim=None):
        super().__init__()
        rdim = max(dim // reduction, 16)
        input_dim = dim * 2
        self.use_temporal = temporal_dim is not None
        if self.use_temporal:
            input_dim = dim * 2 + temporal_dim
        self.conf_joint = nn.Sequential(
            nn.Linear(input_dim, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, 2),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.conf_joint[-2].weight)
        nn.init.zeros_(self.conf_joint[-2].bias)

    def forward(self, x_v_search, x_i_search, temporal_tokens=None):
        if self.use_temporal and temporal_tokens is not None:
            t = temporal_tokens.mean(dim=1, keepdim=True).expand(-1, x_v_search.shape[1], -1)
            joint_inp = torch.cat([x_v_search, x_i_search, t], dim=-1)
        else:
            joint_inp = torch.cat([x_v_search, x_i_search], dim=-1)
        conf_joint = self.conf_joint(joint_inp)
        return conf_joint[:, :, 0:1], conf_joint[:, :, 1:2]


class TBSILayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_degradation=False,
                 use_attn_gate=False, use_temporal_tokens=False, use_dgs=False, dgs_mode="v1"):
        super().__init__()
        self.use_dgs = use_dgs
        self.dgs_mode = dgs_mode

        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.ca_s2t_v2f = CASTBlock(dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, act_layer=act_layer, use_attn_gate=use_attn_gate)
        self.ca_t2s_f2i = CASTBlock(dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, act_layer=act_layer, use_attn_gate=use_attn_gate)
        self.ca_s2t_i2f = CASTBlock(dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, act_layer=act_layer, use_attn_gate=use_attn_gate)
        self.ca_t2s_f2v = CASTBlock(dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, act_layer=act_layer, use_attn_gate=use_attn_gate)
        self.ca_t2t_f2v = CASTBlock(dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, act_layer=act_layer, use_attn_gate=use_attn_gate)
        self.ca_t2t_f2i = CASTBlock(dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, act_layer=act_layer, use_attn_gate=use_attn_gate)

        self.use_degradation = use_degradation
        if use_degradation:
            temporal_dim = dim if use_temporal_tokens else None
            self.degradation_mod = DegradationModulator(dim, temporal_dim=temporal_dim)
        if use_dgs:
            if dgs_mode == "v1":
                self.dgs_router = DivergenceRouter(dim)
            elif dgs_mode == "v2":
                self.dgs_router = DivergenceRouterV2(dim)
            elif dgs_mode == "v3":
                self.dgs_router = CrossAttnConfidence()
            elif dgs_mode == "v4":
                self.dgs_router = DiffProjRouter(dim)
            else:
                raise ValueError(f"Unknown DGS_MODE: {dgs_mode}")
            rp = sum(p.numel() for p in self.dgs_router.parameters())
            print(f"  [DGSFusion] Router active (mode={dgs_mode}, {rp} params)")

    def forward(self, x_v, x_i, lens_z, temporal_tokens=None):
        fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2)
        fused_t = self.t_fusion(fused_t)

        x_v_orig = x_v[:, lens_z:, :]
        x_i_orig = x_i[:, lens_z:, :]

        # ===== Compute per-token quality masks (for CASTBlocks) =====
        qm_v = qm_i = None
        if self.use_dgs:
            α = self.dgs_router(x_v_orig, x_i_orig)  # (B, N_s, 1) for v1, (B, N_s, 2) for v2-v4
            if self.dgs_mode == "v1":
                # v1: quality mask from hardcoded α range
                qm_v = (α - 0.3).sigmoid().clamp(0.1, 0.9) / 0.8
                qm_i = (0.7 - α).sigmoid().clamp(0.1, 0.9) / 0.8
            else:
                # v2/v3/v4: qm = α directly (no hardcoded mapping)
                α_v = α[:, :, 0:1]
                α_i = α[:, :, 1:2]
                qm_v = α_v
                qm_i = α_i
        elif self.use_degradation:
            conf_v, conf_i = self.degradation_mod(x_v_orig, x_i_orig, temporal_tokens=temporal_tokens)
            qm_v, qm_i = conf_v, conf_i

        # 4 CASTBlocks (quality-guided cross-attention)
        fused_t = self.ca_s2t_i2f(torch.cat([fused_t, x_i_orig], dim=1),
                                  quality_mask=qm_i)[:, :lens_z, :]
        temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t, x_v_orig], dim=1),
                                   quality_mask=qm_v)[:, lens_z:, :]
        fused_t = self.ca_s2t_v2f(torch.cat([fused_t, x_v_orig], dim=1),
                                  quality_mask=qm_v)[:, :lens_z, :]
        temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t, x_i_orig], dim=1),
                                   quality_mask=qm_i)[:, lens_z:, :]

        # ===== Apply routing/gate to combine cross-attn output with original =====
        if self.use_dgs:
            if self.dgs_mode == "v1":
                # v1: single symmetric α, hardcoded range [0.3, 0.7]
                x_v_combined = α * temp_x_v + (1 - α) * x_v_orig
                x_i_combined = (1 - α) * temp_x_i + α * x_i_orig
                q_rgb = (α - 0.3).sigmoid().mean(dim=1)
                q_tir = (0.7 - α).sigmoid().mean(dim=1)
                q_global = torch.cat([q_rgb, q_tir], dim=-1)
            else:
                # v2/v3/v4: independent α_v, α_i, free range (0,1)
                α_v = α[:, :, 0:1]
                α_i = α[:, :, 1:2]
                x_v_combined = α_v * temp_x_v + (1 - α_v) * x_v_orig
                x_i_combined = α_i * temp_x_i + (1 - α_i) * x_i_orig
                q_global = torch.cat([α_v.mean(dim=1), α_i.mean(dim=1)], dim=-1)
            x_v = torch.cat([x_v[:, :lens_z, :], x_v_combined], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], x_i_combined], dim=1)
        elif self.use_degradation:
            x_v = torch.cat([x_v[:, :lens_z, :],
                             temp_x_v * (1 - conf_v) + x_v_orig * conf_v], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :],
                             temp_x_i * (1 - conf_i) + x_i_orig * conf_i], dim=1)
            q_global = torch.cat([conf_v.mean(dim=1), conf_i.mean(dim=1)], dim=-1)
        else:
            x_v = torch.cat([x_v[:, :lens_z, :], temp_x_v], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], temp_x_i], dim=1)
            q_global = None

        # Template self-attention
        x_v[:, :lens_z, :] = self.ca_t2t_f2v(
            torch.cat([x_v[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]
        x_i[:, :lens_z, :] = self.ca_t2t_f2i(
            torch.cat([x_i[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]

        return x_v, x_i, q_global
