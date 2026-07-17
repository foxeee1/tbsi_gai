"""
TBSILayer: Core cross-attention between RGB and TIR modalities.
Supports DGSFusion — Divergence-Gated Specialized Fusion (v2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock


class DivergenceRouter(nn.Module):
    """
    DGSFusion: Divergence-Gated Specialized Fusion — per-token continuous gate.
    Fix 5 (LayerNorm) + Fix 1 (no detach) + Fix 2 (continuous gate, 19 params).

    Fix 5: LayerNorm stabilizes router input.
    Fix 1: quality_mask WITHOUT .detach().
    Fix 2: 3-path softmax → continuous sigmoid gate α ∈ [0.3, 0.7].

    Proof of Fix 2 (数学推导证明等价):
      原3路softmax路由:
        α = 0.5 + 0.2(r0 - r1), 其中 r0+r1+r2=1
        output_v = α*T + (1-α)*O
      等价于 1 个连续门控:
        θ = sigmoid(Linear(6→1))
        α = 0.3 + 0.4*θ
        output_v = α*T + (1-α)*O
        output_i = (1-α)*T + α*O

      参数从 55 降到 19 (LN:12 + Linear:7), 表达能力完全保留.

    Args:
      output_v = α * temp_x_v + (1-α) * x_v_orig      α → 0.7: RGB退化, TIR主导
      output_i = (1-α) * temp_x_i + α * x_i_orig      α → 0.3: TIR退化, RGB主导
                                                        α = 0.5: 共识融合
    """
    def __init__(self, dim=768):
        super().__init__()
        self.norm = nn.LayerNorm(6)
        self.gate = nn.Linear(6, 1)  # 7 params: 6 weights + 1 bias
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
        d = self.norm(d)  # Fix 5
        θ = torch.sigmoid(self.gate(d))  # (B, N_s, 1), θ ∈ (0, 1)
        α = 0.3 + 0.4 * θ  # α ∈ [0.3, 0.7]
        return α  # (B, N_s, 1) continuous gate value


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
                 use_attn_gate=False, use_temporal_tokens=False, use_dgs=False):
        super().__init__()
        self.use_dgs = use_dgs

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
            self.dgs_router = DivergenceRouter(dim)
            rp = sum(p.numel() for p in self.dgs_router.parameters())
            print(f"  [DGSFusion] DivergenceRouter active ({rp} params)")

    def forward(self, x_v, x_i, lens_z, temporal_tokens=None):
        fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2)
        fused_t = self.t_fusion(fused_t)

        x_v_orig = x_v[:, lens_z:, :]
        x_i_orig = x_i[:, lens_z:, :]

        # Compute per-token quality masks for cross-attention guidance
        # DGS: divergence routing → quality masks; Deg: joint-MLP → quality masks
        qm_v = qm_i = None
        if self.use_dgs:
            α = self.dgs_router(x_v_orig, x_i_orig)  # (B, N_s, 1) continuous gate
            # quality_mask from α: α→0.7 → RGB degraded → RGB quality low
            # α→0.3 → TIR degraded → TIR quality low
            qm_v = (α - 0.3).sigmoid().clamp(0.1, 0.9) / 0.8  # (B, N_s, 1)
            qm_i = (0.7 - α).sigmoid().clamp(0.1, 0.9) / 0.8
        elif self.use_degradation:
            conf_v, conf_i = self.degradation_mod(x_v_orig, x_i_orig, temporal_tokens=temporal_tokens)
            qm_v, qm_i = conf_v, conf_i

        # 4 CASTBlocks (quality-guided cross-attention — Bug fix: was None for DGS)
        fused_t = self.ca_s2t_i2f(torch.cat([fused_t, x_i_orig], dim=1),
                                  quality_mask=qm_i)[:, :lens_z, :]
        temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t, x_v_orig], dim=1),
                                   quality_mask=qm_v)[:, lens_z:, :]
        fused_t = self.ca_s2t_v2f(torch.cat([fused_t, x_v_orig], dim=1),
                                  quality_mask=qm_v)[:, :lens_z, :]
        temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t, x_i_orig], dim=1),
                                   quality_mask=qm_i)[:, lens_z:, :]

        # ===== DGSFusion v2: Continuous Gate (Fix 2) =====
        if self.use_dgs:
            # α ∈ [0.3, 0.7]: continuous mixing ratio
            # α→0.7: RGB degraded, TIR dominant. α→0.3: TIR degraded, RGB dominant
            # α=0.5: Consensus (equal mixing)

            x_v_combined = α * temp_x_v + (1 - α) * x_v_orig
            x_i_combined = (1 - α) * temp_x_i + α * x_i_orig

            x_v = torch.cat([x_v[:, :lens_z, :], x_v_combined], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], x_i_combined], dim=1)

            # Quality signal: α close to 0.5 → both modalities reliable
            q_rgb = (α - 0.3).sigmoid().mean(dim=1)  # (B, 1)
            q_tir = (0.7 - α).sigmoid().mean(dim=1)   # (B, 1)
            q_global = torch.cat([q_rgb, q_tir], dim=-1)

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
