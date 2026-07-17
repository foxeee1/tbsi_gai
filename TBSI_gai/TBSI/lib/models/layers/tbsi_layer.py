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
    DGSFusion: Divergence-Gated Specialized Fusion — per-token routing router.
    v2: grouped statistics for fine-grained degradation sensing.

    Key improvement over v1:
      v1: collapse 768 channels → 1 std (loses frequency info → MB broken)
      v2: split 768 channels into G=8 groups → 8 independent stds
          → motion blur kills high-freq groups, leaves low-freq → distinguishable

    Architecture:
      per-token divergence signature (B, N, 4*G+2):
        G groups × rgb_mean, rgb_std, tir_mean, tir_std + diff_mean + bias
      → Linear(34→8) → ReLU → Linear(8→3) → Softmax

    Params: 34*8+8 + 8*3+3 = 307 (vs v1 43, still negligible vs 86M backbone)
    """
    NUM_GROUPS = 8  # G=8 groups × 96 channels each

    def __init__(self, dim=768):
        super().__init__()
        g = self.NUM_GROUPS
        self.router = nn.Sequential(
            nn.Linear(4 * g + 2, g),       # 34→8
            nn.ReLU(inplace=True),
            nn.Linear(g, 3),                # 8→3
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    @staticmethod
    def compute_divergence(x_v_search, x_i_search, num_groups=8):
        """
        Per-token grouped divergence signature.
        Args: x_v_search, x_i_search: (B, N_s, C)
        Returns: (B, N_s, 4*num_groups+2)
        """
        C = x_v_search.shape[-1]
        assert C % num_groups == 0, f"C={C} must be divisible by num_groups={num_groups}"
        gs = C // num_groups  # group size (96 for 768/8)

        # Reshape to (B, N, G, gs) and compute per-group statistics
        v = x_v_search.reshape(*x_v_search.shape[:-1], num_groups, gs)
        i = x_i_search.reshape(*x_i_search.shape[:-1], num_groups, gs)

        rgb_mean = v.mean(dim=-1)  # (B, N, G)
        rgb_std = v.std(dim=-1)
        tir_mean = i.mean(dim=-1)
        tir_std = i.std(dim=-1)

        diff = (x_v_search - x_i_search).abs()
        diff_mean = diff.mean(dim=-1, keepdim=True)  # (B, N, 1)
        bias = (x_v_search.mean(dim=-1) - x_i_search.mean(dim=-1)).abs().unsqueeze(-1)

        return torch.cat([rgb_mean, rgb_std, tir_mean, tir_std, diff_mean, bias], dim=-1)

    def forward(self, x_v_search, x_i_search):
        d = self.compute_divergence(x_v_search, x_i_search, self.NUM_GROUPS)
        logits = self.router(d)
        routing = F.softmax(logits, dim=-1)
        return routing


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
            print(f"  [DGSFusion] DivergenceRouter active (43 params)")
            self.dgs_router = DivergenceRouter(dim)

    def forward(self, x_v, x_i, lens_z, temporal_tokens=None):
        fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2)
        fused_t = self.t_fusion(fused_t)

        x_v_orig = x_v[:, lens_z:, :]
        x_i_orig = x_i[:, lens_z:, :]

        # Compute per-token quality masks for cross-attention guidance
        # DGS: divergence routing → quality masks; Deg: joint-MLP → quality masks
        qm_v = qm_i = None
        if self.use_dgs:
            routing = self.dgs_router(x_v_orig, x_i_orig)  # (B, N_s, 3)
            qm_v = (routing[:,:,1:2] + routing[:,:,2:3]).detach()  # (B, N_s, 1) RGB quality
            qm_i = (routing[:,:,0:1] + routing[:,:,2:3]).detach()  # (B, N_s, 1) TIR quality
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

        # ===== DGSFusion: Divergence-Gated Specialized Fusion =====
        if self.use_dgs:
            # 3 specialized paths (fixed mixing, 0 params)
            p0_v = 0.7 * temp_x_v + 0.3 * x_v_orig   # RGB deg: TIR dominant
            p0_i = 0.3 * temp_x_i + 0.7 * x_i_orig
            p1_v = 0.3 * temp_x_v + 0.7 * x_v_orig   # TIR deg: RGB dominant
            p1_i = 0.7 * temp_x_i + 0.3 * x_i_orig
            p2_v = 0.5 * temp_x_v + 0.5 * x_v_orig   # Consensus
            p2_i = 0.5 * temp_x_i + 0.5 * x_i_orig

            r = routing
            x_v_combined = r[:,:,0:1] * p0_v + r[:,:,1:2] * p1_v + r[:,:,2:3] * p2_v
            x_i_combined = r[:,:,0:1] * p0_i + r[:,:,1:2] * p1_i + r[:,:,2:3] * p2_i

            x_v = torch.cat([x_v[:, :lens_z, :], x_v_combined], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], x_i_combined], dim=1)

            # Quality signal for downstream: q_rgb if not in RGB-deg path
            q_rgb = (r[:,:,2] + r[:,:,1]).mean(dim=1, keepdim=True)
            q_tir = (r[:,:,2] + r[:,:,0]).mean(dim=1, keepdim=True)
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
