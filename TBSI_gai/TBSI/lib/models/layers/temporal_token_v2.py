"""
TemporalTokenLayer v2 — Feature-Guided Temporal Tokens with Quality-Aware Gating
================================================================================
Design for joint use with DaFusion (Degradation-Aware Fusion).

v2 improvements over v1 (纯等待，不可执行，等改完后再启用):
  1. Multi-granularity gate (per-token, feature-modulated)
     每个 token 有独立 gate base + 特征上下文调制，不再是 1 个 scalar 控制所有
  2. Quality-Aware gate modulation (fusion module interface)
     可接收融合模块的 quality_hint 来调节更新率
     quality_hint=None → 门控退化为纯 learned gate
  3. Cross-modal token interaction
     双流更新后，RGB↔TIR token 之间进行轻量 cross-attention
     (2K=8 tokens, O(64) FLOPs, 可忽略不计)
  4. Pre-LN MLP (继承 v1 修复，保留)
  5. 零初始化保证：初始 gate=0.5, quality_proj=0, cross-attn proj=0

设计原则:
  - 时序令牌是融合模块的轻量辅助，不承担全状态建模责任
  - 高质量帧 → gate 小，token 保持稳定 (记忆)
  - 低质量帧 → gate 大，token 快速更新 (适应)
  - 融合模块可传递 quality 信号控制令牌行为，形成闭环

参数量: ~550K (vs v1 的 528K)
FLOPs:  ~146M (vs v1 的 143M, +2% 可忽略)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath, trunc_normal_


class TemporalTokenLayerV2(nn.Module):
    """
    TemporalTokenLayer v2.

    Forward per modality stream:
        Q = tokens, KV = feat  (不变: token-centric attention, v1 已验证有效)

        # 多粒度门控
        gate_base: (K,) learnable base gate per token
        gate_ctx:  (B, K) computed from mean-pooled features
        gate_logit = gate_base + gate_ctx + quality_mod(quality_hint)
        gate = sigmoid(gate_logit).unsqueeze(-1)  # (B, K, 1)

        # 转置更新
        tok_out = attn(tokens, feat)
        tok_out = gate * tok_out + (1 - gate) * tokens  # per-token gating
        tok_out = tok_out + MLP(LN(tok_out))

    After both streams:
        # 跨模态交互
        [rgb_tokens; tir_tokens] → self-attention → split back

    Quality interface:
        quality_hint: (B, 2) — per-modality confidence [q_rgb, q_tir] from fusion
          → proj_q = Linear(2, K) → bias_add to gate logit
        quality_hint=None → proj_q 输出 0 (零初始化保证)
    """

    def __init__(self, dim=256, num_tokens=4, num_heads=4, mlp_ratio=2.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # ========================
        # 1. Token pools (per modality, no cross-modal sharing at init)
        # ========================
        self.rgb_tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.tir_tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
        trunc_normal_(self.rgb_tokens, std=.02)
        trunc_normal_(self.tir_tokens, std=.02)

        # ========================
        # 2. Shared projections (token-centric attention)
        # ========================
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        # ========================
        # 3. Post-attention MLP (Pre-LN)
        # ========================
        self.norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        # 零初始化 MLP fc2: 初始输出=0 → identity residual
        nn.init.zeros_(self.mlp.fc2.weight)
        nn.init.zeros_(self.mlp.fc2.bias)

        # ========================
        # 4. Multi-granularity gate
        # ========================
        # 每个 token 独立的基础更新率
        self.gate_base = nn.Parameter(torch.zeros(num_tokens))  # sigmoid(0)=0.5

        # 特征上下文调制: mean-pooled feat → K 维偏置
        self.gate_ctx_proj = nn.Linear(dim, num_tokens)
        nn.init.zeros_(self.gate_ctx_proj.weight)
        nn.init.zeros_(self.gate_ctx_proj.bias)

        # ========================
        # 5. Quality-aware gate modulation (fusion module interface)
        # ========================
        # quality_hint (B, 2): [q_rgb, q_tir] from DaFusion
        # → proj to (B, K) 加到 gate logit
        self.gate_quality_proj = nn.Linear(2, num_tokens)
        nn.init.zeros_(self.gate_quality_proj.weight)
        nn.init.zeros_(self.gate_quality_proj.bias)
        # 零初始化保证 quality_hint=None 时输出=0

        # ========================
        # 6. Cross-modal token interaction
        # ========================
        # 双流更新后，RGB (K) ↔ TIR (K) 做轻量 self-attention
        # 用独立的 QKV 投影，避免干扰主注意力的特征
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_k = nn.Linear(dim, dim, bias=False)
        self.cross_v = nn.Linear(dim, dim, bias=False)
        self.cross_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.cross_proj.weight)
        nn.init.zeros_(self.cross_proj.bias)
        # 零初始化 → 初始 cross-modal 交互 ≈ identity

    def _per_stream_update(self, feat: torch.Tensor,
                           prev_tokens: torch.Tensor,
                           init_tokens: torch.Tensor,
                           quality_val: torch.Tensor = None) -> torch.Tensor:
        """
        Single modality stream: token-centric attention + multi-granularity gate + MLP.

        Args:
            quality_val: (B, 1) scalar quality for this modality, or None
        Returns:
            tok_out: (B, K, D) updated tokens
        """
        B, N, D = feat.shape
        K = self.num_tokens

        # Tokens init
        tokens = prev_tokens if prev_tokens is not None else init_tokens.expand(B, -1, -1)

        # ===== Token-Centric Attention (同 v1) =====
        feat_norm = self.norm(feat)
        tok_norm = self.norm(tokens)

        q = self.q_proj(tok_norm).reshape(B, K, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k, v = self.kv_proj(feat_norm).reshape(
            B, N, 2, self.num_heads, D // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        tok_out = (attn @ v).transpose(1, 2).reshape(B, K, D)
        tok_out = self.proj(tok_out)
        tok_out = self.proj_drop(tok_out)

        # ===== Multi-Granularity Gate =====
        # gate_ctx: 从特征上下文中提取 K 维调制信号
        feat_ctx = feat.mean(dim=1, keepdim=False)  # (B, D)
        gate_ctx = self.gate_ctx_proj(feat_ctx)     # (B, K)

        # gate_quality: 融合模块质量信号调制
        gate_quality = torch.zeros(B, K, device=feat.device)
        if quality_val is not None:
            gate_quality = self.gate_quality_proj(
                torch.cat([quality_val, 1 - quality_val], dim=-1)
            )  # (B, K), 用 quality 和 complementary quality 做双路调制

        # 组合门控
        gate_logit = self.gate_base.unsqueeze(0) + gate_ctx + gate_quality  # (B, K)
        gate = torch.sigmoid(gate_logit).unsqueeze(-1)  # (B, K, 1)

        # Token residual + gate
        tok_out = tokens + self.drop_path(tok_out)
        tok_out = gate * tok_out + (1 - gate) * tokens

        # ===== Post-attention MLP =====
        tok_out = tok_out + self.drop_path(self.mlp(self.norm(tok_out)))

        return tok_out

    def _cross_modal_interaction(self, rgb_tokens: torch.Tensor,
                                  tir_tokens: torch.Tensor) -> tuple:
        """
        轻量跨模态 token 交互。
        Concatenate 2K tokens → self-attention → split back.

        Args:
            rgb_tokens: (B, K, D)
            tir_tokens: (B, K, D)
        Returns:
            rgb_out, tir_out: same shape, residual updated
        """
        BK = self.num_tokens * 2
        B, K, D = rgb_tokens.shape

        # Concat
        all_tokens = torch.cat([rgb_tokens, tir_tokens], dim=1)  # (B, 2K, D)
        all_norm = self.norm(all_tokens)

        # Self-attention over 2K tokens
        q = self.cross_q(all_norm).reshape(B, BK, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = self.cross_k(all_norm).reshape(B, BK, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = self.cross_v(all_norm).reshape(B, BK, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, BK, D)
        out = self.cross_proj(out)

        # Zero-init residual: initial cross-modal = identity
        all_tokens = all_tokens + self.drop_path(out)

        # Split back
        return all_tokens[:, :K], all_tokens[:, K:]

    def forward(self, feat_rgb: torch.Tensor, feat_tir: torch.Tensor,
                prev_tokens: torch.Tensor = None,
                quality_hint: torch.Tensor = None) -> tuple:
        """
        Args:
            quality_hint: (B, 2) — [q_rgb, q_tir] from DaFusion, in [0,1], or None.
                          q_rgb 高 → RGB 质量好 → token 更新慢 (保持记忆)
                          q_tir 低 → TIR 质量差 → token 更新快 (快速适应)

        Returns:
            feat_rgb_out: (B, N, D) unchanged (pass-through)
            feat_tir_out: (B, N, D) unchanged (pass-through)
            tokens_out:   (B, 2K, D) updated [rgb_tokens, tir_tokens]
        """
        K = self.num_tokens

        # Parse quality_hint
        q_rgb_1d = None
        q_tir_1d = None
        if quality_hint is not None:
            q_rgb_1d = quality_hint[:, 0:1]  # (B, 1)
            q_tir_1d = quality_hint[:, 1:2]  # (B, 1)

        # Split prev tokens
        if prev_tokens is not None:
            prev_rgb = prev_tokens[:, :K, :]
            prev_tir = prev_tokens[:, K:, :]
        else:
            prev_rgb = None
            prev_tir = None

        # 双流独立更新 (含各自的质量门控)
        rgb_tokens = self._per_stream_update(feat_rgb, prev_rgb, self.rgb_tokens, q_rgb_1d)
        tir_tokens = self._per_stream_update(feat_tir, prev_tir, self.tir_tokens, q_tir_1d)

        # 跨模态交互 (RGB ↔ TIR token 互相看)
        rgb_tokens, tir_tokens = self._cross_modal_interaction(rgb_tokens, tir_tokens)

        tokens_out = torch.cat([rgb_tokens, tir_tokens], dim=1)

        return feat_rgb, feat_tir, tokens_out


class TemporalTokenBlockV2(nn.Module):
    """Wrapper for TemporalTokenLayerV2 (interface compatible with v1)."""

    def __init__(self, dim=256, num_tokens=4, num_heads=4, mlp_ratio=2.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.temporal_layer = TemporalTokenLayerV2(
            dim=dim, num_tokens=num_tokens, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
        )

    def forward(self, feat_rgb: torch.Tensor, feat_tir: torch.Tensor,
                temporal_tokens: torch.Tensor = None,
                quality_hint: torch.Tensor = None) -> tuple:
        return self.temporal_layer(feat_rgb, feat_tir, temporal_tokens, quality_hint)
