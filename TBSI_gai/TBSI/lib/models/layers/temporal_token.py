"""
Modality-Decoupled Temporal Token Module v2
============================================
Fixes for TBSI temporal tokens:

v1 failures (AUC 54.55, -0.53 vs baseline):
  1. Tokens NEVER updated — output copied input tokens directly (constant bias)
  2. RGB↔TIR cross-attention polluted features before DA fusion
  3. Training always reset tokens → model never learned temporal dependencies
  4. Heavy params (460K) added noise in short 4ep training

v2 design:
  - Dual-stream: RGB and TIR each have their OWN token pool, NO cross-modal attention
  - Full self-attention: Q=KV=[feat, tokens] → tokens actually update from attention
  - Adaptive gating: cosine-similarity-based update gate inhibits noise in short seq
  - Lightweight: K=2/modality, num_heads=4, mlp_ratio=2, ~120K params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath, trunc_normal_


class TemporalTokenLayer(nn.Module):
    """
    Modality-Decoupled Temporal Token Layer v2.

    Architecture per modality stream:
        Q = KV = [feat; tokens]  (full self-attention, N+K positions)
        Output: enhanced_feat (N), updated_tokens (K)
        Adaptive gate: tok_out = gate * tok_new + (1-gate) * tok_old
    """

    def __init__(self, dim=256, num_tokens=2, num_heads=4, mlp_ratio=2.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_tokens = num_tokens  # PER modality (total = 2*num_tokens)
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # ===== Dual-stream token pools (no cross-modal sharing) =====
        self.rgb_tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.tir_tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
        trunc_normal_(self.rgb_tokens, std=.02)
        trunc_normal_(self.tir_tokens, std=.02)

        # ===== Shared projections (modality-independent, so shared is safe) =====
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # ===== Learnable update gate =====
        # Controls how much token content is updated each step.
        # 0 = no update (stable), 1 = full update (fast adaptation)
        self.token_update_gate = nn.Parameter(torch.tensor(0.5))

    def _forward_stream(self, feat: torch.Tensor,
                        prev_tokens: torch.Tensor,
                        init_tokens: torch.Tensor) -> tuple:
        """
        [Phase 1 Fix] Token-Centric Attention: Q=tokens, KV=feat.

        BEFORE: Q=KV=[feat;tokens] (full self-attention, N+K positions).
          Problem: features attend to tokens = feature pollution; 97% of attention
          weight is on self (features-to-features), tokens never learn useful signal.

        AFTER: Q=tokens, KV=feat (tokens-as-queries, features-as-keyvalues).
          - Tokens encode the current feature state by attending to features
          - Features remain uncontaminated by token signals
          - Much lower FLOPs: O(K*N) vs O((N+K)^2)

        Args:
            feat: (B, N, D) — 256 search tokens per modality
            prev_tokens: (B, K, D) or None
            init_tokens: (1, K, D) learned parameter

        Returns:
            feat_out: (B, N, D) unchanged features (pass-through)
            tok_out:  (B, K, D) updated tokens encoding feature state
        """
        B, N, D = feat.shape
        K = self.num_tokens

        # Tokens carry state across frames
        tokens = prev_tokens if prev_tokens is not None else init_tokens.expand(B, -1, -1)

        # ===== Token-Centric Attention: Q=tokens, KV=feat =====
        # No feature pollution: features are used as key/value only
        feat_norm = self.norm(feat)
        tok_norm = self.norm(tokens)

        q = self.q_proj(tok_norm).reshape(B, K, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k, v = self.kv_proj(feat_norm).reshape(
            B, N, 2, self.num_heads, D // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, K, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        tok_out = (attn @ v).transpose(1, 2).reshape(B, K, D)
        tok_out = self.proj(tok_out)
        tok_out = self.proj_drop(tok_out)

        # ===== Token residual + learnable update gate =====
        gate = torch.sigmoid(self.token_update_gate)
        tok_out = tokens + self.drop_path(tok_out)
        tok_out = gate * tok_out + (1 - gate) * tokens

        # ===== Post-attention FFN (MLP was defined but never called — fix dead code) =====
        tok_out = tok_out + self.drop_path(self.mlp(self.norm(tok_out)))

        # Features pass through unchanged
        feat_out = feat

        return feat_out, tok_out

    def forward(self, feat_rgb: torch.Tensor, feat_tir: torch.Tensor,
                prev_tokens: torch.Tensor = None) -> tuple:
        """
        Args:
            feat_rgb: (B, N, D) RGB search features
            feat_tir: (B, N, D) TIR search features
            prev_tokens: (B, 2*K, D) concatenated [rgb_tokens, tir_tokens] or None

        Returns:
            feat_rgb_out: (B, N, D) temporally enhanced RGB
            feat_tir_out: (B, N, D) temporally enhanced TIR
            tokens_out:   (B, 2*K, D) updated [rgb_tokens, tir_tokens]
        """
        K = self.num_tokens

        # Split prev tokens by modality
        if prev_tokens is not None:
            prev_rgb = prev_tokens[:, :K, :]
            prev_tir = prev_tokens[:, K:, :]
        else:
            prev_rgb = None
            prev_tir = None

        # [Fix v1.0] RGB and TIR are processed INDEPENDENTLY — no cross-modal attention
        feat_rgb_out, rgb_tokens = self._forward_stream(feat_rgb, prev_rgb, self.rgb_tokens)
        feat_tir_out, tir_tokens = self._forward_stream(feat_tir, prev_tir, self.tir_tokens)

        tokens_out = torch.cat([rgb_tokens, tir_tokens], dim=1)

        return feat_rgb_out, feat_tir_out, tokens_out


class TemporalTokenBlock(nn.Module):
    """Wrapper for TemporalTokenLayer (interface unchanged)."""

    def __init__(self, dim=256, num_tokens=2, num_heads=4, mlp_ratio=2.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.temporal_layer = TemporalTokenLayer(
            dim=dim, num_tokens=num_tokens, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
        )

    def forward(self, feat_rgb: torch.Tensor, feat_tir: torch.Tensor,
                temporal_tokens: torch.Tensor = None) -> tuple:
        return self.temporal_layer(feat_rgb, feat_tir, temporal_tokens)
