"""
TBSILayer: Core cross-attention between RGB and TIR modalities.
Supports DGSFusion — multiple router modes:
  v1: DivergenceRouter — 6D→1 + α=0.3+0.4*θ (original, 19 params)
  v2: DivergenceRouterV2 — 6D→2, free α ∈ (0,1), qm=α (v2-free, ~14 params)
  v3: CrossAttnConfidence — attention entropy, 0 params (v3-attnent)
  v4: DiffProjRouter — learnable diff projection Linear(768→16) (v4-diffproj, ~12K params)
  v5: SelfQualityRouter — 自质量 (768→8) + 跨模态差异 (768→16) + per-token 4-way路由 (~25K params)
  v6: DiffTemplateRouter — 跨模态差异 + 模板对齐 search@temp + per-token 3-way路由 (~25K params)"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock


# =============================================================================
# DGSFusion Router Variants
# =============================================================================


class DiffProjConservativeRouter(nn.Module):
    """
    DGSFusion v4.5: 保守路由 — 加 β 不确定度门控.

    在 v4 的 DiffProjRouter 基础上:
      - gate 输出从 Linear(16,2) → Linear(16,4)
      - [α_v, α_i, β_v, β_i], β ∈ (0,1) 表示模型对自己路由决策的信心
      - 最终 α = β * α + (1-β) * 0.5
        当 β↓ (不确定) → 拉向 0.5 (保守模式, 等于不做门控)
        当 β↑ (确定)   → 保留原始 α 值 (正常路由)

    Zero-init 保证训练起点: α=0.5, β=0.5 → 初始行为 = 不做门控.
    ~12.4K params: Linear(768,16) + LayerNorm(16) + Linear(16,4).
    """
    def __init__(self, dim=768):
        super().__init__()
        self.diff_proj = nn.Sequential(
            nn.Linear(dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.gate = nn.Linear(16, 4)  # [α_v, α_i, β_v, β_i]
        # Zero-init: 起点 α≈0.5, β≈0.5
        nn.init.zeros_(self.diff_proj[0].weight)
        nn.init.zeros_(self.diff_proj[0].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x_v_search, x_i_search):
        diff = x_v_search - x_i_search        # (B, N_s, 768)
        d = self.diff_proj(diff)              # (B, N_s, 16)
        out = self.gate(d)                     # (B, N_s, 4)

        α_v = torch.sigmoid(out[:, :, 0:1])    # 原始路由权重
        α_i = torch.sigmoid(out[:, :, 1:2])
        β_v = torch.sigmoid(out[:, :, 2:3])    # 路由置信度: 1=相信路由, 0=保守
        β_i = torch.sigmoid(out[:, :, 3:4])

        # 保守模式: 不确定时拉向 0.5
        α_v = β_v * α_v + (1 - β_v) * 0.5
        α_i = β_i * α_i + (1 - β_i) * 0.5

        return torch.cat([α_v, α_i], dim=-1)   # (B, N_s, 2)

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


class SelfQualityRouter(nn.Module):
    """
    DGSFusion v5: 自质量 (Self-Quality) + 跨模态差异 (Cross-modal Diff).

    在 v4 基础上叠加自质量估计:
      Dim 1 — 自质量: 每个模态独立判断"自己好不好"
        quality_rgb/tir: Linear(768→8), vote: Linear(8→2)
      Dim 2 — 跨模态差异: 继承 v4 的 DiffProjRouter

    3 个 vote (rgb_self, tir_self, cross_diff) 通过 per-token 4-way softmax 路由加权.
      w[..., 3] = abstain: 所有信号都不可信时的退路.

    ~25K params: 3×Linear(768→8/16) + 3×vote + per-token router(32→8→4).
    """
    def __init__(self, dim=768):
        super().__init__()
        # Dim 1a: RGB 自质量
        self.quality_rgb = nn.Linear(dim, 8)
        self.vote_rgb = nn.Linear(8, 2)

        # Dim 1b: TIR 自质量
        self.quality_tir = nn.Linear(dim, 8)
        self.vote_tir = nn.Linear(8, 2)

        # Dim 2: 跨模态差异 (继承 v4)
        self.diff_proj = nn.Sequential(
            nn.Linear(dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.vote_diff = nn.Linear(16, 2)

        # Per-token 路由器: bottleneck 层特征 → 4-way softmax (含 abstain)
        self.router = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

        # 零初始化全部
        for m in [self.quality_rgb, self.quality_tir, self.diff_proj[0],
                  self.vote_rgb, self.vote_tir, self.vote_diff,
                  self.router[0], self.router[2]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_v_search, x_i_search):
        # Dim 1: 自质量
        q_rgb = self.quality_rgb(x_v_search)                       # (B,N,8)
        q_tir = self.quality_tir(x_i_search)                       # (B,N,8)
        α_vote_rgb = self.vote_rgb(q_rgb)                          # (B,N,2)
        α_vote_tir = self.vote_tir(q_tir)                          # (B,N,2)

        # Dim 2: 跨模态差异
        d_cross = self.diff_proj(x_v_search - x_i_search)          # (B,N,16)
        α_vote_diff = self.vote_diff(d_cross)                      # (B,N,2)

        # Per-token 路由 (bottleneck 层, 不是 vote 层)
        router_feat = torch.cat([q_rgb, q_tir, d_cross], dim=-1)   # (B,N,32)
        w = F.softmax(self.router(router_feat), dim=-1)            # (B,N,4)
        # w[..., 0:3] = 3个vote的权重, w[..., 3] = abstain

        # 加权融合
        α = (w[..., 0:1] * α_vote_rgb +
             w[..., 1:2] * α_vote_tir +
             w[..., 2:3] * α_vote_diff)                            # (B,N,2)
        # abstain 时加权投票残差自然退火

        return torch.sigmoid(α)


class DiffTemplateRouter(nn.Module):
    """
    DGSFusion v6: 跨模态差异 (DiffProj) + 模板对齐 (Template Alignment).

    Dim 2 — 跨模态差异: 继承 v4 的 diff_proj(768→16) → vote_diff(16→2)
    Dim 3 — 模板对齐: search token 与 template token 的相似度分布
      search_proj & temp_proj: 共享语义空间 (RGB/TIR 共用)
      similarity: bmm(search_proj, temp_proj^T) → 排序分组池化(8) + max(1) → encode(9→8) → vote(8→2)

    3 个 vote (diff, talign_rgb, talign_tir) 通过 per-token 3-way softmax 路由加权.

    ~25K params: diff(12.3K) + temp_align(12.4K) + router(0.3K).
    """
    def __init__(self, dim=768):
        super().__init__()

        # === Dim 2: 跨模态差异 (继承 v4) ===
        self.diff_proj = nn.Sequential(
            nn.Linear(dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.vote_diff = nn.Linear(16, 2)

        # === Dim 3: 模板对齐 (共享语义空间) ===
        self.search_proj = nn.Linear(dim, 8)  # RGB/TIR 共享
        self.temp_proj = nn.Linear(dim, 8)    # RGB/TIR 共享
        self.talign_enc = nn.Linear(9, 8)     # 分布编码: 8组池化 + max(1) → 8
        self.vote_talign_rgb = nn.Linear(8, 2)
        self.vote_talign_tir = nn.Linear(8, 2)

        # === Per-token 路由 (bottleneck 层) ===
        self.router = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 3),   # 3-way: diff, talign_rgb, talign_tir
        )

        # 零初始化全部
        for m in [self.diff_proj[0], self.vote_diff,
                  self.search_proj, self.temp_proj, self.talign_enc,
                  self.vote_talign_rgb, self.vote_talign_tir,
                  self.router[0], self.router[2]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_v_search, x_i_search, x_v_temp, x_i_temp):
        # === Dim 2: 跨模态差异 ===
        d_cross = self.diff_proj(x_v_search - x_i_search)               # (B,N,16)
        α_vote_diff = self.vote_diff(d_cross)                           # (B,N,2)

        # === Dim 3: 模板对齐 (共享语义空间) ===
        T = x_v_temp.shape[1]  # template length (e.g. 64)

        s_rgb = F.gelu(self.search_proj(x_v_search))                    # (B,N,8)
        t_rgb = F.gelu(self.temp_proj(x_v_temp))                        # (B,T,8)
        sim_rgb = torch.bmm(s_rgb, t_rgb.transpose(1, 2))               # (B,N,T)

        s_tir = F.gelu(self.search_proj(x_i_search))
        t_tir = F.gelu(self.temp_proj(x_i_temp))
        sim_tir = torch.bmm(s_tir, t_tir.transpose(1, 2))               # (B,N,T)

        # 相似度分布特征: 排序 → 8组均值 + 最大值 (order-independent)
        def _talign_feat(sim):
            sim_sorted = sim.sort(dim=-1, descending=True).values        # (B,N,T)
            Bp, Np, _ = sim_sorted.shape
            sim_grouped = sim_sorted.reshape(Bp, Np, T // 8, 8).mean(dim=-1)  # (B,N,8)
            sim_max = sim_sorted[:, :, 0:1]                              # (B,N,1)
            return F.gelu(self.talign_enc(
                torch.cat([sim_grouped, sim_max], dim=-1)))              # (B,N,8)

        talign_rgb = _talign_feat(sim_rgb)                               # (B,N,8)
        talign_tir = _talign_feat(sim_tir)

        α_vote_talign_rgb = self.vote_talign_rgb(talign_rgb)            # (B,N,2)
        α_vote_talign_tir = self.vote_talign_tir(talign_tir)            # (B,N,2)

        # === Per-token 路由 (bottleneck 层) ===
        router_feat = torch.cat([d_cross, talign_rgb, talign_tir], dim=-1)  # (B,N,32)
        w = F.softmax(self.router(router_feat), dim=-1)                  # (B,N,3)

        α = (w[:, :, 0:1] * α_vote_diff +
             w[:, :, 1:2] * α_vote_talign_rgb +
             w[:, :, 2:3] * α_vote_talign_tir)                          # (B,N,2)

        return torch.sigmoid(α)


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


class SignalAwareDecoupler(nn.Module):
    """
    Task-aware Feature Decoupling Module.

    Decouples template features into task-relevant signal (only this part
    participates in cross-modal interaction) and task-irrelevant components
    (preserved via residual connection).

    Architecture:
        Linear(dim → dim//4) → ReLU → Linear(dim//4 → dim) → Sigmoid → gate
        signal = gate * x    (task-relevant)
        interference = (1 - gate) * x   (task-irrelevant)

    Zero-init on last linear: sigmoid(0) ≈ 0.5 → balanced split at start.
    ~154K params for dim=768.
    """
    def __init__(self, dim=768, reduction=4, mode="split", residual_scale=0.5,
                 alpha_init=0.1):
        super().__init__()
        self.mode = mode
        self.residual_scale = residual_scale
        rdim = max(dim // reduction, 16)
        self.decouple = nn.Sequential(
            nn.Linear(dim, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, dim),
        )
        if mode == "reliability_residual":
            self.reliability = nn.Sequential(
                nn.Linear(dim * 3, rdim),
                nn.ReLU(inplace=True),
                nn.Linear(rdim, dim),
            )
        if mode in ("conditional_residual", "reliability_guided_residual"):
            self.reliability = nn.Sequential(
                nn.Linear(dim * 4, rdim),
                nn.ReLU(inplace=True),
                nn.Linear(rdim, dim),
            )
        if mode in ("learnable_residual", "conditional_residual", "reliability_guided_residual"):
            alpha_init = min(max(alpha_init, 1e-4), residual_scale - 1e-4)
            alpha_ratio = alpha_init / residual_scale
            alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        # Zero-init last layer: gate ≈ 0.5 at start (balanced split)
        self.reset_last_layer()

    def reset_last_layer(self):
        """Keep the initial gate balanced after model-wide initialization."""
        nn.init.zeros_(self.decouple[-1].weight)
        nn.init.zeros_(self.decouple[-1].bias)
        if hasattr(self, "reliability"):
            nn.init.zeros_(self.reliability[-1].weight)
            nn.init.zeros_(self.reliability[-1].bias)

    def forward(self, x, x_rgb=None, x_tir=None, x_search=None):
        logits = self.decouple(x)
        if self.mode in ("conditional_residual", "reliability_guided_residual"):
            if x_rgb is None or x_tir is None or x_search is None:
                raise ValueError(f"{self.mode} mode requires x_rgb, x_tir, and x_search inputs")
            search_context = x_search.mean(dim=1, keepdim=True).expand_as(x)
            modality_gap = (x_rgb - x_tir).abs()
            reliability_inp = torch.cat([
                x,
                search_context,
                (x - search_context).abs(),
                modality_gap,
            ], dim=-1)
            reliability = torch.sigmoid(self.reliability(reliability_inp))
            alpha = self.residual_scale * torch.sigmoid(self.alpha_logit)
            if self.mode == "reliability_guided_residual":
                modal_agree = F.cosine_similarity(x_rgb, x_tir, dim=-1).unsqueeze(-1)
                search_agree = F.cosine_similarity(x, search_context, dim=-1).unsqueeze(-1)
                reliability_prior = 0.5 * (
                    torch.sigmoid(2.0 * modal_agree) +
                    torch.sigmoid(2.0 * search_agree)
                )
                reliability = 0.5 * reliability + 0.5 * reliability_prior
                modulation = alpha * reliability * torch.tanh(logits)
            else:
                modulation = alpha * (1.0 - reliability) * torch.tanh(logits)
            signal = x * (1.0 + modulation)
            interference = -x * modulation
            return signal, interference

        if self.mode == "reliability_residual":
            if x_rgb is None or x_tir is None:
                raise ValueError("reliability_residual mode requires x_rgb and x_tir inputs")
            reliability_inp = torch.cat([x_rgb, x_tir, (x_rgb - x_tir).abs()], dim=-1)
            reliability = torch.sigmoid(self.reliability(reliability_inp))
            uncertainty = 1.0 - reliability
            modulation = self.residual_scale * uncertainty * torch.tanh(logits)
            signal = x * (1.0 + modulation)
            interference = -x * modulation
            return signal, interference

        if self.mode == "learnable_residual":
            alpha = self.residual_scale * torch.sigmoid(self.alpha_logit)
            modulation = alpha * torch.tanh(logits)
            signal = x * (1.0 + modulation)
            interference = -x * modulation
            return signal, interference

        if self.mode == "residual":
            modulation = self.residual_scale * torch.tanh(logits)
            signal = x * (1.0 + modulation)
            interference = -x * modulation
            return signal, interference

        gate = torch.sigmoid(logits)  # (B, T, C)
        signal = gate * x
        interference = (1 - gate) * x
        return signal, interference


class SoftSearchReliability(nn.Module):
    """
    Soft reliability modulation for search tokens.

    This is a continuous alternative to hard token elimination: every search
    token remains in the bridge, while a small identity-start residual learns
    where RGB/TIR search evidence should be enhanced.
    """
    def __init__(self, dim=768, reduction=4, residual_scale=0.25, alpha_init=0.05):
        super().__init__()
        self.residual_scale = residual_scale
        rdim = max(dim // reduction, 16)
        self.reliability = nn.Sequential(
            nn.Linear(dim * 4, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, 2),
        )
        self.update = nn.Sequential(
            nn.Linear(dim, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, dim),
        )
        alpha_init = min(max(alpha_init, 1e-4), residual_scale - 1e-4)
        alpha_ratio = alpha_init / residual_scale
        alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        self.reset_last_layer()

    def reset_last_layer(self):
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, x_v_search, x_i_search, fused_template):
        template_context = fused_template.mean(dim=1, keepdim=True).expand_as(x_v_search)
        reliability_inp = torch.cat([
            x_v_search,
            x_i_search,
            (x_v_search - x_i_search).abs(),
            template_context,
        ], dim=-1)
        conf = torch.sigmoid(self.reliability(reliability_inp))
        alpha = self.residual_scale * torch.sigmoid(self.alpha_logit)
        update_v = torch.tanh(self.update(x_v_search))
        update_i = torch.tanh(self.update(x_i_search))
        x_v_search = x_v_search + alpha * conf[:, :, 0:1] * update_v
        x_i_search = x_i_search + alpha * conf[:, :, 1:2] * update_i
        return x_v_search, x_i_search, conf.mean(dim=1)


class TemplateSearchCompetition(nn.Module):
    """
    Template-guided soft candidate competition for search tokens.

    Unlike hard token elimination, this module keeps every search token in the
    bridge. Tokens that match the fused template keep nearly full cross-attention
    strength, while weak template candidates are softly pulled toward the
    original search representation.
    """
    def __init__(self, dim=768, residual_scale=0.20, alpha_init=0.10, temperature=4.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.temperature = temperature
        alpha_init = min(max(alpha_init, 1e-4), residual_scale - 1e-4)
        alpha_ratio = alpha_init / residual_scale
        alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

    def reset_last_layer(self):
        pass

    def _candidate_score(self, x_search, fused_template):
        template_context = fused_template.mean(dim=1, keepdim=True)
        sim = F.cosine_similarity(
            F.normalize(x_search, dim=-1),
            F.normalize(template_context, dim=-1),
            dim=-1,
        ).unsqueeze(-1)
        sim = (sim - sim.mean(dim=1, keepdim=True)) / (sim.std(dim=1, keepdim=True) + 1e-6)
        return torch.sigmoid(self.temperature * sim)

    def forward(self, x_v_search, x_i_search, fused_template):
        score_v = self._candidate_score(x_v_search, fused_template)
        score_i = self._candidate_score(x_i_search, fused_template)
        alpha = self.residual_scale * torch.sigmoid(self.alpha_logit)
        gate_v = 1.0 - alpha * (1.0 - score_v)
        gate_i = 1.0 - alpha * (1.0 - score_i)
        return gate_v.clamp(0.0, 1.0), gate_i.clamp(0.0, 1.0), torch.cat([
            score_v.mean(dim=1), score_i.mean(dim=1)
        ], dim=-1)


class TemplateConditionedBridgeMask(nn.Module):
    """
    Pre-softmax template-conditioned bridge bias.

    This module does not reweight features or outputs. It biases the attention
    logits so target-related search tokens participate more strongly when the
    fused template is updated in the bridge interaction.
    """
    def __init__(self, scale=0.2, temperature=4.0, bias_mode="signed", neg_floor=-0.3):
        super().__init__()
        self.scale = scale
        self.temperature = temperature
        self.bias_mode = bias_mode
        self.neg_floor = neg_floor

    def forward(self, x_search, fused_template):
        template_anchor = F.normalize(fused_template.mean(dim=1, keepdim=True), dim=-1)
        search_norm = F.normalize(x_search, dim=-1)
        score = F.cosine_similarity(search_norm, template_anchor, dim=-1)
        score = (score - score.mean(dim=1, keepdim=True)) / (
            score.std(dim=1, keepdim=True) + 1e-6)
        if self.bias_mode == "positive":
            bias = F.relu(score)
        elif self.bias_mode == "clipped_signed":
            bias = torch.tanh(self.temperature * score).clamp(min=self.neg_floor)
        else:
            bias = torch.tanh(self.temperature * score)
        return self.scale * bias.unsqueeze(1)


class CFSReliabilityBridgeBias(nn.Module):
    """
    Cross-modal feature-structure reliability bias for bridge attention.

    The descriptor uses RGB-TIR search-token agreement and modality-internal
    structure statistics, which are complementary to TBSI's search-template
    attention logits.
    """
    def __init__(self, scale=0.2, hidden_dim=32):
        super().__init__()
        self.scale = scale
        self.mlp = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_v, x_i):
        bsz, num_tokens, dim = x_v.shape
        x_v_norm = F.normalize(x_v, dim=-1)
        x_i_norm = F.normalize(x_i, dim=-1)
        mu_v = F.normalize(x_v.mean(dim=1, keepdim=True), dim=-1)
        mu_i = F.normalize(x_i.mean(dim=1, keepdim=True), dim=-1)

        cos_cross = (x_v_norm * x_i_norm).sum(-1, keepdim=True)
        l2_diff = (x_v - x_i).pow(2).mean(dim=-1, keepdim=True).sqrt()
        norm_v = x_v.norm(dim=-1, keepdim=True)
        norm_i = x_i.norm(dim=-1, keepdim=True)
        energy_ratio = torch.log((norm_v + 1e-6) / (norm_i + 1e-6))
        energy_absdiff = (norm_v - norm_i).abs() / (norm_v + norm_i + 1e-6)

        cos_v_self = (x_v_norm * mu_v).sum(-1, keepdim=True)
        cos_i_self = (x_i_norm * mu_i).sum(-1, keepdim=True)
        cos_v_cross = (x_v_norm * mu_i).sum(-1, keepdim=True)
        cos_i_cross = (x_i_norm * mu_v).sum(-1, keepdim=True)

        sim_v = x_v_norm @ x_v_norm.transpose(-2, -1)
        sim_i = x_i_norm @ x_i_norm.transpose(-2, -1)
        p_v = F.softmax(sim_v, dim=-1).clamp_min(1e-6)
        p_i = F.softmax(sim_i, dim=-1).clamp_min(1e-6)
        mix = (0.5 * (p_v + p_i)).clamp_min(1e-6)
        js_struct = 0.5 * (
            (p_v * (p_v.log() - mix.log())).sum(dim=-1, keepdim=True) +
            (p_i * (p_i.log() - mix.log())).sum(dim=-1, keepdim=True))
        ent_v = -(p_v * p_v.log()).sum(dim=-1, keepdim=True) / math.log(num_tokens)
        ent_i = -(p_i * p_i.log()).sum(dim=-1, keepdim=True) / math.log(num_tokens)

        desc = torch.cat([
            cos_cross, l2_diff, energy_ratio, energy_absdiff,
            cos_v_self, cos_i_self, cos_v_cross, cos_i_cross,
            js_struct, ent_v, ent_i, (ent_v - ent_i).abs(),
        ], dim=-1)
        bias = torch.tanh(self.mlp(desc))
        return self.scale * bias.transpose(1, 2)


class CompetitiveBridgeFusion(nn.Module):
    """
    Competitive RGB/TIR search-to-template bridge fusion.

    TBSI computes two s2t interactions independently. This module computes a
    per-template-token modality budget from the two s2t affinity distributions
    and fuses the two candidate template updates before t2s propagation.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, fused_template, x_v_search, x_i_search):
        dim = fused_template.shape[-1]
        q = fused_template
        logits_v = (q @ x_v_search.transpose(-2, -1)) / math.sqrt(dim)
        logits_i = (q @ x_i_search.transpose(-2, -1)) / math.sqrt(dim)
        score_v = torch.logsumexp(logits_v, dim=-1, keepdim=True)
        score_i = torch.logsumexp(logits_i, dim=-1, keepdim=True)
        weights = F.softmax(
            self.temperature * torch.cat([score_i, score_v], dim=-1),
            dim=-1)
        return weights[:, :, 0:1], weights[:, :, 1:2]


class FrequencyGateBias(nn.Module):
    """
    Spatial-frequency guided token bias for RGB/TIR bridge attention.

    The search tokens are reshaped to a square grid. A 2D FFT low-pass
    reconstruction estimates local low-frequency dominance per token. TIR uses
    low-frequency dominance as positive evidence, while RGB uses high-frequency
    dominance as positive evidence.
    """
    def __init__(self, scale=0.1, cutoff=0.25):
        super().__init__()
        self.scale = scale
        self.cutoff = cutoff

    def _low_freq_ratio(self, x):
        bsz, num_tokens, dim = x.shape
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            return x.new_zeros(bsz, 1, num_tokens)

        feat = x.view(bsz, side, side, dim)
        spec = torch.fft.fft2(feat.float(), dim=(1, 2), norm="ortho")
        fy = torch.fft.fftfreq(side, device=x.device)
        fx = torch.fft.fftfreq(side, device=x.device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        mask = (yy.square() + xx.square()).sqrt() <= self.cutoff
        low = torch.fft.ifft2(
            spec * mask.view(1, side, side, 1),
            dim=(1, 2),
            norm="ortho").real.to(dtype=x.dtype)

        low_energy = low.square().mean(dim=-1, keepdim=True)
        total_energy = feat.square().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        ratio = (low_energy / total_energy).clamp(0.0, 1.0)
        return ratio.view(bsz, num_tokens, 1).transpose(1, 2)

    @staticmethod
    def _normalize(raw):
        centered = raw - raw.mean(dim=-1, keepdim=True)
        scale = raw.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return centered / scale

    def forward(self, x_rgb_search, x_tir_search):
        r_rgb = self._low_freq_ratio(x_rgb_search)
        r_tir = self._low_freq_ratio(x_tir_search)
        tir_bias = torch.tanh(self._normalize(2.0 * r_tir - 1.0))
        rgb_bias = torch.tanh(self._normalize(1.0 - 2.0 * r_rgb))
        return self.scale * rgb_bias, self.scale * tir_bias


class OutputResidualGate(nn.Module):
    """
    Delta-level adapter after TBSI interaction outputs.

    The original CAST output is kept as the base interaction. This module learns
    only an extra residual from the interaction delta, with zero-init so the
    initial behavior is exactly the original TBSI bridge output.
    """
    def __init__(self, dim=768, reduction=4, mode="naive", residual_scale=1.0):
        super().__init__()
        self.mode = mode
        self.residual_scale = residual_scale
        rdim = max(dim // reduction, 16)
        self.adapter = nn.Sequential(
            nn.Linear(dim, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, dim),
        )
        if mode == "naive":
            gate_dim = dim * 3
        elif mode == "reliability_template":
            gate_dim = dim * 4 + 4
        else:
            gate_dim = dim * 4 + 3
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, 1),
        )
        self.reset_last_layer()

    def reset_last_layer(self):
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, interaction_out, original_search, template_context=None):
        delta = interaction_out - original_search
        if self.mode in ("reliability", "reliability_template"):
            interaction_norm = F.normalize(interaction_out, dim=-1)
            original_norm = F.normalize(original_search, dim=-1)
            delta_norm = F.normalize(delta, dim=-1)
            agreement = F.cosine_similarity(interaction_norm, original_norm, dim=-1).unsqueeze(-1)
            delta_energy = delta.norm(dim=-1, keepdim=True) / (
                original_search.norm(dim=-1, keepdim=True) + 1e-6)
            delta_alignment = F.cosine_similarity(delta_norm, original_norm, dim=-1).unsqueeze(-1)
            reliability_stats = torch.cat([agreement, delta_energy, delta_alignment], dim=-1)
            if self.mode == "reliability_template":
                if template_context is None:
                    template_agreement = torch.zeros_like(agreement)
                else:
                    template_anchor = template_context.mean(dim=1, keepdim=True)
                    template_anchor = F.normalize(template_anchor, dim=-1)
                    template_agreement = F.cosine_similarity(
                        original_norm, template_anchor, dim=-1).unsqueeze(-1)
                reliability_stats = torch.cat([reliability_stats, template_agreement], dim=-1)
            gate_inp = torch.cat([
                interaction_out, original_search, delta, delta.abs(), reliability_stats
            ], dim=-1)
        else:
            gate_inp = torch.cat([interaction_out, delta, delta.abs()], dim=-1)
        gate = torch.sigmoid(self.gate(gate_inp))
        residual = self.adapter(delta)
        return interaction_out + self.residual_scale * gate * residual, gate.mean(dim=1)


class TBSILayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_degradation=False,
                 use_attn_gate=False, use_temporal_tokens=False, use_dgs=False, dgs_mode="v1",
                 use_signal_decouple=False, signal_decouple_mode="split", signal_decouple_scale=0.5,
                 signal_decouple_alpha_init=0.1, use_soft_search_reliability=False,
                 soft_search_reliability_scale=0.25, soft_search_reliability_alpha_init=0.05,
                 use_template_search_competition=False,
                 template_search_competition_scale=0.20,
                 template_search_competition_alpha_init=0.10,
                 template_search_competition_temperature=4.0,
                 use_output_residual_gate=False,
                 output_residual_gate_mode="naive",
                 output_residual_gate_scale=1.0,
                 use_template_conditioned_bridge=False,
                 template_conditioned_bridge_scale=0.2,
                 template_conditioned_bridge_temperature=4.0,
                 template_conditioned_bridge_bias_mode="signed",
                 template_conditioned_bridge_neg_floor=-0.3,
                 use_cfs_reliability_bridge=False,
                 cfs_reliability_bridge_scale=0.2,
                 cfs_reliability_bridge_hidden=32,
                 use_competitive_bridge=False,
                 competitive_bridge_temperature=1.0,
                 competitive_bridge_residual_scale=1.0,
                 use_freq_gate=False,
                 freq_gate_scale=0.1,
                 freq_gate_cutoff=0.25):
        super().__init__()
        self.use_dgs = use_dgs
        self.dgs_mode = dgs_mode
        self.use_signal_decouple = use_signal_decouple
        self.use_soft_search_reliability = use_soft_search_reliability
        self.use_template_search_competition = use_template_search_competition
        self.use_output_residual_gate = use_output_residual_gate
        self.use_template_conditioned_bridge = use_template_conditioned_bridge
        self.use_cfs_reliability_bridge = use_cfs_reliability_bridge
        self.use_competitive_bridge = use_competitive_bridge
        self.use_freq_gate = use_freq_gate
        self.competitive_bridge_residual_scale = competitive_bridge_residual_scale

        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        if use_signal_decouple:
            self.signal_decoupler = SignalAwareDecoupler(
                dim=dim, mode=signal_decouple_mode, residual_scale=signal_decouple_scale,
                alpha_init=signal_decouple_alpha_init)
            rp = sum(p.numel() for p in self.signal_decoupler.parameters())
            print(f"  [SignalDecouple] Task-aware Feature Decoupling active "
                  f"(mode={signal_decouple_mode}, {rp} params)")

        if use_soft_search_reliability:
            self.soft_search_reliability = SoftSearchReliability(
                dim=dim, residual_scale=soft_search_reliability_scale,
                alpha_init=soft_search_reliability_alpha_init)
            rp = sum(p.numel() for p in self.soft_search_reliability.parameters())
            print(f"  [SoftSearchReliability] Search token soft modulation active "
                  f"({rp} params, scale={soft_search_reliability_scale}, "
                  f"alpha_init={soft_search_reliability_alpha_init})")

        if use_template_search_competition:
            self.template_search_competition = TemplateSearchCompetition(
                dim=dim, residual_scale=template_search_competition_scale,
                alpha_init=template_search_competition_alpha_init,
                temperature=template_search_competition_temperature)
            rp = sum(p.numel() for p in self.template_search_competition.parameters())
            print(f"  [TemplateSearchCompetition] Template-guided search candidate competition active "
                  f"({rp} params, scale={template_search_competition_scale}, "
                  f"alpha_init={template_search_competition_alpha_init}, "
                  f"temperature={template_search_competition_temperature})")

        if use_output_residual_gate:
            self.output_residual_gate_v = OutputResidualGate(
                dim=dim, mode=output_residual_gate_mode,
                residual_scale=output_residual_gate_scale)
            self.output_residual_gate_i = OutputResidualGate(
                dim=dim, mode=output_residual_gate_mode,
                residual_scale=output_residual_gate_scale)
            rp = (sum(p.numel() for p in self.output_residual_gate_v.parameters()) +
                  sum(p.numel() for p in self.output_residual_gate_i.parameters()))
            print(f"  [OutputResidualGate] Delta-level bridge output adapter active "
                  f"({rp} params, mode={output_residual_gate_mode}, "
                  f"scale={output_residual_gate_scale}, zero-init)")

        if use_template_conditioned_bridge:
            self.template_conditioned_bridge = TemplateConditionedBridgeMask(
                scale=template_conditioned_bridge_scale,
                temperature=template_conditioned_bridge_temperature,
                bias_mode=template_conditioned_bridge_bias_mode,
                neg_floor=template_conditioned_bridge_neg_floor)
            print(f"  [TemplateConditionedBridge] Pre-softmax bridge bias active "
                  f"(scale={template_conditioned_bridge_scale}, "
                  f"temperature={template_conditioned_bridge_temperature}, "
                  f"bias_mode={template_conditioned_bridge_bias_mode}, "
                  f"neg_floor={template_conditioned_bridge_neg_floor})")

        if use_cfs_reliability_bridge:
            self.cfs_reliability_bridge = CFSReliabilityBridgeBias(
                scale=cfs_reliability_bridge_scale,
                hidden_dim=cfs_reliability_bridge_hidden)
            rp = sum(p.numel() for p in self.cfs_reliability_bridge.parameters())
            print(f"  [CFSReliabilityBridge] Cross-modal feature-structure bridge bias active "
                  f"({rp} params, scale={cfs_reliability_bridge_scale}, "
                  f"hidden={cfs_reliability_bridge_hidden}, zero-init)")

        if use_competitive_bridge:
            self.competitive_bridge = CompetitiveBridgeFusion(
                temperature=competitive_bridge_temperature)
            print(f"  [CompetitiveBridge] RGB/TIR s2t competition active "
                  f"(temperature={competitive_bridge_temperature}, "
                  f"residual_scale={competitive_bridge_residual_scale}, 0 params)")

        if use_freq_gate:
            self.freq_gate = FrequencyGateBias(
                scale=freq_gate_scale,
                cutoff=freq_gate_cutoff)
            print(f"  [FreqGATE] Spatial-frequency bridge bias active "
                  f"(scale={freq_gate_scale}, cutoff={freq_gate_cutoff}, 0 params)")

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
            elif dgs_mode == "v5":
                self.dgs_router = SelfQualityRouter(dim)
            elif dgs_mode == "v6":
                self.dgs_router = DiffTemplateRouter(dim)
            elif dgs_mode == "v4.5":
                self.dgs_router = DiffProjConservativeRouter(dim)
            else:
                raise ValueError(f"Unknown DGS_MODE: {dgs_mode}")
            rp = sum(p.numel() for p in self.dgs_router.parameters())
            print(f"  [DGSFusion] Router active (mode={dgs_mode}, {rp} params)")

    def forward(self, x_v, x_i, lens_z, temporal_tokens=None):
        x_v_template = x_v[:, :lens_z, :]
        x_i_template = x_i[:, :lens_z, :]
        fused_t = torch.cat([x_v_template, x_i_template], dim=2)
        fused_t = self.t_fusion(fused_t)

        # ===== Signal-aware Decoupling: only task-relevant signal participates in cross-attn =====
        fused_t_interference = None
        if self.use_signal_decouple:
            x_v_search = x_v[:, lens_z:, :]
            x_i_search = x_i[:, lens_z:, :]
            x_search_context = 0.5 * (x_v_search + x_i_search)
            fused_t_signal, fused_t_interference = self.signal_decoupler(
                fused_t, x_rgb=x_v_template, x_tir=x_i_template, x_search=x_search_context)
            # Use signal in cross-attention; interference preserved as residual
            fused_t_attn = fused_t_signal
        else:
            fused_t_attn = fused_t

        x_v_orig = x_v[:, lens_z:, :]
        x_i_orig = x_i[:, lens_z:, :]
        search_quality_signal = None
        if self.use_soft_search_reliability:
            x_v_orig, x_i_orig, search_quality_signal = self.soft_search_reliability(
                x_v_orig, x_i_orig, fused_t)

        tsc_gate_v = tsc_gate_i = None
        if self.use_template_search_competition:
            tsc_gate_v, tsc_gate_i, search_quality_signal = self.template_search_competition(
                x_v_orig, x_i_orig, fused_t)

        # ===== Compute per-token quality masks (for CASTBlocks) =====
        qm_v = qm_i = None
        if self.use_dgs:
            if self.dgs_mode == "v6":
                α = self.dgs_router(x_v_orig, x_i_orig,
                                    x_v[:, :lens_z, :], x_i[:, :lens_z, :])
            else:
                α = self.dgs_router(x_v_orig, x_i_orig)  # (B, N_s, 1) for v1, (B, N_s, 2) for v2-v5
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
        elif self.use_template_search_competition:
            qm_v, qm_i = tsc_gate_v, tsc_gate_i

        tcb_bias_v = tcb_bias_i = None
        if self.use_template_conditioned_bridge:
            tcb_bias_i = self.template_conditioned_bridge(x_i_orig, fused_t)
            tcb_bias_v = self.template_conditioned_bridge(x_v_orig, fused_t)
        if self.use_cfs_reliability_bridge:
            cfs_bias = self.cfs_reliability_bridge(x_v_orig, x_i_orig)
            tcb_bias_i = cfs_bias if tcb_bias_i is None else tcb_bias_i + cfs_bias
            tcb_bias_v = cfs_bias if tcb_bias_v is None else tcb_bias_v + cfs_bias
        if self.use_freq_gate:
            freq_bias_v, freq_bias_i = self.freq_gate(x_v_orig, x_i_orig)
            tcb_bias_i = freq_bias_i if tcb_bias_i is None else tcb_bias_i + freq_bias_i
            tcb_bias_v = freq_bias_v if tcb_bias_v is None else tcb_bias_v + freq_bias_v

        # 4 CASTBlocks (quality-guided cross-attention, using decoupled signal)
        if self.use_competitive_bridge:
            fused_t_base = fused_t_attn
            comp_i, comp_v = self.competitive_bridge(fused_t_base, x_v_orig, x_i_orig)
            fused_i = self.ca_s2t_i2f(torch.cat([fused_t_base, x_i_orig], dim=1),
                                      quality_mask=qm_i, attn_bias=tcb_bias_i)[:, :lens_z, :]
            fused_v = self.ca_s2t_v2f(torch.cat([fused_t_base, x_v_orig], dim=1),
                                      quality_mask=qm_v, attn_bias=tcb_bias_v)[:, :lens_z, :]
            fused_t_comp = comp_i * fused_i + comp_v * fused_v
            fused_t_seq = self.ca_s2t_i2f(torch.cat([fused_t_base, x_i_orig], dim=1),
                                          quality_mask=qm_i, attn_bias=tcb_bias_i)[:, :lens_z, :]
            fused_t_seq = self.ca_s2t_v2f(torch.cat([fused_t_seq, x_v_orig], dim=1),
                                          quality_mask=qm_v, attn_bias=tcb_bias_v)[:, :lens_z, :]
            fused_t_attn = fused_t_seq + self.competitive_bridge_residual_scale * (
                fused_t_comp - fused_t_seq)
            search_quality_signal = torch.cat([comp_v.mean(dim=1), comp_i.mean(dim=1)], dim=-1)
            temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t_attn, x_v_orig], dim=1),
                                       quality_mask=qm_v)[:, lens_z:, :]
            temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t_attn, x_i_orig], dim=1),
                                       quality_mask=qm_i)[:, lens_z:, :]
        else:
            fused_t_attn = self.ca_s2t_i2f(torch.cat([fused_t_attn, x_i_orig], dim=1),
                                           quality_mask=qm_i, attn_bias=tcb_bias_i)[:, :lens_z, :]
            temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t_attn, x_v_orig], dim=1),
                                       quality_mask=qm_v)[:, lens_z:, :]
            fused_t_attn = self.ca_s2t_v2f(torch.cat([fused_t_attn, x_v_orig], dim=1),
                                           quality_mask=qm_v, attn_bias=tcb_bias_v)[:, :lens_z, :]
            temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t_attn, x_i_orig], dim=1),
                                       quality_mask=qm_i)[:, lens_z:, :]

        output_gate_signal = None
        if self.use_output_residual_gate:
            temp_x_v, gate_v = self.output_residual_gate_v(temp_x_v, x_v_orig, fused_t)
            temp_x_i, gate_i = self.output_residual_gate_i(temp_x_i, x_i_orig, fused_t)
            output_gate_signal = torch.cat([gate_v, gate_i], dim=-1)

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
        elif self.use_template_search_competition:
            x_v = torch.cat([x_v[:, :lens_z, :],
                             tsc_gate_v * temp_x_v + (1.0 - tsc_gate_v) * x_v_orig], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :],
                             tsc_gate_i * temp_x_i + (1.0 - tsc_gate_i) * x_i_orig], dim=1)
            q_global = search_quality_signal
        else:
            x_v = torch.cat([x_v[:, :lens_z, :], temp_x_v], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], temp_x_i], dim=1)
            q_global = output_gate_signal if output_gate_signal is not None else search_quality_signal

        # ===== Restore interference for template self-attention (not cross-modal) =====
        if self.use_signal_decouple and fused_t_interference is not None:
            fused_t_attn = fused_t_attn + fused_t_interference

        # Template self-attention (with full information: signal + interference)
        x_v[:, :lens_z, :] = self.ca_t2t_f2v(
            torch.cat([x_v[:, :lens_z, :], fused_t_attn], dim=1))[:, :lens_z, :]
        x_i[:, :lens_z, :] = self.ca_t2t_f2i(
            torch.cat([x_i[:, :lens_z, :], fused_t_attn], dim=1))[:, :lens_z, :]

        return x_v, x_i, q_global
