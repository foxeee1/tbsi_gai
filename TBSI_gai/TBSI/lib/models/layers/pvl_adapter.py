"""PVL-style probabilistic cross-modal adapter for RGB-TIR features.

The probabilistic key/value formulation follows the public MedCLIPSeg PVL
idea (Apache-2.0, HealthX-Lab/MedCLIPSeg, commit 31d2220); this tracking
adapter is an independent implementation for paired visual modalities.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProbabilisticCrossAttention(nn.Module):
    """单头概率交叉注意力，使用 Key 方差惩罚不确定匹配。"""

    def __init__(self, dim, beta=2.35, sample_seed=42):
        super().__init__()
        self.dim = int(dim)
        self.beta = float(beta)
        self.sample_seed = int(sample_seed)
        self._sample_index = 0
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim * 2)
        self.v_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

    def forward(self, query, context):
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError('PVL tokens must be rank-3 [B,N,C] tensors')
        if query.shape[0] != context.shape[0] or query.shape[2] != self.dim or context.shape[2] != self.dim:
            raise ValueError('PVL query/context batch and channel dimensions must match')
        if query.device != context.device or query.dtype != context.dtype:
            raise ValueError('PVL query/context must share device and dtype')

        q = self.q_proj(query)
        k_mu, k_logvar = self.k_proj(context).chunk(2, dim=-1)
        v_mu, v_logvar = self.v_proj(context).chunk(2, dim=-1)
        k_mu = self.norm_k(k_mu)
        v_mu = self.norm_v(v_mu)
        k_var = F.softplus(k_logvar) + 1e-6
        v_var = F.softplus(v_logvar) + 1e-6

        mean_score = torch.matmul(q, k_mu.transpose(1, 2)) / math.sqrt(self.dim)
        variance_penalty = torch.matmul(q.square(), k_var.transpose(1, 2)) / self.dim
        score = mean_score - self.beta * torch.sqrt(variance_penalty.clamp_min(1e-8))
        attention = torch.softmax(score, dim=-1)

        # 训练期用模块私有种子重参数采样，不消耗共享模型的随机数流；推理期
        # 取 Value 期望，避免闭环跟踪中逐帧随机改变决策。
        if self.training:
            generator = torch.Generator(device=v_var.device)
            generator.manual_seed(self.sample_seed + self._sample_index)
            self._sample_index += 1
            noise = torch.randn(v_var.shape, device=v_var.device,
                                dtype=v_var.dtype, generator=generator)
            value = v_mu + torch.sqrt(v_var) * noise
        else:
            value = v_mu
        return self.out_proj(torch.matmul(attention, value))


class PVLAdapter(nn.Module):
    """瓶颈化双向 PVL 适配器，保持两路输入输出的形状与通道不变。"""

    def __init__(self, dim=768, adapter_dim=64, beta=2.35):
        super().__init__()
        if dim <= 0 or adapter_dim <= 0:
            raise ValueError('PVL dim and adapter_dim must be positive')
        self.dim = int(dim)
        self.adapter_dim = int(adapter_dim)
        self.rgb_down = nn.Linear(dim, adapter_dim, bias=False)
        self.tir_down = nn.Linear(dim, adapter_dim, bias=False)
        self.rgb_to_tir = _ProbabilisticCrossAttention(adapter_dim, beta, sample_seed=42)
        self.tir_to_rgb = _ProbabilisticCrossAttention(adapter_dim, beta, sample_seed=1042)
        self.rgb_up = nn.Linear(adapter_dim, dim, bias=False)
        self.tir_up = nn.Linear(adapter_dim, dim, bias=False)
        # 零初始化外层残差门：新模块初始严格等价于输入直通。
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, rgb, tir):
        if rgb.ndim != 3 or tir.ndim != 3:
            raise ValueError('PVL inputs must be [B,N,C] tokens')
        if rgb.shape != tir.shape:
            raise ValueError('PVL RGB/TIR token shapes must match, got {} and {}'.format(
                tuple(rgb.shape), tuple(tir.shape)))
        if rgb.shape[-1] != self.dim:
            raise ValueError('PVL expected {} input channels, got {}'.format(self.dim, rgb.shape[-1]))
        if rgb.device != tir.device or rgb.dtype != tir.dtype:
            raise ValueError('PVL RGB/TIR tensors must share device and dtype')

        rgb_low, tir_low = self.rgb_down(rgb), self.tir_down(tir)
        # 两个方向都读取原始瓶颈特征，避免顺序更新造成单向偏置。
        rgb_message = self.tir_to_rgb(rgb_low, tir_low)
        tir_message = self.rgb_to_tir(tir_low, rgb_low)
        rgb_out = rgb + self.residual_scale * self.rgb_up(rgb_message)
        tir_out = tir + self.residual_scale * self.tir_up(tir_message)
        if rgb_out.shape != rgb.shape or tir_out.shape != tir.shape:
            raise RuntimeError('PVL adapter changed its input tensor shape')
        return rgb_out, tir_out
