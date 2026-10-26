"""Fixed-position bidirectional cross-modal residual for the FFM P0."""
import torch
from torch import nn


class CrossAttentionMessage(nn.Module):
    def __init__(self, dim=768, hidden_dim=192, num_heads=4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, hidden_dim, bias=True)
        self.k = nn.Linear(dim, hidden_dim, bias=True)
        self.v = nn.Linear(dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, query, source):
        if query.ndim != 3 or source.ndim != 3:
            raise ValueError('cross attention expects [B,N,C] tensors')
        if query.shape[0] != source.shape[0] or query.shape[2] != self.dim or source.shape[2] != self.dim:
            raise ValueError('cross attention tensor contract mismatch')
        b, nq, _ = query.shape
        nk = source.shape[1]
        q = self.q(query).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(source).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(source).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        message = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, nq, self.hidden_dim)
        return self.out(message)


class PostTBSICrossModalResidual(nn.Module):
    """Bidirectional zero-gated residual at post-block-5 search tokens."""
    def __init__(self, dim=768, mode='spatial', hidden_dim=192, num_heads=4, force_zero=False):
        super().__init__()
        if mode not in ('diag', 'spatial'):
            raise ValueError('mode must be diag or spatial')
        self.dim = int(dim)
        self.mode = mode
        self.force_zero = bool(force_zero)
        if mode == 'diag':
            self.rgb_from_tir = nn.Linear(dim, dim, bias=False)
            self.tir_from_rgb = nn.Linear(dim, dim, bias=False)
        else:
            self.rgb_from_tir = CrossAttentionMessage(dim, hidden_dim, num_heads)
            self.tir_from_rgb = CrossAttentionMessage(dim, hidden_dim, num_heads)
        self.gamma_rgb = nn.Parameter(torch.zeros(()))
        self.gamma_tir = nn.Parameter(torch.zeros(()))

    def forward(self, rgb, tir):
        if rgb.shape != tir.shape or rgb.ndim != 3 or rgb.shape[-1] != self.dim:
            raise ValueError('post-TBSI residual requires matching [B,N,dim] RGB/TIR tokens')
        if self.force_zero:
            return rgb, tir
        if self.mode == 'diag':
            rgb_message = self.rgb_from_tir(tir)
            tir_message = self.tir_from_rgb(rgb)
        else:
            rgb_message = self.rgb_from_tir(rgb, tir)
            tir_message = self.tir_from_rgb(tir, rgb)
        return rgb + self.gamma_rgb * rgb_message, tir + self.gamma_tir * tir_message
