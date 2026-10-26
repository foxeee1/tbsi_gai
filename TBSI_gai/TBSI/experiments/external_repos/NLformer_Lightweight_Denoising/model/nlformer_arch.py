import torch
import os
import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np
import numbers
from einops import rearrange
from torch import einsum
from torch.nn.init import trunc_normal_

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################

def lazy_inversion_2d(U, eps=1e-4):
    A = U @ U.transpose(-2, -1)
    A = A + torch.eye(2, device=U.device, dtype=U.dtype).expand(A.shape)
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]

    # det(A) = ad - bc
    inv_det = 1.0 / (a * d - b * c + eps)

    invA = torch.empty_like(A)
    invA[..., 0, 0] =  d * inv_det
    invA[..., 0, 1] = -b * inv_det
    invA[..., 1, 0] = -c * inv_det
    invA[..., 1, 1] =  a * inv_det

    return invA

class Attention_block(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, bias):
        super(Attention_block, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim // num_heads
        self.scale = self.dim_head**-0.5

        self.pad_size = (self.overlap_win_size - window_size) // 2
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=0)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_c = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim*3, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = nn.Parameter(torch.zeros(1, self.window_size**2, self.overlap_win_size**2))
        trunc_normal_(self.rel_pos_emb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        
        self.phi = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # First-order statistics
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: F.pad(t, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='circular'), (ks, vs))
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.dim), (ks, vs))

        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out_s = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # Second-order statistics
        qv_c = self.qkv_dwconv(self.qkv_c(x))
        qc, vc = qv_c.chunk(2, dim=1)
        qc, vc = map(lambda t: rearrange(t, 'b (head c) h w -> b head c (h w)', head = self.num_heads), (qc, vc))

        qc = F.normalize(qc, dim=-1)

        cov = (qc @ qc.transpose(-2, -1)) * self.temperature
        cov = cov.softmax(dim=-1)
        out = (cov @ vc)

        out_c = rearrange(out, 'b head c (h w) -> b (head c) h w', head = self.num_heads, h = h, w = w)

        # Complementary information
        phi_3, phi_4 = self.phi(x).chunk(2, dim=1)
        phi_3, phi_4 = map(lambda t: rearrange(t, 'b (head c) h w -> b head (h w) 1 c', head = self.num_heads), (phi_3, phi_4))
        vc = rearrange(vc, 'b head c (h w) -> b head (h w) c 1', head = self.num_heads, h = h, w = w)

        # Woodbury Inversion
        U = torch.cat([phi_3, phi_4], dim=-2)
        inv_A = lazy_inversion_2d(U)
        
        out_phi = (U.transpose(-2, -1) @ inv_A) @ (U @ vc)
        out_phi = rearrange(out_phi, 'b head (h w) c 1 -> b (head c) h w', head = self.num_heads, h = h, w = w) 

        # merge first- and second-order statistics
        out = torch.cat([out_s, out_c, out_phi], dim=1) # dim*3
        out = self.project_out(out)

        return out

##########################################################################
class transformer_block(nn.Module):
    def __init__(self, n, head_num, in_dim, ffn_expansion_factor, bias, o_ratio, ln_bias):
        super(transformer_block, self).__init__()
        self.ln_1 = LayerNorm(in_dim, ln_bias)
        self.attention = Attention_block(in_dim, n, o_ratio, head_num, bias)

        self.ln_2 = LayerNorm(in_dim, ln_bias)
        self.ffn = FeedForward(in_dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

##########################################################################
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class NLformer(nn.Module):
    def __init__(self, channel=3, win=8, ln_bias='BiasFree', dim=20, ffn_factor=2.5, head_num=[1,2,4,8], num_blocks=[2,2,2,2], num_refine_blocks=2):
        super(NLformer, self).__init__()
        self.win = win if isinstance(win, list) else [win, win, win, win]
        self.chan = channel
        self.expand_factor = ffn_factor

        self.conv_first = nn.Conv2d(channel, dim, kernel_size=3, padding=1, bias=False)
        self.conv_last = nn.Conv2d(int(dim*2**1), channel, kernel_size=3, padding=1, bias=False)

        self.encoder_level1 = nn.Sequential(*[transformer_block(self.win[0], head_num[0], dim, ffn_factor, False, 0.5, ln_bias) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[transformer_block(self.win[1], head_num[1], int(dim*2**1), ffn_factor, False, 0.5, ln_bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[transformer_block(self.win[2], head_num[2], int(dim*2**2), ffn_factor, False, 0.5, ln_bias) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[transformer_block(self.win[3], head_num[3], int(dim*2**3), ffn_factor, False, 0, ln_bias) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(*[transformer_block(self.win[2], head_num[2], int(dim*2**2), ffn_factor, False, 0.5, ln_bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[transformer_block(self.win[1], head_num[1], int(dim*2**1), ffn_factor, False, 0.5, ln_bias) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[transformer_block(self.win[0], head_num[0], int(dim*2**1), ffn_factor, False, 0.5, ln_bias) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[transformer_block(self.win[0], head_num[0], int(dim*2**1), ffn_factor, False, 0.5, ln_bias) for i in range(num_refine_blocks)])

    def forward(self, x_noisy):
        inp_enc_level1 = self.conv_first(x_noisy)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.conv_last(out_dec_level1)

        return out_dec_level1 + x_noisy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NLformer(channel=3, win=8, ln_bias='BiasFree', dim=20, ffn_factor=2.5, head_num=[1,2,4,8], num_blocks=[2,2,2,2], num_refine_blocks=2).to(device)
    # model = NLformer(channel=3, win=8, ln_bias='BiasFree', dim=48, ffn_factor=2.5, head_num=[1,2,4,8], num_blocks=[2,3,4,6], num_refine_blocks=2).to(device)

    input_img = torch.randn(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_img)
    print(flop_count_table(flops))
