# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# H-WeightFormer: hierarchical WeightFormer
# Hierarchical macro design follows H-ViT^3, while the token mixer and the
# channel mixer are built from the dynamic weight-generation blocks of wfm.py.
# No attention is used anywhere in this model.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DynLinear(nn.Module):
    """Linear layer whose weight is a static parameter plus an input-conditioned residual.

    The dynamic part is predicted from a 2x downsampled version of the token map, then
    factorized through ``w1``/``w2`` so the generator stays cheap.
    """

    def __init__(self, in_features, out_features=None, rate=4):
        super().__init__()
        self.scale = in_features**-0.5
        self.rate = rate
        out_features = out_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))
        self.wfm = nn.Linear(in_features, in_features // rate + out_features // rate)
        self.w1 = nn.Parameter(torch.randn(in_features // rate, in_features))
        self.w2 = nn.Parameter(torch.randn(out_features // rate, out_features))
        self.pool = nn.AvgPool2d(2)
        trunc_normal_(self.w, std=0.02)
        trunc_normal_(self.b, std=0.02)
        trunc_normal_(self.w1, std=0.02)
        trunc_normal_(self.w2, std=0.02)

    def forward(self, x, h, w):
        B, N, C = x.shape
        x_pool = self.pool(x.transpose(1, 2).reshape(B, C, h, w)).flatten(2).transpose(1, 2)
        wfm = self.wfm(x_pool)
        wfm1, wfm2 = torch.split(
            wfm, [self.in_features // self.rate, self.out_features // self.rate], dim=-1
        )
        wfm1 = F.silu(wfm1) @ self.w1
        wfm2 = F.silu(wfm2) @ self.w2
        weight = wfm1.transpose(-2, -1) @ wfm2 * self.scale / N
        weight = weight / (weight.norm(dim=-2, keepdim=True) + 1.0)
        weight = weight + self.w
        return x @ weight + self.b


class DynCnn(nn.Module):
    """Depth-wise 3x3 conv whose kernel is a static parameter plus a predicted residual."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim, 1, 3, 3))
        self.b = nn.Parameter(torch.randn(dim))
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.wfm = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        trunc_normal_(self.w, std=0.02)
        trunc_normal_(self.b, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.wfm(self.pool(x)).reshape(B * C, 1, 3, 3)
        weight = weight / (weight.norm(dim=(-2, -1), keepdim=True) + 1.0)
        weight = weight + self.w.repeat(B, 1, 1, 1)
        x = F.conv2d(
            x.reshape(1, B * C, H, W), weight, self.b.repeat(B), padding=1, groups=B * C
        )
        return x.reshape(B, C, H, W)


class Mlp(nn.Module):
    """Channel mixer with an inner depth-wise conv, matching the H-ViT^3 FFN.

    When ``dynamic`` is set, the expansion projection becomes a ``DynLinear``.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., dynamic=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dynamic = dynamic
        if dynamic:
            self.fc1 = DynLinear(in_features, hidden_features)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwc = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                             groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x, h, w) if self.dynamic else self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + self.dwc(x.reshape(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class WFMBlock(nn.Module):
    r""" H-WeightFormer block.

    Token mixing is done by a depth-wise conv, which becomes a ``DynCnn`` on dynamic
    layers; channel mixing is the H-ViT^3 FFN, whose expansion becomes a ``DynLinear``
    on dynamic layers.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        dynamic (bool): Whether this block uses dynamic weight generation.
    """

    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, dynamic=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.dynamic = dynamic

        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        if self.dynamic:
            self.dwc = DynCnn(dim)
        else:
            self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop, dynamic=self.dynamic)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # Token mixing
        shortcut = x
        x = self.norm1(x)
        x = self.dwc(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"mlp_ratio={self.mlp_ratio}, dynamic={self.dynamic}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
    """

    def __init__(self, input_resolution, dim, dim_out, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        in_channels = dim
        out_channels = dim_out
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x


class BasicLayer(nn.Module):
    """ A basic H-WeightFormer layer for one stage.

    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        dynamic (list[bool]): Per-block flags telling which blocks are dynamic.
    """

    def __init__(self, dim, dim_out, input_resolution, depth, mlp_ratio=4., drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 dynamic=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        if dynamic is None:
            dynamic = [False] * depth

        # build blocks
        self.blocks = nn.ModuleList([
            WFMBlock(dim=dim, input_resolution=input_resolution, mlp_ratio=mlp_ratio, drop=drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, dynamic=dynamic[i])
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, dim_out=dim_out)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        self.conv = nn.Sequential(
            ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class HWFM(nn.Module):
    r""" H-WeightFormer

    Hierarchical WeightFormer: an attention-free model whose macro design follows
    H-ViT^3 and whose mixers are the dynamic weight-generation blocks of WeightFormer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        dim (tuple(int)): Patch embedding dimension of each stage.
        depths (tuple(int)): Depth of each stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        dyn_period (int | tuple(int)): One block out of every ``dyn_period`` is dynamic.
            A per-stage tuple is allowed; use 1 to make every block of a stage dynamic,
            or 0 to disable dynamic blocks in that stage. Default: 2
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 dim=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 mlp_ratio=4., dyn_period=2, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dim[0]
        self.mlp_ratio = mlp_ratio

        if isinstance(dyn_period, (int, float)):
            dyn_period = [int(dyn_period)] * self.num_layers
        self.dyn_period = list(dyn_period)
        # Place dynamic blocks at the end of each period so that, as in wfm.py, a stage
        # of exactly one period ends with its dynamic block.
        self.dynamic = [
            [p > 0 and (i + 1) % p == 0 for i in range(d)]
            for d, p in zip(depths, self.dyn_period)
        ]

        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim[0])
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=dim[i_layer],
                               dim_out=dim[i_layer + 1] if i_layer < self.num_layers - 1 else None,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               dynamic=self.dynamic[i_layer])
            self.layers.append(layer)

        self.norm = nn.BatchNorm1d(dim[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x.transpose(1, 2))
        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def h_wfm_tiny(**kwargs):
    model = HWFM(dim=[64, 128, 320, 512], depths=[2, 4, 14, 3], **kwargs)
    return model


def h_wfm_small(**kwargs):
    model = HWFM(dim=[64, 128, 320, 512], depths=[4, 8, 28, 6], **kwargs)
    return model


def h_wfm_base(**kwargs):
    model = HWFM(dim=[96, 192, 448, 640], depths=[4, 8, 25, 6], **kwargs)
    return model
