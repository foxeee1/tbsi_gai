"""SCSU spatial-channel selection operator used by the exploratory Cscsu path."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSU(nn.Module):
    def __init__(self, channels, k=3, bias=False):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("k must be an odd integer")
        strip_k = 3 * k + 2
        self.dw_square = nn.Conv2d(channels, channels, k, padding=k // 2,
                                   groups=channels, bias=bias)
        self.dw_hstrip = nn.Conv2d(channels, channels, (1, strip_k),
                                   padding=(0, strip_k // 2),
                                   groups=channels, bias=bias)
        self.dw_vstrip = nn.Conv2d(channels, channels, (strip_k, 1),
                                   padding=(strip_k // 2, 0),
                                   groups=channels, bias=bias)
        self.weight_gen = nn.Conv2d(channels, channels * 3, 1, bias=True)

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.weight_gen(F.adaptive_avg_pool2d(x, 1))
        weights = torch.softmax(weights.view(b, 3, c, 1, 1), dim=1)
        branches = torch.stack((self.dw_square(x), self.dw_hstrip(x),
                                self.dw_vstrip(x)), dim=1)
        return (branches * weights).sum(dim=1)


class SpatialChannelSelection(nn.Module):
    """Split-channel SCSU operator, preserving the input tensor contract."""

    def __init__(self, channels, k1=3, k2=5, bias=False):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be divisible by 2")
        hidden = channels // 2
        self.scsu1 = SCSU(hidden, k=k1, bias=bias)
        self.scsu2 = SCSU(hidden, k=k2, bias=bias)
        self.proj = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return self.proj(torch.cat((self.scsu1(x1), self.scsu2(x2)), dim=1))
