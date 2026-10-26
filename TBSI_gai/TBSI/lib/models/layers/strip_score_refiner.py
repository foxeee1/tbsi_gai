"""Task-adapted strip spatial gate for the CENTER score branch."""
import torch
from torch import nn


class StripScoreRefiner(nn.Module):
    def __init__(self, dim=256, grid_size=16, strip_size=7, alpha_init=0.0):
        super().__init__()
        if strip_size < 3 or strip_size % 2 != 1 or strip_size > grid_size:
            raise ValueError('strip_size must be odd and within the feature grid')
        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.strip_horizontal = nn.Conv2d(dim, dim, (1, strip_size),
                                           padding=(0, strip_size // 2), groups=dim)
        self.strip_vertical = nn.Conv2d(dim, dim, (strip_size, 1),
                                         padding=(strip_size // 2, 0), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (self.dim, self.grid_size, self.grid_size):
            raise ValueError('StripScoreRefiner expects [B,{},{},{}], got {}'.format(
                self.dim, self.grid_size, self.grid_size, tuple(x.shape)))
        attention = self.conv1(self.strip_vertical(self.strip_horizontal(self.conv0(x))))
        return x + self.alpha * (x * attention)
