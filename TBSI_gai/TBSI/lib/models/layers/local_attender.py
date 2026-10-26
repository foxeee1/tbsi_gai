"""Same-resolution local weighted aggregation adapted from UPLiFT.

This module intentionally keeps the operator independent of the tracker. It
does not contain upsampling logic and uses conv_res=False in the experiment.
"""
import torch
from torch import nn
import torch.nn.functional as F


class LocalAttender(nn.Module):
    def __init__(self, in_channels, num_connected=5, conv_res=False, low_mem=False):
        super().__init__()
        if num_connected == 5:
            self.offsets = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
            self.padding = 1
        else:
            raise ValueError('G2 LocalAttender is fixed to the 5-point stencil')
        self.in_channels = in_channels
        self.num_connected = num_connected
        self.conv_res = conv_res
        self.low_mem = low_mem
        self.attender = nn.Conv2d(in_channels, num_connected, kernel_size=1)
        self.pad = nn.ReplicationPad2d(self.padding)

    def make_offsets(self, value):
        b, c, h, w = value.shape
        value = self.pad(value)
        offsets = [value[:, :, self.padding + dh:self.padding + dh + h,
                         self.padding + dw:self.padding + dw + w]
                   for dh, dw in self.offsets]
        return torch.stack(offsets, dim=2)

    def forward(self, guide, value):
        if guide.ndim != 4 or value.ndim != 4:
            raise ValueError('guide and value must be BCHW tensors')
        if guide.shape[0] != value.shape[0] or guide.shape[2:] != value.shape[2:]:
            raise ValueError('guide/value batch and spatial shapes must match')
        logits = self.attender(guide)
        if self.conv_res:
            raise ValueError('conv_res=True is intentionally disabled for G2')
        weights = F.softmax(logits, dim=1)
        if self.low_mem:
            offsets = self.make_offsets(value)
            result = torch.zeros_like(value)
            for index in range(self.num_connected):
                result = result + offsets[:, :, index] * weights[:, index:index + 1]
            return result
        offsets = self.make_offsets(value)
        return (offsets * weights.unsqueeze(1)).sum(dim=2)


__all__ = ['LocalAttender']
