# Description: Utility functions for model
import torch
import os
import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np

# Count parameters
def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]

    top = slice(0, (h//2//refield+1)*refield)
    bottom = slice(h - (h//2//refield+1)*refield, h)
    left = slice(0, (w//2//refield+1)*refield)
    right = slice(w - (w//2//refield+1)*refield, w)
    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    with torch.no_grad():
        Es = [model(Ls[i]) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
    E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
    E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
    E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E


def test_nlmb(model, img_L, pt, n):
    h, w = img_L.size()[-2:]
    img_R = torch.zeros_like(img_L).to(img_L.device)
    pad = n // 2

    for i in range(0, h, pt):
        for j in range(0, w, pt):
            # Locate current coordinate
            patch_len_i = pt
            patch_len_j = pt

            start_i_est = min(i, h - pt)
            start_j_est = min(j, w - pt)

            start_i_padded = max(0, i - pad)
            start_j_padded = max(0, j - pad)

            end_i_est = min(h, i + pt)
            end_j_est = min(w, j + pt)

            end_i_padded = min(h, i + pt + pad)
            end_j_padded = min(w, j + pt + pad)

            # Extract patch
            patch_L = img_L[:, :, start_i_padded:end_i_padded, start_j_padded:end_j_padded]
            with torch.no_grad():
                patch_R = model(patch_L)

            # Fill in the patch
            img_R[:, :, start_i_est:end_i_est, start_j_est:end_j_est] = patch_R[:, :, start_i_est - start_i_padded:end_i_est - start_i_padded, start_j_est - start_j_padded:end_j_est - start_j_padded]

    return img_R