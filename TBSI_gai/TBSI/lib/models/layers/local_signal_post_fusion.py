"""Explicit S1 localization with S2 no-op/RGB/TIR utility decisions."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.search = nn.Linear(768, 64, bias=False)
        self.template = nn.Linear(768, 64, bias=False)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, search, template):
        query = F.normalize(self.search(F.layer_norm(search, (768,))), dim=-1)
        key = F.normalize(self.template(F.layer_norm(template, (768,))), dim=-1)
        return self.scale * (query * key[:, None, None, :]).sum(-1) + self.bias


class LocalSignalPostFusion(nn.Module):
    def __init__(self, mode, artifact_dir, rho=0.05):
        super().__init__()
        if mode not in ('fixed_rgb', 'fixed_tir', 'random', 's1s2', 's1s2s3'):
            raise ValueError('unsupported local signal mode: ' + str(mode))
        self.mode = mode
        self.rho = float(rho)
        self.template_fg = None
        self.runtime_s3_features = None
        self.last_decision = None
        self.random_counter = 0
        self.rgb_response = None
        self.tir_response = None
        self.models = None
        if mode in ('fixed_rgb', 'fixed_tir', 'random', 's1s2', 's1s2s3'):
            artifact = Path(artifact_dir)
            heads = torch.load(str(artifact / 's1_response_heads.pth'), map_location='cpu')
            self.rgb_response = _Response()
            self.tir_response = _Response()
            self.rgb_response.load_state_dict(heads['rgb'], strict=True)
            self.tir_response.load_state_dict(heads['tir'], strict=True)
            self.rgb_response.eval()
            self.tir_response.eval()
            if mode in ('s1s2', 's1s2s3'):
                self.models = json.loads((artifact / 'local_utility_models.json').read_text())
                if mode not in self.models:
                    raise RuntimeError('missing local utility model: ' + mode)

    def train(self, mode=True):
        super().train(mode)
        if self.rgb_response is not None:
            self.rgb_response.eval()
            self.tir_response.eval()
        return self

    @staticmethod
    def _summary(score):
        flat = score.float().reshape(-1)
        peak_idx = int(flat.argmax().item())
        y, x = divmod(peak_idx, 16)
        yy, xx = torch.meshgrid(torch.arange(16, device=score.device),
                                torch.arange(16, device=score.device), indexing='ij')
        valid = ((xx - x) ** 2 + (yy - y) ** 2) > 4
        other = score.float()[valid]
        prob = torch.softmax(flat, dim=0)
        entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum() / math.log(256.)
        return (flat[peak_idx], flat[peak_idx] - other.max(),
                flat[peak_idx] - flat.mean(), entropy,
                torch.tensor(float(x), device=score.device),
                torch.tensor(float(y), device=score.device))

    def _signals(self, cat_feature, rgb, tir, fused):
        if self.template_fg is None:
            raise RuntimeError('template foreground weights not initialized')
        weights = self.template_fg.to(device=cat_feature.device,
                                      dtype=cat_feature.dtype).reshape(1, 64, 1)
        denom = weights.sum(1).clamp_min(1e-6)
        template_rgb = (cat_feature[:, :64] * weights).sum(1) / denom
        template_tir = (cat_feature[:, 320:384] * weights).sum(1) / denom
        rgb_map = self.rgb_response(rgb.reshape(1, 16, 16, 768), template_rgb)[0]
        tir_map = self.tir_response(tir.reshape(1, 16, 16, 768), template_tir)[0]
        rgb_summary = self._summary(rgb_map)
        tir_summary = self._summary(tir_map)
        rgb_norm = rgb.float().norm(dim=-1)
        tir_norm = tir.float().norm(dim=-1)
        rgb_similarity = F.cosine_similarity(rgb.float(), template_rgb[:, None, :].float(), dim=-1)
        tir_similarity = F.cosine_similarity(tir.float(), template_tir[:, None, :].float(), dim=-1)
        base = list(rgb_summary) + list(tir_summary) + [
            rgb_norm.mean(), rgb_norm.std(unbiased=False), rgb_similarity.max(), rgb_similarity.std(unbiased=False),
            tir_norm.mean(), tir_norm.std(unbiased=False), tir_similarity.max(), tir_similarity.std(unbiased=False),
            fused.float().norm(dim=1).mean(),
        ]
        if self.mode == 's1s2s3':
            if self.runtime_s3_features is None:
                raise RuntimeError('S3 image features not initialized for current crop')
            base.extend(torch.as_tensor(self.runtime_s3_features, device=cat_feature.device,
                                        dtype=torch.float32).flatten().unbind())
        x = torch.stack([v.float() if torch.is_tensor(v) else torch.tensor(float(v), device=cat_feature.device)
                         for v in base])
        return rgb_map, tir_map, x

    @staticmethod
    def _local_mask(score):
        idx = int(score.detach().float().argmax().item())
        y, x = divmod(idx, 16)
        mask = torch.zeros_like(score, dtype=torch.float32)
        mask[max(0, y - 1):min(16, y + 2), max(0, x - 1):min(16, x + 2)] = 1.
        return mask[None, None]

    def _utility(self, x):
        model = self.models[self.mode]
        mean = torch.as_tensor(model['mean'], device=x.device, dtype=x.dtype)
        scale = torch.as_tensor(model['scale'], device=x.device, dtype=x.dtype).clamp_min(1e-6)
        weights = torch.as_tensor(model['weights'], device=x.device, dtype=x.dtype)
        xn = ((x - mean) / scale).clamp(-8., 8.)
        return weights[:, 0] + weights[:, 1:] @ xn

    def forward(self, fused, feat_rgb, feat_tir, rgb_only, tir_only, cat_feature):
        rgb_map = tir_map = None
        if self.mode in ('s1s2', 's1s2s3'):
            rgb_map, tir_map, signals = self._signals(cat_feature, feat_rgb.flatten(2).transpose(1, 2),
                                                       feat_tir.flatten(2).transpose(1, 2), fused)
            utilities = self._utility(signals)
            choice = int(torch.cat((utilities.new_zeros(1), utilities)).argmax().item()) - 1
            self.last_decision = {'choice': choice, 'utility_rgb': float(utilities[0]),
                                  'utility_tir': float(utilities[1])}
        elif self.mode == 'fixed_rgb':
            choice = 0
        elif self.mode == 'fixed_tir':
            choice = 1
        else:
            choice = (self.random_counter * 1103515245 + 12345) & 1
            self.random_counter += 1
        if choice < 0:
            return fused
        source = rgb_only if choice == 0 else tir_only
        score = (rgb_map if choice == 0 else tir_map)
        if score is None:
            # Fixed controls still localize with the same frozen S1 map.
            rgb_map, tir_map, _ = self._signals(cat_feature,
                feat_rgb.flatten(2).transpose(1, 2),
                feat_tir.flatten(2).transpose(1, 2), fused)
            score = rgb_map if choice == 0 else tir_map
        mask = self._local_mask(score).to(device=fused.device, dtype=fused.dtype)
        return fused + self.rho * mask * (source - fused)
