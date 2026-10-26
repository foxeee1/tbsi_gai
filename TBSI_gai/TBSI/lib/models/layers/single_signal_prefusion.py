"""Read-only single-signal routing before the existing search fusion convolution."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Response(nn.Module):
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


def image_features_from_patch(patch, state):
    values = []
    for image in (patch[:, :, :3], patch[:, :, 3:]):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        center = gray[80:176, 80:176]
        ring = np.concatenate((gray[:64].reshape(-1), gray[-64:].reshape(-1),
                               gray[:, :64].reshape(-1), gray[:, -64:].reshape(-1)))
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        values.extend((float(gray.mean()), float(gray.std()),
                       float(((gray < 0.03) | (gray > 0.97)).mean()),
                       float(np.log1p(lap.var())),
                       float(abs(center.mean() - ring.mean()))))
    w, h = float(state[2]), float(state[3])
    values.extend((math.log(max(1.0, w * h)), math.log(max(1.0, min(w, h))),
                   math.log(max(w, h) / max(1.0, min(w, h)))))
    return values


class SingleSignalPrefusion(nn.Module):
    def __init__(self, mode, artifact_dir, alpha=0.15):
        super().__init__()
        if mode not in ('fixed_rgb', 'fixed_tir', 's1', 's2', 's3', 's1s2', 's1s2s3'):
            raise ValueError('unsupported signal mode: ' + str(mode))
        if not 0 < alpha < 1:
            raise ValueError('alpha must be in (0, 1)')
        if mode in ('fixed_rgb', 'fixed_tir'):
            self.mode = mode
            self.alpha = float(alpha)
            self.mean = self.scale = self.beta = None
            self.template_fg = None
            self.runtime_s3_features = None
            self.last_signal = None
            self.components = None
            return
        artifact = Path(artifact_dir)
        if mode in ('s1s2', 's1s2s3'):
            self.mode = mode
            self.alpha = float(alpha)
            self.mean = self.scale = self.beta = None
            self.template_fg = None
            self.runtime_s3_features = None
            self.last_signal = None
            names = ('s1', 's2') if mode == 's1s2' else ('s1', 's2', 's3')
            self.components = nn.ModuleList([SingleSignalPrefusion(name, artifact_dir, alpha)
                                             for name in names])
            return
        calibration = json.loads((artifact / 'signal_calibration.json').read_text())[mode]
        self.mode = mode
        self.alpha = float(alpha)
        self.mean = np.asarray(calibration['mean'], dtype=np.float64)
        self.scale = np.asarray(calibration['scale'], dtype=np.float64)
        self.beta = np.asarray(calibration['beta'], dtype=np.float64)
        self.template_fg = None
        self.runtime_s3_features = None
        self.last_signal = None
        self.components = None
        if mode == 's1':
            heads = torch.load(str(artifact / 's1_response_heads.pth'), map_location='cpu')
            self.rgb_response = Response()
            self.tir_response = Response()
            self.rgb_response.load_state_dict(heads['rgb'], strict=True)
            self.tir_response.load_state_dict(heads['tir'], strict=True)
            self.rgb_response.eval()
            self.tir_response.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 's1':
            self.rgb_response.eval()
            self.tir_response.eval()
        return self

    def _predict(self, cat_feature, rgb, tir):
        if self.mode == 'fixed_rgb':
            return 1.0
        if self.mode == 'fixed_tir':
            return -1.0
        features = self._features(cat_feature, rgb, tir)
        if len(features) != len(self.mean) or len(self.beta) != len(features) + 1:
            raise RuntimeError('signal/calibration dimension mismatch')
        prediction = float(self.beta[0] + ((features - self.mean) / self.scale) @ self.beta[1:])
        if not np.isfinite(prediction):
            raise RuntimeError('non-finite signal')
        return prediction

    @staticmethod
    def _response_summary(score):
        flat = score.reshape(-1)
        peak = int(flat.argmax().item())
        y, x = divmod(peak, 16)
        yy, xx = np.mgrid[:16, :16]
        other = score.detach().float().cpu().numpy()[((xx - x) ** 2 + (yy - y) ** 2) > 4]
        value = float(flat[peak].item())
        return (value, value - float(other.max()),
                value - float(flat.mean().item()),
                float(math.hypot(x - 7.5, y - 7.5)))

    @staticmethod
    def _feature_summary(search, template):
        pooled = template.mean(0)
        a = search.reshape(-1, search.shape[-1]).float()
        q = pooled.float()
        norms = a.norm(dim=1)
        similarity = (a @ q) / (norms * q.norm() + 1e-8)
        top = similarity.topk(8).values
        return (float(norms.mean().item()), float(norms.std(unbiased=False).item()),
                float(top[0].item()), float((top[0] - top[1:].mean()).item()),
                float(similarity.std(unbiased=False).item()))

    def _template_pools(self, cat_feature):
        if self.template_fg is None:
            raise RuntimeError('template foreground weights not initialized')
        weights = self.template_fg.to(device=cat_feature.device,
                                      dtype=cat_feature.dtype).reshape(1, 64, 1)
        denom = weights.sum(1).clamp_min(1e-6)
        rgb = (cat_feature[:, :64] * weights).sum(1) / denom
        tir = (cat_feature[:, 320:384] * weights).sum(1) / denom
        return rgb, tir

    def _features(self, cat_feature, rgb, tir):
        if cat_feature.shape[0] != 1 or rgb.shape[1:] != (256, 768):
            raise RuntimeError('single-signal tracker requires B=1 and 256x768 search tokens')
        if self.mode == 's3':
            if self.runtime_s3_features is None:
                raise RuntimeError('S3 image features not initialized for current frame')
            return np.asarray(self.runtime_s3_features, dtype=np.float64)
        trgb, ttir = self._template_pools(cat_feature)
        if self.mode == 's1':
            with torch.no_grad():
                r = self.rgb_response(rgb.reshape(1, 16, 16, 768), trgb)[0]
                t = self.tir_response(tir.reshape(1, 16, 16, 768), ttir)[0]
            return np.asarray(self._response_summary(r) + self._response_summary(t),
                              dtype=np.float64)
        with torch.no_grad():
            return np.asarray(self._feature_summary(rgb[0], trgb) +
                              self._feature_summary(tir[0], ttir), dtype=np.float64)

    def forward(self, cat_feature, rgb, tir):
        if self.components is not None:
            for component in self.components:
                component.template_fg = self.template_fg
                component.runtime_s3_features = self.runtime_s3_features
            predictions = [component._predict(cat_feature, rgb, tir)
                           for component in self.components]
            directions = [1.0 if value >= 0 else -1.0 for value in predictions]
            vote = sum(directions)
            direction = 1.0 if vote > 0 else (-1.0 if vote < 0 else 0.0)
            prediction = float(np.mean(predictions))
            self.last_signal = {'prediction': prediction, 'direction': direction,
                                'component_predictions': predictions,
                                'component_directions': directions,
                                'vote': vote, 'alpha': self.alpha}
        else:
            prediction = self._predict(cat_feature, rgb, tir)
            direction = 1.0 if prediction > 0 else (-1.0 if prediction < 0 else 0.0)
            self.last_signal = {'prediction': prediction, 'direction': direction,
                                'alpha': self.alpha}
        return rgb * (1.0 + self.alpha * direction), tir * (1.0 - self.alpha * direction)
