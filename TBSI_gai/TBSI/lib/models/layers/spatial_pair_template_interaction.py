"""Search-conditioned template pair redistribution; source axis is always last.

Keeps the parent's A@(gV) evaluation order and adds a zero-exact residual.
Spatial agreement is a condition, not a target identity or utility label.
"""
import torch
from torch import nn


class SpatialPairTemplateInteraction(nn.Module):
    def __init__(self, mode='pair', temperature=0.1, max_bias=1.0):
        super().__init__()
        if mode not in ('pair', 'sep') or temperature <= 0 or max_bias <= 0:
            raise ValueError('Invalid spatial pair configuration')
        self.mode = mode
        self.temperature = float(temperature)
        self.max_bias = float(max_bias)
        self.strength = nn.Parameter(torch.zeros(()))
        self.last_diagnostics = None
        self.capture_diagnostics = False

    def condition(self, rgb, tir):
        if rgb.ndim != 3 or tir.shape != rgb.shape:
            raise ValueError('Expected paired [B,template,search] cosine maps')
        p = (rgb.float() / self.temperature).softmax(-1).sqrt()
        q = (tir.float() / self.temperature).softmax(-1).sqrt()
        p = p - p.mean(-1, keepdim=True)
        q = q - q.mean(-1, keepdim=True)
        c = p @ q.transpose(-1, -2)
        if self.mode == 'sep':
            c = c.mean(-1, keepdim=True) + c.mean(-2, keepdim=True) - c.mean((-1,-2), keepdim=True)
        return c

    def correction(self, attention, gain, condition):
        if attention.ndim != 4 or gain.shape != (attention.shape[0], attention.shape[-1]):
            raise ValueError('A=[B,H,receiver,source]; gain=[B,source]')
        if condition.shape != (attention.shape[0], attention.shape[-2], attention.shape[-1]):
            raise ValueError('Condition receiver/source axes differ')
        # FP32 expm1 avoids softmax(log B) roundtrip at zero and AMP underflow.
        b = attention.float() * gain.float()[:, None, None, :]
        mass = b.sum(-1, keepdim=True)
        t = torch.expm1(self.max_bias * self.strength.tanh() * condition[:, None])
        mean_t = (b*t).sum(-1, keepdim=True) / mass.clamp_min(torch.finfo(b.dtype).tiny)
        return b * (t - mean_t) / (1 + mean_t)

    def forward(self, msg_rt, msg_tr, a_rt, a_tr, v_tir, v_rgb, gate, s_rgb, s_tir):
        with torch.cuda.amp.autocast(enabled=False):
            c = self.condition(s_rgb, s_tir)
            drt = self.correction(a_rt, gate[...,1], c)
            dtr = self.correction(a_tr, gate[...,0], c.transpose(-1,-2))
            rt = drt @ v_tir.float()
            tr = dtr @ v_rgb.float()
            if self.capture_diagnostics:
                self.last_diagnostics = {
                    'strength': float(self.strength.detach()),
                    'effective_strength': float((self.max_bias*self.strength.tanh()).detach()),
                    'condition_rms': float(c.detach().square().mean().sqrt()),
                    'nonseparable_rms': float((c-c.mean(-1,keepdim=True)-c.mean(-2,keepdim=True)+c.mean((-1,-2),keepdim=True)).detach().square().mean().sqrt()),
                    'coefficient_delta_rms': float(drt.detach().square().mean().sqrt()),
                    'mass_residual_max': float(drt.detach().sum(-1).abs().max()),
                    'message_delta_rms': float(rt.detach().square().mean().sqrt())}
        return msg_rt + rt.to(msg_rt.dtype), msg_tr + tr.to(msg_tr.dtype)
