"""Target-weighted template HMoE adapted from SEATrack.

Original source: AutoLab-SAI-SJTU/SEATrack, MIT license.
Copyright and license notice retained in SEATrack_LICENSE.
This module changes only the Dispatch normalization to accept token weights.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRP(nn.Module):
    def __init__(self, in_dim=768, hid_dim=4, out_dim=768, drp1=0.1, drp2=0.1,
                 use_bias=True, if_act=False):
        super().__init__()
        self.lora_a = nn.Parameter(torch.empty(in_dim, hid_dim))
        self.lora_b = nn.Parameter(torch.empty(hid_dim, out_dim))
        self.use_bias = use_bias
        self.if_act = if_act
        if use_bias:
            self.bias_a = nn.Parameter(torch.empty(hid_dim))
            self.bias_b = nn.Parameter(torch.empty(out_dim))
        self.act = nn.GELU() if if_act else nn.Identity()
        for name, parameter in self.named_parameters():
            if 'lora' in name:
                nn.init.xavier_normal_(parameter, gain=math.sqrt(2))
            else:
                bound = 1 / math.sqrt(in_dim) if 'a' in name else 1 / math.sqrt(hid_dim)
                nn.init.uniform_(parameter, -bound, bound)
        self.drop1 = nn.Dropout(drp1)
        self.drop2 = nn.Dropout(drp2)

    def forward(self, x):
        if self.use_bias:
            x = torch.matmul(x, self.lora_a) + self.bias_a.view(1, 1, -1)
            x = self.drop1(self.act(x))
            return self.drop2(torch.matmul(x, self.lora_b) + self.bias_b.view(1, 1, -1))
        return self.drop2(self.drop1(self.act(x @ self.lora_a)) @ self.lora_b)


class MultiExpertLinear(nn.Module):
    def __init__(self, num_experts, in_dim, hid_dim, drp1, drp2,
                 use_bias=True, if_act=False):
        super().__init__()
        self.lora_a = nn.Parameter(torch.empty(num_experts, in_dim, hid_dim))
        self.lora_b = nn.Parameter(torch.empty(num_experts, hid_dim, in_dim))
        self.drop1 = nn.Dropout(drp1)
        self.drop2 = nn.Dropout(drp2)
        self.act = nn.GELU() if if_act else nn.Identity()
        self.use_bias = use_bias
        if use_bias:
            self.bias_a = nn.Parameter(torch.empty(num_experts, hid_dim))
            self.bias_b = nn.Parameter(torch.empty(num_experts, in_dim))
        else:
            self.register_parameter('bias', None)
        for parameter in self.parameters():
            nn.init.xavier_normal_(parameter, gain=math.sqrt(2))

    def forward(self, x):
        if self.use_bias:
            x = torch.matmul(x, self.lora_a) + self.bias_a.unsqueeze(1).unsqueeze(0)
            x = torch.matmul(self.drop1(self.act(x)), self.lora_b)
            x = x + self.bias_b.unsqueeze(1).unsqueeze(0)
            return self.drop2(x)
        return self.drop2(self.drop1(self.act(x @ self.lora_a)) @ self.lora_b)


class MaskedTemplateHMoE(nn.Module):
    def __init__(self, dim=768, experts=4, slots=4, hid_dim=4, prior_mode='hard'):
        super().__init__()
        if dim % slots:
            raise ValueError('dim must be divisible by slots')
        self.size_slots = slots
        self.size_experts = experts
        if prior_mode not in ('hard', 'disp_only', 'soft', 'none'):
            raise ValueError('prior_mode must be hard, disp_only, soft, or none')
        self.prior_mode = prior_mode
        thi = torch.empty(dim // slots, experts * slots)
        thi = nn.init.normal_(thi, mean=0, std=math.sqrt(1.0 / thi.size(0)))
        self.norm = nn.LayerNorm(dim)
        self.gate_thi = nn.Parameter(thi)
        self.linear1 = LoRP(in_dim=dim, hid_dim=hid_dim, out_dim=dim)
        self.experts = MultiExpertLinear(experts, dim // slots, hid_dim, 0.1, 0.1, use_bias=True)
        self.D_temp = nn.Parameter(torch.ones(1))
        self.C_temp = nn.Parameter(torch.ones(1))
        self.linear2 = LoRP(in_dim=dim, hid_dim=hid_dim, out_dim=dim)

    def forward(self, x, token_weight):
        if x.ndim != 3:
            raise ValueError('x must be [B,N,D]')
        if token_weight is None or token_weight.shape != x.shape[:2]:
            raise ValueError('token_weight must be [B,N]')
        if not torch.isfinite(token_weight).all():
            raise ValueError('non-finite template weights')
        token_weight = token_weight.to(device=x.device, dtype=x.dtype).clamp(0, 1)
        if (token_weight.sum(dim=1) <= 0).any():
            raise ValueError('a template contains no target-weighted token')

        batch, tokens, dim = x.shape
        thi = self.gate_thi.unsqueeze(0).expand(batch, -1, -1)
        routed = self.linear1(self.norm(x)).reshape(batch, tokens * self.size_slots, dim // self.size_slots)
        logits = torch.bmm(routed, thi)
        if self.prior_mode == 'soft':
            dispatch_weight = 1.0 + 0.5 * token_weight
        elif self.prior_mode == 'none':
            dispatch_weight = torch.ones_like(token_weight)
        else:
            dispatch_weight = token_weight
        slot_weight = dispatch_weight.repeat_interleave(self.size_slots, dim=1)
        dispatch_logits = logits / self.D_temp.clamp_min(0.05)
        dispatch_logits = dispatch_logits + torch.log(slot_weight.clamp_min(1e-4)).unsqueeze(-1)
        dispatch = F.softmax(dispatch_logits, dim=1)
        combine = logits.reshape(batch, tokens, self.size_slots,
                                 self.size_slots * self.size_experts).sum(dim=2)
        combine = combine.reshape(batch, tokens, self.size_experts, self.size_slots).sum(dim=-1)
        combine = F.softmax(combine / self.C_temp.clamp_min(0.05), dim=-1)
        experts_inputs = torch.bmm(dispatch.transpose(1, 2), routed).reshape(
            batch, self.size_experts, self.size_slots, dim // self.size_slots)
        experts_outputs = self.experts(experts_inputs).reshape(batch, self.size_experts, -1)
        experts_outputs = self.linear2(experts_outputs)
        moe_out = torch.bmm(combine, experts_outputs)
        if self.prior_mode == 'hard':
            return moe_out * token_weight.unsqueeze(-1)
        return moe_out
