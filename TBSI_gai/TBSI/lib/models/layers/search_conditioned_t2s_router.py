"""Search-conditioned template-to-search residual routing.

This module is deliberately independent from the native template MHSA.  It
operates on the real search axis (256 tokens) after ``tbsi_e`` template
fusion, so its effect can be audited at the same representation consumed by
the CENTER head.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F


class SearchConditionedT2SRouter(nn.Module):
    """Bounded T2S residual with a plain and an expert-routed mode.

    RGB search queries read TIR template keys/values and vice versa.  STSER
    adds shared/private response experts plus an explicit null response.  The
    outer ``gamma`` is zero initialized, preserving pretrained behavior while
    leaving a gradient path through the scalar residual strength.
    """

    def __init__(self, dim, num_heads=12, mode='stser', rank=64,
                 max_delta=0.10, temperature=0.07, competition_topk=4,
                 router_temperature=None, content_strength=0.0,
                 content_match_temperature=0.2, content_prior_floor=0.05,
                 content_search_factor=4.0, content_neighborhood='object'):
        super().__init__()
        mode = str(mode).lower()
        if mode not in ('plain_t2s', 'stser', 'stser_competition',
                        'stser_competition_content'):
            raise ValueError('unsupported search-conditioned T2S mode: {}'.format(mode))
        if dim % int(num_heads) != 0:
            raise ValueError('dim must be divisible by num_heads')
        self.mode = mode
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.temperature = max(float(temperature), 1e-4)
        self.router_temperature = (self.temperature if router_temperature is None else
                                   max(float(router_temperature), 1e-4))
        self.max_delta = max(float(max_delta), 0.0)
        self.competition_topk = max(int(competition_topk), 1)
        self.content_strength = float(content_strength)
        if mode != 'stser_competition_content' and self.content_strength != 0:
            raise ValueError('nonzero content strength requires stser_competition_content mode')
        self.content_match_temperature = float(content_match_temperature)
        self.content_prior_floor = float(content_prior_floor)
        self.content_search_factor = float(content_search_factor)
        self.content_neighborhood = str(content_neighborhood)
        if mode == 'stser_competition_content':
            if not 0.0 <= self.content_strength <= 1.0:
                raise ValueError('content strength must be in [0, 1]')
            if self.content_match_temperature <= 0 or self.content_search_factor <= 0:
                raise ValueError('content matching temperature and search factor must be positive')
            if not 0.0 < self.content_prior_floor <= 1.0:
                raise ValueError('content prior floor must be in (0, 1]')
            if self.content_neighborhood not in ('object', 'fixed3'):
                raise ValueError('unsupported content neighborhood')
            # Sigmoid(logit(0.1)) gives a bounded, nonzero initial correction.
            self.content_logit = nn.Parameter(torch.full((2,), -2.1972245773362196))
        self.q_rgb = nn.Linear(dim, dim, bias=False)
        self.k_tir = nn.Linear(dim, dim, bias=False)
        self.v_tir = nn.Linear(dim, dim, bias=False)
        self.q_tir = nn.Linear(dim, dim, bias=False)
        self.k_rgb = nn.Linear(dim, dim, bias=False)
        self.v_rgb = nn.Linear(dim, dim, bias=False)
        self.null_key = nn.Parameter(torch.zeros(2, self.num_heads, self.head_dim))
        self.null_value = nn.Parameter(torch.zeros(2, self.num_heads, self.head_dim))
        self.out_rgb = nn.Linear(dim, dim, bias=False)
        self.out_tir = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.zeros(2))
        if self.mode in ('stser', 'stser_competition', 'stser_competition_content'):
            hidden = max(8, int(rank))
            self.shared_expert = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                nn.Linear(hidden, dim))
            self.private_rgb = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                nn.Linear(hidden, dim))
            self.private_tir = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                nn.Linear(hidden, dim))
            router_input_dim = dim * (3 if self.mode in (
                'stser_competition', 'stser_competition_content') else 2) + 1
            # Router sees query, cross-modal message and source agreement. The
            # competition variant adds a relative search-position descriptor.
            self.router_rgb = nn.Sequential(nn.Linear(router_input_dim, hidden),
                                            nn.GELU(), nn.Linear(hidden, 3))
            self.router_tir = nn.Sequential(nn.Linear(router_input_dim, hidden),
                                            nn.GELU(), nn.Linear(hidden, 3))
        self.last_stats = {'mode': 'off'}

    def _attend(self, query, key, value, null_index, return_projected=False):
        bsz, n_search, _ = query.shape
        n_template = key.shape[1]
        q = self.q_rgb(query) if null_index == 0 else self.q_tir(query)
        k = self.k_tir(key) if null_index == 0 else self.k_rgb(key)
        v = self.v_tir(value) if null_index == 0 else self.v_rgb(value)
        q = q.view(bsz, n_search, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n_template, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n_template, self.num_heads, self.head_dim).transpose(1, 2)
        null_k = self.null_key[null_index].to(dtype=k.dtype, device=k.device)
        null_v = self.null_value[null_index].to(dtype=v.dtype, device=v.device)
        k_all = torch.cat((k, null_k[None, :, None, :].expand(bsz, -1, -1, -1)), dim=2)
        v_all = torch.cat((v, null_v[None, :, None, :].expand(bsz, -1, -1, -1)), dim=2)
        weights = torch.softmax(torch.matmul(q, k_all.transpose(-2, -1)) * self.scale, dim=-1)
        message = torch.matmul(weights, v_all).transpose(1, 2).reshape(bsz, n_search, self.dim)
        if return_projected:
            return message, weights[..., :n_template], weights[..., -1], q, k, v
        return message, weights[..., :n_template], weights[..., -1]

    def _content_correction(self, q, k, v, foreground, modality_index):
        bsz, _, n_search, _ = q.shape
        n_template = k.shape[2]
        side = int(round(n_search ** 0.5))
        template_side = int(round(n_template ** 0.5))
        if side * side != n_search or template_side * template_side != n_template:
            raise ValueError('content correction requires square template and search grids')
        if foreground is None or foreground.shape != (bsz, n_template):
            raise ValueError('content correction requires a [B,N_template] foreground mask')
        if foreground.device != q.device or not torch.isfinite(foreground).all():
            raise ValueError('foreground mask must be finite and on the query device')
        if (foreground < 0).any() or (foreground > 1).any() or (foreground.sum(dim=1) <= 0).any():
            raise ValueError('foreground coverage must contain target pixels in [0, 1]')
        probe = os.environ.get('TBSI_STSER_CONTENT_PROBE', 'correct')
        if probe not in ('correct', 'off', 'roll_mask', 'roll_competition', 'scale_only'):
            raise ValueError('unsupported STSER content probe: ' + probe)
        if self.training and probe != 'correct':
            raise RuntimeError('STSER content probes are inference only')
        foreground = foreground.float().clamp(0.0, 1.0)
        if probe == 'roll_mask':
            foreground = torch.roll(foreground, shifts=template_side, dims=1)
        prior = self.content_prior_floor + (1.0 - self.content_prior_floor) * foreground
        scores = torch.matmul(F.normalize(q.float(), dim=-1, eps=1e-6),
                              F.normalize(k.float(), dim=-1, eps=1e-6).transpose(-2, -1))
        scores = scores / self.content_match_temperature

        if self.content_neighborhood == 'object':
            patch = foreground.reshape(bsz, template_side, template_side)
            width = (patch > 0).any(dim=1).float().sum(dim=1).clamp_min(1e-3)
            height = (patch > 0).any(dim=2).float().sum(dim=1).clamp_min(1e-3)
            ratio = (width / height).clamp(1e-3, 1e3)
            radius_x = torch.ceil(side * torch.sqrt(ratio) / self.content_search_factor).long()
            radius_y = torch.ceil(side / (self.content_search_factor * torch.sqrt(ratio))).long()
        else:
            radius_x = torch.ones(bsz, device=q.device, dtype=torch.long)
            radius_y = torch.ones(bsz, device=q.device, dtype=torch.long)
        pos = torch.arange(n_search, device=q.device)
        x = pos.remainder(side).unsqueeze(0)
        y = (pos // side).unsqueeze(0)
        left = (x - radius_x.unsqueeze(1)).clamp(min=0, max=side)
        right = (x + radius_x.unsqueeze(1) + 1).clamp(min=0, max=side)
        top = (y - radius_y.unsqueeze(1)).clamp(min=0, max=side)
        bottom = (y + radius_y.unsqueeze(1) + 1).clamp(min=0, max=side)
        reference_count = n_search - (right - left) * (bottom - top)

        # A summed-area table excludes each query's object-scale rectangle
        # without materializing a [B,H,N_search,N_search,N_template] tensor.
        score_max = scores.amax(dim=2)
        mass = (scores - score_max.unsqueeze(2)).exp().permute(0, 1, 3, 2)
        mass = mass.reshape(bsz, self.num_heads, n_template, side, side)
        # Float64 accumulation avoids cancellation when a large local window
        # is subtracted from the global mass during the reference lookup.
        integral = F.pad(mass.double(), (1, 0, 1, 0)).cumsum(dim=-2).cumsum(dim=-1)
        integral = integral.flatten(start_dim=-2)

        def corners(yy, xx):
            index = (yy * (side + 1) + xx)[:, None, None, :]
            return integral.gather(-1, index.expand(-1, self.num_heads, n_template, -1))

        local = (corners(bottom, right) - corners(top, right) -
                 corners(bottom, left) + corners(top, left))
        total = mass.double().sum(dim=(-2, -1)).unsqueeze(-1)
        outside = (total - local).clamp_min(1e-12)
        log_background = (outside.log() + score_max.unsqueeze(-1) -
                          reference_count.clamp_min(1).float().log()[:, None, None, :])
        contrast = torch.tanh(scores - log_background.permute(0, 1, 3, 2).float())
        valid_reference = (reference_count > 0)[:, None, :, None] & (
            outside.permute(0, 1, 3, 2) > total.permute(0, 1, 3, 2) * 1e-6)
        contrast = contrast.masked_fill(~valid_reference, 0.0)
        if probe == 'roll_competition':
            contrast = torch.roll(contrast, shifts=side, dims=2)

        base = torch.softmax(scores + prior.log()[:, None, None, :], dim=-1)
        coefficient = torch.sigmoid(self.content_logit[modality_index])
        changed = torch.softmax(scores + prior.log()[:, None, None, :] +
                                coefficient * contrast, dim=-1)
        source_shift = changed - base
        correction = torch.matmul(source_shift, v.float())
        if probe == 'scale_only':
            base_message = torch.matmul(base, v.float())
            correction = base_message * (correction.float().norm(dim=-1, keepdim=True) /
                                         base_message.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        correction = correction.transpose(1, 2).reshape(bsz, n_search, self.dim)
        with torch.no_grad():
            stats = {
                'lambda': float(coefficient.detach()),
                'probe': probe,
                'source_shift_l1': float(source_shift.abs().sum(dim=-1).mean().detach()),
                'foreground_mass_base': float((base * foreground[:, None, None, :]).sum(dim=-1).mean().detach()),
                'foreground_mass_changed': float((changed * foreground[:, None, None, :]).sum(dim=-1).mean().detach()),
                'valid_reference_fraction': float(valid_reference.float().mean().detach()),
                'content_message_norm': float(correction.norm(dim=-1).mean().detach()),
                'content_message_cosine_base': float(F.cosine_similarity(
                    correction.float(), torch.matmul(base, v.float()).transpose(1, 2).reshape(
                        bsz, n_search, self.dim), dim=-1, eps=1e-6).mean().detach()),
            }
        return correction.to(dtype=v.dtype), stats

    def _bounded(self, delta, source):
        if self.max_delta <= 0:
            return torch.zeros_like(delta)
        source_norm = source.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta.float()
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        limit = self.max_delta * source_norm
        return (delta * (limit / delta_norm).clamp(max=1.0)).to(dtype=source.dtype)

    def _competition_descriptor(self, search):
        """Return a weighted hard-negative descriptor for every search token."""
        bsz, n_tokens, dim = search.shape
        side = int(round(n_tokens ** 0.5))
        if side * side != n_tokens:
            raise ValueError('competition routing expects a square search grid')
        normalized = F.normalize(search.float(), dim=-1, eps=1e-6)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        coords = torch.arange(side, device=search.device)
        gy, gx = torch.meshgrid(coords, coords)
        dy = (gy.reshape(-1, 1) - gy.reshape(1, -1)).abs()
        dx = (gx.reshape(-1, 1) - gx.reshape(1, -1)).abs()
        blocked = (dy <= 1) & (dx <= 1)
        similarity = similarity.masked_fill(blocked.unsqueeze(0), -1e4)
        k = min(self.competition_topk, max(1, n_tokens - int(blocked[0].sum())))
        values, indices = torch.topk(similarity, k=k, dim=-1)
        weights = torch.softmax(values / self.temperature, dim=-1)
        source = search.unsqueeze(1).expand(-1, n_tokens, -1, -1)
        competitors = torch.gather(
            source, 2, indices.unsqueeze(-1).expand(-1, -1, -1, dim))
        descriptor = (weights.unsqueeze(-1) * competitors).sum(dim=2)
        return descriptor.to(dtype=search.dtype)

    def _route(self, query, message, other, competitor=None):
        agreement = F.cosine_similarity(query.float(), other.float(), dim=-1,
                                        eps=1e-6).unsqueeze(-1)
        if self.mode in ('stser_competition', 'stser_competition_content'):
            if competitor is None:
                raise ValueError('competition descriptor is required')
            features = torch.cat((query, message, query - competitor,
                                  agreement.to(dtype=query.dtype)), dim=-1)
        else:
            features = torch.cat((query, message, agreement.to(dtype=query.dtype)), dim=-1)
        logits = self.router_rgb(features) if other is not None and self._routing_side == 0 else self.router_tir(features)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)
        shared = self.shared_expert(message)
        private = self.private_rgb(message) if self._routing_side == 0 else self.private_tir(message)
        null = torch.zeros_like(private)
        return weights[..., 0:1] * shared + weights[..., 1:2] * private + weights[..., 2:3] * null, weights

    def forward(self, template_rgb, template_tir, search_rgb, search_tir,
                template_fg_masks=None):
        if template_rgb.ndim != 3 or template_tir.ndim != 3 or search_rgb.ndim != 3 or search_tir.ndim != 3:
            raise ValueError('STSER expects four [B,N,C] token tensors')
        if template_rgb.shape[0] != search_rgb.shape[0] or template_tir.shape[0] != search_tir.shape[0]:
            raise ValueError('STSER batch dimensions must match')
        probe = os.environ.get('TBSI_STSER_CONTENT_PROBE', 'correct')
        if self.mode == 'stser_competition_content' and probe != 'correct' and self.training:
            raise RuntimeError('STSER content probes are inference only')
        content_active = (self.mode == 'stser_competition_content' and
                          self.content_strength > 0 and probe != 'off')
        if content_active:
            if template_fg_masks is None or len(template_fg_masks) != 2:
                raise ValueError('content correction requires RGB/TIR template foreground masks')
            attended_rgb = self._attend(search_rgb, template_tir, template_tir, 0,
                                        return_projected=True)
            attended_tir = self._attend(search_tir, template_rgb, template_rgb, 1,
                                        return_projected=True)
            msg_rgb, attn_rgb, null_rgb, q_rgb, k_tir, v_tir = attended_rgb
            msg_tir, attn_tir, null_tir, q_tir, k_rgb, v_rgb = attended_tir
        else:
            msg_rgb, attn_rgb, null_rgb = self._attend(search_rgb, template_tir, template_tir, 0)
            msg_tir, attn_tir, null_tir = self._attend(search_tir, template_rgb, template_rgb, 1)
        content_stats = None
        if self.mode == 'plain_t2s':
            delta_rgb = self.out_rgb(msg_rgb)
            delta_tir = self.out_tir(msg_tir)
            route_rgb = route_tir = None
        else:
            competitor_rgb = competitor_tir = None
            if self.mode in ('stser_competition', 'stser_competition_content'):
                competitor_rgb = self._competition_descriptor(search_rgb)
                competitor_tir = self._competition_descriptor(search_tir)
            self._routing_side = 0
            delta_rgb, route_rgb = self._route(search_rgb, msg_rgb, search_tir,
                                               competitor_rgb)
            self._routing_side = 1
            delta_tir, route_tir = self._route(search_tir, msg_tir, search_rgb,
                                               competitor_tir)
            if content_active:
                # RGB search reads the TIR template and vice versa. Both masks
                # currently originate from the same first-frame target box.
                correction_rgb, stats_rgb = self._content_correction(
                    q_rgb, k_tir, v_tir, template_fg_masks[1], 0)
                correction_tir, stats_tir = self._content_correction(
                    q_tir, k_rgb, v_rgb, template_fg_masks[0], 1)
                delta_rgb = delta_rgb + self.content_strength * (1 - route_rgb[..., 2:3]) * correction_rgb
                delta_tir = delta_tir + self.content_strength * (1 - route_tir[..., 2:3]) * correction_tir
                content_stats = (stats_rgb, stats_tir)
            delta_rgb = self.out_rgb(delta_rgb)
            delta_tir = self.out_tir(delta_tir)
        delta_rgb = self._bounded(delta_rgb, search_rgb)
        delta_tir = self._bounded(delta_tir, search_tir)
        out_rgb = search_rgb + self.gamma[0].to(dtype=search_rgb.dtype) * delta_rgb
        out_tir = search_tir + self.gamma[1].to(dtype=search_tir.dtype) * delta_tir
        with torch.no_grad():
            self.last_stats = {
                'mode': self.mode,
                'gamma_rgb': float(self.gamma[0].detach()),
                'gamma_tir': float(self.gamma[1].detach()),
                'rgb_message_norm': float(msg_rgb.float().norm(dim=-1).mean().detach()),
                'tir_message_norm': float(msg_tir.float().norm(dim=-1).mean().detach()),
                'rgb_bounded_delta_norm': float(delta_rgb.float().norm(dim=-1).mean().detach()),
                'tir_bounded_delta_norm': float(delta_tir.float().norm(dim=-1).mean().detach()),
                'rgb_null_mass': float(null_rgb.mean().detach()),
                'tir_null_mass': float(null_tir.mean().detach()),
                'rgb_route_private': (float(route_rgb[..., 1].mean().detach())
                                      if route_rgb is not None else None),
                'rgb_route_shared': (float(route_rgb[..., 0].mean().detach())
                                     if route_rgb is not None else None),
                'rgb_route_null': (float(route_rgb[..., 2].mean().detach())
                                   if route_rgb is not None else None),
                'tir_route_private': (float(route_tir[..., 1].mean().detach())
                                      if route_tir is not None else None),
                'tir_route_shared': (float(route_tir[..., 0].mean().detach())
                                     if route_tir is not None else None),
                'tir_route_null': (float(route_tir[..., 2].mean().detach())
                                   if route_tir is not None else None),
                'competition_mode': self.mode in ('stser_competition', 'stser_competition_content'),
            }
            if self.mode == 'stser_competition_content':
                self.last_stats['content_strength'] = self.content_strength
                self.last_stats['rgb_content_lambda'] = float(torch.sigmoid(self.content_logit[0]).detach())
                self.last_stats['tir_content_lambda'] = float(torch.sigmoid(self.content_logit[1]).detach())
                if content_stats is not None:
                    for modality, side_stats in zip(('rgb', 'tir'), content_stats):
                        for key, value in side_stats.items():
                            self.last_stats[modality + '_content_' + key] = value
        return out_rgb, out_tir
