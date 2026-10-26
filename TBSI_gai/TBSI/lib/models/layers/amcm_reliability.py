"""AMCM-lite target-conditioned spatial modality reliability gate.

This is an engineering adaptation, not an official AMCM implementation. It
keeps M1's learned fusion and only applies a zero-initialized, bounded,
per-location modulation to the two search streams.
"""
import torch
import torch.nn.functional as F
from torch import nn


class _NLformerAttentionCore(nn.Module):
    """Faithful three-branch NLformer attention core for 2-D token maps.

    This keeps the published operator's local first-order, channel covariance,
    and low-rank complementary branches. The host AMCM wrapper supplies the
    packed RGB/TIR representation and restores the two-stream contract.
    """
    def __init__(self, dim, heads=8, window=8, overlap_ratio=0.5):
        super().__init__()
        if dim % heads:
            raise ValueError('NLformer packed dim must be divisible by heads')
        self.dim, self.heads = int(dim), int(heads)
        self.window = int(window)
        self.dim_head = self.dim // self.heads
        self.overlap = int(window * overlap_ratio) + window
        self.pad = (self.overlap - self.window) // 2
        self.scale = self.dim_head ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_c = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 2, dim * 2, 3, padding=1,
                                groups=dim * 2, bias=False)
        self.phi = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.rel_pos = nn.Parameter(torch.zeros(1, window * window,
                                                self.overlap * self.overlap))
        self.project = nn.Conv2d(dim * 3, dim, 1, bias=False)

    @staticmethod
    def _woodbury(u, v):
        # U: [B,H,N,2,C], V: [B,H,N,C,1], matching the published low-rank
        # complementary branch while avoiding an external dependency.
        # torch.linalg.inv has no half-precision CPU/CUDA implementation in
        # the project's PyTorch version. Keep this tiny 2x2 solve in FP32 and
        # restore the caller dtype for mixed-precision inference.
        input_dtype = u.dtype
        u = u.float()
        v = v.float()
        a = torch.matmul(u, u.transpose(-2, -1))
        eye = torch.eye(2, device=u.device, dtype=u.dtype).view(1, 1, 1, 2, 2)
        a = a + eye
        # Explicit 2x2 inverse avoids the PyTorch 1.9 CUDA cuSPARSE path
        # used by linalg.inv, which is unavailable in some runtime images.
        a00, a01 = a[..., 0, 0], a[..., 0, 1]
        a10, a11 = a[..., 1, 0], a[..., 1, 1]
        det = (a00 * a11 - a01 * a10).clamp_min(1e-6)
        inv = torch.stack((a11 / det, -a01 / det,
                           -a10 / det, a00 / det), dim=-1).reshape_as(a)
        out = torch.matmul(torch.matmul(u.transpose(-2, -1), inv),
                           torch.matmul(u, v))
        return out.to(input_dtype)

    def forward(self, x):
        b, c, h, w = x.shape
        if h % self.window or w % self.window:
            raise ValueError('NLformer window must divide AMCM spatial size')
        q, k, v = self.qkv(x).chunk(3, dim=1)
        # Non-overlapping queries and overlapping keys/values, as in the
        # official implementation.
        q = q.view(b, self.heads, self.dim_head, h, w)
        q = q.unfold(3, self.window, self.window).unfold(4, self.window, self.window)
        q = q.permute(0, 3, 4, 1, 2, 5, 6).reshape(-1, self.heads,
                                                     self.window * self.window,
                                                     self.dim_head)
        pad = F.pad(torch.cat((k, v), dim=1), (self.pad,) * 4, mode='circular')
        k, v = pad.chunk(2, dim=1)
        k = F.unfold(k, self.overlap, stride=self.window)
        v = F.unfold(v, self.overlap, stride=self.window)
        k = k.transpose(1, 2).reshape(-1, self.heads, self.overlap * self.overlap,
                                       self.dim_head)
        v = v.transpose(1, 2).reshape(-1, self.heads, self.overlap * self.overlap,
                                       self.dim_head)
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = (attn + self.rel_pos.unsqueeze(0)).softmax(dim=-1)
        local = torch.matmul(attn, v).reshape(b, h // self.window, w // self.window,
                                               self.heads, self.window, self.window,
                                               self.dim_head)
        local = local.permute(0, 3, 6, 1, 4, 2, 5).reshape(b, c, h, w)

        qc, vc = self.qkv_dw(self.qkv_c(x)).chunk(2, dim=1)
        qc = qc.view(b, self.heads, self.dim_head, h * w)
        vc = vc.view(b, self.heads, self.dim_head, h * w)
        qc = F.normalize(qc, dim=-1)
        cov = torch.matmul(qc, qc.transpose(-2, -1)) * self.temperature
        cov = cov.softmax(dim=-1)
        channel = torch.matmul(cov, vc).reshape(b, c, h, w)

        p3, p4 = self.phi(x).chunk(2, dim=1)
        p3 = p3.view(b, self.heads, self.dim_head, h * w).permute(0, 1, 3, 2).unsqueeze(-2)
        p4 = p4.view(b, self.heads, self.dim_head, h * w).permute(0, 1, 3, 2).unsqueeze(-2)
        u = torch.cat((p3, p4), dim=-2)
        vv = vc.permute(0, 1, 3, 2).unsqueeze(-1)
        complementary = self._woodbury(u, vv).squeeze(-1)
        complementary = complementary.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.project(torch.cat((local, channel, complementary), dim=1)), {
            'local_live': local, 'channel_live': channel,
            'complementary_live': complementary, 'attention_live': attn,
        }


class AMCMLiteReliability(nn.Module):
    def __init__(self, dim=768, hidden_dim=32, max_delta=0.10, mode='legacy'):
        super().__init__()
        self.dim = int(dim)
        self.max_delta = float(max_delta)
        self.mode = str(mode)
        hidden_dim = max(8, int(hidden_dim))

        # Multi-scale local context, shared per channel to keep the adapter small.
        self.context3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.context5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.context7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.context_mix = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.GELU())

        # [rgb context, tir context, disagreement, target similarity x2].
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 3 + 2, hidden_dim, 1), nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        if self.mode in ('directed_residual', 'directed_residual_fg',
                         'directed_residual_mix', 'directed_residual_adaptive'):
            # The message is a learned cross-modal correction. It changes the
            # token representation before TBSI fusion instead of scaling the
            # already fused tensor.
            message_in = dim * 3 + 2
            self.message_rgb = nn.Sequential(
                nn.Conv2d(message_in, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1))
            self.message_tir = nn.Sequential(
                nn.Conv2d(message_in, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1))
            # Match the standard zero-initialized residual adapter contract:
            # enabling this branch starts exactly at the parent representation.
            nn.init.zeros_(self.message_rgb[-1].weight)
            nn.init.zeros_(self.message_rgb[-1].bias)
            nn.init.zeros_(self.message_tir[-1].weight)
            nn.init.zeros_(self.message_tir[-1].bias)
        elif self.mode == 'lcgm':
            # LCGM-inspired local self-gating and cross-modal interaction.
            # The paper emits one RGB-enhanced map; AMCM keeps its dual-stream
            # contract by projecting the shared local interaction per stream.
            self.lcgm_rgb_in = nn.Conv2d(dim, hidden_dim, 1, bias=False)
            self.lcgm_tir_in = nn.Conv2d(dim, hidden_dim, 1, bias=False)
            self.lcgm_rgb_gate = nn.Conv2d(hidden_dim, hidden_dim, 1)
            self.lcgm_tir_gate = nn.Conv2d(hidden_dim, hidden_dim, 1)
            self.lcgm_rgb_local = nn.Conv2d(
                hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
            self.lcgm_tir_local = nn.Conv2d(
                hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
            self.lcgm_rgb_out = nn.Conv2d(hidden_dim, dim, 1)
            self.lcgm_tir_out = nn.Conv2d(hidden_dim, dim, 1)
            nn.init.zeros_(self.lcgm_rgb_out.weight)
            nn.init.zeros_(self.lcgm_rgb_out.bias)
            nn.init.zeros_(self.lcgm_tir_out.weight)
            nn.init.zeros_(self.lcgm_tir_out.bias)
        elif self.mode in ('cbam', 'focal_modulation', 'swinfusion'):
            # The three screen candidates share the AMCM boundary and differ
            # only in the spatial evidence used to route a learned residual.
            # The last residual projections are zero initialized, preserving
            # exact parent behavior before training.
            signal_channels = {
                'cbam': 7,             # avg/max RGB/TIR + 2 template maps + disagreement
                'focal_modulation': 5, # 3 context scales + 2 template maps
                'swinfusion': 5,       # 2 context scales + agreement + 2 template maps
            }[self.mode]
            self.screen_router = nn.Sequential(
                nn.Conv2d(signal_channels, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, 2, 1))
            nn.init.zeros_(self.screen_router[-1].weight)
            nn.init.zeros_(self.screen_router[-1].bias)
            message_in = dim * 3 + 2
            self.screen_message_rgb = nn.Sequential(
                nn.Conv2d(message_in, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1))
            self.screen_message_tir = nn.Sequential(
                nn.Conv2d(message_in, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1))
            nn.init.zeros_(self.screen_message_rgb[-1].weight)
            nn.init.zeros_(self.screen_message_rgb[-1].bias)
            nn.init.zeros_(self.screen_message_tir[-1].weight)
            nn.init.zeros_(self.screen_message_tir[-1].bias)
        elif self.mode in ('ngram_local_fusion', 'ngram_local_fusion_v2'):
            # N-gram local fusion: aggregate each search stream over three
            # token neighborhoods, exchange the resulting local evidence, and
            # produce a target-conditioned residual before TBSI fusion. The
            # final projections are zero initialized so the disabled/untrained
            # contract is exactly identity.
            self.ngram_rgb_mix = nn.Sequential(
                nn.Conv2d(dim * 3, dim, 1, bias=False), nn.GELU())
            self.ngram_tir_mix = nn.Sequential(
                nn.Conv2d(dim * 3, dim, 1, bias=False), nn.GELU())
            message_in = dim * 3 + 2
            self.ngram_message_rgb = nn.Sequential(
                nn.Conv2d(message_in, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1))
            self.ngram_message_tir = nn.Sequential(
                nn.Conv2d(message_in, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1))
            nn.init.zeros_(self.ngram_message_rgb[-1].weight)
            nn.init.zeros_(self.ngram_message_rgb[-1].bias)
            nn.init.zeros_(self.ngram_message_tir[-1].weight)
            nn.init.zeros_(self.ngram_message_tir[-1].bias)
        elif self.mode == 'orthogonal_uncertainty':
            # Minimal cross-modal adapter: separate shared content from each
            # stream's orthogonal increment, then route only the increments
            # with a learned spatial reliability map.
            self.ou_router = nn.Sequential(
                nn.Conv2d(5, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, 2, 1))
            self.ou_message_rgb = nn.Sequential(
                nn.Conv2d(dim * 3, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1))
            self.ou_message_tir = nn.Sequential(
                nn.Conv2d(dim * 3, hidden_dim, 1), nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1))
            nn.init.zeros_(self.ou_router[-1].weight)
            nn.init.zeros_(self.ou_router[-1].bias)
            nn.init.zeros_(self.ou_message_rgb[-1].weight)
            nn.init.zeros_(self.ou_message_rgb[-1].bias)
            nn.init.zeros_(self.ou_message_tir[-1].weight)
            nn.init.zeros_(self.ou_message_tir[-1].bias)
        elif self.mode in ('nlformer', 'nlformer_spatial', 'nlformer_spatial_residual_gate', 'nlformer_consensus_residual_gate'):
            # Official NLformer core over packed RGB/TIR channels. The final
            # residual scale is zero so enabling the mode is exact identity;
            # audits can set it nonzero to inspect branch behavior.
            self.nlformer_inner_dim = 256
            self.nlformer_in = nn.Conv2d(dim * 2, self.nlformer_inner_dim, 1, bias=False)
            self.nlformer_norm = nn.GroupNorm(1, self.nlformer_inner_dim, affine=True)
            self.nlformer_core = _NLformerAttentionCore(self.nlformer_inner_dim,
                                                        heads=8)
            self.nlformer_out = nn.Conv2d(self.nlformer_inner_dim, dim * 2, 1, bias=False)
            self.nlformer_template = nn.Linear(dim * 2, self.nlformer_inner_dim, bias=False)
            nn.init.zeros_(self.nlformer_template.weight)
            if self.mode in ('nlformer_spatial', 'nlformer_spatial_residual_gate', 'nlformer_consensus_residual_gate'):
                # Preserve the NLformer core but replace global-only template
                # conditioning with two per-location template-match maps.
                # Zero initialization keeps the parent contract exact while
                # allowing the training loss to learn spatially selective use.
                self.nlformer_template_spatial = nn.Conv2d(
                    2, self.nlformer_inner_dim, 1, bias=False)
                nn.init.zeros_(self.nlformer_template_spatial.weight)
            self.nlformer_alpha = nn.Parameter(torch.zeros(1))

    def _context(self, x):
        return self.context_mix(self.context3(x) + self.context5(x) + self.context7(x))

    @staticmethod
    def _orthogonal_increment(x, shared):
        """Remove the component of x parallel to the shared feature."""
        basis = F.normalize(shared, dim=1, eps=1e-6)
        projection = (x * basis).sum(dim=1, keepdim=True) * basis
        return x - projection

    @staticmethod
    def _template_token_match(search, template, temperature=0.07,
                              token_weights=None, return_entropy=False):
        """Return per-search-token soft matching against the template tokens."""
        batch, channels, height, width = search.shape
        query = F.normalize(search, dim=1).flatten(2).transpose(1, 2)
        key = F.normalize(template, dim=-1)
        similarity = torch.bmm(query, key.transpose(1, 2))
        logits = similarity / temperature
        if token_weights is not None:
            weights = token_weights.to(device=logits.device, dtype=logits.dtype)
            if weights.ndim == 3:
                weights = weights.squeeze(1)
            if weights.ndim != 2 or weights.shape != (batch, key.shape[1]):
                raise ValueError('template foreground weights must be [B,N]')
            weights = weights.clamp_min(1e-4)
            weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
            logits = logits + weights[:, None, :].log()
        attention = torch.softmax(logits, dim=-1)
        match = (attention * similarity).sum(dim=-1)
        match = match.view(batch, 1, height, width)
        if return_entropy:
            # FP16 cannot represent 1e-8; compute entropy in FP32 so
            # zero-probability tokens cannot produce 0 * log(0) = NaN.
            attention_float = attention.float()
            entropy = -(attention_float * attention_float.clamp_min(1e-8).log()).sum(dim=-1)
            entropy = entropy / torch.log(torch.tensor(
                float(key.shape[1]), device=entropy.device, dtype=torch.float32))
            return match, entropy.view(batch, 1, height, width).clamp(0.0, 1.0).to(match.dtype)
        return match

    def forward(self, search_rgb, search_tir, template_rgb, template_tir,
                template_fg_masks=None):
        if search_rgb.ndim != 4 or search_tir.shape != search_rgb.shape:
            raise ValueError('AMCM-lite search features must match [B,C,H,W]')
        if template_rgb.ndim != 3 or template_tir.shape != template_rgb.shape:
            raise ValueError('AMCM-lite template tokens must match [B,N,C]')
        if search_rgb.shape[1] != self.dim or template_rgb.shape[2] != self.dim:
            raise ValueError('AMCM-lite channel contract violated')

        fg_rgb = fg_tir = None
        if template_fg_masks is not None:
            if not isinstance(template_fg_masks, (tuple, list)) or len(template_fg_masks) != 2:
                raise ValueError('template_fg_masks must be (rgb_mask, tir_mask)')
            fg_rgb, fg_tir = template_fg_masks
        if self.mode == 'orthogonal_uncertainty':
            shared = 0.5 * (search_rgb + search_tir)
            rgb_increment = self._orthogonal_increment(search_rgb - shared, shared)
            tir_increment = self._orthogonal_increment(search_tir - shared, shared)
            disagreement = torch.abs(search_rgb - search_tir)
            agreement = (F.normalize(search_rgb, dim=1) *
                         F.normalize(search_tir, dim=1)).sum(dim=1, keepdim=True)
            rgb_energy = search_rgb.norm(dim=1, keepdim=True)
            tir_energy = search_tir.norm(dim=1, keepdim=True)
            router_input = torch.cat((
                disagreement.mean(dim=1, keepdim=True),
                agreement, rgb_energy, tir_energy,
                torch.abs(rgb_energy - tir_energy)), dim=1)
            reliability = torch.softmax(self.ou_router(router_input), dim=1)
            message_input = torch.cat((shared, rgb_increment, tir_increment), dim=1)
            delta_rgb = torch.tanh(self.ou_message_rgb(message_input)) * self.max_delta
            delta_tir = torch.tanh(self.ou_message_tir(message_input)) * self.max_delta
            # A stream receives the other stream's complementary evidence.
            out_rgb = search_rgb + reliability[:, 1:2] * delta_rgb
            out_tir = search_tir + reliability[:, 0:1] * delta_tir
            return out_rgb, out_tir, {
                'rgb_gate_mean': float(reliability[:, 0:1].detach().mean()),
                'tir_gate_mean': float(reliability[:, 1:2].detach().mean()),
                'gate_abs_mean': float((reliability - 0.5).detach().abs().mean()),
                'rgb_gate_map_live': reliability[:, 0:1],
                'tir_gate_map_live': reliability[:, 1:2],
                'rgb_residual_live': out_rgb - search_rgb,
                'tir_residual_live': out_tir - search_tir,
                'rgb_input_live': search_rgb,
                'tir_input_live': search_tir,
                'rgb_output_live': out_rgb,
                'tir_output_live': out_tir,
                'shared_live': shared,
                'rgb_orthogonal_live': rgb_increment,
                'tir_orthogonal_live': tir_increment,
                'agreement_live': agreement,
                'router_input_live': router_input,
            }

        if self.mode in ('nlformer', 'nlformer_spatial', 'nlformer_spatial_residual_gate', 'nlformer_consensus_residual_gate'):
            packed = torch.cat((search_rgb, search_tir), dim=1)
            template = torch.cat((template_rgb.mean(dim=1),
                                  template_tir.mean(dim=1)), dim=-1)
            # Template conditioning is a zero-initialized bias adapter; the
            # audited NLformer branches remain the published computation.
            packed_inner = self.nlformer_in(packed)
            packed_inner = packed_inner + self.nlformer_template(template).unsqueeze(-1).unsqueeze(-1)
            if self.mode in ('nlformer_spatial', 'nlformer_spatial_residual_gate', 'nlformer_consensus_residual_gate'):
                rgb_match = self._template_token_match(search_rgb, template_rgb)
                tir_match = self._template_token_match(search_tir, template_tir)
                spatial_match = torch.cat((rgb_match, tir_match), dim=1)
                # Remove the per-sample DC component so the condition carries
                # location information rather than a global feature shift.
                spatial_match = spatial_match - spatial_match.mean(dim=(2, 3), keepdim=True)
                packed_inner = packed_inner + self.nlformer_template_spatial(spatial_match)
            normalized = self.nlformer_norm(packed_inner)
            core, branch = self.nlformer_core(normalized)
            residual = torch.tanh(self.nlformer_out(core)) * self.nlformer_alpha
            if self.mode in ('nlformer_spatial_residual_gate', 'nlformer_consensus_residual_gate'):
                # Single-variable follow-up: use the template match only as a
                # normalized spatial amplitude for the already-generated
                # residual. The mean is one, so this cannot act as a global
                # scale; it tests whether the residual content is useful once
                # its spatial support is target-conditioned.
                if self.mode == 'nlformer_consensus_residual_gate':
                    # A location is trusted only when both modalities support
                    # it. The minimum is a monotone soft-consensus test and
                    # rejects modality-specific background matches.
                    consensus = torch.minimum(spatial_match[:, 0:1],
                                              spatial_match[:, 1:2])
                    route = torch.exp(consensus / 0.07)
                else:
                    route = torch.exp(spatial_match / 0.07)
                route = route / route.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                route = route.clamp(0.25, 4.0)
                route = route / route.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                if self.mode == 'nlformer_consensus_residual_gate':
                    route = route.expand(-1, 2, -1, -1)
                residual = residual * torch.cat((
                    route[:, 0:1].expand(-1, self.dim, -1, -1),
                    route[:, 1:2].expand(-1, self.dim, -1, -1)), dim=1)
            output = packed + residual
            out_rgb, out_tir = output.chunk(2, dim=1)
            return out_rgb, out_tir, {
                'nlformer_alpha': float(self.nlformer_alpha.detach()),
                'nlformer_residual_live': residual,
                'nlformer_local_live': branch['local_live'],
                'nlformer_channel_live': branch['channel_live'],
                'nlformer_complementary_live': branch['complementary_live'],
                'nlformer_attention_live': branch['attention_live'],
                'nlformer_spatial_match_live': spatial_match if self.mode in ('nlformer_spatial', 'nlformer_spatial_residual_gate') else None,
                'nlformer_residual_route_live': route if self.mode in ('nlformer_spatial_residual_gate', 'nlformer_consensus_residual_gate') else None,
                'rgb_input_live': search_rgb, 'tir_input_live': search_tir,
                'rgb_output_live': out_rgb, 'tir_output_live': out_tir,
            }

        if self.mode in ('ngram_local_fusion', 'ngram_local_fusion_v2'):
            # Depthwise convolutions preserve the channel semantics while the
            # three receptive fields act as 3/5/7-token local n-grams.
            rgb_grams = torch.cat((self.context3(search_rgb),
                                   self.context5(search_rgb),
                                   self.context7(search_rgb)), dim=1)
            tir_grams = torch.cat((self.context3(search_tir),
                                   self.context5(search_tir),
                                   self.context7(search_tir)), dim=1)
            rgb_local = self.context_mix(self.ngram_rgb_mix(rgb_grams))
            tir_local = self.context_mix(self.ngram_tir_mix(tir_grams))
            rgb_sim = self._template_token_match(search_rgb, template_rgb)
            tir_sim = self._template_token_match(search_tir, template_tir)
            evidence = torch.cat((rgb_local, tir_local,
                                  torch.abs(rgb_local - tir_local),
                                  rgb_sim, tir_sim), dim=1)
            reliability = torch.softmax(self.gate(evidence), dim=1)
            # Remove spatially constant components: this branch must learn a
            # local correction rather than an uninformative global rescaling.
            msg_rgb = torch.tanh(self.ngram_message_rgb(evidence)) * self.max_delta
            msg_tir = torch.tanh(self.ngram_message_tir(evidence)) * self.max_delta
            msg_rgb = msg_rgb - msg_rgb.mean(dim=(2, 3), keepdim=True)
            msg_tir = msg_tir - msg_tir.mean(dim=(2, 3), keepdim=True)
            # The opposite stream's template match routes the message, so a
            # modality can borrow local evidence where its partner recognizes
            # the target instead of enhancing every background token.
            if self.mode == 'ngram_local_fusion_v2':
                # Sparse excess matching: only evidence above each search
                # map's spatial mean can route a residual. Normalization keeps
                # the route useful when the target occupies few cells while
                # suppressing broad background responses.
                tir_excess = F.relu(tir_sim - tir_sim.mean(dim=(2, 3), keepdim=True))
                rgb_excess = F.relu(rgb_sim - rgb_sim.mean(dim=(2, 3), keepdim=True))
                rgb_route = torch.clamp(
                    tir_excess / (tir_excess.mean(dim=(2, 3), keepdim=True) + 1e-6), 0.0, 3.0)
                tir_route = torch.clamp(
                    rgb_excess / (rgb_excess.mean(dim=(2, 3), keepdim=True) + 1e-6), 0.0, 3.0)
            else:
                rgb_route = torch.sigmoid((tir_sim - tir_sim.mean(dim=(2, 3), keepdim=True)) / 0.07)
                tir_route = torch.sigmoid((rgb_sim - rgb_sim.mean(dim=(2, 3), keepdim=True)) / 0.07)
            rgb_route = rgb_route * (0.5 + reliability[:, 1:2])
            tir_route = tir_route * (0.5 + reliability[:, 0:1])
            delta_rgb = msg_rgb * rgb_route
            delta_tir = msg_tir * tir_route
            out_rgb = search_rgb + delta_rgb
            out_tir = search_tir + delta_tir
            return out_rgb, out_tir, {
                'rgb_gate_mean': float(rgb_route.detach().mean()),
                'tir_gate_mean': float(tir_route.detach().mean()),
                'gate_abs_mean': float((delta_rgb.abs().mean() + delta_tir.abs().mean()).detach() * 0.5),
                'rgb_gate_map_live': rgb_route,
                'tir_gate_map_live': tir_route,
                'rgb_sim_map_live': rgb_sim,
                'tir_sim_map_live': tir_sim,
                'rgb_residual_live': delta_rgb,
                'tir_residual_live': delta_tir,
                'rgb_input_live': search_rgb,
                'tir_input_live': search_tir,
                'rgb_output_live': out_rgb,
                'tir_output_live': out_tir,
                'ngram_rgb_local_live': rgb_local,
                'ngram_tir_local_live': tir_local,
                'ngram_evidence_live': evidence,
            }
        if self.mode == 'lcgm':
            rgb_proj = self.lcgm_rgb_in(search_rgb)
            tir_proj = self.lcgm_tir_in(search_tir)
            rgb_gate = torch.sigmoid(self.lcgm_rgb_gate(rgb_proj))
            tir_gate = torch.sigmoid(self.lcgm_tir_gate(tir_proj))
            rgb_self = self.lcgm_rgb_local(rgb_proj * rgb_gate)
            tir_self = self.lcgm_tir_local(tir_proj * tir_gate)
            # LCGM's two reported directional products are algebraically the
            # same elementwise product. The AMCM adapter gives each stream its
            # own learned residual projection while sharing that interaction.
            cross_local = rgb_self * tir_self
            delta_rgb = torch.tanh(self.lcgm_rgb_out(cross_local)) * self.max_delta
            delta_tir = torch.tanh(self.lcgm_tir_out(cross_local)) * self.max_delta
            out_rgb = search_rgb + delta_rgb
            out_tir = search_tir + delta_tir
            return out_rgb, out_tir, {
                'rgb_gate_mean': float(rgb_gate.detach().mean()),
                'tir_gate_mean': float(tir_gate.detach().mean()),
                'gate_abs_mean': float((delta_rgb.abs().mean() +
                                        delta_tir.abs().mean()).detach() * 0.5),
                'rgb_gate_map_live': rgb_gate,
                'tir_gate_map_live': tir_gate,
                'rgb_residual_live': delta_rgb,
                'tir_residual_live': delta_tir,
                'rgb_input_live': search_rgb,
                'tir_input_live': search_tir,
                'rgb_output_live': out_rgb,
                'tir_output_live': out_tir,
                'lcgm_cross_local_live': cross_local,
                'lcgm_rgb_self_live': rgb_self,
                'lcgm_tir_self_live': tir_self,
            }
        rgb_ctx = self._context(search_rgb)
        tir_ctx = self._context(search_tir)
        if self.mode in ('directed_residual', 'directed_residual_fg',
                         'directed_residual_mix', 'directed_residual_adaptive'):
            if self.mode == 'directed_residual_fg':
                rgb_sim = self._template_token_match(search_rgb, template_rgb,
                                                     token_weights=fg_rgb)
                tir_sim = self._template_token_match(search_tir, template_tir,
                                                     token_weights=fg_tir)
            elif self.mode == 'directed_residual_mix':
                # Keep global context as a fallback: foreground matching is
                # useful when the mask is aligned, but should not erase all
                # template evidence when the mask is stale or contaminated.
                rgb_global = self._template_token_match(search_rgb, template_rgb)
                tir_global = self._template_token_match(search_tir, template_tir)
                rgb_fg = self._template_token_match(search_rgb, template_rgb,
                                                    token_weights=fg_rgb)
                tir_fg = self._template_token_match(search_tir, template_tir,
                                                    token_weights=fg_tir)
                rgb_sim = 0.5 * (rgb_global + rgb_fg)
                tir_sim = 0.5 * (tir_global + tir_fg)
            elif self.mode == 'directed_residual_adaptive':
                rgb_global = self._template_token_match(search_rgb, template_rgb)
                tir_global = self._template_token_match(search_tir, template_tir)
                rgb_fg, rgb_entropy = self._template_token_match(
                    search_rgb, template_rgb, token_weights=fg_rgb,
                    return_entropy=True)
                tir_fg, tir_entropy = self._template_token_match(
                    search_tir, template_tir, token_weights=fg_tir,
                    return_entropy=True)
                # Trust foreground matching only when its attention is sharp
                # in both modalities; otherwise fall back toward global match.
                confidence = 1.0 - torch.maximum(rgb_entropy, tir_entropy)
                rgb_sim = confidence * rgb_fg + (1.0 - confidence) * rgb_global
                tir_sim = confidence * tir_fg + (1.0 - confidence) * tir_global
            else:
                rgb_sim = self._template_token_match(search_rgb, template_rgb)
                tir_sim = self._template_token_match(search_tir, template_tir)
        else:
            rgb_template = F.normalize(template_rgb.mean(dim=1), dim=1)
            tir_template = F.normalize(template_tir.mean(dim=1), dim=1)
            rgb_sim = (F.normalize(search_rgb, dim=1) * rgb_template[:, :, None, None]).sum(dim=1, keepdim=True)
            tir_sim = (F.normalize(search_tir, dim=1) * tir_template[:, :, None, None]).sum(dim=1, keepdim=True)
        features = torch.cat((rgb_ctx, tir_ctx, torch.abs(rgb_ctx - tir_ctx), rgb_sim, tir_sim), dim=1)
        if self.mode in ('cbam', 'focal_modulation', 'swinfusion'):
            mean_sim = 0.5 * (rgb_sim + tir_sim)
            if self.mode == 'cbam':
                signal = torch.cat((
                    search_rgb.mean(1, keepdim=True), search_rgb.amax(1, keepdim=True),
                    search_tir.mean(1, keepdim=True), search_tir.amax(1, keepdim=True),
                    rgb_sim, tir_sim,
                    torch.abs(rgb_sim - tir_sim)), dim=1)
            elif self.mode == 'focal_modulation':
                signal = torch.cat((
                    F.avg_pool2d(mean_sim, 3, 1, 1),
                    F.avg_pool2d(mean_sim, 5, 1, 2),
                    F.avg_pool2d(mean_sim, 7, 1, 3), rgb_sim, tir_sim), dim=1)
            else:
                signal = torch.cat((
                    F.avg_pool2d(mean_sim, 3, 1, 1),
                    F.avg_pool2d(mean_sim, 5, 1, 2),
                    (F.normalize(search_rgb, dim=1) * F.normalize(search_tir, dim=1)).sum(1, keepdim=True),
                    rgb_sim, tir_sim), dim=1)
            route = torch.softmax(self.screen_router(signal), dim=1)
            msg_rgb = torch.tanh(self.screen_message_rgb(features)) * self.max_delta
            msg_tir = torch.tanh(self.screen_message_tir(features)) * self.max_delta
            # Cross-modal route: RGB correction is supported by TIR evidence,
            # and vice versa. This keeps the candidate as feature modulation,
            # not post-fusion score scaling.
            out_rgb = search_rgb + route[:, 1:2] * msg_rgb
            out_tir = search_tir + route[:, 0:1] * msg_tir
            return out_rgb, out_tir, {
                'rgb_gate_mean': float(route[:, 0:1].detach().mean()),
                'tir_gate_mean': float(route[:, 1:2].detach().mean()),
                'gate_abs_mean': float((route - 0.5).detach().abs().mean()),
                'rgb_gate_map_live': route[:, 0:1],
                'tir_gate_map_live': route[:, 1:2],
                'rgb_sim_map_live': rgb_sim,
                'tir_sim_map_live': tir_sim,
                'rgb_residual_live': out_rgb - search_rgb,
                'tir_residual_live': out_tir - search_tir,
                'rgb_input_live': search_rgb,
                'tir_input_live': search_tir,
                'rgb_output_live': out_rgb,
                'tir_output_live': out_tir,
                'screen_signal_live': signal,
            }
        logits = self.gate(features)
        if self.mode in ('directed_residual', 'directed_residual_fg',
                         'directed_residual_mix', 'directed_residual_adaptive'):
            # p_m is the learned modality reliability distribution. The less
            # reliable stream receives a correction routed from the other
            # stream, which is the operation the supervision describes.
            probs = torch.softmax(logits, dim=1)
            msg_rgb = torch.tanh(self.message_rgb(features)) * self.max_delta
            msg_tir = torch.tanh(self.message_tir(features)) * self.max_delta
            # Remove spatially constant corrections so the adapter can only
            # change feature content through location-dependent evidence.
            msg_rgb = msg_rgb - msg_rgb.mean(dim=(2, 3), keepdim=True)
            msg_tir = msg_tir - msg_tir.mean(dim=(2, 3), keepdim=True)
            # Route each correction with the opposite modality's target match:
            # a stream can contribute a spatial correction where it recognizes
            # the template, rather than spreading that message uniformly.
            rgb_route = 2.0 * torch.sigmoid(
                (tir_sim - tir_sim.mean(dim=(2, 3), keepdim=True)) / 0.07)
            tir_route = 2.0 * torch.sigmoid(
                (rgb_sim - rgb_sim.mean(dim=(2, 3), keepdim=True)) / 0.07)
            out_rgb = search_rgb + probs[:, 1:2] * msg_rgb * rgb_route
            out_tir = search_tir + probs[:, 0:1] * msg_tir * tir_route
            return out_rgb, out_tir, {
                'rgb_gate_mean': float(probs[:, 0:1].detach().mean()),
                'tir_gate_mean': float(probs[:, 1:2].detach().mean()),
                'gate_abs_mean': float((probs - 0.5).detach().abs().mean()),
                'rgb_gate_map': probs[:, 0:1].detach(),
                'tir_gate_map': probs[:, 1:2].detach(),
                'rgb_gate_map_live': probs[:, 0:1],
                'tir_gate_map_live': probs[:, 1:2],
                'rgb_logits_live': logits[:, 0:1],
                'tir_logits_live': logits[:, 1:2],
                'rgb_prob_live': probs[:, 0:1],
                'tir_prob_live': probs[:, 1:2],
                'rgb_sim_map_live': rgb_sim,
                'tir_sim_map_live': tir_sim,
                'rgb_message_norm_live': msg_rgb,
                'tir_message_norm_live': msg_tir,
                'rgb_residual_live': out_rgb - search_rgb,
                'tir_residual_live': out_tir - search_tir,
                'rgb_input_live': search_rgb,
                'tir_input_live': search_tir,
                'rgb_output_live': out_rgb,
                'tir_output_live': out_tir,
            }
        delta = torch.tanh(logits) * self.max_delta
        return search_rgb * (1.0 + delta[:, 0:1]), search_tir * (1.0 + delta[:, 1:2]), {
            'rgb_gate_mean': float(delta[:, 0:1].detach().mean()),
            'tir_gate_mean': float(delta[:, 1:2].detach().mean()),
            'gate_abs_mean': float(delta.detach().abs().mean()),
            'rgb_gate_map': delta[:, 0:1].detach(),
            'tir_gate_map': delta[:, 1:2].detach(),
            'rgb_gate_map_live': delta[:, 0:1],
            'tir_gate_map_live': delta[:, 1:2],
            'rgb_sim_map_live': rgb_sim,
            'tir_sim_map_live': tir_sim,
        }
