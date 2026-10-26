"""
TBSILayer: Core cross-attention between RGB and TIR modalities.
Supports degradation-aware modulation at the per-layer level.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock
from lib.models.layers.masked_template_hmoe import MaskedTemplateHMoE
from lib.models.layers.pvl_adapter import PVLAdapter
from lib.models.layers.search_conditioned_t2s_router import SearchConditionedT2SRouter


def template_box_to_token_weights(boxes, grid_size=8):
    """Return per-token patch coverage for normalized xywh boxes."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('template boxes must be [B,4] normalized xywh')
    device, dtype = boxes.device, boxes.dtype
    b = boxes.clamp(0, 1)
    x, y, w, h = b.unbind(-1)
    x2, y2 = (x + w).clamp(0, 1), (y + h).clamp(0, 1)
    edges = torch.arange(grid_size + 1, device=device, dtype=dtype) / grid_size
    left, right = edges[:-1].view(1, 1, -1), edges[1:].view(1, 1, -1)
    top, bottom = edges[:-1].view(1, -1, 1), edges[1:].view(1, -1, 1)
    ix = (torch.minimum(right, x2[:, None, None]) - torch.maximum(left, x[:, None, None])).clamp_min(0)
    iy = (torch.minimum(bottom, y2[:, None, None]) - torch.maximum(top, y[:, None, None])).clamp_min(0)
    # Normalize by each patch area so a fully covered token has weight 1.
    return (ix * iy * (grid_size * grid_size)).flatten(1).clamp(0, 1)


class EarlyTargetMarginGate(nn.Module):
    """Zero-initialized residual from early target-vs-hard-background matches."""

    def __init__(self, hidden_dim=16, hard_negative_k=4, trainable_direct_rho=False):
        super().__init__()
        self.hard_negative_k = max(1, int(hard_negative_k))
        hidden = max(8, int(hidden_dim))
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.direct_rho = (nn.Parameter(torch.tensor(0.0))
                           if trainable_direct_rho else None)

    @staticmethod
    def foreground_candidate_margin(rgb_similarity, tir_similarity, fg_weights):
        """Use foreground template evidence at one shared candidate location."""
        weights = fg_weights.float().clamp_min(0.0)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        rgb_map = (rgb_similarity.float() * weights.unsqueeze(-1)).sum(1) / denom
        tir_map = (tir_similarity.float() * weights.unsqueeze(-1)).sum(1) / denom
        joint_map = 0.5 * (rgb_map + tir_map)
        candidate = joint_map.argmax(dim=-1)
        side = int(round(joint_map.shape[-1] ** 0.5))
        if side * side != joint_map.shape[-1]:
            raise ValueError('direct margin requires square search token grids')
        yy, xx = torch.meshgrid(torch.arange(side, device=joint_map.device),
                                torch.arange(side, device=joint_map.device))
        row, col = candidate // side, candidate % side
        local = ((yy.flatten()[None] - row[:, None]).abs() <= 1) & \
                ((xx.flatten()[None] - col[:, None]).abs() <= 1)
        def margin(score_map):
            target = score_map.gather(1, candidate[:, None]).squeeze(1)
            background = score_map.masked_fill(local, float('-inf')).max(dim=1).values
            return target - background
        return margin(rgb_map), margin(tir_map), candidate

    def _target_background_margin(self, similarity):
        _, _, search_tokens = similarity.shape
        grid = int(round(search_tokens ** 0.5))
        if grid * grid != search_tokens:
            raise ValueError('early target margin requires square search token grids')
        peak_values, peak_indices = similarity.max(dim=-1)
        rows = torch.div(peak_indices, grid, rounding_mode='floor')
        cols = peak_indices % grid
        yy, xx = torch.meshgrid(torch.arange(grid, device=similarity.device),
                                torch.arange(grid, device=similarity.device))
        local_peak = ((yy.reshape(1, 1, -1) - rows.unsqueeze(-1)).abs() <= 1) & \
            ((xx.reshape(1, 1, -1) - cols.unsqueeze(-1)).abs() <= 1)
        background = similarity.masked_fill(local_peak, float('-inf'))
        k = min(self.hard_negative_k, max(1, search_tokens - 9))
        hard_values = background.topk(k, dim=-1).values
        finite = torch.isfinite(hard_values)
        hard_mean = torch.where(finite, hard_values, torch.zeros_like(hard_values)).sum(-1) / \
            finite.sum(-1).clamp_min(1)
        return peak_values, peak_indices, peak_values - hard_mean

    def forward(self, rgb_similarity, tir_similarity, agreement):
        rgb_peak, rgb_index, rgb_margin = self._target_background_margin(rgb_similarity)
        tir_peak, tir_index, tir_margin = self._target_background_margin(tir_similarity)
        rgb_at_tir_peak = rgb_similarity.gather(-1, tir_index.unsqueeze(-1)).squeeze(-1)
        tir_at_rgb_peak = tir_similarity.gather(-1, rgb_index.unsqueeze(-1)).squeeze(-1)
        evidence = torch.stack((rgb_peak, tir_peak, agreement, rgb_margin,
                                tir_margin, rgb_at_tir_peak, tir_at_rgb_peak), dim=-1)
        return (torch.tanh(self.scale) * torch.tanh(self.net(evidence))).to(
            dtype=rgb_similarity.dtype)


class DiscriminativeEvidenceRouter(nn.Module):
    """DiMP-style dense target scoring used to condition MSHA cross messages."""

    def __init__(self, dim, score_dim=64, hidden_dim=8):
        super().__init__()
        self.rgb_projection = nn.Linear(dim, score_dim, bias=False)
        self.tir_projection = nn.Linear(dim, score_dim, bias=False)
        self.router = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(),
                                    nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _score(template, search, projection, fg_weights):
        template = F.normalize(projection(template.float()), dim=-1, eps=1e-6)
        search = F.normalize(projection(search.float()), dim=-1, eps=1e-6)
        weights = fg_weights.float().clamp_min(0)
        log_weights = torch.log(weights + 1e-6)
        logits = torch.matmul(template, search.transpose(1, 2)) / 0.07
        scores = torch.logsumexp(logits + log_weights.unsqueeze(-1), dim=1)
        scores = scores - torch.log(weights.sum(dim=1, keepdim=True) + 1e-6)
        return scores

    @staticmethod
    def _candidate_margin(score_map):
        bsz, tokens = score_map.shape
        side = int(round(tokens ** 0.5))
        if side * side != tokens:
            raise ValueError('DER requires a square search token grid')
        values, indices = score_map.max(dim=-1)
        yy, xx = torch.meshgrid(torch.arange(side, device=score_map.device),
                                torch.arange(side, device=score_map.device))
        row, col = indices // side, indices % side
        neighborhood = ((yy.flatten()[None] - row[:, None]).abs() <= 1) & \
            ((xx.flatten()[None] - col[:, None]).abs() <= 1)
        competitor = score_map.masked_fill(neighborhood, float('-inf')).max(dim=-1).values
        return values - competitor

    def forward(self, early_context, fg_weights):
        rgb_t, tir_t, rgb_s, tir_s = early_context
        rgb_score = self._score(rgb_t, rgb_s, self.rgb_projection, fg_weights)
        tir_score = self._score(tir_t, tir_s, self.tir_projection, fg_weights)
        rgb_margin = self._candidate_margin(rgb_score)
        tir_margin = self._candidate_margin(tir_score)
        evidence = torch.stack((rgb_margin, tir_margin,
                                rgb_score.max(-1).values, tir_score.max(-1).values), -1)
        correction = torch.tanh(self.scale) * torch.tanh(self.router(evidence))
        return rgb_score, tir_score, correction


class EarlyTemplateValueReconstruction(nn.Module):
    """Low-rank, identity-initialized token transform for MSHA Value messages."""

    def __init__(self, dim, rank=8, max_delta=0.10):
        super().__init__()
        rank = max(1, int(rank))
        self.dim = int(dim)
        self.max_delta = float(max_delta)
        # Each direction receives its own transform; the shared evidence is
        # appended after the source, counterpart, and complementary channels.
        in_dim = dim * 3 + 3
        self.down_rgb = nn.Linear(in_dim, rank, bias=False)
        self.up_rgb = nn.Linear(rank, dim, bias=False)
        self.down_tir = nn.Linear(in_dim, rank, bias=False)
        self.up_tir = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up_rgb.weight)
        nn.init.zeros_(self.up_tir.weight)

    def _transform(self, source, counterpart, evidence, down, up):
        diff = source - counterpart
        inp = torch.cat((source, counterpart, diff, evidence), dim=-1)
        delta = up(F.gelu(down(inp)))
        # Bound the residual relative to the source Value norm per token. This
        # keeps the first screen from becoming an uncontrolled feature rewrite.
        scale = self.max_delta * source.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta.float()
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (scale / delta_norm).clamp(max=1.0)
        return (source.float() + delta).to(dtype=source.dtype), delta_norm.mean()

    def forward(self, v_rgb, v_tir, evidence):
        if v_rgb.shape != v_tir.shape or evidence.shape[:2] != v_rgb.shape[:2] or evidence.shape[-1] != 3:
            raise ValueError('early template Value reconstruction shape mismatch')
        out_rgb, norm_rgb = self._transform(v_rgb, v_tir, evidence, self.down_rgb, self.up_rgb)
        out_tir, norm_tir = self._transform(v_tir, v_rgb, evidence, self.down_tir, self.up_tir)
        return out_rgb, out_tir, {'delta_rgb': float(norm_rgb.detach()),
                                  'delta_tir': float(norm_tir.detach())}


class QueryConditionedCrossValue(nn.Module):
    """Pairwise low-rank residual for a cross-modal Value message.

    The residual is zero at initialization.  Unlike the legacy early gate,
    the transform is conditioned on the receiver query and source key, so a
    source token can be useful for one search candidate and harmful for
    another without changing the native QK attention distribution.
    """

    def __init__(self, head_dim, rank=8, max_delta=0.10):
        super().__init__()
        self.head_dim = int(head_dim)
        self.rank = max(1, int(rank))
        self.max_delta = float(max_delta)
        self.down = nn.Linear(2 * self.head_dim + 3, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.head_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, query, key, value, evidence, condition_mode='normal'):
        # query/key/value: [B,H,N,D], evidence: [B,N,3].
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError('query-conditioned cross-value tensors must be 4D')
        if query.shape[:2] != key.shape[:2] or key.shape != value.shape:
            raise ValueError('query-conditioned cross-value attention shape mismatch')
        if evidence.shape != (value.shape[0], value.shape[2], 3):
            raise ValueError('query-conditioned cross-value evidence shape mismatch')
        if condition_mode == 'query_shuffled':
            query_cond = torch.roll(query, shifts=1, dims=2)
            evidence_cond = evidence
        elif condition_mode == 'evidence_shuffled':
            query_cond = query
            evidence_cond = torch.roll(evidence, shifts=1, dims=1)
        elif condition_mode == 'normal':
            query_cond, evidence_cond = query, evidence
        else:
            raise ValueError('unsupported query-conditioned condition mode: {}'.format(condition_mode))
        bsz, heads, n_query, head_dim = query.shape
        n_source = key.shape[2]
        q = query_cond.unsqueeze(3).expand(-1, -1, -1, n_source, -1)
        k = key.unsqueeze(2).expand(-1, -1, n_query, -1, -1)
        e = evidence_cond.to(dtype=query.dtype)[:, None, None, :, :].expand(
            -1, heads, n_query, -1, -1)
        inp = torch.cat((q, k, e), dim=-1)
        delta = self.up(F.gelu(self.down(inp)))
        # Bound the residual relative to each source Value token.  The bound
        # is measured before the pairwise expansion and is deterministic.
        source_norm = value.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta.float()
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        limit = self.max_delta * source_norm.unsqueeze(2)
        delta = delta * (limit / delta_norm).clamp(max=1.0)
        return delta.to(dtype=value.dtype)


class V2TemplateFusion(nn.Module):
    """V2-style joint template MHSA, kept behind an explicit feature gate."""

    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, expert_mode='off', expert_count=4,
                 expert_slots=4, expert_hidden=4, cfvi_mode='off',
                 cfvi_lambda_init=0.001, cfvi_gate_hidden=64,
                 cfvi_band_mask=None, cfvi_band_renorm=True,
                 freq_exchange_mode='off', freq_exchange_diagnostics_path=None,
                 message_freq_refine_mode='off', freq_relation_mode='off',
                 freq_relation_hidden=64, freq_relation_max_bias=0.5,
                 freq_relation_calibration='off', freq_relation_topk=8,
                 freq_relation_clip=3.0, freq_relation_eps=1e-6,
                 pvl_adapter_dim=0, pvl_beta=2.35,
                 early_conditioned_cross_gate=False,
                 early_gate_hidden=32, early_target_margin_gate=False,
                 early_margin_hidden=16, early_margin_hard_negative_k=4,
                 early_direct_margin_gate=False, early_direct_margin_rho=0.0,
                 early_direct_margin_temperature=0.25,
                 early_direct_margin_trainable=False,
                 early_template_value_reconstruction=False,
                 early_template_value_rank=8,
                 early_template_value_max_delta=0.10,
                 query_conditioned_cross_value=False,
                 query_conditioned_cross_value_rank=8,
                 query_conditioned_cross_value_max_delta=0.10,
                 discriminative_evidence_router=False, der_score_dim=64,
                 der_hidden_dim=8, role_pair_bias=False,
                 early_search_relation_bias=False, early_search_relation_dim=32,
                 early_search_relation_temperature=0.07,
                 early_search_relation_max_bias=0.5,
                 search_conditioned_t2s_mode='off',
                 search_conditioned_t2s_loc=False,
                 search_conditioned_t2s_rank=64,
                 search_conditioned_t2s_max_delta=0.10,
                 search_conditioned_t2s_temperature=0.07,
                 search_conditioned_t2s_competition_topk=4):
        super().__init__()
        self.norm = norm_layer(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.early_conditioned_cross_gate = bool(early_conditioned_cross_gate)
        self.early_template_value_reconstruction = bool(early_template_value_reconstruction)
        if self.early_template_value_reconstruction:
            self.early_value_reconstruction = EarlyTemplateValueReconstruction(
                dim=dim, rank=early_template_value_rank,
                max_delta=early_template_value_max_delta)
        self.query_conditioned_cross_value = bool(
            query_conditioned_cross_value or
            os.environ.get('TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE', '0') == '1')
        if self.query_conditioned_cross_value:
            self.query_conditioned_cross_value_rank = int(query_conditioned_cross_value_rank)
            self.query_conditioned_cross_value_max_delta = float(query_conditioned_cross_value_max_delta)
            self.query_conditioned_rgb_from_tir = QueryConditionedCrossValue(
                self.head_dim, rank=self.query_conditioned_cross_value_rank,
                max_delta=self.query_conditioned_cross_value_max_delta)
            self.query_conditioned_tir_from_rgb = QueryConditionedCrossValue(
                self.head_dim, rank=self.query_conditioned_cross_value_rank,
                max_delta=self.query_conditioned_cross_value_max_delta)
        self.last_query_conditioned_stats = {'mode': 'off'}
        self._token_reliability_diag_counter = 0
        if self.early_conditioned_cross_gate:
            self.early_gate = nn.Sequential(
                nn.Linear(3, max(8, int(early_gate_hidden))), nn.GELU(),
                nn.Linear(max(8, int(early_gate_hidden)), 2))
            nn.init.zeros_(self.early_gate[-1].weight)
            nn.init.zeros_(self.early_gate[-1].bias)
            # Keeps the candidate close to M1 at initialization while allowing
            # the gate head to receive gradients from the cross-modal message.
            self.early_gate_scale = nn.Parameter(torch.tensor(0.1))
        self.early_target_margin_gate = bool(early_target_margin_gate)
        self.early_direct_margin_gate = bool(early_direct_margin_gate)
        self.early_direct_margin_rho = float(early_direct_margin_rho)
        self.early_direct_margin_temperature = max(float(early_direct_margin_temperature), 1e-6)
        if self.early_target_margin_gate:
            self.early_margin_gate = EarlyTargetMarginGate(
                hidden_dim=early_margin_hidden,
                hard_negative_k=early_margin_hard_negative_k,
                trainable_direct_rho=early_direct_margin_trainable)
        self.discriminative_evidence_router = bool(discriminative_evidence_router)
        if self.discriminative_evidence_router:
            self.der = DiscriminativeEvidenceRouter(
                dim=dim, score_dim=der_score_dim, hidden_dim=der_hidden_dim)
        self.last_der_scores = None
        self.role_pair_bias = bool(role_pair_bias)
        if self.role_pair_bias:
            # Zero initialization preserves the pretrained attention path.
            self.role_pair_bias_gain = nn.Parameter(torch.zeros(num_heads))
        self.early_search_relation_bias = bool(early_search_relation_bias)
        if self.early_search_relation_bias:
            rel_dim = max(8, int(early_search_relation_dim))
            self.early_search_relation_temperature = max(float(early_search_relation_temperature), 1e-4)
            self.early_search_relation_max_bias = float(early_search_relation_max_bias)
            self.early_search_relation_rgb = nn.Linear(dim, rel_dim, bias=False)
            self.early_search_relation_tir = nn.Linear(dim, rel_dim, bias=False)
            self.early_search_relation_rgb_search = nn.Linear(dim, rel_dim, bias=False)
            self.early_search_relation_tir_search = nn.Linear(dim, rel_dim, bias=False)
            self.early_search_relation_gain = nn.Parameter(torch.zeros(num_heads))
            self.last_early_search_relation_stats = {}
        if pvl_adapter_dim > 0:
            # 新模块初始化不能推进 M1 共享层的随机数状态。
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261020)
                self.pvl_adapter = PVLAdapter(dim, pvl_adapter_dim, pvl_beta)
        else:
            self.pvl_adapter = None
        if freq_relation_mode == 'off':
            self.freq_relation = None
        elif freq_relation_calibration == 'reliable':
            from lib.models.layers.freq_relation_bias import HFRelationBiasR1

            self.freq_relation = HFRelationBiasR1(
                dim=dim, hidden=freq_relation_hidden, topk=freq_relation_topk,
                max_bias=freq_relation_max_bias)
        elif freq_relation_calibration == 'scale_calibrated':
            from lib.models.layers.frequency_relation_bias_v2 import ScaleCalibratedFreqRelationBias

            self.freq_relation = ScaleCalibratedFreqRelationBias(
                feature_dim=dim, max_alpha=freq_relation_max_bias,
                clip_value=freq_relation_clip, eps=freq_relation_eps)
        else:
            from lib.models.layers.freq_relation_bias import FrequencyRelationBias

            self.freq_relation = FrequencyRelationBias(
                dim=dim, num_heads=num_heads, hidden=freq_relation_hidden,
                basis=freq_relation_mode, max_bias=freq_relation_max_bias)
        self.freq_relation_calibration = freq_relation_calibration
        if freq_exchange_mode == 'target_ll_high':
            from lib.models.layers.target_freq_exchange import TargetAnchoredFreqExchange

            self.freq_exchange = TargetAnchoredFreqExchange(
                dim=dim, diagnostics_path=freq_exchange_diagnostics_path)
        elif freq_exchange_mode == 'off':
            self.freq_exchange = None
        else:
            raise ValueError('unsupported frequency exchange mode: {}'.format(freq_exchange_mode))
        if cfvi_mode == 'fixed':
            from lib.models.layers.cfvi_module import FixedCrossModalFreqInjection

            self.cfvi = FixedCrossModalFreqInjection(
                dim, cfvi_lambda_init, cfvi_band_mask, cfvi_band_renorm)
        elif cfvi_mode == 'raw':
            from lib.models.layers.cfvi_module import RawCrossModalValueInjection

            self.cfvi = RawCrossModalValueInjection(dim, cfvi_lambda_init)
        elif cfvi_mode == 'structured':
            from lib.models.layers.cfvi_module import StructuredCrossModalFreqInjection

            self.cfvi = StructuredCrossModalFreqInjection(dim, cfvi_lambda_init,
                                                           cfvi_band_renorm)
        elif cfvi_mode == 'dynamic':
            from lib.models.layers.cfvi_module import DynamicCrossModalFreqInjection

            self.cfvi = DynamicCrossModalFreqInjection(
                dim, cfvi_gate_hidden, cfvi_lambda_init)
        elif cfvi_mode == 'off':
            self.cfvi = None
        else:
            raise ValueError('unsupported CFVI mode: {}'.format(cfvi_mode))
        self.cfvi_mode = cfvi_mode
        if message_freq_refine_mode not in ('off', 'high_refine'):
            raise ValueError('unsupported message frequency refine mode: {}'.format(
                message_freq_refine_mode))
        if message_freq_refine_mode == 'high_refine':
            from lib.models.layers.cfvi_module import MessageFrequencyRefine

            self.message_freq_refine = MessageFrequencyRefine(dim)
        else:
            self.message_freq_refine = None
        # Inference-only diagnostics. The default path does not allocate or
        # retain attention statistics and therefore remains parity-safe.
        self.analysis_enabled = False
        self.analysis_collector = None
        self.output_analysis_collector = None
        self.expert_mode = expert_mode
        if expert_mode not in ('off', 'global', 'target', 'target_disp_only', 'target_soft'):
            raise ValueError('unsupported template expert mode: {}'.format(expert_mode))
        if expert_mode != 'off':
            prior_mode = {
                'global': 'none',
                'target': 'hard',
                'target_disp_only': 'disp_only',
                'target_soft': 'soft',
            }[expert_mode]
            self.template_expert = MaskedTemplateHMoE(
                dim=dim, experts=expert_count, slots=expert_slots, hid_dim=expert_hidden,
                prior_mode=prior_mode)
            self.expert_gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _condition_search_message(template, search, fg_mask, alpha, mode, temperature):
        """Inject a bounded template-to-search message on the 16x16 axis."""
        bsz, n_template, dim = template.shape
        n_search = search.shape[1]
        side = int(round(n_search ** 0.5))
        if side * side != n_search or fg_mask.shape != (bsz, n_template):
            raise ValueError('conditioned search message shape contract mismatch')
        t_norm = F.normalize(template.float(), dim=-1, eps=1e-6)
        s_norm = F.normalize(search.float(), dim=-1, eps=1e-6)
        weights = fg_mask.to(device=template.device, dtype=torch.float32).clamp_min(0.0)
        proto = F.normalize((t_norm * weights.unsqueeze(-1)).sum(1) /
                            weights.sum(1, keepdim=True).clamp_min(1e-6), dim=-1, eps=1e-6)
        similarity = torch.matmul(proto.unsqueeze(1), s_norm.transpose(1, 2)).squeeze(1)
        peak = similarity.argmax(dim=-1)
        yy, xx = torch.meshgrid(torch.arange(side, device=template.device),
                                torch.arange(side, device=template.device), indexing='ij')
        local = ((yy.reshape(1, -1) - (peak // side).unsqueeze(1)).abs() <= 1) & \
                ((xx.reshape(1, -1) - (peak % side).unsqueeze(1)).abs() <= 1)
        background = similarity.masked_fill(local, float('-inf')).max(dim=-1, keepdim=True).values
        evidence = torch.tanh((similarity - background) / max(float(temperature), 1e-6))
        if mode == 'shifted':
            evidence = torch.roll(evidence.reshape(bsz, side, side),
                                  shifts=(max(1, side // 2), max(1, side // 2)),
                                  dims=(1, 2)).reshape(bsz, n_search)
        elif mode != 'target':
            raise ValueError('unsupported conditioned search message mode: {}'.format(mode))
        message = evidence.to(dtype=search.dtype).unsqueeze(-1) * proto.to(dtype=search.dtype).unsqueeze(1)
        return search + float(alpha) * message, {
            'mode': mode,
            'alpha': float(alpha),
            'evidence_mean': float(evidence.mean().detach()),
            'evidence_std': float(evidence.std().detach()),
            'peak': peak.detach().cpu(),
        }

    @staticmethod
    def _early_gate_token_reliability(early_context, mode='balanced'):
        """Build bounded token reliability scales for cross-modal source Values."""
        if early_context is None or len(early_context) != 4:
            raise ValueError('early token reliability requires four early token tensors')
        rgb_t, tir_t, rgb_s, tir_s = [value.float() for value in early_context]

        def confidence(template, search):
            similarity = torch.matmul(
                F.normalize(template, dim=-1, eps=1e-6),
                F.normalize(search, dim=-1, eps=1e-6).transpose(1, 2))
            probability = torch.softmax(similarity / 0.07, dim=-1)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
            entropy = entropy / math.log(float(similarity.shape[-1]))
            top2 = similarity.topk(k=2, dim=-1).values
            margin_confidence = torch.sigmoid((top2[..., 0] - top2[..., 1]) / 0.07)
            return (1.0 - entropy.clamp(0.0, 1.0)) * margin_confidence

        agreement = torch.sigmoid(
            (F.normalize(rgb_t, dim=-1, eps=1e-6) *
             F.normalize(tir_t, dim=-1, eps=1e-6)).sum(-1) / 0.10)
        rgb_confidence = confidence(rgb_t, rgb_s) * (0.5 + 0.5 * agreement)
        tir_confidence = confidence(tir_t, tir_s) * (0.5 + 0.5 * agreement)

        def centered_scale(score):
            log_score = torch.log(score.clamp_min(1e-6))
            centered = torch.tanh(log_score - log_score.mean(dim=1, keepdim=True))
            centered = centered - centered.mean(dim=1, keepdim=True)
            peak = centered.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
            residual = centered / peak
            return 1.0 + 0.10 * residual, residual

        def low_confidence_suppression(score):
            log_score = torch.log(score.clamp_min(1e-6))
            median = log_score.median(dim=1, keepdim=True).values
            deviation = (log_score - median).abs()
            mad = deviation.median(dim=1, keepdim=True).values.clamp_min(1e-3)
            risk = torch.tanh(F.relu(median - log_score) / mad)
            residual = -risk
            return 1.0 + 0.10 * residual, residual

        if mode == 'balanced':
            scale_fn = centered_scale
        elif mode == 'suppress_low':
            scale_fn = low_confidence_suppression
        else:
            raise ValueError('unsupported token reliability mode: {}'.format(mode))
        rgb_scale, rgb_residual = scale_fn(rgb_confidence)
        tir_scale, tir_residual = scale_fn(tir_confidence)
        stats = {
            'mode': mode,
            'rgb_scale_mean': rgb_scale.mean().detach(),
            'tir_scale_mean': tir_scale.mean().detach(),
            'rgb_scale_min': rgb_scale.min().detach(),
            'rgb_scale_max': rgb_scale.max().detach(),
            'tir_scale_min': tir_scale.min().detach(),
            'tir_scale_max': tir_scale.max().detach(),
            'rgb_residual_std': rgb_residual.std(unbiased=False).detach(),
            'tir_residual_std': tir_residual.std(unbiased=False).detach(),
        }
        return rgb_scale, tir_scale, stats

    @staticmethod
    def _dcvr_route_values(v, early_context, fg_masks, num_heads, head_dim):
        """Route native cross-modal Value tokens using early evidence.

        This is deliberately functional and feature-gated by the caller.  It
        keeps Q/K and the attention map untouched, then applies a zero-mean,
        foreground-budgeted source-token reweighting to the two cross-modal
        Value blocks.  The evidence combines matching entropy, local peak
        competition and RGB/TIR template agreement instead of a single max.
        """
        if early_context is None or len(early_context) != 4:
            raise ValueError('DCVR requires four early token tensors')
        rgb_t, tir_t, rgb_s, tir_s = [z.float() for z in early_context]
        if v.ndim != 3:
            raise ValueError('DCVR token route expects [B,N,C] values')
        bsz, length, _ = v.shape
        n_rgb = rgb_t.shape[1]
        if length != 2 * n_rgb or tir_t.shape[1] != n_rgb:
            raise ValueError('DCVR expects equal RGB/TIR template Value axes')
        if fg_masks is None:
            fg_masks = (rgb_t.new_ones((bsz, n_rgb)), tir_t.new_ones((bsz, n_rgb)))
            mask_fallback = True
        elif len(fg_masks) != 2:
            raise ValueError('DCVR foreground masks must be a pair')
        else:
            mask_fallback = False
        def evidence(template, search, fg):
            t = F.normalize(template, dim=-1, eps=1e-6)
            s = F.normalize(search, dim=-1, eps=1e-6)
            sim = torch.matmul(t, s.transpose(1, 2))
            p = torch.softmax(sim / 0.07, dim=-1)
            ent = -(p * p.clamp_min(1e-8).log()).sum(-1)
            ent = ent / math.log(float(sim.shape[-1]))
            top2 = sim.topk(k=2, dim=-1).values
            competition = torch.sigmoid((top2[..., 0] - top2[..., 1]) / 0.07)
            confidence = (1.0 - ent) * competition
            fg = fg.to(device=template.device, dtype=template.dtype).clamp(0, 1)
            # Keep a small background allowance so the route cannot erase
            # context, then normalize total message budget to one.
            confidence = confidence * (0.75 + 0.25 * fg)
            confidence = confidence / confidence.mean(dim=1, keepdim=True).clamp_min(1e-6)
            return confidence, ent, competition
        rgb_w, rgb_ent, rgb_comp = evidence(rgb_t, rgb_s, fg_masks[0])
        tir_w, tir_ent, tir_comp = evidence(tir_t, tir_s, fg_masks[1])
        rgb_n = F.normalize(rgb_t, dim=-1, eps=1e-6)
        tir_n = F.normalize(tir_t, dim=-1, eps=1e-6)
        agreement = (rgb_n * tir_n).sum(-1).clamp(-1, 1)
        pair = torch.sigmoid(agreement / 0.10)
        # A source is admitted when it is reliable and agrees with its paired
        # template token.  The opposite direction receives the corresponding
        # source reliability, preserving asymmetric RGB/TIR degradation.
        rgb_source = rgb_w * (0.5 + 0.5 * pair)
        tir_source = tir_w * (0.5 + 0.5 * pair)
        rgb_source = rgb_source / rgb_source.mean(1, keepdim=True).clamp_min(1e-6)
        tir_source = tir_source / tir_source.mean(1, keepdim=True).clamp_min(1e-6)
        source_scale = float(os.environ.get('TBSI_DCVR_ALPHA', '0.25'))
        rgb_source = 1.0 + source_scale * (rgb_source - 1.0)
        tir_source = 1.0 + source_scale * (tir_source - 1.0)
        # v order is [RGB template, TIR template].  Each receiver's cross
        # message uses the other modality's routed source Value block.
        routed = v.clone()
        routed[:, :n_rgb] = v[:, :n_rgb] * rgb_source[:, :, None].to(v.dtype)
        routed[:, n_rgb:] = v[:, n_rgb:] * tir_source[:, :, None].to(v.dtype)
        with torch.no_grad():
            stats = {
                'mode': 'dcvr', 'alpha': source_scale,
                'fg_mask_fallback_uniform': bool(mask_fallback),
                'rgb_entropy_mean': float(rgb_ent.mean()),
                'tir_entropy_mean': float(tir_ent.mean()),
                'rgb_competition_mean': float(rgb_comp.mean()),
                'tir_competition_mean': float(tir_comp.mean()),
                'agreement_mean': float(agreement.mean()),
                'rgb_source_scale_mean': float(rgb_source.mean()),
                'tir_source_scale_mean': float(tir_source.mean()),
                'rgb_source_scale_std': float(rgb_source.std(unbiased=False)),
                'tir_source_scale_std': float(tir_source.std(unbiased=False)),
            }
            path = os.environ.get('TBSI_DCVR_DIAGNOSTICS_PATH', '')
            if path:
                import json
                with open(path, 'a') as handle:
                    handle.write(json.dumps(stats, sort_keys=True) + '\n')
        return routed, stats

    def forward(self, x_rgb_template, x_tir_template, fg_masks=None,
                x_rgb_search=None, x_tir_search=None,
                early_context=None, early_utility_logits=None):
        # The two modality template streams interact jointly, then are split
        # back so the surrounding ViT and search path remain unchanged.
        if self.pvl_adapter is not None:
            x_rgb_template, x_tir_template = self.pvl_adapter(
                x_rgb_template, x_tir_template)
        if self.freq_exchange is not None:
            if fg_masks is None or len(fg_masks) != 2:
                raise ValueError('TAFH requires RGB/TIR template foreground masks')
            x_rgb_template, x_tir_template = self.freq_exchange(
                x_rgb_template, x_tir_template, fg_masks[0], fg_masks[1])
        x = torch.cat([x_rgb_template, x_tir_template], dim=1)
        residual = x
        x = self.norm(x)
        normalized_tokens = x
        bsz, length, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        self.last_dcvr_stats = {'mode': 'off'}
        if os.environ.get('TBSI_DCVR_MODE', 'off').lower() in ('on', 'target'):
            v_tokens = v.transpose(1, 2).reshape(bsz, length, self.num_heads * self.head_dim)
            v_tokens, self.last_dcvr_stats = self._dcvr_route_values(
                v_tokens, early_context, fg_masks, self.num_heads, self.head_dim)
            v = v_tokens.reshape(bsz, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # FCSV-R: redistribute only foreground cross-modal template Value
        # sources using early RGB/TIR search co-support. The default is off,
        # preserving the parent path exactly.
        self.last_fcsv_r_stats = {'mode': 'off'}
        fcsv_cross_value_gate = None
        fcsv_mode = os.environ.get('TBSI_FCSV_R_MODE', 'off').lower()
        if fcsv_mode not in (
                'off', 'correct', 'shifted', 'mask_only',
                'continuous_correct', 'continuous_shifted'):
            raise ValueError('unsupported TBSI_FCSV_R_MODE: {}'.format(fcsv_mode))
        if fcsv_mode != 'off':
            if early_context is None or len(early_context) != 4:
                raise ValueError('FCSV-R requires early context')
            if fg_masks is None or len(fg_masks) != 2:
                raise ValueError('FCSV-R requires RGB/TIR foreground masks')
            n_rgb, n_tir = x_rgb_template.shape[1], x_tir_template.shape[1]
            if n_rgb != n_tir or n_rgb != 64:
                raise ValueError('FCSV-R expects 64 RGB and 64 TIR template tokens')
            early_rgb_t, early_tir_t, early_rgb_s, early_tir_s = early_context
            tau = float(os.environ.get('TBSI_FCSV_R_TEMPERATURE', '0.07'))
            alpha = float(os.environ.get('TBSI_FCSV_R_ALPHA', '0.1'))
            if tau <= 0.0 or not torch.isfinite(x.new_tensor(tau)):
                raise ValueError('TBSI_FCSV_R_TEMPERATURE must be positive and finite')
            if not torch.isfinite(x.new_tensor(alpha)) or abs(alpha) > 0.1 + 1e-7:
                raise ValueError('TBSI_FCSV_R_ALPHA must be finite and within [-0.1, 0.1]')
            rgb_t = F.normalize(early_rgb_t.float(), dim=-1, eps=1e-6)
            tir_t = F.normalize(early_tir_t.float(), dim=-1, eps=1e-6)
            rgb_s = F.normalize(early_rgb_s.float(), dim=-1, eps=1e-6)
            tir_s = F.normalize(early_tir_s.float(), dim=-1, eps=1e-6)
            p_rgb = torch.softmax(torch.matmul(rgb_t, rgb_s.transpose(1, 2)) / tau, dim=-1)
            p_tir = torch.softmax(torch.matmul(tir_t, tir_s.transpose(1, 2)) / tau, dim=-1)
            if fcsv_mode in ('shifted', 'continuous_shifted'):
                side = int(round(p_tir.shape[-1] ** 0.5))
                if side * side != p_tir.shape[-1]:
                    raise ValueError('FCSV-R shifted mode requires square search grid')
                p_tir = p_tir.reshape(bsz, n_tir, side, side)
                p_tir = torch.roll(p_tir, shifts=(side // 2, side // 2), dims=(-2, -1))
                p_tir = p_tir.reshape(bsz, n_tir, -1)
            co = torch.matmul(p_rgb, p_tir.transpose(1, 2))
            w_rgb = fg_masks[0].to(device=x.device, dtype=torch.float32).clamp(0, 1)
            w_tir = fg_masks[1].to(device=x.device, dtype=torch.float32).clamp(0, 1)
            if w_rgb.shape != (bsz, n_rgb) or w_tir.shape != (bsz, n_tir):
                raise ValueError('FCSV-R foreground masks must match template tokens')

            def source_gate(receiver_prior, source_prior, relation):
                receiver_prior = receiver_prior / receiver_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                source_prior = source_prior / source_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                score = torch.matmul(receiver_prior.unsqueeze(1), relation).squeeze(1)
                score = torch.log(score.clamp_min(1e-6))
                centered = score - (source_prior * score).sum(dim=-1, keepdim=True)
                z = torch.tanh(centered)
                if fcsv_mode == 'mask_only':
                    gate = (source_prior > 0).to(source_prior.dtype)
                elif fcsv_mode in ('continuous_correct', 'continuous_shifted'):
                    # Continuous simplex: zero-prior tokens retain native
                    # cross-Value contribution; only positive-prior sources
                    # receive a normalized relative reallocation.
                    weights = torch.exp(alpha * z).masked_fill(source_prior <= 0, 0.0)
                    denom = (source_prior * weights).sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    gate = torch.where(source_prior > 0, weights / denom, torch.ones_like(source_prior))
                else:
                    logits = torch.log(source_prior.clamp_min(1e-6)) + alpha * z
                    q_src = torch.softmax(logits.masked_fill(source_prior <= 0, -1e4), dim=-1)
                    gate = q_src / source_prior.clamp_min(1e-6)
                return gate, score, z

            g_tir, score_tir, z_tir = source_gate(w_rgb, w_tir, co)
            g_rgb, score_rgb, z_rgb = source_gate(w_tir, w_rgb, co.transpose(1, 2))
            # Contract check: alpha=0 must be an exact identity path,
            # including tokens with zero foreground prior.
            if alpha == 0.0:
                g_rgb = torch.ones_like(w_rgb)
                g_tir = torch.ones_like(w_tir)
            # Keep the native Value tensor untouched.  The gates are applied
            # only to the two cross-modal message terms below, after the
            # native QK attention distribution has been formed.
            fcsv_cross_value_gate = torch.stack((g_rgb, g_tir), dim=-1)
            with torch.no_grad():
                self.last_fcsv_r_stats = {
                    'mode': fcsv_mode, 'alpha': alpha, 'temperature': tau,
                    'co_mean': float(co.mean()), 'co_std': float(co.std(unbiased=False)),
                    'g_rgb_min': float(g_rgb.min()), 'g_rgb_max': float(g_rgb.max()),
                    'g_tir_min': float(g_tir.min()), 'g_tir_max': float(g_tir.max()),
                    'rgb_budget': float((w_rgb / w_rgb.sum(-1, keepdim=True).clamp_min(1e-6) * g_rgb).sum(-1).mean()),
                    'tir_budget': float((w_tir / w_tir.sum(-1, keepdim=True).clamp_min(1e-6) * g_tir).sum(-1).mean()),
                }
        token_reliability_mode = os.environ.get(
            'TBSI_EARLY_GATE_TOKEN_REL_MODE', 'off').lower()
        if token_reliability_mode not in ('off', 'on', 'suppress_low'):
            raise ValueError('TBSI_EARLY_GATE_TOKEN_REL_MODE must be off, on, or suppress_low')
        token_reliability = None
        if token_reliability_mode in ('on', 'suppress_low'):
            if not self.early_conditioned_cross_gate:
                raise ValueError('early token reliability requires EARLY_CONDITIONED_CROSS_GATE')
            operator_mode = 'balanced' if token_reliability_mode == 'on' else 'suppress_low'
            token_reliability = self._early_gate_token_reliability(early_context, operator_mode)
            token_diag_path = os.environ.get('TBSI_EARLY_GATE_TOKEN_REL_DIAGNOSTICS_PATH', '')
            if token_diag_path:
                try:
                    token_diag_stride = max(1, int(os.environ.get(
                        'TBSI_EARLY_GATE_TOKEN_REL_DIAGNOSTICS_STRIDE', '16')))
                except ValueError:
                    raise ValueError('TBSI_EARLY_GATE_TOKEN_REL_DIAGNOSTICS_STRIDE must be a positive integer')
                self._token_reliability_diag_counter += 1
                if self._token_reliability_diag_counter % token_diag_stride == 0:
                    import json
                    stats = {
                        key: (value if isinstance(value, str) else float(value))
                        for key, value in token_reliability[2].items()
                    }
                    stats['layer_index'] = getattr(self, 'path_effect_layer_index', None)
                    with open(token_diag_path, 'a') as handle:
                        handle.write(json.dumps(stats, sort_keys=True) + '\n')
        if self.cfvi is not None:
            n_rgb = x_rgb_template.shape[1]
            v_before_cfvi = v
            local_rgb, local_tir = self.cfvi(x[:, :n_rgb], x[:, n_rgb:])
            local_rgb = local_rgb.reshape(bsz, n_rgb, self.num_heads, self.head_dim)
            local_tir = local_tir.reshape(bsz, n_rgb, self.num_heads, self.head_dim)
            local_rgb = local_rgb.permute(0, 2, 1, 3)
            local_tir = local_tir.permute(0, 2, 1, 3)
            v = torch.cat((v[:, :, :n_rgb] + local_rgb,
                           v[:, :, n_rgb:] + local_tir), dim=2)
            diag_path = os.environ.get('CFVI_DIAGNOSTICS_PATH', '')
            if diag_path:
                import json
                rho_rgb = (local_rgb.float().norm() /
                           v_before_cfvi[:, :, :n_rgb].float().norm().clamp_min(1e-8)).item()
                rho_tir = (local_tir.float().norm() /
                           v_before_cfvi[:, :, n_rgb:].float().norm().clamp_min(1e-8)).item()
                row = {'layer': getattr(self, 'path_effect_layer_index', None),
                       'rho_rgb': rho_rgb, 'rho_tir': rho_tir,
                       'lambda_rgb': float(self.cfvi.lambda_rgb.detach().item()),
                       'lambda_tir': float(self.cfvi.lambda_tir.detach().item()),
                       'stats': getattr(self.cfvi, 'last_stats', {})}
                with open(diag_path, 'a') as handle:
                    handle.write(json.dumps(row) + '\n')
        logits = (q @ k.transpose(-2, -1)) * self.scale
        if self.early_search_relation_bias:
            if early_context is None or len(early_context) != 4:
                raise ValueError('early search relation bias requires early context')
            early_rgb_t, early_tir_t, early_rgb_s, early_tir_s = early_context
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if n_rgb != n_tir or early_rgb_t.shape[1] != n_rgb or early_tir_t.shape[1] != n_tir:
                raise ValueError('early search relation bias template axis mismatch')
            def relation(t, s, tp, sp):
                q_t = F.normalize(tp(t.float()), dim=-1, eps=1e-6)
                q_s = F.normalize(sp(s.float()), dim=-1, eps=1e-6)
                return torch.softmax(torch.matmul(q_t, q_s.transpose(1, 2)) /
                                     self.early_search_relation_temperature, dim=-1)
            p_rgb = relation(early_rgb_t, early_rgb_s,
                             self.early_search_relation_rgb, self.early_search_relation_rgb_search)
            p_tir = relation(early_tir_t, early_tir_s,
                             self.early_search_relation_tir, self.early_search_relation_tir_search)
            rel = torch.matmul(p_rgb, p_tir.transpose(1, 2))
            rel = rel / rel.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            rel = rel - rel.mean(dim=-1, keepdim=True)
            gain = self.early_search_relation_max_bias * torch.tanh(self.early_search_relation_gain)
            logits = logits.clone()
            logits[:, :, :n_rgb, n_rgb:] = logits[:, :, :n_rgb, n_rgb:] + gain[None, :, None, None] * rel[:, None]
            logits[:, :, n_rgb:, :n_rgb] = logits[:, :, n_rgb:, :n_rgb] + gain[None, :, None, None] * rel.transpose(-1, -2)[:, None]
            with torch.no_grad():
                self.last_early_search_relation_stats = {
                    'relation_mean': float(rel.mean().detach()),
                    'relation_std': float(rel.std(unbiased=False).detach()),
                    'gain_mean': float(gain.mean().detach()),
                }
        if self.role_pair_bias:
            if fg_masks is None or len(fg_masks) != 2:
                raise ValueError('role-pair bias requires RGB/TIR template foreground masks')
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            fg_rgb = fg_masks[0].to(device=logits.device, dtype=logits.dtype)
            fg_tir = fg_masks[1].to(device=logits.device, dtype=logits.dtype)
            if (n_rgb != n_tir or fg_rgb.shape != (bsz, n_rgb) or
                    fg_tir.shape != (bsz, n_tir) or logits.shape[-1] != n_rgb + n_tir):
                raise ValueError('role-pair bias expects equal template token axes and matching masks')
            gain = torch.tanh(self.role_pair_bias_gain).to(dtype=logits.dtype)
            pair = fg_rgb[:, None, :, None] * fg_tir[:, None, None, :]
            logits = logits.clone()
            logits[:, :, :n_rgb, n_rgb:] = (
                logits[:, :, :n_rgb, n_rgb:] + gain[None, :, None, None] * pair)
            logits[:, :, n_rgb:, :n_rgb] = (
                logits[:, :, n_rgb:, :n_rgb] + gain[None, :, None, None] * pair.transpose(-1, -2))
        # Optional soft participation for uncertain template keys. This is a
        # read-only, feature-gated MSHA diagnostic: default mode is off and
        # therefore preserves the original M1 attention path exactly.
        uncertainty_mode = os.environ.get('TBSI_TEMPLATE_UNCERTAINTY_MODE', 'off').lower()
        if uncertainty_mode not in ('off', 'soft', 'query_soft', 'target_context_soft',
                                    'message_gate'):
            raise ValueError('unsupported TBSI_TEMPLATE_UNCERTAINTY_MODE: {}'.format(
                uncertainty_mode))
        participation = None
        if uncertainty_mode in ('soft', 'query_soft', 'target_context_soft', 'message_gate'):
            try:
                threshold = float(os.environ.get('TBSI_TEMPLATE_UNCERTAINTY_THRESHOLD', '0.5'))
                temperature = float(os.environ.get('TBSI_TEMPLATE_UNCERTAINTY_TEMPERATURE', '0.15'))
            except ValueError:
                raise ValueError('template uncertainty threshold/temperature must be finite floats')
            if temperature <= 0.0 or not torch.isfinite(logits.new_tensor(threshold)):
                raise ValueError('template uncertainty threshold/temperature are invalid')
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if n_rgb != n_tir:
                raise ValueError('template uncertainty requires equal RGB/TIR template lengths')
            rgb_norm = F.normalize(x_rgb_template, dim=-1, eps=1e-6)
            tir_norm = F.normalize(x_tir_template, dim=-1, eps=1e-6)
            agreement = (rgb_norm * tir_norm).sum(dim=-1)
            if uncertainty_mode in ('query_soft', 'target_context_soft', 'message_gate'):
                if x_rgb_search is None or x_tir_search is None:
                    raise ValueError('{} uncertainty requires RGB/TIR search tokens'.format(
                        uncertainty_mode))
                rgb_search_norm = F.normalize(x_rgb_search, dim=-1, eps=1e-6)
                tir_search_norm = F.normalize(x_tir_search, dim=-1, eps=1e-6)
                rgb_similarity = torch.matmul(rgb_norm, rgb_search_norm.transpose(1, 2))
                tir_similarity = torch.matmul(tir_norm, tir_search_norm.transpose(1, 2))
                if uncertainty_mode in ('query_soft', 'message_gate'):
                    query_match = 0.5 * torch.amax(rgb_similarity, dim=-1) + \
                        0.5 * torch.amax(tir_similarity, dim=-1)
                else:
                    if fg_masks is None or len(fg_masks) != 2:
                        raise ValueError('target_context_soft requires RGB/TIR template masks')
                    fg_r = fg_masks[0].to(device=x.device, dtype=x.dtype).clamp(0, 1)
                    fg_t = fg_masks[1].to(device=x.device, dtype=x.dtype).clamp(0, 1)
                    if fg_r.shape != (bsz, n_rgb) or fg_t.shape != (bsz, n_tir):
                        raise ValueError('template masks must be [B,lens_z]')

                    def role_contrast(similarity, tokens, search_tokens, foreground):
                        context = 1.0 - foreground
                        fg_proto = F.normalize(
                            (tokens * foreground.unsqueeze(-1)).sum(1) /
                            foreground.sum(1, keepdim=True).clamp_min(1e-6), dim=-1, eps=1e-6)
                        context_proto = F.normalize(
                            (tokens * context.unsqueeze(-1)).sum(1) /
                            context.sum(1, keepdim=True).clamp_min(1e-6), dim=-1, eps=1e-6)
                        fg_match = torch.amax(torch.matmul(
                            fg_proto.unsqueeze(1), search_tokens.transpose(1, 2)), dim=-1)
                        context_match = torch.amax(torch.matmul(
                            context_proto.unsqueeze(1), search_tokens.transpose(1, 2)), dim=-1)
                        token_match = torch.amax(similarity, dim=-1)
                        fg_score = token_match - context_match.expand_as(token_match)
                        context_score = token_match - fg_match.expand_as(token_match)
                        return foreground * fg_score + context * context_score

                    query_match = 0.5 * role_contrast(
                        rgb_similarity, rgb_norm, rgb_search_norm, fg_r) + \
                        0.5 * role_contrast(
                            tir_similarity, tir_norm, tir_search_norm, fg_t)
                reliability = 0.5 * agreement + 0.5 * query_match
            else:
                query_match = None
                reliability = agreement
            participation = torch.sigmoid((reliability - threshold) / temperature).clamp_min(1e-4)
            if uncertainty_mode != 'message_gate':
                key_bias = torch.log(torch.cat((participation, participation), dim=1))
                logits = logits + key_bias[:, None, None, :]
            with torch.no_grad():
                self.last_uncertainty_stats = {
                    'mode': uncertainty_mode,
                    'agreement_mean': float(agreement.mean().detach()),
                    'agreement_std': float(agreement.std().detach()),
                    'query_match_mean': (float(query_match.mean().detach())
                                         if query_match is not None else None),
                    'reliability_mean': float(reliability.mean().detach()),
                    'participation_mean': float(participation.mean().detach()),
                    'participation_min': float(participation.min().detach()),
                    'participation_max': float(participation.max().detach()),
                }
        else:
            self.last_uncertainty_stats = {'mode': 'off'}
        if self.freq_relation is not None:
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if n_rgb != 64 or n_tir != 64:
                raise ValueError('frequency relation requires 64+64 template tokens')
            cross_rt = logits[:, :, :n_rgb, n_rgb:]
            cross_tr = logits[:, :, n_rgb:, :n_rgb]
            if self.freq_relation_calibration == 'scale_calibrated':
                relation_bias = self.freq_relation(
                    x[:, :n_rgb, :], x[:, n_rgb:, :], cross_rt, cross_tr,
                    num_heads=self.num_heads)
            else:
                relation_bias = self.freq_relation(
                    x[:, :n_rgb, :], x[:, n_rgb:, :], cross_rt, cross_tr)
            logits = logits + relation_bias
            self.last_frequency_stats = getattr(self.freq_relation, 'last_stats', None)
            self.last_r2_native_logits = (cross_rt.detach(), cross_tr.detach())
        # Frozen route probe: test-only logit biases, disabled by default.
        # The zero-bias path intentionally skips the extra operation for exact
        # parity with the trained M1 model.
        try:
            delta_r_to_t = float(os.environ.get('TBSI_TEMPLATE_ROUTE_R2T', '0.0'))
            delta_t_to_r = float(os.environ.get('TBSI_TEMPLATE_ROUTE_T2R', '0.0'))
        except ValueError:
            raise ValueError('TBSI_TEMPLATE_ROUTE_R2T/T2R must be finite floats')
        if delta_r_to_t != 0.0 or delta_t_to_r != 0.0:
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if n_rgb != n_tir or logits.shape[-1] != n_rgb + n_tir:
                raise ValueError('template route probe requires equal RGB/TIR template lengths')
            route_bias = logits.new_zeros((logits.shape[-2], logits.shape[-1]))
            route_bias[:n_rgb, n_rgb:] = delta_r_to_t
            route_bias[n_rgb:, :n_rgb] = delta_t_to_r
            logits = logits + route_bias[None, None]
        try:
            cross_scale = float(os.environ.get('TBSI_TEMPLATE_CROSS_SCALE', '1.0'))
        except ValueError:
            raise ValueError('TBSI_TEMPLATE_CROSS_SCALE must be a finite float')
        if cross_scale < 0.0 or not torch.isfinite(logits.new_tensor(cross_scale)):
            raise ValueError('TBSI_TEMPLATE_CROSS_SCALE must be non-negative and finite')
        cross_support_mode = os.environ.get('TBSI_MSHA_CROSS_SUPPORT_MODE', 'off').lower()
        try:
            cross_support_layer = int(os.environ.get('TBSI_MSHA_CROSS_SUPPORT_LAYER', '-1'))
        except ValueError:
            raise ValueError('TBSI_MSHA_CROSS_SUPPORT_LAYER must be an integer')
        if cross_support_mode not in ('off', 'aligned_window', 'shifted_window'):
            raise ValueError('unsupported TBSI_MSHA_CROSS_SUPPORT_MODE: {}'.format(
                cross_support_mode))
        apply_cross_support = (cross_support_mode != 'off' and
                               (cross_support_layer < 0 or
                                cross_support_layer == getattr(self, 'path_effect_layer_index', -1)))
        if apply_cross_support:
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            side = int(round(n_rgb ** 0.5))
            if n_rgb != n_tir or side * side != n_rgb or logits.shape[-1] != n_rgb + n_tir:
                raise ValueError('cross-support diagnostic requires equal square RGB/TIR grids')
            token = torch.arange(n_rgb, device=logits.device)
            row_y, row_x = torch.div(token, side, rounding_mode='floor'), token % side
            if cross_support_mode == 'shifted_window':
                row_y = (row_y + side // 2) % side
                row_x = (row_x + side // 2) % side
            key_y, key_x = torch.div(token, side, rounding_mode='floor'), token % side
            dy = (row_y[:, None] - key_y[None, :]).abs()
            dx = (row_x[:, None] - key_x[None, :]).abs()
            support = (torch.minimum(dy, side - dy) <= 1) & (torch.minimum(dx, side - dx) <= 1)
            logits = logits.clone()
            logits[:, :, :n_rgb, n_rgb:] = logits[:, :, :n_rgb, n_rgb:].masked_fill(
                ~support[None, None], -1e4)
            logits[:, :, n_rgb:, :n_rgb] = logits[:, :, n_rgb:, :n_rgb].masked_fill(
                ~support.T[None, None], -1e4)
            self.last_cross_support_stats = {
                'mode': cross_support_mode,
                'layer': getattr(self, 'path_effect_layer_index', None),
                'allowed_sources_per_query': int(support.sum(dim=-1)[0].item()),
            }
        else:
            self.last_cross_support_stats = {
                'mode': 'off' if cross_support_mode == 'off' else 'not_target_layer',
                'requested_mode': cross_support_mode,
                'layer': getattr(self, 'path_effect_layer_index', None),
                'target_layer': cross_support_layer,
            }
        if cross_scale != 1.0:
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if n_rgb != n_tir or logits.shape[-1] != n_rgb + n_tir:
                raise ValueError('cross-scale probe requires equal RGB/TIR template lengths')
            logits = logits.clone()
            if cross_scale == 0.0:
                logits[:, :, :n_rgb, n_rgb:] = -1e4
                logits[:, :, n_rgb:, :n_rgb] = -1e4
            else:
                cross_bias = math.log(cross_scale)
                logits[:, :, :n_rgb, n_rgb:] += cross_bias
                logits[:, :, n_rgb:, :n_rgb] += cross_bias
            attn_prob = logits.softmax(dim=-1)
            if self.analysis_enabled:
                # Current-forward probability snapshot for read-only diagnostics.
                self.last_attn_prob = attn_prob.detach()
            if self.analysis_enabled and self.analysis_collector is not None:
                n_rgb = x_rgb_template.shape[1]
                v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
                a_rr = attn_prob[:, :, :n_rgb, :n_rgb]
                a_rt = attn_prob[:, :, :n_rgb, n_rgb:]
                a_tr = attn_prob[:, :, n_rgb:, :n_rgb]
                a_tt = attn_prob[:, :, n_rgb:, n_rgb:]
                msg_rgb_self = a_rr @ v_rgb
                msg_rgb_cross = a_rt @ v_tir
                msg_tir_self = a_tt @ v_tir
                msg_tir_cross = a_tr @ v_rgb
                message = (attn_prob @ v).transpose(1, 2).reshape(bsz, length, dim)
                projected_message = self.proj(message)
                residual_norm = residual.norm(dim=-1).mean().clamp_min(1e-8)
                output_change_norm = projected_message.norm(dim=-1).mean() / residual_norm

                def norm(value):
                    return value.norm(dim=-1).mean()

                self.analysis_collector(
                    attn_prob=attn_prob.detach(),
                    rgb_template=x_rgb_template.detach(),
                    tir_template=x_tir_template.detach(),
                    fg_masks=fg_masks,
                    rgb_self_norm=norm(msg_rgb_self).detach(),
                    rgb_cross_norm=norm(msg_rgb_cross).detach(),
                    tir_self_norm=norm(msg_tir_self).detach(),
                    tir_cross_norm=norm(msg_tir_cross).detach(),
                    output_change_norm=output_change_norm.detach(),
                    attention_output=projected_message.detach(),
                    final_output=(residual + projected_message).detach(),
                    cross_scale=cross_scale,
                )
                # Keep the actual output path identical to the disabled/default
                # path. Diagnostics must not change when the collector is enabled
                # only at selected frames in a closed-loop tracker.
                attn = self.attn_drop(logits.softmax(dim=-1))
        else:
            attn = self.attn_drop(logits.softmax(dim=-1))
        # Read-only evidence audit for the proposed differential MSHA route.
        # It deliberately runs after the native attention path and never changes
        # logits, attention, or output tensors.  The audit asks whether the
        # self- and cross-modal routes carry a shared response on the same
        # template-token axis before introducing a subtractive operator.
        diff_diag_path = os.environ.get('TBSI_MSHA_DIFF_DIAGNOSTICS_PATH', '')
        if diff_diag_path:
            import json
            with torch.no_grad():
                n_rgb = x_rgb_template.shape[1]
                n_tir = x_tir_template.shape[1]
                if n_rgb != n_tir or n_rgb * 2 != length:
                    raise ValueError('differential MSHA diagnostics require equal RGB/TIR template lengths')
                p = attn.float()
                v_float = v.float()
                a_rr = p[:, :, :n_rgb, :n_rgb]
                a_rt = p[:, :, :n_rgb, n_rgb:]
                a_tr = p[:, :, n_rgb:, :n_rgb]
                a_tt = p[:, :, n_rgb:, n_rgb:]
                eps = 1e-6

                def row_cond(a):
                    return a / a.sum(dim=-1, keepdim=True).clamp_min(eps)

                def overlap(a, b):
                    # Bhattacharyya overlap of two distributions on the
                    # aligned 8x8 template-token axis.
                    return torch.sqrt(a.clamp_min(0) * b.clamp_min(0)).sum(dim=-1)

                def cosine(a, b):
                    return F.cosine_similarity(a, b, dim=-1, eps=eps)

                rr, rt = row_cond(a_rr), row_cond(a_rt)
                tt, tr = row_cond(a_tt), row_cond(a_tr)
                msg_rr = torch.matmul(a_rr, v_float[:, :, :n_rgb])
                msg_rt = torch.matmul(a_rt, v_float[:, :, n_rgb:])
                msg_tt = torch.matmul(a_tt, v_float[:, :, n_rgb:])
                msg_tr = torch.matmul(a_tr, v_float[:, :, :n_rgb])

                rows = {
                    'layer': getattr(self, 'path_effect_layer_index', None),
                    'tag': os.environ.get('TBSI_MSHA_DIFF_DIAGNOSTIC_TAG', ''),
                    'batch': int(bsz),
                    'heads': int(self.num_heads),
                    'tokens_per_modality': int(n_rgb),
                    'rgb_query_attention_overlap_mean': float(overlap(rr, rt).mean().cpu()),
                    'tir_query_attention_overlap_mean': float(overlap(tt, tr).mean().cpu()),
                    'rgb_query_message_cosine_mean': float(cosine(msg_rr, msg_rt).mean().cpu()),
                    'tir_query_message_cosine_mean': float(cosine(msg_tt, msg_tr).mean().cpu()),
                    'rgb_cross_mass_mean': float(a_rt.sum(dim=-1).mean().cpu()),
                    'tir_cross_mass_mean': float(a_tr.sum(dim=-1).mean().cpu()),
                }
                if fg_masks is not None and len(fg_masks) == 2:
                    fg_r = fg_masks[0].to(device=p.device, dtype=p.dtype).clamp(0, 1)
                    fg_t = fg_masks[1].to(device=p.device, dtype=p.dtype).clamp(0, 1)
                    if fg_r.shape == (bsz, n_rgb) and fg_t.shape == (bsz, n_tir):
                        fg_r_q = fg_r[:, None, :]
                        fg_t_q = fg_t[:, None, :]
                        fg_r_den = fg_r_q.sum().clamp_min(eps) * float(self.num_heads)
                        fg_t_den = fg_t_q.sum().clamp_min(eps) * float(self.num_heads)
                        rows.update({
                            'rgb_fg_attention_overlap_mean': float((overlap(rr, rt) * fg_r_q).sum().cpu() / fg_r_den.cpu()),
                            'tir_fg_attention_overlap_mean': float((overlap(tt, tr) * fg_t_q).sum().cpu() / fg_t_den.cpu()),
                            'rgb_fg_query_mass_mean': float((a_rt.sum(dim=-1) * fg_r_q).sum().cpu() / fg_r_den.cpu()),
                            'tir_fg_query_mass_mean': float((a_tr.sum(dim=-1) * fg_t_q).sum().cpu() / fg_t_den.cpu()),
                        })
                with open(diff_diag_path, 'a') as handle:
                    handle.write(json.dumps(rows, sort_keys=True) + '\n')
        with torch.no_grad():
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            self.last_route_stats = {
                'r_to_t_mass': float(attn[:, :, :n_rgb, n_rgb:].sum().item() /
                                     (attn.shape[0] * attn.shape[1] * n_rgb)),
                't_to_r_mass': float(attn[:, :, n_rgb:, :n_rgb].sum().item() /
                                     (attn.shape[0] * attn.shape[1] * n_tir)),
                'delta_r_to_t': delta_r_to_t,
                'delta_t_to_r': delta_t_to_r,
            }
        early_gate_strength = 1.0
        if self.early_conditioned_cross_gate:
            try:
                early_gate_strength = float(os.environ.get('TBSI_TEST_EARLY_GATE_STRENGTH', '1.0'))
            except ValueError:
                raise ValueError('TBSI_TEST_EARLY_GATE_STRENGTH must be a finite float in [0, 1]')
            if (not math.isfinite(early_gate_strength) or
                    early_gate_strength < 0.0 or early_gate_strength > 1.0):
                raise ValueError('TBSI_TEST_EARLY_GATE_STRENGTH must be a finite float in [0, 1]')
        cross_value_gate = None
        try:
            fixed_r2t = float(os.environ.get('TBSI_NATIVE_MSHA_FIXED_R2T', '1.0'))
            fixed_t2r = float(os.environ.get('TBSI_NATIVE_MSHA_FIXED_T2R', '1.0'))
        except ValueError:
            raise ValueError('fixed native MSHA message scales must be finite floats')
        if fixed_r2t != 1.0 or fixed_t2r != 1.0:
            cross_value_gate = x_rgb_template.new_tensor([fixed_r2t, fixed_t2r]).view(1, 1, 2)
            cross_value_gate = cross_value_gate.expand(bsz, x_rgb_template.shape[1], 2)
            self.last_native_message_stats = {
                'mode': 'fixed', 'fixed_r2t': fixed_r2t, 'fixed_t2r': fixed_t2r,
                'gate_rgb_mean': fixed_r2t, 'gate_tir_mean': fixed_t2r,
            }
        native_mode = os.environ.get('TBSI_NATIVE_MSHA_MESSAGE_MODE', 'off').lower()
        if native_mode != 'off' and cross_value_gate is None:
            if early_context is None or len(early_context) != 4 or fg_masks is None:
                raise ValueError('native MSHA message gate requires early context and foreground masks')
            try:
                native_alpha = float(os.environ.get('TBSI_NATIVE_MSHA_MESSAGE_ALPHA', '0.25'))
                native_temperature = float(os.environ.get('TBSI_NATIVE_MSHA_MESSAGE_TEMPERATURE', '0.25'))
            except ValueError:
                raise ValueError('native MSHA message parameters must be finite floats')
            evidence = []
            for early_t, early_s, weights in ((early_context[0], early_context[2], fg_masks[0]),
                                               (early_context[1], early_context[3], fg_masks[1])):
                t_norm = F.normalize(early_t.float(), dim=-1, eps=1e-6)
                s_norm = F.normalize(early_s.float(), dim=-1, eps=1e-6)
                w = weights.to(device=x.device, dtype=torch.float32).clamp_min(0.0)
                if native_mode.startswith('p2_'):
                    temperature = float(os.environ.get('TBSI_P2_SCORE_TEMPERATURE', '0.07'))
                    log_w = torch.log(w + 1e-6)
                    similarity = torch.matmul(t_norm, s_norm.transpose(1, 2)) / temperature
                    score = torch.logsumexp(similarity + log_w.unsqueeze(-1), dim=1)
                    score = score - torch.logsumexp(log_w, dim=1, keepdim=True)
                else:
                    proto = F.normalize((t_norm * w.unsqueeze(-1)).sum(1) /
                                        w.sum(1, keepdim=True).clamp_min(1e-6), dim=-1, eps=1e-6)
                    score = torch.matmul(proto.unsqueeze(1), s_norm.transpose(1, 2)).squeeze(1)
                if native_mode == 'shifted':
                    side = int(round(score.shape[-1] ** 0.5))
                    score = torch.roll(score.reshape(bsz, side, side),
                                       shifts=(max(1, side // 2), max(1, side // 2)),
                                       dims=(1, 2)).reshape_as(score)
                elif native_mode not in ('target', 'p2_independent', 'p2_shared', 'p2_swapped'):
                    raise ValueError('unsupported native MSHA message mode: {}'.format(native_mode))
                peak = score.argmax(dim=-1)
                side = int(round(score.shape[-1] ** 0.5))
                yy, xx = torch.meshgrid(torch.arange(side, device=x.device),
                                        torch.arange(side, device=x.device), indexing='ij')
                local = ((yy.reshape(1, -1) - (peak // side).unsqueeze(1)).abs() <= 1) & \
                        ((xx.reshape(1, -1) - (peak % side).unsqueeze(1)).abs() <= 1)
                hard_bg = score.masked_fill(local, float('-inf')).max(dim=-1).values
                evidence.append({
                    'confidence': torch.tanh((score.gather(1, peak[:, None]).squeeze(1) - hard_bg) /
                                             max(native_temperature, 1e-6)),
                    'peak': peak,
                    'margin': score.gather(1, peak[:, None]).squeeze(1) - hard_bg,
                })
            c_rgb, c_tir = evidence[0]['confidence'], evidence[1]['confidence']
            if native_mode == 'p2_shared':
                gate_rgb = gate_tir = 0.5 * (c_rgb + c_tir)
            elif native_mode == 'p2_swapped':
                gate_rgb, gate_tir = c_tir, c_rgb
            else:
                gate_rgb, gate_tir = c_rgb, c_tir
            cross_value_gate = torch.stack((1.0 + native_alpha * gate_rgb,
                                            1.0 + native_alpha * gate_tir), dim=-1)
            cross_value_gate = cross_value_gate[:, None, :].expand(-1, x_rgb_template.shape[1], -1)
            self.last_native_message_stats = {
                'mode': native_mode,
                'alpha': native_alpha,
                'rgb_evidence_margin_mean': float(evidence[0]['margin'].mean().detach()),
                'tir_evidence_margin_mean': float(evidence[1]['margin'].mean().detach()),
                'rgb_evidence_mean': float(c_rgb.mean().detach()),
                'tir_evidence_mean': float(c_tir.mean().detach()),
                'gate_rgb_source_confidence_mean': float(gate_rgb.mean().detach()),
                'gate_tir_source_confidence_mean': float(gate_tir.mean().detach()),
                'rgb_candidate_index_mean': float(evidence[0]['peak'].float().mean().detach()),
                'tir_candidate_index_mean': float(evidence[1]['peak'].float().mean().detach()),
                'gate_rgb_mean': float(cross_value_gate[..., 0].mean().detach()),
                'gate_tir_mean': float(cross_value_gate[..., 1].mean().detach()),
            }
            native_diag_path = os.environ.get('TBSI_NATIVE_MSHA_DIAGNOSTICS_PATH', '')
            if native_diag_path:
                import json
                with open(native_diag_path, 'a') as handle:
                    handle.write(json.dumps(self.last_native_message_stats) + '\n')
        else:
            self.last_native_message_stats = {'mode': 'off'}
        if self.early_conditioned_cross_gate and early_gate_strength > 0.0:
            if early_context is None or len(early_context) != 4:
                raise ValueError('early conditioned gate requires four early token tensors')
            early_rgb_t, early_tir_t, early_rgb_s, early_tir_s = early_context
            if (early_rgb_t.shape != x_rgb_template.shape or
                    early_tir_t.shape != x_tir_template.shape or
                    early_rgb_s.ndim != 3 or early_tir_s.ndim != 3 or
                    early_rgb_s.shape[0] != x_rgb_template.shape[0] or
                    early_tir_s.shape[0] != x_tir_template.shape[0]):
                raise ValueError('early conditioned gate token shape mismatch')
            rgb_t = F.normalize(early_rgb_t.float(), dim=-1, eps=1e-6)
            tir_t = F.normalize(early_tir_t.float(), dim=-1, eps=1e-6)
            rgb_s = F.normalize(early_rgb_s.float(), dim=-1, eps=1e-6)
            tir_s = F.normalize(early_tir_s.float(), dim=-1, eps=1e-6)
            rgb_similarity = torch.matmul(rgb_t, rgb_s.transpose(1, 2))
            tir_similarity = torch.matmul(tir_t, tir_s.transpose(1, 2))
            rgb_query = torch.amax(rgb_similarity, dim=-1)
            tir_query = torch.amax(tir_similarity, dim=-1)
            agreement = (rgb_t * tir_t).sum(dim=-1)
            gate_logits = self.early_gate(torch.stack(
                (rgb_query, tir_query, agreement), dim=-1).to(x_rgb_template.dtype))
            if self.early_direct_margin_gate:
                if fg_masks is None or len(fg_masks) != 2:
                    raise ValueError('direct margin gate requires foreground template masks')
                weights = (0.5 * (fg_masks[0].float() + fg_masks[1].float())).clamp_min(0.0)
                rgb_evidence, tir_evidence, _ = self.early_margin_gate.foreground_candidate_margin(
                    rgb_similarity, tir_similarity, weights)
                rho = (self.early_margin_gate.direct_rho
                        if self.early_margin_gate.direct_rho is not None
                        else self.early_direct_margin_rho)
                if self.early_margin_gate.direct_rho is None:
                    try:
                        rho = float(os.environ.get('TBSI_TEST_EARLY_DIRECT_MARGIN_RHO', str(rho)))
                    except ValueError:
                        raise ValueError('TBSI_TEST_EARLY_DIRECT_MARGIN_RHO must be finite')
                temp = self.early_direct_margin_temperature
                # gate[..., 0] admits TIR into RGB; gate[..., 1] admits RGB into TIR.
                direct = torch.stack((torch.tanh(torch.as_tensor(rho, device=gate_logits.device, dtype=gate_logits.dtype) * torch.tanh(tir_evidence / temp)),
                                      torch.tanh(torch.as_tensor(rho, device=gate_logits.device, dtype=gate_logits.dtype) * torch.tanh(rgb_evidence / temp))), dim=-1)
                gate_logits = gate_logits + direct[:, None, :]
            if self.early_target_margin_gate:
                gate_logits = gate_logits + self.early_margin_gate(
                    rgb_similarity, tir_similarity, agreement)
            self.last_der_scores = None
            if self.discriminative_evidence_router:
                if fg_masks is None or len(fg_masks) != 2:
                    raise ValueError('DER requires foreground template weights')
                fg_weights = 0.5 * (fg_masks[0].float() + fg_masks[1].float())
                der_rgb, der_tir, der_correction = self.der(early_context, fg_weights)
                fg_scale = fg_weights.clamp(0, 1).unsqueeze(-1)
                gate_logits = gate_logits + fg_scale * der_correction.unsqueeze(1)
                self.last_der_scores = (der_rgb, der_tir)
            gate_delta = torch.tanh(self.early_gate_scale) * torch.tanh(gate_logits)
            if early_gate_strength != 1.0:
                gate_delta = early_gate_strength * gate_delta
            # Gate source template values independently for the two directions:
            # TIR evidence is admitted into RGB queries by gate[..., 1], and
            # RGB evidence is admitted into TIR queries by gate[..., 0].
            cross_value_gate = torch.exp(gate_delta).clamp(0.25, 4.0)
            self._early_value_evidence = torch.stack(
                (rgb_query, tir_query, agreement), dim=-1)
        elif self.early_template_value_reconstruction:
            raise ValueError('early Value reconstruction requires early conditioned gate context')
        if early_utility_logits is not None:
            if self.early_conditioned_cross_gate:
                raise ValueError('EarlyUtilityRouter and legacy early gate are mutually exclusive')
            if early_utility_logits.shape != (bsz, 2):
                raise ValueError('early utility logits must have shape [B,2]')
            utility_prob = torch.softmax(early_utility_logits.float(), dim=-1)
            utility_scale = torch.exp(0.25 * (2.0 * utility_prob - 1.0)).to(v.dtype)
            # Gate order follows source modality: RGB -> TIR queries, TIR -> RGB queries.
            cross_value_gate = utility_scale[:, None, :].expand(
                -1, x_rgb_template.shape[1], -1)
            self.last_utility_router_probs = utility_prob.detach()
        else:
            self.last_utility_router_probs = None
        if fcsv_cross_value_gate is not None:
            if cross_value_gate is None:
                cross_value_gate = fcsv_cross_value_gate
            else:
                cross_value_gate = cross_value_gate * fcsv_cross_value_gate
        with torch.no_grad():
            if self.early_conditioned_cross_gate and cross_value_gate is not None:
                self.last_early_gate_stats = {
                    'rgb_query_mean': float(rgb_query.mean().detach()),
                    'tir_query_mean': float(tir_query.mean().detach()),
                    'agreement_mean': float(agreement.mean().detach()),
                    'rgb_margin_mean': float((rgb_query - self.early_margin_gate._target_background_margin(rgb_similarity)[2]).mean().detach()) if self.early_target_margin_gate else None,
                    'tir_margin_mean': float((tir_query - self.early_margin_gate._target_background_margin(tir_similarity)[2]).mean().detach()) if self.early_target_margin_gate else None,
                    'gate_rgb_mean': float(cross_value_gate[..., 0].mean().detach()),
                    'gate_tir_mean': float(cross_value_gate[..., 1].mean().detach()),
                }
            else:
                self.last_early_gate_stats = {'mode': 'off'}
        if self.early_template_value_reconstruction:
            if not hasattr(self, '_early_value_evidence'):
                raise RuntimeError('early Value reconstruction evidence was not constructed')
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if n_rgb != n_tir:
                raise ValueError('early Value reconstruction requires equal template lengths')
            v_rgb_token = v[:, :, :n_rgb].transpose(1, 2).reshape(bsz, n_rgb, dim)
            v_tir_token = v[:, :, n_rgb:].transpose(1, 2).reshape(bsz, n_tir, dim)
            v_rgb_token, v_tir_token, recon_stats = self.early_value_reconstruction(
                v_rgb_token, v_tir_token, self._early_value_evidence.to(v_rgb_token.dtype))
            v = torch.cat((v_rgb_token.reshape(bsz, n_rgb, self.num_heads, self.head_dim).permute(0, 2, 1, 3),
                           v_tir_token.reshape(bsz, n_tir, self.num_heads, self.head_dim).permute(0, 2, 1, 3)), dim=2)
            self.last_early_value_reconstruction_stats = recon_stats
        else:
            self.last_early_value_reconstruction_stats = {'mode': 'off'}
        query_conditioned_active = (self.query_conditioned_cross_value and
                                    os.environ.get('TBSI_MSHA_QUERY_CONDITIONED_DISABLE', '0') != '1')
        if query_conditioned_active:
            if early_context is None or len(early_context) != 4:
                raise ValueError('query-conditioned cross-value requires early context')
            if cross_value_gate is not None:
                raise ValueError('query-conditioned cross-value is mutually exclusive with scalar cross gate')
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if n_rgb != n_tir:
                raise ValueError('query-conditioned cross-value requires equal template lengths')
            early_rgb_t, early_tir_t, early_rgb_s, early_tir_s = early_context
            rgb_t = F.normalize(early_rgb_t.float(), dim=-1, eps=1e-6)
            tir_t = F.normalize(early_tir_t.float(), dim=-1, eps=1e-6)
            rgb_s = F.normalize(early_rgb_s.float(), dim=-1, eps=1e-6)
            tir_s = F.normalize(early_tir_s.float(), dim=-1, eps=1e-6)
            rgb_similarity = torch.matmul(rgb_t, rgb_s.transpose(1, 2))
            tir_similarity = torch.matmul(tir_t, tir_s.transpose(1, 2))
            evidence = torch.stack((torch.amax(rgb_similarity, dim=-1),
                                    torch.amax(tir_similarity, dim=-1),
                                    (rgb_t * tir_t).sum(dim=-1)), dim=-1)
            condition_mode = os.environ.get(
                'TBSI_MSHA_QUERY_CONDITIONED_CONDITION', 'normal').lower()
            v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
            q_rgb, q_tir = q[:, :, :n_rgb], q[:, :, n_rgb:]
            k_rgb, k_tir = k[:, :, :n_rgb], k[:, :, n_rgb:]
            a_rr, a_rt = attn[:, :, :n_rgb, :n_rgb], attn[:, :, :n_rgb, n_rgb:]
            a_tr, a_tt = attn[:, :, n_rgb:, :n_rgb], attn[:, :, n_rgb:, n_rgb:]
            delta_tir = self.query_conditioned_rgb_from_tir(
                q_rgb, k_tir, v_tir, evidence, condition_mode)
            delta_rgb = self.query_conditioned_tir_from_rgb(
                q_tir, k_rgb, v_rgb, evidence, condition_mode)
            msg_rgb = a_rr @ v_rgb + (a_rt.unsqueeze(-1) * (v_tir.unsqueeze(2) + delta_tir)).sum(dim=3)
            msg_tir = a_tt @ v_tir + (a_tr.unsqueeze(-1) * (v_rgb.unsqueeze(2) + delta_rgb)).sum(dim=3)
            x = torch.cat((msg_rgb, msg_tir), dim=2)
            x = x.transpose(1, 2).reshape(bsz, length, dim)
            with torch.no_grad():
                self.last_query_conditioned_stats = {
                    'mode': condition_mode,
                    'delta_rgb_rms': float(delta_rgb.float().pow(2).mean().sqrt().detach()),
                    'delta_tir_rms': float(delta_tir.float().pow(2).mean().sqrt().detach()),
                    'delta_rgb_relative': float(delta_rgb.float().norm().detach() /
                                                  v_rgb.float().unsqueeze(2).norm().detach().clamp_min(1e-6)),
                    'delta_tir_relative': float(delta_tir.float().norm().detach() /
                                                  v_tir.float().unsqueeze(2).norm().detach().clamp_min(1e-6)),
                }
        elif (uncertainty_mode == 'message_gate' and participation is not None):
            # 只衰减跨模态 Value 消息；self message、query、attention 排序和
            # 外层 residual 保持原路径，避免把 uncertainty 误当成目标选择器。
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if participation.shape[1] != n_rgb or n_rgb != n_tir:
                raise ValueError('message_gate requires equal RGB/TIR template lengths')
            v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
            a_rr, a_rt = attn[:, :, :n_rgb, :n_rgb], attn[:, :, :n_rgb, n_rgb:]
            a_tr, a_tt = attn[:, :, n_rgb:, :n_rgb], attn[:, :, n_rgb:, n_rgb:]
            # reliability 对应每个 RGB/TIR 配对 token，共享同一 token gate。
            p_rgb = participation[:, None, :, None]
            p_tir = participation[:, None, :, None]
            if cross_value_gate is not None:
                p_tir = p_tir * cross_value_gate[:, None, :, 1:2]
                p_rgb = p_rgb * cross_value_gate[:, None, :, 0:1]
            msg_rgb = a_rr @ v_rgb + a_rt @ (v_tir * p_tir)
            msg_tir = a_tt @ v_tir + a_tr @ (v_rgb * p_rgb)
            x = torch.cat((msg_rgb, msg_tir), dim=2)
            with torch.no_grad():
                self.last_uncertainty_stats.update({
                    'gate_rgb_mean': float(participation[:, :n_rgb].mean().detach()),
                    'gate_tir_mean': float(participation.mean().detach()),
                    'gate_cross_value_applied': True,
                })
            x = x.transpose(1, 2).reshape(bsz, length, dim)
        else:
            if cross_value_gate is None:
                fg_cross_mode = os.environ.get('TBSI_MSHA_FG_CROSS_MODE', 'off').lower()
                if fg_cross_mode == 'off':
                    x = (attn @ v).transpose(1, 2).reshape(bsz, length, dim)
                elif fg_cross_mode == 'foreground_queries':
                    if fg_masks is None or len(fg_masks) != 2:
                        raise ValueError('foreground query cross mode requires template masks')
                    n_rgb = x_rgb_template.shape[1]
                    n_tir = x_tir_template.shape[1]
                    if fg_masks[0].shape != (bsz, n_rgb) or fg_masks[1].shape != (bsz, n_tir):
                        raise ValueError('foreground masks must match RGB/TIR template tokens')
                    v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
                    a_rr, a_rt = attn[:, :, :n_rgb, :n_rgb], attn[:, :, :n_rgb, n_rgb:]
                    a_tr, a_tt = attn[:, :, n_rgb:, :n_rgb], attn[:, :, n_rgb:, n_rgb:]
                    fg_rgb = fg_masks[0].to(device=attn.device, dtype=attn.dtype).clamp(0, 1)
                    fg_tir = fg_masks[1].to(device=attn.device, dtype=attn.dtype).clamp(0, 1)
                    msg_rgb = a_rr @ v_rgb + fg_rgb[:, None, :, None] * (a_rt @ v_tir)
                    msg_tir = a_tt @ v_tir + fg_tir[:, None, :, None] * (a_tr @ v_rgb)
                    x = torch.cat((msg_rgb, msg_tir), dim=2)
                    x = x.transpose(1, 2).reshape(bsz, length, dim)
                    with torch.no_grad():
                        self.last_fg_cross_stats = {
                            'mode': fg_cross_mode,
                            'rgb_foreground_mean': float(fg_rgb.mean().detach()),
                            'tir_foreground_mean': float(fg_tir.mean().detach()),
                        }
                elif fg_cross_mode == 'shifted_queries':
                    if fg_masks is None or len(fg_masks) != 2:
                        raise ValueError('shifted query cross mode requires template masks')
                    n_rgb = x_rgb_template.shape[1]
                    n_tir = x_tir_template.shape[1]
                    if n_rgb != n_tir or fg_masks[0].shape != (bsz, n_rgb) or fg_masks[1].shape != (bsz, n_tir):
                        raise ValueError('shifted query cross mode requires equal template grids and matching masks')
                    side = int(n_rgb ** 0.5)
                    if side * side != n_rgb:
                        raise ValueError('shifted query control requires a square template token grid')
                    v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
                    a_rr, a_rt = attn[:, :, :n_rgb, :n_rgb], attn[:, :, :n_rgb, n_rgb:]
                    a_tr, a_tt = attn[:, :, n_rgb:, :n_rgb], attn[:, :, n_rgb:, n_rgb:]
                    # 保持软前景 mask 的权重分布与总质量，只把 query 区域平移到非目标位置。
                    shift = max(1, side // 2)
                    fg_rgb = torch.roll(fg_masks[0].to(device=attn.device, dtype=attn.dtype).reshape(bsz, side, side),
                                        shifts=(shift, shift), dims=(1, 2)).reshape(bsz, n_rgb).clamp(0, 1)
                    fg_tir = torch.roll(fg_masks[1].to(device=attn.device, dtype=attn.dtype).reshape(bsz, side, side),
                                        shifts=(shift, shift), dims=(1, 2)).reshape(bsz, n_tir).clamp(0, 1)
                    msg_rgb = a_rr @ v_rgb + fg_rgb[:, None, :, None] * (a_rt @ v_tir)
                    msg_tir = a_tt @ v_tir + fg_tir[:, None, :, None] * (a_tr @ v_rgb)
                    x = torch.cat((msg_rgb, msg_tir), dim=2)
                    x = x.transpose(1, 2).reshape(bsz, length, dim)
                    with torch.no_grad():
                        self.last_fg_cross_stats = {
                            'mode': fg_cross_mode,
                            'rgb_foreground_mean': float(fg_rgb.mean().detach()),
                            'tir_foreground_mean': float(fg_tir.mean().detach()),
                            'rgb_mask_overlap': float((fg_rgb * fg_masks[0].to(device=attn.device, dtype=attn.dtype)).sum().detach() /
                                                      fg_rgb.sum().clamp_min(1e-8)),
                            'tir_mask_overlap': float((fg_tir * fg_masks[1].to(device=attn.device, dtype=attn.dtype)).sum().detach() /
                                                      fg_tir.sum().clamp_min(1e-8)),
                        }
                else:
                    raise ValueError('unsupported TBSI_MSHA_FG_CROSS_MODE: {}'.format(fg_cross_mode))
            else:
                n_rgb = x_rgb_template.shape[1]
                n_tir = x_tir_template.shape[1]
                v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
                a_rr, a_rt = attn[:, :, :n_rgb, :n_rgb], attn[:, :, :n_rgb, n_rgb:]
                a_tr, a_tt = attn[:, :, n_rgb:, :n_rgb], attn[:, :, n_rgb:, n_rgb:]
                gate_tir = cross_value_gate[:, None, :, 1:2]
                gate_rgb = cross_value_gate[:, None, :, 0:1]
                if token_reliability is not None:
                    rgb_scale, tir_scale, _ = token_reliability
                    gate_rgb = gate_rgb * rgb_scale[:, None, :, None].to(gate_rgb.dtype)
                    gate_tir = gate_tir * tir_scale[:, None, :, None].to(gate_tir.dtype)
                msg_rgb = a_rr @ v_rgb + a_rt @ (v_tir * gate_tir)
                msg_tir = a_tt @ v_tir + a_tr @ (v_rgb * gate_rgb)
                x = torch.cat((msg_rgb, msg_tir), dim=2)
                x = x.transpose(1, 2).reshape(bsz, length, dim)
        if os.environ.get('TBSI_PRIORITY2_CAPTURE', '0') == '1':
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            a_rt = attn[:, :, :n_rgb, n_rgb:]
            a_tr = attn[:, :, n_rgb:, :n_rgb]
            v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
            msg_tir_to_rgb = a_rt @ v_tir
            msg_rgb_to_tir = a_tr @ v_rgb
            if cross_value_gate is None:
                gate_rgb_source = torch.ones_like(msg_rgb_to_tir[..., :1])
                gate_tir_source = torch.ones_like(msg_tir_to_rgb[..., :1])
            else:
                gate_rgb_source = cross_value_gate[:, None, :, 0:1]
                gate_tir_source = cross_value_gate[:, None, :, 1:2]
            def norm_ratio(after, before):
                return float((after.float().norm() / before.float().norm().clamp_min(1e-8)).detach())
            self.last_native_message_stats.update({
                'rgb_to_tir_message_norm_before': float(msg_rgb_to_tir.float().norm().detach()),
                'rgb_to_tir_message_norm_after': float((msg_rgb_to_tir * gate_rgb_source).float().norm().detach()),
                'rgb_to_tir_message_norm_ratio': norm_ratio(msg_rgb_to_tir * gate_rgb_source,
                                                            msg_rgb_to_tir),
                'tir_to_rgb_message_norm_before': float(msg_tir_to_rgb.float().norm().detach()),
                'tir_to_rgb_message_norm_after': float((msg_tir_to_rgb * gate_tir_source).float().norm().detach()),
                'tir_to_rgb_message_norm_ratio': norm_ratio(msg_tir_to_rgb * gate_tir_source,
                                                            msg_tir_to_rgb),
            })
        # Receiver-conditioned centered Value diagnostic.  It preserves the
        # native QK attention and adds only a bounded, zero-initializable
        # cross-message content residual.  This path is off by default and is
        # used first as a fixed-state mechanism screen before adding trainable
        # parameters.
        receiver_mode = os.environ.get(
            'TBSI_RECEIVER_CENTERED_VALUE_MODE', 'off').lower()
        self.last_receiver_centered_stats = {'mode': 'off'}
        if receiver_mode != 'off':
            if receiver_mode not in ('correct', 'shifted', 'row_shuffled'):
                raise ValueError('unsupported receiver-centered Value mode: {}'.format(receiver_mode))
            if early_context is None or len(early_context) != 4 or fg_masks is None or len(fg_masks) != 2:
                raise ValueError('receiver-centered Value requires early context and foreground masks')
            n_rgb = x_rgb_template.shape[1]
            n_tir = x_tir_template.shape[1]
            if n_rgb != n_tir or n_rgb != 64:
                raise ValueError('receiver-centered Value expects 64 RGB/TIR template tokens')
            early_rgb_t, early_tir_t, early_rgb_s, early_tir_s = early_context
            tau = float(os.environ.get('TBSI_RECEIVER_CENTERED_TEMPERATURE', '0.07'))
            alpha = float(os.environ.get('TBSI_RECEIVER_CENTERED_ALPHA', '1.0'))
            rho = float(os.environ.get('TBSI_RECEIVER_CENTERED_RHO', '0.1'))
            if tau <= 0.0 or alpha < 0.0 or rho < 0.0:
                raise ValueError('receiver-centered Value parameters must be nonnegative and temperature positive')
            rgb_t = F.normalize(early_rgb_t.float(), dim=-1, eps=1e-6)
            tir_t = F.normalize(early_tir_t.float(), dim=-1, eps=1e-6)
            rgb_s = F.normalize(early_rgb_s.float(), dim=-1, eps=1e-6)
            tir_s = F.normalize(early_tir_s.float(), dim=-1, eps=1e-6)
            p_rgb = torch.softmax(torch.matmul(rgb_t, rgb_s.transpose(1, 2)) / tau, dim=-1)
            p_tir = torch.softmax(torch.matmul(tir_t, tir_s.transpose(1, 2)) / tau, dim=-1)
            if receiver_mode == 'shifted':
                side = int(round(p_tir.shape[-1] ** 0.5))
                if side * side != p_tir.shape[-1]:
                    raise ValueError('receiver-centered shifted mode requires square search grid')
                p_tir = torch.roll(p_tir.reshape(bsz, n_tir, side, side),
                                   shifts=(side // 2, side // 2), dims=(-2, -1)).reshape_as(p_tir)
            relation = torch.matmul(p_rgb, p_tir.transpose(1, 2))
            if receiver_mode == 'row_shuffled':
                relation = torch.roll(relation, shifts=1, dims=1)
            pi_rgb = fg_masks[0].to(device=x.device, dtype=torch.float32).clamp(0, 1)
            pi_tir = fg_masks[1].to(device=x.device, dtype=torch.float32).clamp(0, 1)
            pi_rgb = pi_rgb / pi_rgb.sum(-1, keepdim=True).clamp_min(1e-6)
            pi_tir = pi_tir / pi_tir.sum(-1, keepdim=True).clamp_min(1e-6)

            def receiver_transport(rel, source_prior):
                score = torch.log(rel.clamp_min(1e-6))
                centered = score - (source_prior[:, None, :] * score).sum(-1, keepdim=True)
                logits = torch.log(source_prior[:, None, :].clamp_min(1e-6)) + alpha * torch.tanh(centered)
                return torch.softmax(logits.masked_fill(source_prior[:, None, :] <= 0, -1e4), dim=-1)

            t_rgb_from_tir = receiver_transport(relation, pi_tir)
            t_tir_from_rgb = receiver_transport(relation.transpose(1, 2), pi_rgb)
            n_rgb = x_rgb_template.shape[1]
            v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
            if cross_value_gate is None:
                gate_rgb = torch.ones((bsz, n_rgb), device=v.device, dtype=v.dtype)
                gate_tir = torch.ones((bsz, n_tir), device=v.device, dtype=v.dtype)
            else:
                gate_rgb = cross_value_gate[..., 0].to(dtype=v.dtype)
                gate_tir = cross_value_gate[..., 1].to(dtype=v.dtype)
            u_rgb, u_tir = v_rgb * gate_rgb[:, None, :, None], v_tir * gate_tir[:, None, :, None]
            bar_tir = (pi_tir.to(dtype=u_tir.dtype)[:, None, :, None] * u_tir).sum(2)
            bar_rgb = (pi_rgb.to(dtype=u_rgb.dtype)[:, None, :, None] * u_rgb).sum(2)
            d_rgb = torch.einsum('bik,bhkd->bhid', t_rgb_from_tir.to(u_tir.dtype), u_tir) - bar_tir[:, :, None, :]
            d_tir = torch.einsum('bik,bhkd->bhid', t_tir_from_rgb.to(u_rgb.dtype), u_rgb) - bar_rgb[:, :, None, :]
            a_rt = attn[:, :, :n_rgb, n_rgb:]
            a_tr = attn[:, :, n_rgb:, :n_rgb]
            m_rgb = a_rt @ u_tir
            m_tir = a_tr @ u_rgb
            fg_rgb = fg_masks[0].to(device=x.device, dtype=v.dtype).clamp(0, 1)
            fg_tir = fg_masks[1].to(device=x.device, dtype=v.dtype).clamp(0, 1)
            def bounded(delta, native, fg, cross_mass):
                r = delta * fg[:, None, :, None] * cross_mass[:, :, :, None]
                rn = r.float().norm(dim=-1, keepdim=True)
                mn = native.float().norm(dim=-1, keepdim=True)
                return r * (rho * mn / (rn + rho * mn + 1e-6)).to(dtype=r.dtype)
            r_rgb = bounded(d_rgb, m_rgb, fg_rgb, a_rt.sum(-1))
            r_tir = bounded(d_tir, m_tir, fg_tir, a_tr.sum(-1))
            msg_rgb = x[:, :n_rgb].reshape(bsz, n_rgb, self.num_heads, self.head_dim).transpose(1, 2) + r_rgb
            msg_tir = x[:, n_rgb:].reshape(bsz, n_tir, self.num_heads, self.head_dim).transpose(1, 2) + r_tir
            x = torch.cat((msg_rgb, msg_tir), dim=2).transpose(1, 2).reshape(bsz, length, dim)
            with torch.no_grad():
                self.last_receiver_centered_stats = {
                    'mode': receiver_mode, 'alpha': alpha, 'rho': rho,
                    'relation_mean': float(relation.mean()),
                    'relation_std': float(relation.std(unbiased=False)),
                    'rgb_residual_ratio': float(r_rgb.float().norm() / m_rgb.float().norm().clamp_min(1e-6)),
                    'tir_residual_ratio': float(r_tir.float().norm() /
                                                m_tir.float().norm().clamp_min(1e-6)),
                }
        projected_message = self.proj(x)
        # Optional read-only audit for the template uncertainty experiment.
        # It is deliberately outside the model output path and disabled unless
        # an explicit file is supplied by the experiment runner.
        uncertainty_diag_path = os.environ.get('TBSI_TEMPLATE_UNCERTAINTY_DIAGNOSTICS_PATH', '')
        if uncertainty_diag_path:
            import json
            with torch.no_grad():
                entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(-1).mean()
                message_norm = projected_message.float().norm(dim=-1).mean()
                input_norm = residual.float().norm(dim=-1).mean().clamp_min(1e-8)
                row = {
                    'layer': getattr(self, 'path_effect_layer_index', None),
                    'mode': uncertainty_mode,
                    'uncertainty': getattr(self, 'last_uncertainty_stats', {}),
                    'route': getattr(self, 'last_route_stats', {}),
                    'attention_entropy': float(entropy.detach()),
                    'message_to_input_norm': float((message_norm / input_norm).detach()),
                    'message_norm': float(message_norm.detach()),
                }
                with open(uncertainty_diag_path, 'a') as handle:
                    handle.write(json.dumps(row) + '\n')
        if self.freq_relation is not None:
            cross_attn = torch.cat((attn[:, :, :n_rgb, n_rgb:].mean(1),
                                    attn[:, :, n_rgb:, :n_rgb].mean(1)), dim=-1)
            entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(-1).mean()
            self.last_r2_snapshot = {
                'attn': attn.detach().cpu(),
                'message': projected_message.detach().cpu(),
                'cross_mass': float(cross_attn.mean().detach().cpu()),
                'attention_entropy': float(entropy.detach().cpu()),
                'frequency_stats': getattr(self, 'last_frequency_stats', None),
            }
        # Read-only diagnostics for the default route. The older collector
        # path is inside the cross-scale probe branch; keep this callback
        # gated so normal forward behavior and tensor contracts are unchanged.
        if (self.analysis_enabled and self.analysis_collector is not None and
                cross_scale == 1.0):
            analysis_prob = logits.softmax(dim=-1)
            n_rgb = x_rgb_template.shape[1]
            v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
            a_rr = analysis_prob[:, :, :n_rgb, :n_rgb]
            a_rt = analysis_prob[:, :, :n_rgb, n_rgb:]
            a_tr = analysis_prob[:, :, n_rgb:, :n_rgb]
            a_tt = analysis_prob[:, :, n_rgb:, n_rgb:]
            def norm(value):
                return value.norm(dim=-1).mean()
            self.analysis_collector(
                attn_prob=analysis_prob.detach(),
                rgb_template=x_rgb_template.detach(),
                tir_template=x_tir_template.detach(),
                fg_masks=fg_masks,
                rgb_self_norm=norm(a_rr @ v_rgb).detach(),
                rgb_cross_norm=norm(a_rt @ v_tir).detach(),
                tir_self_norm=norm(a_tt @ v_tir).detach(),
                tir_cross_norm=norm(a_tr @ v_rgb).detach(),
                output_change_norm=(projected_message.norm(dim=-1).mean() /
                                    residual.norm(dim=-1).mean().clamp_min(1e-8)).detach(),
                attention_output=projected_message.detach(),
                final_output=(residual + projected_message).detach(),
                cross_scale=cross_scale,
            )
        if self.message_freq_refine is not None:
            n_rgb = x_rgb_template.shape[1]
            refined_rgb, refined_tir = self.message_freq_refine(
                projected_message[:, :n_rgb], projected_message[:, n_rgb:])
            projected_message = torch.cat((refined_rgb, refined_tir), dim=1)
        x = residual + self.proj_drop(projected_message)
        if self.expert_mode != 'off':
            if self.expert_mode == 'global':
                token_weight = x.new_ones((x.shape[0], x.shape[1]))
            else:
                if fg_masks is None or len(fg_masks) != 2:
                    raise ValueError('target expert mode requires RGB/TIR template masks')
                mask_r, mask_t = fg_masks
                n = x_rgb_template.shape[1]
                if mask_r.shape != (x.shape[0], n) or mask_t.shape != (x.shape[0], n):
                    raise ValueError('template masks must be [B,lens_z]')
                token_weight = torch.cat([mask_r, mask_t], dim=1)
            x = x + self.expert_gamma * self.template_expert(x, token_weight)
        if (self.analysis_enabled and self.output_analysis_collector is not None and
                cross_scale == 1.0):
            with torch.no_grad():
                self.output_analysis_collector(
                    final_output=x.detach(), fg_masks=fg_masks,
                    mode=os.environ.get('TBSI_MSHA_FG_CROSS_MODE', 'off').lower())
        return x[:, :x_rgb_template.shape[1]], x[:, x_rgb_template.shape[1]:]


class DegradationModulator(nn.Module):
    """
    [Phase 1 Fix] Joint modality confidence estimator for TBSILayer.

    First-principles fix:
      - BEFORE: independent MLPs for RGB and TIR — violates "reliability is a
        relationship property" (a token is reliable RELATIVE to the other modality)
      - AFTER: single joint MLP on cat(rgb, tir) → [conf_v, conf_i]
        The joint representation captures cross-modal contrast, so a token with
        poor RGB but good TIR correctly gets low conf_v and high conf_i.

    Architecture:
      - Input: cat(x_v_search, x_i_search)  →  (B, N_s, 2*C)
      - Joint MLP: Linear(2C → rdim) → ReLU → Linear(rdim → 2)
      - Output split: [conf_v, conf_i], each (B, N_s, 1) in [0,1]
    """
    def __init__(self, dim, reduction=4, temporal_dim=None):
        super().__init__()
        rdim = max(dim // reduction, 16)
        input_dim = dim * 2  # always joint: cat(rgb, tir)
        self.use_temporal = temporal_dim is not None
        if self.use_temporal:
            input_dim = dim * 2 + temporal_dim  # cat(rgb, tir, temporal)

        self.conf_joint = nn.Sequential(
            nn.Linear(input_dim, rdim),
            nn.ReLU(inplace=True),
            nn.Linear(rdim, 2),  # 2 outputs: [conf_v, conf_i]
            nn.Sigmoid(),
        )
        # Zero-init last layer: initial sigmoid(0) ≈ 0.5 → uniform confidence
        nn.init.zeros_(self.conf_joint[-2].weight)
        nn.init.zeros_(self.conf_joint[-2].bias)

    def forward(self, x_v_search, x_i_search, temporal_tokens=None):
        """
        Args:
            x_v_search: (B, N_s, C) visible search tokens
            x_i_search: (B, N_s, C) infrared search tokens
        Returns:
            conf_v, conf_i: each (B, N_s, 1) confidence in [0,1]
        """
        if self.use_temporal and temporal_tokens is not None:
            t = temporal_tokens.mean(dim=1, keepdim=True).expand(-1, x_v_search.shape[1], -1)
            joint_inp = torch.cat([x_v_search, x_i_search, t], dim=-1)
        else:
            joint_inp = torch.cat([x_v_search, x_i_search], dim=-1)

        conf_joint = self.conf_joint(joint_inp)  # (B, N_s, 2)
        conf_v = conf_joint[:, :, 0:1]           # (B, N_s, 1)
        conf_i = conf_joint[:, :, 1:2]           # (B, N_s, 1)
        return conf_v, conf_i


class TBSILayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_degradation=False,
                 use_attn_gate=False, use_temporal_tokens=False,
                 use_smsa=False, use_smsa_rgb=None, use_smsa_tir=None,
                 smsa_alpha=1.3, smsa_fp32=True, fusion_mode='legacy',
                 use_bridge_token_pruner=False, pruner_cfg=None,
                 template_expert_mode='off', template_expert_count=4,
                 template_expert_slots=4, template_expert_hidden=4,
                 cfvi_mode='off', cfvi_lambda_init=0.001,
                 cfvi_gate_hidden=64, cfvi_band_mask=None,
                 cfvi_band_renorm=True, freq_exchange_mode='off',
                 freq_exchange_diagnostics_path=None,
                 message_freq_refine_mode='off', freq_relation_mode='off',
                 freq_relation_hidden=64, freq_relation_max_bias=0.5,
                 freq_relation_calibration='off', freq_relation_topk=8,
                 freq_relation_clip=3.0, freq_relation_eps=1e-6,
                 pvl_adapter_dim=0, pvl_beta=2.35,
                 early_conditioned_cross_gate=False,
                 early_gate_hidden=32, early_target_margin_gate=False,
                 early_margin_hidden=16, early_margin_hard_negative_k=4,
                 early_direct_margin_gate=False, early_direct_margin_rho=0.0,
                 early_direct_margin_temperature=0.25,
                 early_direct_margin_trainable=False,
                 early_template_value_reconstruction=False,
                 early_template_value_rank=8,
                 early_template_value_max_delta=0.10,
                 discriminative_evidence_router=False, der_score_dim=64,
                 der_hidden_dim=8, role_pair_bias=False,
                 early_search_relation_bias=False, early_search_relation_dim=32,
                 early_search_relation_temperature=0.07,
                 early_search_relation_max_bias=0.5,
                 search_conditioned_t2s_mode='off',
                 search_conditioned_t2s_loc=False,
                 search_conditioned_t2s_rank=64,
                 search_conditioned_t2s_max_delta=0.10,
                 search_conditioned_t2s_temperature=0.07,
                 search_conditioned_t2s_competition_topk=4,
                 search_conditioned_t2s_router_temperature=None,
                 search_conditioned_t2s_content_strength=0.0,
                 search_conditioned_t2s_content_match_temperature=0.2,
                 search_conditioned_t2s_content_prior_floor=0.05,
                 search_conditioned_t2s_content_search_factor=4.0,
                 search_conditioned_t2s_content_neighborhood='object'):
        super().__init__()

        self.fusion_mode = fusion_mode
        if fusion_mode == 'tbsi_e':
            # TBSI-E: one template-only joint MHSA, followed by ordinary ViT
            # blocks. The legacy six CAST paths are intentionally absent.
            self.v2_template_fusion = V2TemplateFusion(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, norm_layer=norm_layer,
                expert_mode=template_expert_mode,
                expert_count=template_expert_count,
                expert_slots=template_expert_slots,
                expert_hidden=template_expert_hidden,
                cfvi_mode=cfvi_mode,
                cfvi_lambda_init=cfvi_lambda_init,
                cfvi_gate_hidden=cfvi_gate_hidden,
                cfvi_band_mask=cfvi_band_mask,
                cfvi_band_renorm=cfvi_band_renorm,
                freq_exchange_mode=freq_exchange_mode,
                freq_exchange_diagnostics_path=freq_exchange_diagnostics_path,
                message_freq_refine_mode=message_freq_refine_mode,
                freq_relation_mode=freq_relation_mode,
                freq_relation_hidden=freq_relation_hidden,
                freq_relation_max_bias=freq_relation_max_bias,
                freq_relation_calibration=freq_relation_calibration,
                freq_relation_topk=freq_relation_topk,
                freq_relation_clip=freq_relation_clip,
                freq_relation_eps=freq_relation_eps,
                pvl_adapter_dim=pvl_adapter_dim, pvl_beta=pvl_beta,
                early_conditioned_cross_gate=early_conditioned_cross_gate,
                early_gate_hidden=early_gate_hidden,
                early_target_margin_gate=early_target_margin_gate,
                early_margin_hidden=early_margin_hidden,
                early_margin_hard_negative_k=early_margin_hard_negative_k,
                early_direct_margin_gate=early_direct_margin_gate,
                early_direct_margin_rho=early_direct_margin_rho,
                early_direct_margin_temperature=early_direct_margin_temperature,
                early_direct_margin_trainable=early_direct_margin_trainable,
                early_template_value_reconstruction=early_template_value_reconstruction,
                early_template_value_rank=early_template_value_rank,
                early_template_value_max_delta=early_template_value_max_delta,
                discriminative_evidence_router=discriminative_evidence_router,
                der_score_dim=der_score_dim, der_hidden_dim=der_hidden_dim,
                role_pair_bias=role_pair_bias,
                early_search_relation_bias=early_search_relation_bias,
                early_search_relation_dim=early_search_relation_dim,
                early_search_relation_temperature=early_search_relation_temperature,
                early_search_relation_max_bias=early_search_relation_max_bias)
            self.t_fusion = None
            self.bridge_token_pruner = None
            self.use_degradation = False
            self.ca_s2t_i2f = None
            self.ca_t2s_f2i = None
            self.ca_s2t_v2f = None
            self.ca_t2s_f2v = None
            self.ca_t2t_f2v = None
            self.ca_t2t_f2i = None
            self.search_conditioned_t2s = (
                SearchConditionedT2SRouter(
                    dim=dim, num_heads=num_heads,
                    mode=search_conditioned_t2s_mode,
                    rank=search_conditioned_t2s_rank,
                    max_delta=search_conditioned_t2s_max_delta,
                    temperature=search_conditioned_t2s_temperature,
                    competition_topk=search_conditioned_t2s_competition_topk,
                    router_temperature=search_conditioned_t2s_router_temperature,
                    content_strength=search_conditioned_t2s_content_strength,
                    content_match_temperature=search_conditioned_t2s_content_match_temperature,
                    content_prior_floor=search_conditioned_t2s_content_prior_floor,
                    content_search_factor=search_conditioned_t2s_content_search_factor,
                    content_neighborhood=search_conditioned_t2s_content_neighborhood)
                if search_conditioned_t2s_mode != 'off' and search_conditioned_t2s_loc
                else None)
            return

        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.bridge_token_pruner = None
        if use_bridge_token_pruner:
            from lib.models.layers.bridge_token_pruner import BridgeTokenPruner
            pruner_cfg = pruner_cfg or {}
            self.bridge_token_pruner = BridgeTokenPruner(
                dim=pruner_cfg.get('dim', dim),
                template_keep=pruner_cfg.get('template_keep', 0.75),
                search_keep=pruner_cfg.get('search_keep', 0.75),
                hidden_dim=pruner_cfg.get('hidden_dim', 192),
                temperature=pruner_cfg.get('temperature', 0.15),
                hard_eval=pruner_cfg.get('hard_eval', False),
            )
        if fusion_mode == 'v2':
            self.v2_template_fusion = V2TemplateFusion(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, norm_layer=norm_layer)

        self.ca_s2t_v2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate,
            use_smsa=use_smsa if use_smsa_tir is None else use_smsa_tir,
            smsa_alpha=smsa_alpha,
            smsa_fp32=smsa_fp32
        )
        self.ca_t2s_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )
        self.ca_s2t_i2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate,
            use_smsa=use_smsa if use_smsa_rgb is None else use_smsa_rgb,
            smsa_alpha=smsa_alpha,
            smsa_fp32=smsa_fp32
        )
        self.ca_t2s_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )
        self.ca_t2t_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )
        self.ca_t2t_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
            use_attn_gate=use_attn_gate
        )

        self.use_degradation = use_degradation
        if use_degradation:
            # Only enable temporal context in DM when temporal tokens are actually running
            temporal_dim = dim if use_temporal_tokens else None
            self.degradation_mod = DegradationModulator(dim, temporal_dim=temporal_dim)

    def forward(self, x_v, x_i, lens_z, temporal_tokens=None, template_fg_masks=None,
                early_context=None, early_utility_logits=None):
        # x_v: [B, N, C], N = 320 (64 template + 256 search)
        # x_i: [B, N, C]
        if self.fusion_mode == 'tbsi_e':
            fused_v, fused_i = self.v2_template_fusion(
                x_v[:, :lens_z, :], x_i[:, :lens_z, :],
                fg_masks=template_fg_masks,
                x_rgb_search=x_v[:, lens_z:, :],
                x_tir_search=x_i[:, lens_z:, :],
                early_context=early_context,
                early_utility_logits=early_utility_logits)
            search_v, search_i = x_v[:, lens_z:, :], x_i[:, lens_z:, :]
            conditioned_mode = os.environ.get('TBSI_T2S_CONDITIONED_MESSAGE_MODE', 'off').lower()
            try:
                conditioned_alpha = float(os.environ.get('TBSI_T2S_CONDITIONED_MESSAGE_ALPHA', '0.0'))
                conditioned_temperature = float(os.environ.get('TBSI_T2S_CONDITIONED_MESSAGE_TEMPERATURE', '0.25'))
            except ValueError:
                raise ValueError('TBSI_T2S conditioned message parameters must be finite floats')
            if conditioned_mode != 'off' and conditioned_alpha != 0.0:
                if template_fg_masks is None or len(template_fg_masks) != 2:
                    raise ValueError('conditioned search message requires template foreground masks')
                search_v, self.last_conditioned_message_rgb = self.v2_template_fusion._condition_search_message(
                    fused_v, search_v, template_fg_masks[0], conditioned_alpha,
                    conditioned_mode, conditioned_temperature)
                search_i, self.last_conditioned_message_tir = self.v2_template_fusion._condition_search_message(
                    fused_i, search_i, template_fg_masks[1], conditioned_alpha,
                    conditioned_mode, conditioned_temperature)
            else:
                self.last_conditioned_message_rgb = {'mode': 'off'}
                self.last_conditioned_message_tir = {'mode': 'off'}
            if self.search_conditioned_t2s is not None:
                search_v, search_i = self.search_conditioned_t2s(
                    fused_v, fused_i, search_v, search_i,
                    template_fg_masks=template_fg_masks)
            else:
                self.last_search_conditioned_t2s_stats = {'mode': 'off'}
            if self.search_conditioned_t2s is not None:
                self.last_search_conditioned_t2s_stats = dict(
                    self.search_conditioned_t2s.last_stats)
            return (torch.cat([fused_v, search_v], dim=1),
                    torch.cat([fused_i, search_i], dim=1))

        if self.bridge_token_pruner is not None:
            x_v, x_i, _ = self.bridge_token_pruner(x_v, x_i, lens_z)
        if self.fusion_mode == 'v2':
            fused_v, fused_i = self.v2_template_fusion(
                x_v[:, :lens_z, :], x_i[:, :lens_z, :])
            x_v = torch.cat([fused_v, x_v[:, lens_z:, :]], dim=1)
            x_i = torch.cat([fused_i, x_i[:, lens_z:, :]], dim=1)
            # Continue through the standard TBSI interaction path. V2 changes
            # only the template initialization; search/template interaction
            # and its optional SMSA remain active and attributable.

        fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2)
        fused_t = self.t_fusion(fused_t)  # [B, 64, C]

        if self.use_degradation:
            # Per-patch confidence for search regions (with optional temporal context)
            conf_v, conf_i = self.degradation_mod(x_v[:, lens_z:, :], x_i[:, lens_z:, :],
                                                  temporal_tokens=temporal_tokens)
            # conf_v: (B, N_s, 1) high where RGB reliable
            # conf_i: (B, N_s, 1) high where TIR reliable
            # qm_v/qm_i: quality masks for cross-attention modulation
            qm_v = conf_v  # visible quality → modulate visible-related CA
            qm_i = conf_i  # infrared quality → modulate infrared-related CA

        # Search-to-Template: infrared search → fused template (use ir quality mask)
        fused_t = self.ca_s2t_i2f(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1),
                                  quality_mask=qm_i if self.use_degradation else None)[:, :lens_z, :]

        # Opt-in diagnostic: suppress only one template-to-search attention
        # message in one TBSI layer; the CAST residual and MLP remain active.
        path_action = os.environ.get('TBSI_PATH_EFFECT_ACTION', 'F').upper()
        try:
            path_layer = int(os.environ.get('TBSI_PATH_EFFECT_LAYER', '-1'))
        except ValueError:
            path_layer = -1
        layer_index = getattr(self, 'path_effect_layer_index', -1)
        try:
            path_scale = float(os.environ.get('TBSI_PATH_EFFECT_SCALE', '0.0'))
        except ValueError:
            path_scale = 0.0
        suppress_rgb_readout = path_action == 'GR' and layer_index == path_layer
        suppress_tir_readout = path_action == 'GT' and layer_index == path_layer

        # Template-to-Search: fused template → visible search (use vis quality mask)
        conditioned_mode = os.environ.get('TBSI_T2S_CONDITIONED_MESSAGE_MODE', 'off').lower()
        try:
            conditioned_alpha = float(os.environ.get('TBSI_T2S_CONDITIONED_MESSAGE_ALPHA', '0.0'))
            conditioned_temperature = float(os.environ.get('TBSI_T2S_CONDITIONED_MESSAGE_TEMPERATURE', '0.25'))
        except ValueError:
            raise ValueError('TBSI_T2S conditioned message parameters must be finite floats')
        temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1),
                                   quality_mask=qm_v if self.use_degradation else None,
                                   attention_scale=path_scale if suppress_rgb_readout else 1.0,
                                   template_fg_mask=(template_fg_masks[0] if template_fg_masks is not None else None),
                                   conditioned_message_alpha=conditioned_alpha,
                                   conditioned_message_mode=conditioned_mode,
                                   conditioned_message_temperature=conditioned_temperature)[:, lens_z:, :]

        # Search-to-Template: visible search → fused template (use vis quality mask)
        fused_t = self.ca_s2t_v2f(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1),
                                  quality_mask=qm_v if self.use_degradation else None)[:, :lens_z, :]

        # Template-to-Search: fused template → infrared search (use ir quality mask)
        temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1),
                                   quality_mask=qm_i if self.use_degradation else None,
                                   attention_scale=path_scale if suppress_tir_readout else 1.0,
                                   template_fg_mask=(template_fg_masks[1] if template_fg_masks is not None else None),
                                   conditioned_message_alpha=conditioned_alpha,
                                   conditioned_message_mode=conditioned_mode,
                                   conditioned_message_temperature=conditioned_temperature)[:, lens_z:, :]

        # Apply degradation-aware gating if enabled (实验1)
        if self.use_degradation:
            x_v = torch.cat([x_v[:, :lens_z, :], temp_x_v * (1 - conf_v) + x_v[:, lens_z:, :] * conf_v], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], temp_x_i * (1 - conf_i) + x_i[:, lens_z:, :] * conf_i], dim=1)
        else:
            x_v = torch.cat([x_v[:, :lens_z, :], temp_x_v], dim=1)
            x_i = torch.cat([x_i[:, :lens_z, :], temp_x_i], dim=1)

        # Template self-attention (no quality_mask — operates on template tokens)
        x_v[:, :lens_z, :] = self.ca_t2t_f2v(torch.cat([x_v[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]
        x_i[:, :lens_z, :] = self.ca_t2t_f2i(torch.cat([x_i[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]

        return x_v, x_i

    def get_smsa_stats(self):
        if self.ca_s2t_i2f is None or self.ca_s2t_v2f is None:
            return None
        stats = []
        for block in (self.ca_s2t_i2f, self.ca_s2t_v2f):
            value = getattr(block.attn_reshape, 'smsa_stats', None)
            if value is not None:
                stats.append(value)
        if not stats:
            return None
        return {key: sum(item[key] for item in stats) / len(stats) for key in stats[0]}
