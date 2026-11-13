"""
TBSILayer: Core cross-attention between RGB and TIR modalities.
Supports degradation-aware modulation at the per-layer level.
"""
import os
import torch
import torch.nn as nn
from lib.models.layers.attn_blocks import CASTBlock
from lib.models.layers.early_conditioned_msha_gate import (
    EarlyConditionedMSHAGate, early_msha_gate_evidence,
)
from lib.models.layers.fixed_source_value_gate import FixedSourceValueGate
from lib.models.layers.early_source_evidence import EarlySourceEvidence
from lib.models.layers.structured_spatial_evidence import StructuredSpatialEvidence
from lib.models.layers.dual_message_modulator import DualMessageModulator
from lib.models.layers.fine_grained_unified_allocator import FineGrainedUnifiedAllocator
from lib.models.layers.receiver_cross_write import ReceiverCrossWrite
from lib.models.layers.msha_message_normalization import MSHAMessageNormalization
from lib.models.layers.cross_mass_preserving_relation import (
    CrossMassPreservingReceiverSourceRelation,
)
from lib.models.layers.conservative_source_allocation import ConservativeSourceAllocation
from lib.models.layers.early_gate_evidence_gradient_isolation import EarlyGateEvidenceGradientIsolation
from lib.models.layers.query_source_conditional_write import QuerySourceConditionalWrite
from lib.models.layers.receiver_source_mass_reallocator import ReceiverSourceMassReallocator


class V2TemplateFusion(nn.Module):
    """V2-style joint template MHSA, kept behind an explicit feature gate."""

    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, early_conditioned_cross_gate=False,
                 early_gate_hidden=32, early_gate_source_mode='dynamic',
                 early_gate_search_context_mode='off',
                 early_gate_search_context_layer_aligned=False,
                 early_gate_center_source_log_gain=False,
                 early_source_evidence=False,
                 early_source_evidence_hidden=24, early_source_evidence_delta=0.25,
                 early_source_evidence_topk=4, structured_spatial_evidence=False,
                 structured_evidence_hidden=8, dual_message_mode='off',
                 dual_message_max_delta=0.1, fu_enabled=False, fu_rank=8,
                 fu_max_relation=0.5, fu_max_rejection=0.5,
                 receiver_cross_write=False, receiver_write_max_delta=0.1,
                 msha_message_normalization='off',
                 early_gate_value_modulation=True,
                 early_gate_conservative_allocation=False,
                 early_gate_evidence_gradient_isolation=False,
                 cross_mass_preserving_relation=False,
                 cross_relation_max_bias=0.1,
                 query_source_write=False, query_source_write_hidden=32,
                 query_source_write_max_delta=0.15,
                 mass_redist_enable=False, mass_redist_mode="pair",
                 early_target_support_mode="max",
                 early_target_support_temperature=0.1,
                 spatial_pair_mode="off", spatial_pair_temperature=0.1,
                 spatial_pair_max_bias=1.0):
        super().__init__()
        self.norm = norm_layer(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.capture_gate_utility = False
        self.last_gate_log_gain = None
        self.spatial_pair_mode = str(spatial_pair_mode)
        if self.spatial_pair_mode not in ('off', 'pair', 'sep'):
            raise ValueError('Unknown spatial pair mode')
        if self.spatial_pair_mode != 'off':
            if (not early_conditioned_cross_gate or not early_gate_value_modulation or
                    early_target_support_mode != 'max' or early_gate_source_mode != 'dynamic' or
                    early_source_evidence or structured_spatial_evidence or fu_enabled or
                    receiver_cross_write or msha_message_normalization != 'off' or
                    cross_mass_preserving_relation or early_gate_conservative_allocation or
                    early_gate_evidence_gradient_isolation):
                raise ValueError('Spatial pair experiment requires unmodified native P gate path')
            from .spatial_pair_template_interaction import SpatialPairTemplateInteraction
            self.spatial_pair = SpatialPairTemplateInteraction(
                self.spatial_pair_mode, spatial_pair_temperature, spatial_pair_max_bias)
        self.early_target_support_mode = str(early_target_support_mode)
        if self.early_target_support_mode not in ('max', 'uniform', 'target'):
            raise ValueError('unknown Early target support mode')
        if self.early_target_support_mode != 'max':
            if not early_conditioned_cross_gate or not early_gate_value_modulation:
                raise ValueError('target evidence requires original dynamic Value gate')
            if (early_source_evidence or structured_spatial_evidence or fu_enabled or
                    receiver_cross_write or msha_message_normalization != 'off' or
                    early_gate_conservative_allocation or early_gate_evidence_gradient_isolation or
                    cross_mass_preserving_relation or early_gate_source_mode != 'dynamic'):
                raise ValueError('evidence-only experiment cannot combine other candidate actions')
            from .early_target_support_evidence import EarlyTargetSupportEvidence
            self.early_target_support = EarlyTargetSupportEvidence(
                self.early_target_support_mode, early_target_support_temperature)
        self.early_conditioned_cross_gate = bool(early_conditioned_cross_gate)
        self.early_gate_search_context_mode = str(early_gate_search_context_mode)
        self.early_gate_search_context_layer_aligned = bool(early_gate_search_context_layer_aligned)
        self.early_gate_center_source_log_gain = bool(early_gate_center_source_log_gain)
        if self.early_gate_search_context_mode not in ('off', 'summary', 'cross', 'global_mean'):
            raise ValueError('unknown Early gate search-context mode')
        if self.early_conditioned_cross_gate and early_gate_value_modulation:
            if (not self.early_conditioned_cross_gate or not early_gate_value_modulation or
                    early_gate_source_mode != 'dynamic' or early_source_evidence or
                    structured_spatial_evidence or dual_message_mode != 'off' or
                    fu_enabled or receiver_cross_write or cross_mass_preserving_relation or
                    spatial_pair_mode != 'off' or early_gate_conservative_allocation or
                    early_gate_evidence_gradient_isolation):
                raise ValueError('search-context gain requires the native Early gate action')
            from .early_gate_search_context import EarlyGateSearchContext
            # Keep this matched, inactive-capable branch from advancing the
            # RNG used to initialize any existing tracker parameter.
            with torch.random.fork_rng(devices=[]):
                self.search_context_gain = EarlyGateSearchContext(
                    dim=dim, mode=self.early_gate_search_context_mode)
        self.early_gate_value_modulation = bool(early_gate_value_modulation)
        self.early_gate_conservative_allocation = bool(early_gate_conservative_allocation)
        self.early_gate_evidence_gradient_isolation = bool(early_gate_evidence_gradient_isolation)
        if self.early_gate_evidence_gradient_isolation:
            self.early_gate_evidence_isolation = EarlyGateEvidenceGradientIsolation()
        if self.early_gate_conservative_allocation:
            self.conservative_source_allocation = ConservativeSourceAllocation()
        self.cross_mass_preserving_relation_enabled = bool(cross_mass_preserving_relation)
        self.query_source_write_enabled = bool(query_source_write)
        self.mass_redist_enabled = bool(mass_redist_enable)
        self.mass_redist_mode = str(mass_redist_mode)
        if self.mass_redist_mode not in ('pair', 'lr_joint', 'lr_conserved', 'lr_expected'):
            raise ValueError('unknown Mass-Redist mode')
        if self.mass_redist_mode != 'pair':
            from .low_rank_joint_mass_reallocator import fusion_conflict
        if self.mass_redist_mode != 'pair' and (not self.mass_redist_enabled or
                fusion_conflict(early_gate_source_mode, early_source_evidence,
                                structured_spatial_evidence, dual_message_mode,
                                fu_enabled, receiver_cross_write,
                                msha_message_normalization, spatial_pair_mode,
                                early_gate_conservative_allocation,
                                early_gate_evidence_gradient_isolation,
                                early_target_support_mode,
                                early_gate_center_source_log_gain)):
            raise ValueError('low-rank MR requires the isolated MR-only g=1 path')
        if self.mass_redist_enabled:
            if not (self.early_conditioned_cross_gate and self.early_gate_value_modulation and
                    self.early_gate_search_context_mode == 'cross'):
                raise ValueError('Mass-Redist requires C cross-context source Value gating')
            if self.cross_mass_preserving_relation_enabled or self.query_source_write_enabled:
                raise ValueError('Mass-Redist cannot be combined with another relation/write action')
            with torch.random.fork_rng(devices=[]):
                if self.mass_redist_mode == 'pair':
                    allocator = ReceiverSourceMassReallocator
                    options = {}
                elif self.mass_redist_mode == 'lr_expected':
                    from .expected_score_mass_reallocator import ExpectedScoreMassReallocator
                    allocator = ExpectedScoreMassReallocator
                    options = {}
                else:
                    from .low_rank_joint_mass_reallocator import LowRankJointMassReallocator
                    allocator = LowRankJointMassReallocator
                    options = {'allocation': ('joint' if self.mass_redist_mode == 'lr_joint'
                                               else 'conserved')}
                self.mass_reallocator_rgb = allocator(dim=dim, **options)
                self.mass_reallocator_tir = allocator(dim=dim, **options)
        if self.query_source_write_enabled:
            self.query_source_write = QuerySourceConditionalWrite(
                self.head_dim, query_source_write_hidden, query_source_write_max_delta)
        self.analysis_enabled = False
        self.analysis_collector = None
        self.last_relation_diagnostics = None
        self.last_fusion_stats = None
        self.structured_spatial_evidence_enabled = bool(structured_spatial_evidence)
        self.dual_message_mode = str(dual_message_mode).lower()
        self.msha_message_normalization = str(msha_message_normalization).lower()
        self.message_normalization = MSHAMessageNormalization(
            self.msha_message_normalization)
        if self.cross_mass_preserving_relation_enabled and self.msha_message_normalization != 'off':
            raise ValueError('cross-mass-preserving relation requires native message path')
        if self.dual_message_mode in ('false', 'none'):
            self.dual_message_mode = 'off'
        if self.dual_message_mode != 'off' and not self.structured_spatial_evidence_enabled:
            raise ValueError('active D mode requires the structured E encoder')
        self.structured_support_logits = None
        if self.structured_spatial_evidence_enabled:
            self.structured_evidence = StructuredSpatialEvidence(structured_evidence_hidden)
        if self.dual_message_mode != 'off':
            if self.dual_message_mode not in ('cross_summary', 'cross_structured', 'dual_structured'):
                raise ValueError('unknown dual message mode: ' + self.dual_message_mode)
            self.dual_message = DualMessageModulator(
                dim=dim, evidence_dim=32, max_delta=dual_message_max_delta)
        if self.early_conditioned_cross_gate:
            self.early_gate_source_mode = str(early_gate_source_mode).lower()
            if self.early_gate_source_mode not in ('dynamic', 'fixed'):
                raise ValueError('unknown early gate source mode: ' + self.early_gate_source_mode)
            if self.early_gate_value_modulation:
                self.early_gate = EarlyConditionedMSHAGate(early_gate_hidden)
                if self.early_gate_source_mode == 'fixed':
                    self.fixed_source_gate = FixedSourceValueGate()
                self.early_gate_scale = nn.Parameter(torch.tensor(0.1))
                if self.mass_redist_mode != 'pair':
                    # Explicit MR-only contract, never inferred from weight values.
                    self.early_gate_scale.data.zero_()
                    self.early_gate_scale.requires_grad_(False)
                    self.early_gate.requires_grad_(False)
                    self.search_context_gain.readout.requires_grad_(False)
            if self.cross_mass_preserving_relation_enabled:
                if self.early_gate_value_modulation:
                    raise ValueError(
                        'cross-mass-preserving relation requires Value gate action disabled')
                self.cross_relation_rgb = CrossMassPreservingReceiverSourceRelation(
                    evidence_index=1, max_bias=cross_relation_max_bias)
                self.cross_relation_tir = CrossMassPreservingReceiverSourceRelation(
                    evidence_index=0, max_bias=cross_relation_max_bias)
            self.early_source_evidence_enabled = bool(early_source_evidence)
            if self.early_source_evidence_enabled:
                self.early_source_evidence = EarlySourceEvidence(
                    hidden_dim=early_source_evidence_hidden,
                    max_delta=early_source_evidence_delta,
                    topk=early_source_evidence_topk)
            self.fu_enabled = bool(fu_enabled)
            if self.fu_enabled:
                self.fu_allocator = FineGrainedUnifiedAllocator(
                    dim=dim, rank=fu_rank, max_relation=fu_max_relation,
                    max_rejection=fu_max_rejection)
            self.receiver_cross_write_enabled = bool(receiver_cross_write)
            if self.receiver_cross_write_enabled:
                self.receiver_write_rgb = ReceiverCrossWrite(receiver_write_max_delta)
                self.receiver_write_tir = ReceiverCrossWrite(receiver_write_max_delta)
        else:
            self.early_source_evidence_enabled = False
            self.fu_enabled = False
            self.receiver_cross_write_enabled = False

    def forward(self, x_rgb_template, x_tir_template, early_context=None,
                template_boxes=None, fusion_context=None):
        self.last_gate_log_gain = None
        # The two modality template streams interact jointly, then are split
        # back so the surrounding ViT and search path remain unchanged.
        x = torch.cat([x_rgb_template, x_tir_template], dim=1)
        residual = x
        x = self.norm(x)
        bsz, length, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) * self.scale
        attention = logits.softmax(dim=-1)
        attn = self.attn_drop(attention)
        if self.early_conditioned_cross_gate:
            evidence, rgb_similarity, tir_similarity = early_msha_gate_evidence(
                early_context, x_rgb_template, x_tir_template)
            if self.early_target_support_mode != 'max':
                evidence = self.early_target_support(
                    evidence, rgb_similarity, tir_similarity, template_boxes)
            if self.early_source_evidence_enabled:
                evidence = self.early_source_evidence(
                    evidence, rgb_similarity, tir_similarity)
            if self.early_gate_value_modulation:
                gate_evidence = (self.early_gate_evidence_isolation(evidence)
                                 if self.early_gate_evidence_gradient_isolation else evidence)
                if self.early_gate_source_mode == 'dynamic':
                    if self.mass_redist_mode != 'pair':
                        condition_context = (fusion_context
                                             if self.early_gate_search_context_layer_aligned
                                             else early_context)
                        if self.early_gate_search_context_layer_aligned and condition_context is None:
                            raise ValueError('layer-aligned MR requires fusion_context')
                        self.search_context_values = self.search_context_gain.context_only(
                            gate_evidence, early_context, condition_context,
                            template_boxes=template_boxes)
                        gate_logits = None
                    else:
                        gate_logits = self.early_gate(gate_evidence.to(x.dtype))
                    if (self.early_gate_search_context_mode != 'off' and
                            self.mass_redist_mode == 'pair'):
                        condition_context = (fusion_context
                                             if self.early_gate_search_context_layer_aligned
                                             else early_context)
                        if self.early_gate_search_context_layer_aligned and condition_context is None:
                            raise ValueError('layer-aligned search context requires fusion_context')
                        if self.mass_redist_enabled:
                            context_logits, self.search_context_values = self.search_context_gain(
                                gate_evidence, early_context,
                                condition_context=condition_context, return_context=True)
                            gate_logits = gate_logits + context_logits
                        else:
                            gate_logits = gate_logits + self.search_context_gain(
                                gate_evidence, early_context,
                                condition_context=condition_context)
                else:
                    gate_logits = self.fixed_source_gate(
                        evidence.shape[0], evidence.shape[1], x)
                if self.mass_redist_mode != 'pair':
                    gate = x.new_ones((evidence.shape[0], evidence.shape[1], 2))
                    gate_log_gain = None
                elif self.early_gate_conservative_allocation:
                    gate = self.conservative_source_allocation(
                        gate_logits, self.early_gate_scale)
                    gate_log_gain = None
                else:
                    gate_log_gain = (torch.tanh(self.early_gate_scale) *
                                     torch.tanh(gate_logits))
                    if self.early_gate_center_source_log_gain:
                        from .source_log_gain_centering import center_source_log_gain
                        gate_log_gain = center_source_log_gain(gate_log_gain)
                    gate = torch.exp(gate_log_gain).clamp(0.25, 4.0)
            else:
                gate_log_gain = None
                gate = x.new_ones((evidence.shape[0], evidence.shape[1], 2))
            if self.capture_gate_utility and self.early_gate_value_modulation:
                if gate_log_gain is None:
                    raise RuntimeError('utility capture requires the native dynamic gate')
                self.last_gate_log_gain = gate_log_gain
            n_rgb = x_rgb_template.shape[1]
            if self.structured_spatial_evidence_enabled:
                if template_boxes is None:
                    raise ValueError('E/D requires transformed first-frame template boxes')
                state_rgb, state_tir, support_logits, coverage, _ = self.structured_evidence(
                    rgb_similarity, tir_similarity, template_boxes, evidence,
                    action_mode=('summary' if self.dual_message_mode == 'cross_summary'
                                 else 'structured'))
                self.structured_support_logits = support_logits
            else:
                state_rgb = state_tir = None
                self.structured_support_logits = None
            v_rgb, v_tir = v[:, :, :n_rgb], v[:, :, n_rgb:]
            cross_rgb, cross_tir = v_rgb, v_tir
            if hasattr(self, 'value_direction') and self.value_direction_enabled:
                mode = self.value_direction_mode
                rgb_kwargs, tir_kwargs = {}, {}
                if getattr(self, 'value_direction_aligned_enabled', False):
                    rgb_kwargs['receiver_attention'] = attn[:, :, n_rgb:, :n_rgb]
                    tir_kwargs['receiver_attention'] = attn[:, :, :n_rgb, n_rgb:]
                cross_rgb = self.value_direction(
                    cross_rgb, x_rgb_template, x_tir_template, evidence, mode=mode, **rgb_kwargs)
                # Canonical evidence is relative to the source role.
                cross_tir = self.value_direction(
                    cross_tir, x_tir_template, x_rgb_template,
                    evidence[..., [1, 0, 2]], mode=mode, **tir_kwargs)
            attn_rt = attn[:, :, :n_rgb, n_rgb:]
            attn_tr = attn[:, :, n_rgb:, :n_rgb]
            attn_rr = attn[:, :, :n_rgb, :n_rgb]
            attn_tt = attn[:, :, n_rgb:, n_rgb:]
            if self.early_gate_value_modulation:
                gated_tir = cross_tir * gate[:, None, :, 1:2]
                gated_rgb = cross_rgb * gate[:, None, :, 0:1]
            else:
                # Relation-only mode keeps the original Value content intact.
                gated_tir, gated_rgb = cross_tir, cross_rgb
            relation_rt, relation_tr = attn_rt, attn_tr
            relation_diag = None
            if self.cross_mass_preserving_relation_enabled:
                relation_rt, diag_rt = self.cross_relation_rgb(
                    q[:, :, :n_rgb], k[:, :, n_rgb:], attn_rt,
                    evidence)
                relation_tr, diag_tr = self.cross_relation_tir(
                    q[:, :, n_rgb:], k[:, :, :n_rgb], attn_tr,
                    evidence)
                relation_diag = {'rgb_receiver': diag_rt, 'tir_receiver': diag_tr}
            if self.mass_redist_enabled:
                self_rgb = (attn_rr @ v_rgb).transpose(1, 2).reshape(bsz, n_rgb, dim)
                self_tir = (attn_tt @ v_tir).transpose(1, 2).reshape(bsz, n_rgb, dim)
                if self.mass_redist_mode != 'pair':
                    relation_rt, self_scale_rgb = self.mass_reallocator_rgb(
                        attention[:, :, :n_rgb, n_rgb:],
                        attention[:, :, :n_rgb, :n_rgb].sum(-1, keepdim=True),
                        self_rgb, v_tir, self.search_context_values[..., 1],
                        evidence[..., [1, 0, 2]], cross_attention=attn_rt)
                    relation_tr, self_scale_tir = self.mass_reallocator_tir(
                        attention[:, :, n_rgb:, :n_rgb],
                        attention[:, :, n_rgb:, n_rgb:].sum(-1, keepdim=True),
                        self_tir, v_rgb, self.search_context_values[..., 0],
                        evidence, cross_attention=attn_tr)
                else:
                    relation_rt = self.mass_reallocator_rgb(
                        attn_rt * gate[:, None, None, :, 1], self_rgb, v_tir,
                        self.search_context_values[..., 1], evidence[..., [1, 0, 2]])
                    relation_tr = self.mass_reallocator_tir(
                        attn_tr * gate[:, None, None, :, 0], self_tir, v_rgb,
                        self.search_context_values[..., 0], evidence)
                gated_tir, gated_rgb = cross_tir, cross_rgb
            if self.msha_message_normalization != 'off':
                msg_rr, msg_rt, _ = self.message_normalization(
                    attn_rr, attn_rt, v_rgb, gated_tir)
                msg_tt, msg_tr, _ = self.message_normalization(
                    attn_tt, attn_tr, v_tir, gated_rgb)
            else:
                msg_rr = attn_rr @ v_rgb
                msg_tt = attn_tt @ v_tir
                if self.mass_redist_mode != 'pair':
                    msg_rr = msg_rr * self_scale_rgb
                    msg_tt = msg_tt * self_scale_tir
                msg_rt = relation_rt @ gated_tir
                msg_tr = relation_tr @ gated_rgb
            if self.query_source_write_enabled:
                write_rt = self.query_source_write(q[:, :, :n_rgb], k[:, :, n_rgb:], v_tir)
                write_tr = self.query_source_write(q[:, :, n_rgb:], k[:, :, :n_rgb], v_rgb)
                msg_rt = (relation_rt * write_rt) @ gated_tir
                msg_tr = (relation_tr * write_tr) @ gated_rgb
            if self.spatial_pair_mode != 'off':
                msg_rt, msg_tr = self.spatial_pair(
                    msg_rt, msg_tr, attn_rt, attn_tr, cross_tir, cross_rgb,
                    gate, rgb_similarity, tir_similarity)
            if self.fu_enabled:
                if template_boxes is None or early_context is None:
                    raise ValueError('FU requires early features and transformed template boxes')
                early_rgb_t, early_tir_t, early_rgb_s, early_tir_s = early_context
                msg_rt = self.fu_allocator(
                    attn_rt, v_tir, gate[..., 1], x_rgb_template, x_tir_template,
                    template_boxes[:, 0], template_boxes[:, 1], evidence,
                    evidence[..., [1, 0, 2]], tir_similarity, early_tir_s)
                msg_tr = self.fu_allocator(
                    attn_tr, v_rgb, gate[..., 0], x_tir_template, x_rgb_template,
                    template_boxes[:, 1], template_boxes[:, 0], evidence[..., [1, 0, 2]],
                    evidence, rgb_similarity, early_rgb_s)
            if self.dual_message_mode != 'off':
                scale_rr, scale_rt, _message_diag_rgb = self.dual_message(
                    state_rgb, state_tir, coverage[:, 0], coverage[:, 1],
                    msg_rr, msg_rt, attention[:, :, :n_rgb, :n_rgb],
                    attention[:, :, :n_rgb, n_rgb:],
                    enable_self=self.dual_message_mode == 'dual_structured')
                scale_tt, scale_tr, _message_diag_tir = self.dual_message(
                    state_tir, state_rgb, coverage[:, 1], coverage[:, 0],
                    msg_tt, msg_tr, attention[:, :, n_rgb:, n_rgb:],
                    attention[:, :, n_rgb:, :n_rgb],
                    enable_self=self.dual_message_mode == 'dual_structured')
                msg_rr = msg_rr * scale_rr[:, None, :, :]
                msg_rt = msg_rt * scale_rt[:, None, :, :]
                msg_tt = msg_tt * scale_tt[:, None, :, :]
                msg_tr = msg_tr * scale_tr[:, None, :, :]
            if self.receiver_cross_write_enabled:
                msg_rt, _ = self.receiver_write_rgb(msg_rr, msg_rt)
                msg_tr, _ = self.receiver_write_tir(msg_tt, msg_tr)
            rgb = msg_rr + msg_rt
            tir = msg_tt + msg_tr
            x = torch.cat((rgb, tir), dim=2).transpose(1, 2).reshape(bsz, length, dim)
            if self.analysis_enabled:
                self.last_relation_diagnostics = relation_diag
                self.last_pair_attention = attention.detach()
                self.last_pair_q = q.detach()
                self.last_pair_k = k.detach()
                def _norm(tensor):
                    return tensor.detach().float().norm(dim=-1)
                self.last_fusion_stats = {
                    'cross_mass_rgb_to_tir': float(attention[:, :, :n_rgb, n_rgb:].sum(-1).mean().detach()),
                    'cross_mass_tir_to_rgb': float(attention[:, :, n_rgb:, :n_rgb].sum(-1).mean().detach()),
                    'cross_entropy_rgb_to_tir': float((-(attn_rt.clamp_min(1e-8) * attn_rt.clamp_min(1e-8).log()).sum(-1).mean()).detach()),
                    'cross_entropy_tir_to_rgb': float((-(attn_tr.clamp_min(1e-8) * attn_tr.clamp_min(1e-8).log()).sum(-1).mean()).detach()),
                    'self_message_norm_rgb': float(_norm(msg_rr).mean()),
                    'cross_message_norm_rgb': float(_norm(msg_rt).mean()),
                    'self_message_norm_tir': float(_norm(msg_tt).mean()),
                    'cross_message_norm_tir': float(_norm(msg_tr).mean()),
                    'value_gate_mean_rgb_source': float(gate[:, :, 0].mean().detach()),
                    'value_gate_mean_tir_source': float(gate[:, :, 1].mean().detach()),
                }
                if callable(self.analysis_collector):
                    self.analysis_collector(
                        attention=attention.detach(), attn_used=attn.detach(), q=q.detach(), k=k.detach(),
                        v=v.detach(), evidence=evidence.detach(), gate=gate.detach(),
                        rgb_similarity=rgb_similarity.detach(),
                        tir_similarity=tir_similarity.detach(),
                        msg_rr=msg_rr.detach(), msg_rt=msg_rt.detach(), msg_tt=msg_tt.detach(),
                        msg_tr=msg_tr.detach(), relation_diagnostics=relation_diag,
                        rgb_template=x_rgb_template.detach(), tir_template=x_tir_template.detach())
        else:
            x = (attn @ v).transpose(1, 2).reshape(bsz, length, dim)
        x = residual + self.proj_drop(self.proj(x))
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
                 early_conditioned_cross_gate=False, early_gate_hidden=32,
                 early_gate_source_mode='dynamic',
                 early_gate_search_context_mode='off',
                 early_gate_search_context_layer_aligned=False,
                 early_gate_center_source_log_gain=False,
                 early_source_evidence=False, early_source_evidence_hidden=24,
                 early_source_evidence_delta=0.25, early_source_evidence_topk=4,
                 structured_spatial_evidence=False, structured_evidence_hidden=8,
                 dual_message_mode='off', dual_message_max_delta=0.1,
                 fu_enabled=False, fu_rank=8, fu_max_relation=0.5,
                 fu_max_rejection=0.5, receiver_cross_write=False,
                 receiver_write_max_delta=0.1,
                 msha_message_normalization='off',
                 early_gate_value_modulation=True,
                 early_gate_conservative_allocation=False,
                 early_gate_evidence_gradient_isolation=False,
                 cross_mass_preserving_relation=False,
                 cross_relation_max_bias=0.1,
                 query_source_write=False, query_source_write_hidden=32,
                 query_source_write_max_delta=0.15,
                 mass_redist_enable=False, mass_redist_mode="pair",
                 early_target_support_mode="max",
                 early_target_support_temperature=0.1,
                 spatial_pair_mode="off", spatial_pair_temperature=0.1,
                 spatial_pair_max_bias=1.0):
        super().__init__()
        self.fusion_mode = fusion_mode
        if fusion_mode == 'tbsi_e':
            # Match the archived parent state: no dead legacy bridge weights.
            self.v2_template_fusion = V2TemplateFusion(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, norm_layer=norm_layer,
                early_conditioned_cross_gate=early_conditioned_cross_gate,
                early_gate_hidden=early_gate_hidden,
                early_gate_source_mode=early_gate_source_mode,
                early_gate_search_context_mode=early_gate_search_context_mode,
                early_gate_search_context_layer_aligned=early_gate_search_context_layer_aligned,
                early_gate_center_source_log_gain=early_gate_center_source_log_gain,
                early_source_evidence=early_source_evidence,
                early_source_evidence_hidden=early_source_evidence_hidden,
                early_source_evidence_delta=early_source_evidence_delta,
                early_source_evidence_topk=early_source_evidence_topk,
                structured_spatial_evidence=structured_spatial_evidence,
                structured_evidence_hidden=structured_evidence_hidden,
                dual_message_mode=dual_message_mode,
                dual_message_max_delta=dual_message_max_delta,
                fu_enabled=fu_enabled, fu_rank=fu_rank,
                fu_max_relation=fu_max_relation, fu_max_rejection=fu_max_rejection,
                receiver_cross_write=receiver_cross_write,
                receiver_write_max_delta=receiver_write_max_delta,
                msha_message_normalization=msha_message_normalization,
                early_gate_value_modulation=early_gate_value_modulation,
                early_gate_conservative_allocation=early_gate_conservative_allocation,
                early_gate_evidence_gradient_isolation=early_gate_evidence_gradient_isolation,
                cross_mass_preserving_relation=cross_mass_preserving_relation,
                cross_relation_max_bias=cross_relation_max_bias,
                query_source_write=query_source_write,
                query_source_write_hidden=query_source_write_hidden,
                query_source_write_max_delta=query_source_write_max_delta,
                mass_redist_enable=mass_redist_enable,
                mass_redist_mode=mass_redist_mode,
                early_target_support_mode=early_target_support_mode,
                early_target_support_temperature=early_target_support_temperature,
                spatial_pair_mode=spatial_pair_mode,
                spatial_pair_temperature=spatial_pair_temperature,
                spatial_pair_max_bias=spatial_pair_max_bias)
            return

        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.fusion_mode = fusion_mode
        if fusion_mode in ('v2', 'tbsi_e'):
            self.v2_template_fusion = V2TemplateFusion(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, norm_layer=norm_layer,
                early_conditioned_cross_gate=early_conditioned_cross_gate,
                early_gate_hidden=early_gate_hidden,
                early_gate_source_mode=early_gate_source_mode,
                early_gate_search_context_mode=early_gate_search_context_mode,
                early_gate_search_context_layer_aligned=early_gate_search_context_layer_aligned,
                early_gate_center_source_log_gain=early_gate_center_source_log_gain,
                early_source_evidence=early_source_evidence,
                early_source_evidence_hidden=early_source_evidence_hidden,
                early_source_evidence_delta=early_source_evidence_delta,
                early_source_evidence_topk=early_source_evidence_topk,
                structured_spatial_evidence=structured_spatial_evidence,
                structured_evidence_hidden=structured_evidence_hidden,
                dual_message_mode=dual_message_mode,
                dual_message_max_delta=dual_message_max_delta,
                fu_enabled=fu_enabled, fu_rank=fu_rank,
                fu_max_relation=fu_max_relation, fu_max_rejection=fu_max_rejection,
                receiver_cross_write=receiver_cross_write,
                receiver_write_max_delta=receiver_write_max_delta,
                msha_message_normalization=msha_message_normalization,
                early_gate_value_modulation=early_gate_value_modulation,
                early_gate_conservative_allocation=early_gate_conservative_allocation,
                early_gate_evidence_gradient_isolation=early_gate_evidence_gradient_isolation,
                cross_mass_preserving_relation=cross_mass_preserving_relation,
                cross_relation_max_bias=cross_relation_max_bias,
                query_source_write=query_source_write,
                query_source_write_hidden=query_source_write_hidden,
                query_source_write_max_delta=query_source_write_max_delta,
                mass_redist_enable=mass_redist_enable,
                mass_redist_mode=mass_redist_mode,
                early_target_support_mode=early_target_support_mode,
                early_target_support_temperature=early_target_support_temperature,
                spatial_pair_mode=spatial_pair_mode,
                spatial_pair_temperature=spatial_pair_temperature,
                spatial_pair_max_bias=spatial_pair_max_bias)

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

    def forward(self, x_v, x_i, lens_z, temporal_tokens=None, early_context=None,
                template_boxes=None, fusion_context=None):
        # x_v: [B, N, C], N = 320 (64 template + 256 search)
        # x_i: [B, N, C]
        if self.fusion_mode == 'tbsi_e':
            fused_v, fused_i = self.v2_template_fusion(
                x_v[:, :lens_z, :], x_i[:, :lens_z, :], early_context=early_context,
                template_boxes=template_boxes, fusion_context=fusion_context)
            # Retained TBSI-E updates templates only. Subsequent ViT blocks
            # propagate these changes to search tokens; legacy bridges do not run.
            return (torch.cat([fused_v, x_v[:, lens_z:, :]], dim=1),
                    torch.cat([fused_i, x_i[:, lens_z:, :]], dim=1))
        if self.fusion_mode == 'v2':
            fused_v, fused_i = self.v2_template_fusion(
                x_v[:, :lens_z, :], x_i[:, :lens_z, :], early_context=early_context,
                template_boxes=template_boxes, fusion_context=fusion_context)
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

        # Template-to-Search: fused template → visible search (use vis quality mask)
        temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1),
                                   quality_mask=qm_v if self.use_degradation else None)[:, lens_z:, :]

        # Search-to-Template: visible search → fused template (use vis quality mask)
        fused_t = self.ca_s2t_v2f(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1),
                                  quality_mask=qm_v if self.use_degradation else None)[:, :lens_z, :]

        # Template-to-Search: fused template → infrared search (use ir quality mask)
        temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1),
                                   quality_mask=qm_i if self.use_degradation else None)[:, lens_z:, :]

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
        if self.fusion_mode == 'tbsi_e':
            return None
        stats = []
        for block in (self.ca_s2t_i2f, self.ca_s2t_v2f):
            value = getattr(block.attn_reshape, 'smsa_stats', None)
            if value is not None:
                stats.append(value)
        if not stats:
            return None
        return {key: sum(item[key] for item in stats) / len(stats) for key in stats[0]}
