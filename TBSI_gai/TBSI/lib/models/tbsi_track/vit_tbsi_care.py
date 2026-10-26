""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman

Modified by Botao Ye
"""
import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.tbsi_track.base_backbone import BaseBackbone
from lib.models.tbsi_track.utils import combine_tokens, recover_tokens

from lib.models.layers.tbsi_layer import TBSILayer
from lib.models.layers.pvl_adapter import PVLAdapter
from lib.models.layers.early_search_residual import EarlySearchResidual
from lib.models.layers.early_utility_router import EarlyUtilityRouter
from lib.models.layers.post_tbsi_cross_modal_residual import PostTBSICrossModalResidual
from lib.models.layers.local_cross_modal_correspondence import LocalCrossModalCorrespondence


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _forward_impl_attn(self, x):
        feat, attn = self.attn(self.norm1(x), True)
        x = x + self.drop_path(feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

    def forward(self, x, return_attention=False):
        if return_attention:
            return self._forward_impl_attn(x)
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class VisionTransformerTBSI(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 tbsi_loc=None, tbsi_drop_path=None,
                 da_in_layer=False, use_attn_gate=False,
                 use_checkpoint=False, use_smsa=False, use_smsa_rgb=None, use_smsa_tir=None,
                 smsa_alpha=1.3, smsa_fp32=True, fusion_mode='legacy',
                 bridge_token_pruner_enable=False, bridge_token_pruner_layer_ids=None,
                 bridge_token_pruner_cfg=None,
                 template_expert_mode='off', template_expert_loc=None,
                 template_expert_count=4, template_expert_slots=4,
                 template_expert_hidden=4, cfvi_mode='off',
                 cfvi_lambda_init=0.001, cfvi_gate_hidden=64, cfvi_loc=None,
                 cfvi_band_mask=None, cfvi_band_renorm=True,
                 freq_exchange_mode='off', freq_exchange_loc=None,
                 freq_exchange_diagnostics_path=None,
                 message_freq_refine_mode='off', message_freq_refine_loc=None,
                 freq_relation_mode='off', freq_relation_loc=None,
                 freq_relation_hidden=64, freq_relation_max_bias=0.5,
                 freq_relation_calibration='off', freq_relation_topk=8,
                 freq_relation_clip=3.0, freq_relation_eps=1e-6,
                 early_search_residual_enable=False, early_search_residual_early_block=2,
                 early_search_residual_late_block=9, early_search_residual_force_zero=False,
                 early_search_residual_use_scsu=False, early_search_residual_scsu_k1=3,
                 early_search_residual_scsu_k2=5,
                 post_tbsi_residual_enable=False, post_tbsi_residual_mode='spatial',
                 post_tbsi_residual_block=5, post_tbsi_residual_hidden_dim=192,
                 post_tbsi_residual_heads=4, post_tbsi_residual_force_zero=False,
                 pvl_enable=False, pvl_position='template_loc6',
                 pvl_loc=6, pvl_adapter_dim=64, pvl_beta=2.35,
                 early_conditioned_cross_gate=False, early_gate_hidden=32,
                 early_target_margin_gate=False, early_margin_hidden=16,
                 early_margin_hard_negative_k=4,
                 early_direct_margin_gate=False, early_direct_margin_rho=0.0,
                 early_direct_margin_temperature=0.25,
                 early_direct_margin_trainable=False,
                 early_template_value_reconstruction=False,
                 early_template_value_rank=8,
                 early_template_value_max_delta=0.10,
                 discriminative_evidence_router=False, der_score_dim=64,
                 der_hidden_dim=8, early_utility_router_enable=False,
                 early_utility_descriptor_dim=32, early_utility_hidden_dim=64,
                 msha_role_pair_bias_enable=False, msha_role_pair_bias_loc=9,
                 early_search_relation_bias_enable=False, early_search_relation_bias_loc=9,
                 early_search_relation_dim=32, early_search_relation_temperature=0.07,
                 early_search_relation_max_bias=0.5,
                 search_conditioned_t2s_mode='off',
                 search_conditioned_t2s_loc=None,
                 search_conditioned_t2s_rank=64,
                 search_conditioned_t2s_max_delta=0.10,
                 search_conditioned_t2s_temperature=0.07,
                 search_conditioned_t2s_competition_topk=4,
                 search_conditioned_t2s_router_temperature=None,
                 search_conditioned_t2s_content_strength=0.0,
                 search_conditioned_t2s_content_match_temperature=0.2,
                 search_conditioned_t2s_content_prior_floor=0.05,
                 search_conditioned_t2s_content_search_factor=4.0,
                 search_conditioned_t2s_content_neighborhood='object',
                 local_cross_modal_correspondence_enable=False,
                 local_cross_modal_correspondence_block=2,
                 local_cross_modal_correspondence_hidden_dim=96,
                 local_cross_modal_correspondence_relevance_dim=64,
                 local_cross_modal_correspondence_max_delta=0.05,
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                use_checkpoint=use_checkpoint)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.early_search_residual = None
        self.early_search_residual_early_block = int(early_search_residual_early_block)
        self.early_search_residual_late_block = int(early_search_residual_late_block)
        if early_search_residual_enable:
            self.early_search_residual = EarlySearchResidual(
                dim=embed_dim, force_zero=early_search_residual_force_zero,
                use_scsu=early_search_residual_use_scsu,
                scsu_k1=early_search_residual_scsu_k1,
                scsu_k2=early_search_residual_scsu_k2)
        self.post_tbsi_residual = None
        self.post_tbsi_residual_block = int(post_tbsi_residual_block)
        if post_tbsi_residual_enable:
            self.post_tbsi_residual = PostTBSICrossModalResidual(
                dim=embed_dim, mode=post_tbsi_residual_mode,
                hidden_dim=post_tbsi_residual_hidden_dim,
                num_heads=post_tbsi_residual_heads,
                force_zero=post_tbsi_residual_force_zero)
        self.local_cross_modal_correspondence = None
        self.local_cross_modal_correspondence_block = int(local_cross_modal_correspondence_block)
        if local_cross_modal_correspondence_enable:
            self.local_cross_modal_correspondence = LocalCrossModalCorrespondence(
                dim=embed_dim,
                hidden_dim=local_cross_modal_correspondence_hidden_dim,
                relevance_dim=local_cross_modal_correspondence_relevance_dim,
                max_delta=local_cross_modal_correspondence_max_delta)

        # Template-bridged search region interaction (TBSI)
        self.tbsi_loc = tbsi_loc
        self.early_utility_router_enable = bool(early_utility_router_enable)
        self.early_utility_router = (
            EarlyUtilityRouter(embed_dim, early_utility_descriptor_dim,
                               early_utility_hidden_dim)
            if self.early_utility_router_enable else None)
        self.tbsi_fusion = fusion_mode
        self.tbsi_drop_path = tbsi_drop_path
        self.tbsi_layers = nn.ModuleList()
        self.da_in_layer = da_in_layer
        self.use_checkpoint = use_checkpoint
        if self.tbsi_loc is not None and type(self.tbsi_loc) == list:
            for i in range(len(self.tbsi_loc)):
                use_pruner = (bridge_token_pruner_enable and
                              i in (bridge_token_pruner_layer_ids or []))
                self.tbsi_layers.append(TBSILayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=self.tbsi_drop_path[i], norm_layer=norm_layer, act_layer=act_layer,
                    use_degradation=da_in_layer, use_attn_gate=use_attn_gate,
                    use_smsa=use_smsa, use_smsa_rgb=use_smsa_rgb, use_smsa_tir=use_smsa_tir,
                    smsa_alpha=smsa_alpha, smsa_fp32=smsa_fp32,
                    fusion_mode=fusion_mode,
                    use_bridge_token_pruner=use_pruner,
                    pruner_cfg=bridge_token_pruner_cfg,
                    template_expert_mode=(template_expert_mode
                                          if self.tbsi_loc[i] in (template_expert_loc or [])
                                          else 'off'),
                    template_expert_count=template_expert_count,
                    template_expert_slots=template_expert_slots,
                    template_expert_hidden=template_expert_hidden,
                    cfvi_mode=(cfvi_mode if not cfvi_loc or self.tbsi_loc[i] in cfvi_loc else 'off'),
                    cfvi_lambda_init=cfvi_lambda_init,
                    cfvi_gate_hidden=cfvi_gate_hidden,
                    cfvi_band_mask=cfvi_band_mask,
                    cfvi_band_renorm=cfvi_band_renorm,
                    freq_exchange_mode=(freq_exchange_mode
                                        if not freq_exchange_loc or self.tbsi_loc[i] in freq_exchange_loc
                                        else 'off'),
                    freq_exchange_diagnostics_path=freq_exchange_diagnostics_path,
                    message_freq_refine_mode=(message_freq_refine_mode
                        if self.tbsi_loc[i] in (message_freq_refine_loc or []) else 'off'),
                    freq_relation_mode=(freq_relation_mode
                        if self.tbsi_loc[i] in (freq_relation_loc or []) else 'off'),
                    freq_relation_hidden=freq_relation_hidden,
                    freq_relation_max_bias=freq_relation_max_bias,
                    freq_relation_calibration=freq_relation_calibration,
                    freq_relation_topk=freq_relation_topk,
                    freq_relation_clip=freq_relation_clip,
                    freq_relation_eps=freq_relation_eps,
                    pvl_adapter_dim=(pvl_adapter_dim if pvl_enable and
                                     pvl_position == 'template_loc6' and
                                     self.tbsi_loc[i] == pvl_loc else 0),
                    pvl_beta=pvl_beta,
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
                    role_pair_bias=(msha_role_pair_bias_enable and
                                    self.tbsi_loc[i] == msha_role_pair_bias_loc),
                    early_search_relation_bias=(early_search_relation_bias_enable and
                                                self.tbsi_loc[i] == early_search_relation_bias_loc),
                    early_search_relation_dim=early_search_relation_dim,
                    early_search_relation_temperature=early_search_relation_temperature,
                    early_search_relation_max_bias=early_search_relation_max_bias,
                    search_conditioned_t2s_mode=search_conditioned_t2s_mode,
                    search_conditioned_t2s_loc=(self.tbsi_loc[i] in
                                                (search_conditioned_t2s_loc or [])),
                    search_conditioned_t2s_rank=search_conditioned_t2s_rank,
                    search_conditioned_t2s_max_delta=search_conditioned_t2s_max_delta,
                    search_conditioned_t2s_temperature=search_conditioned_t2s_temperature,
                    search_conditioned_t2s_competition_topk=search_conditioned_t2s_competition_topk,
                    search_conditioned_t2s_router_temperature=search_conditioned_t2s_router_temperature,
                    search_conditioned_t2s_content_strength=search_conditioned_t2s_content_strength,
                    search_conditioned_t2s_content_match_temperature=search_conditioned_t2s_content_match_temperature,
                    search_conditioned_t2s_content_prior_floor=search_conditioned_t2s_content_prior_floor,
                    search_conditioned_t2s_content_search_factor=search_conditioned_t2s_content_search_factor,
                    search_conditioned_t2s_content_neighborhood=search_conditioned_t2s_content_neighborhood,
                    ))
                self.tbsi_layers[-1].path_effect_layer_index = i

        self.init_weights(weight_init)
        if self.local_cross_modal_correspondence is not None:
            self.local_cross_modal_correspondence.reset_output()

    def forward_features(self, z, x, temporal_tokens=None, template_fg_masks=None,
                         diagnostic=False, diagnostic_early_block=2,
                         diagnostic_late_block=9, diagnostic_tbsi_index=0,
                         diagnostic_post_block=5):
        B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]

        x_v = self.patch_embed(x[0])
        z_v = self.patch_embed(z[0])
        x_i = self.patch_embed(x[1])
        z_i = self.patch_embed(z[1]) 

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # Visible and infrared data share the positional encoding and other parameters in ViT
        z_v += self.pos_embed_z
        x_v += self.pos_embed_x
        z_i += self.pos_embed_z
        x_i += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
        x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x_v = self.pos_drop(x_v)
        x_i = self.pos_drop(x_i)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        
        tbsi_index = 0
        diagnostic_features = {} if diagnostic else None
        der_score_maps = []
        early_gate_stats = []
        early_rgb_for_residual = None
        early_tir_for_residual = None
        early_template_rgb_for_gate = None
        early_template_tir_for_gate = None
        early_search_rgb_for_gate = None
        early_search_tir_for_gate = None
        early_utility_logits = None
        for i, blk in enumerate(self.blocks):
            x_v = blk(x_v)
            x_i = blk(x_i)
            if i == 2:
                early_template_rgb_for_gate = x_v[:, :lens_z, :]
                early_template_tir_for_gate = x_i[:, :lens_z, :]
                early_search_rgb_for_gate = x_v[:, lens_z:, :]
                early_search_tir_for_gate = x_i[:, lens_z:, :]
                if self.early_utility_router is not None:
                    if template_fg_masks is None:
                        raise ValueError('early utility router requires template foreground masks')
                    early_utility_logits = self.early_utility_router(
                        (early_template_rgb_for_gate, early_template_tir_for_gate,
                         early_search_rgb_for_gate, early_search_tir_for_gate),
                         template_fg_masks)
            if (self.local_cross_modal_correspondence is not None and
                    i == self.local_cross_modal_correspondence_block):
                x_v[:, lens_z:, :], x_i[:, lens_z:, :] = self.local_cross_modal_correspondence(
                    x_v[:, lens_z:, :], x_i[:, lens_z:, :],
                    x_v[:, :lens_z, :], x_i[:, :lens_z, :], template_fg_masks)
            if self.early_search_residual is not None and i == self.early_search_residual_early_block:
                early_rgb_for_residual = x_v[:, lens_z:, :]
                early_tir_for_residual = x_i[:, lens_z:, :]
            if diagnostic and i == diagnostic_early_block:
                diagnostic_features['early_rgb'] = x_v[:, lens_z:, :].detach()
                diagnostic_features['early_tir'] = x_i[:, lens_z:, :].detach()
            if diagnostic and i == diagnostic_post_block:
                diagnostic_features['post_block_rgb'] = x_v[:, lens_z:, :].detach()
                diagnostic_features['post_block_tir'] = x_i[:, lens_z:, :].detach()
            if self.post_tbsi_residual is not None and i == self.post_tbsi_residual_block:
                x_v[:, lens_z:, :], x_i[:, lens_z:, :] = self.post_tbsi_residual(
                    x_v[:, lens_z:, :], x_i[:, lens_z:, :])
            if self.tbsi_loc is not None and i in self.tbsi_loc:
                if diagnostic and i == diagnostic_late_block:
                    diagnostic_features['late_input_rgb'] = x_v[:, lens_z:, :].detach()
                    diagnostic_features['late_input_tir'] = x_i[:, lens_z:, :].detach()
                if (self.early_search_residual is not None and
                        i == self.early_search_residual_late_block):
                    if early_rgb_for_residual is None or early_tir_for_residual is None:
                        raise RuntimeError('early residual late block precedes early capture')
                    late_rgb = x_v[:, lens_z:, :]
                    late_tir = x_i[:, lens_z:, :]
                    x_v[:, lens_z:, :], x_i[:, lens_z:, :] = self.early_search_residual(
                        late_rgb, late_tir, early_rgb_for_residual, early_tir_for_residual)
                x_v, x_i = self.tbsi_layers[tbsi_index](
                    x_v, x_i, lens_z, temporal_tokens=temporal_tokens,
                    template_fg_masks=template_fg_masks,
                    early_context=(early_template_rgb_for_gate,
                                   early_template_tir_for_gate,
                                   early_search_rgb_for_gate,
                                   early_search_tir_for_gate),
                    early_utility_logits=early_utility_logits)
                der_scores = getattr(
                    self.tbsi_layers[tbsi_index].v2_template_fusion,
                    'last_der_scores', None)
                gate_stats = getattr(
                    self.tbsi_layers[tbsi_index].v2_template_fusion,
                    'last_early_gate_stats', None)
                if gate_stats is not None:
                    early_gate_stats.append(gate_stats)
                if der_scores is not None:
                    der_score_maps.append(der_scores)
                if diagnostic and tbsi_index == diagnostic_tbsi_index:
                    diagnostic_features['post_tbsi_rgb'] = x_v[:, lens_z:, :].detach()
                    diagnostic_features['post_tbsi_tir'] = x_i[:, lens_z:, :].detach()
                tbsi_index += 1

        x_v = recover_tokens(x_v, lens_z, lens_x, mode=self.cat_mode)
        x_i = recover_tokens(x_i, lens_z, lens_x, mode=self.cat_mode)
        x = torch.cat([x_v, x_i], dim=1)
        
        aux_dict = {"attn": None}
        if early_utility_logits is not None:
            aux_dict['early_utility_logits'] = early_utility_logits
        # Optional sparse cross-modal fusion receives the earliest available
        # paired tokens without changing the default M1 output contract.
        if os.environ.get('TBSI_SPARSE_CROSS_MODE', 'off').lower() != 'off':
            if early_search_rgb_for_gate is None or early_search_tir_for_gate is None:
                raise RuntimeError('sparse cross route requires block02 search features')
            aux_dict['early_sparse_context'] = (
                early_template_rgb_for_gate.detach(),
                early_template_tir_for_gate.detach(),
                early_search_rgb_for_gate.detach(),
                early_search_tir_for_gate.detach())
            self._last_early_sparse_context = aux_dict['early_sparse_context']
        else:
            self._last_early_sparse_context = None
        if der_score_maps:
            aux_dict['der_score_maps'] = der_score_maps
        if early_gate_stats:
            aux_dict['early_gate_stats'] = early_gate_stats
        if diagnostic:
            diagnostic_features['late_output_rgb'] = x_v[:, lens_z:, :].detach()
            diagnostic_features['late_output_tir'] = x_i[:, lens_z:, :].detach()
            aux_dict['diagnostic_features'] = diagnostic_features
        return self.norm(x), aux_dict

    def get_smsa_stats(self):
        values = [layer.get_smsa_stats() for layer in self.tbsi_layers]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return {key: sum(value[key] for value in values) / len(values) for key in values[0]}

    def get_tsms_attention(self):
        """Return training-graph SMSA attentions with their TBSI locations.

        The actor consumes this immediately to build the TSMS coverage loss.
        It is deliberately separate from the detached inference diagnostics.
        """
        values = []
        for index, layer in enumerate(self.tbsi_layers):
            location = self.tbsi_loc[index] if index < len(self.tbsi_loc) else index
            for name, block in (('RGB2F', layer.ca_s2t_i2f),
                                ('TIR2F', layer.ca_s2t_v2f)):
                if block is None:
                    continue
                attention = getattr(block.attn_reshape,
                                     'last_smsa_attention_for_loss', None)
                if attention is not None:
                    values.append((location, name, attention))
        return values

    def get_smsa_reference_attention(self):
        """Return detached SMSA maps from a frozen reference backbone."""
        values = []
        for index, layer in enumerate(self.tbsi_layers):
            location = self.tbsi_loc[index] if index < len(self.tbsi_loc) else index
            for name, block in (('RGB2F', layer.ca_s2t_i2f),
                                ('TIR2F', layer.ca_s2t_v2f)):
                if block is None:
                    continue
                attention = getattr(block.attn_reshape, 'last_smsa_attention', None)
                if attention is not None:
                    values.append((location, name, attention.detach()))
        return values

    def clear_tsms_attention(self):
        """Release the previous training graph after backward has started."""
        for layer in self.tbsi_layers:
            for block in (layer.ca_s2t_i2f, layer.ca_s2t_v2f):
                if block is None:
                    continue
                block.attn_reshape.last_smsa_attention_for_loss = None

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            # PVL 参数在自己的初始化范围内，不消耗 M1 主干的 RNG 序列。
            def initialize_without_pvl(module):
                for child in module.children():
                    if isinstance(child, PVLAdapter):
                        continue
                    initialize_without_pvl(child)
                _init_vit_weights(module)

            initialize_without_pvl(self)
        for module in self.modules():
            if module.__class__.__name__ == 'MessageAdapter':
                nn.init.zeros_(module.fc2.weight)
                nn.init.zeros_(module.fc2.bias)
            if module.__class__.__name__ == 'EarlyTargetMarginGate':
                nn.init.zeros_(module.net[-1].weight)
                nn.init.zeros_(module.net[-1].bias)
            if isinstance(module, EarlyUtilityRouter):
                nn.init.zeros_(module.predictor[-1].weight)
                nn.init.zeros_(module.predictor[-1].bias)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformerTBSI, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = VisionTransformerTBSI(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_tbsi(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def vit_small_patch16_224_tbsi(pretrained=False, **kwargs):
    """
    ViT-Small model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_tiny_patch16_224_tbsi(pretrained=False, **kwargs):
    """
    ViT-Tiny model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
