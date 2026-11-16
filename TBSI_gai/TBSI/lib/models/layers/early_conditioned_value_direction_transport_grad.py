"""Receiver-conditioned Value repair with an isolated transport gradient.

The aligned operator's forward values, parameters, and checkpoint keys are
preserved. Only the extra condition route through the attention argument is
detached. Attention in the host's original message readout remains trainable.
This is a single-stage joint-learning comparison, not a frozen backbone.
"""
import torch

from lib.models.layers.early_conditioned_value_direction_aligned import (
    AttentionAlignedValueDirection,
)


class TransportGradientIsolatedValueDirection(AttentionAlignedValueDirection):
    def forward(self, values, source_tokens, receiver_tokens, evidence,
                receiver_attention, mode='direction', enabled=True):
        return super().forward(
            values, source_tokens, receiver_tokens, evidence,
            receiver_attention.detach(), mode=mode, enabled=enabled)


def install_transport_gradient_isolated_value_direction(model, rank=8,
                                                       max_step=0.10):
    targets = [m for m in model.modules()
               if m.__class__.__name__ == 'V2TemplateFusion']
    if not targets or any(not m.early_conditioned_cross_gate for m in targets):
        raise ValueError('Transport gradient isolation requires an Early gate parent')
    if any(hasattr(m, 'value_direction') for m in targets):
        raise ValueError('Value direction is already installed')
    for module in targets:
        dim, heads = module.qkv.in_features, module.num_heads
        if dim % heads or not 0 < rank <= dim // heads:
            raise ValueError('invalid rank/head contract')
    devices = sorted({m.qkv.weight.device.index for m in targets
                      if m.qkv.weight.is_cuda})
    with torch.random.fork_rng(devices=devices):
        for module in targets:
            operator = TransportGradientIsolatedValueDirection(
                module.qkv.in_features, module.num_heads, rank, max_step)
            operator.to(device=module.qkv.weight.device,
                        dtype=module.qkv.weight.dtype)
            operator.train(module.training)
            module.add_module('value_direction', operator)
            module.value_direction_enabled = True
            module.value_direction_aligned_enabled = True
            module.value_direction_mode = 'direction'
    return len(targets)
