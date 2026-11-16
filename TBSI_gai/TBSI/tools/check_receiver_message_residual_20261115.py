#!/usr/bin/env python3
"""Implementation-only audit for the receiver-conditioned MR message residual."""

import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.models.layers.receiver_conditioned_message_residual import (
    ReceiverConditionedMessageResidual,
)
from lib.models.layers.tbsi_layer import V2TemplateFusion


OUT = ROOT / "output/experiments/MSHA-MR-RECEIVER-MESSAGE-RESIDUAL-AUDIT-S42-20261115"


def main():
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {}

    # Direct tensor contract and exact identity at initialization.
    module = ReceiverConditionedMessageResidual(dim=32, rank=8, max_delta=0.1).to(device)
    receiver = torch.randn(2, 16, 32, device=device)
    message = torch.randn(2, 4, 16, 8, device=device, requires_grad=True)
    native = module(receiver, message)
    result["direct_identity_max_abs"] = float((native - message).abs().max())
    assert result["direct_identity_max_abs"] == 0.0
    assert native.shape == message.shape

    # The branch must be trainable even though it starts as identity.
    loss = native.square().mean()
    loss.backward()
    readout_grad = module.readout.weight.grad
    result["readout_grad_finite"] = bool(torch.isfinite(readout_grad).all())
    result["readout_grad_l1"] = float(readout_grad.abs().sum())
    assert result["readout_grad_finite"] and result["readout_grad_l1"] > 0.0

    # A bounded correction cannot exceed max_delta times the message RMS per channel.
    with torch.no_grad():
        module.readout.weight.normal_(0, 2)
        module.readout.bias.normal_(0, 2)
        corrected = module(receiver, message.detach())
        rms = message.detach().float().transpose(1, 2).reshape(2, 16, 32).square().mean(-1, keepdim=True).sqrt()
        delta = (corrected - message.detach()).float().transpose(1, 2).reshape(2, 16, 32)
        bound_ratio = float((delta.abs() / rms.clamp_min(1e-6)).max())
    result["max_relative_channel_delta"] = bound_ratio
    assert bound_ratio <= 0.1000002

    # Save/load must preserve the learned branch exactly.
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    buffer.seek(0)
    restored = ReceiverConditionedMessageResidual(dim=32, rank=8, max_delta=0.1).to(device)
    restored.load_state_dict(torch.load(buffer, map_location=device), strict=True)
    with torch.no_grad():
        repeat = restored(receiver, message.detach())
    result["save_load_max_abs"] = float((repeat - corrected).abs().max())
    assert result["save_load_max_abs"] == 0.0

    # Disabled and enabled V2 paths share all public state. The enabled path
    # has new zero-readout parameters, so its output must remain bitwise equal.
    kwargs = dict(
        dim=768, num_heads=12, qkv_bias=True,
        early_conditioned_cross_gate=True,
        early_gate_search_context_mode="cross",
        mass_redist_enable=True, mass_redist_mode="pair",
    )
    disabled = V2TemplateFusion(**kwargs, receiver_message_residual=False).to(device).eval()
    enabled = V2TemplateFusion(
        **kwargs, receiver_message_residual=True,
        receiver_message_residual_rank=16,
        receiver_message_residual_max_delta=0.1,
    ).to(device).eval()
    common = disabled.state_dict()
    missing, unexpected = enabled.load_state_dict(common, strict=False)
    result["enabled_new_parameter_keys"] = len(missing)
    result["unexpected_parameter_keys"] = len(unexpected)
    assert unexpected == [] and len(missing) > 0
    inputs = [torch.randn(2, 64, 768, device=device) for _ in range(2)]
    early = tuple(torch.randn(2, n, 768, device=device) for n in (64, 64, 256, 256))
    boxes = torch.tensor([[[0.25, 0.25, 0.5, 0.5]] * 2] * 2, device=device)
    with torch.no_grad():
        out_disabled = disabled(*inputs, early_context=early, template_boxes=boxes)
        out_enabled = enabled(*inputs, early_context=early, template_boxes=boxes)
    result["v2_disabled_enabled_max_abs"] = max(
        float((a - b).abs().max()) for a, b in zip(out_disabled, out_enabled))
    assert result["v2_disabled_enabled_max_abs"] == 0.0
    result["v2_residual_module_count"] = sum(
        isinstance(m, ReceiverConditionedMessageResidual) for m in enabled.modules())
    assert result["v2_residual_module_count"] == 2
    result["device"] = str(device)
    result["optimizer_steps"] = 0
    result["scope"] = "Implementation contracts only; no efficacy claim"
    OUT.joinpath("contracts").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("contracts/implementation_checks.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
