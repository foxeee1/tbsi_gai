#!/usr/bin/env python3
"""P0 contract checks for the fixed post-TBSI cross-modal residual.

This is an implementation gate only. It does not train or evaluate tracking.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.config.tbsi_track.config import cfg as base_cfg, update_config_from_file
from lib.models.layers.post_tbsi_cross_modal_residual import PostTBSICrossModalResidual
from lib.models.tbsi_track.tbsi_track import build_tbsi_track


def tensor_digest(value):
    if not torch.is_tensor(value):
        return repr(value)
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(module, exclude='post_tbsi_residual'):
    items = []
    for name, value in module.state_dict().items():
        if exclude in name:
            continue
        items.append((name, tensor_digest(value)))
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def finite_grads(module):
    return all(torch.isfinite(p.grad).all().item()
               for p in module.parameters() if p.grad is not None)


def module_checks(device):
    torch.manual_seed(42)
    b, n, c = 2, 256, 768
    rgb = torch.randn(b, n, c, device=device)
    tir = torch.randn(b, n, c, device=device)
    module = PostTBSICrossModalResidual(dim=c, mode='spatial', hidden_dim=192,
                                        num_heads=4).to(device).train()
    with torch.no_grad():
        out_rgb, out_tir = module(rgb, tir)
    zero_diff = max((out_rgb - rgb).abs().max().item(),
                    (out_tir - tir).abs().max().item())

    module.zero_grad(set_to_none=True)
    out_rgb, out_tir = module(rgb, tir)
    loss = out_rgb.square().mean() + out_tir.square().mean()
    loss.backward()
    gate_grads_first = {
        'rgb': float(module.gamma_rgb.grad.abs().max().item()),
        'tir': float(module.gamma_tir.grad.abs().max().item()),
    }
    qkv_first = max((p.grad.abs().max().item() for name, p in module.named_parameters()
                     if any(k in name for k in ('.q.', '.k.', '.v.')) and p.grad is not None),
                    default=0.0)
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    opt.step()
    module.zero_grad(set_to_none=True)
    out_rgb, out_tir = module(rgb, tir)
    (out_rgb.square().mean() + out_tir.square().mean()).backward()
    qkv_after = max((p.grad.abs().max().item() for name, p in module.named_parameters()
                     if any(k in name for k in ('.q.', '.k.', '.v.')) and p.grad is not None),
                    default=0.0)
    result = {
        'input_shape': [b, n, c],
        'zero_gate_max_abs_diff': zero_diff,
        'zero_gate_identity': zero_diff == 0.0,
        'output_finite': bool(torch.isfinite(out_rgb).all() and torch.isfinite(out_tir).all()),
        'gate_grad_first_max': gate_grads_first,
        'qkv_grad_first_max': qkv_first,
        'qkv_grad_after_gate_update_max': qkv_after,
        'all_grads_finite': finite_grads(module),
        'trainable_parameters': sum(p.numel() for p in module.parameters()),
    }
    result['passed'] = bool(result['zero_gate_identity'] and result['output_finite']
                            and result['all_grads_finite']
                            and min(gate_grads_first.values()) > 0.0
                            and qkv_first == 0.0 and qkv_after > 0.0)
    return result


def tracker_checks(config_path, device):
    cfg = copy.deepcopy(base_cfg)
    update_config_from_file(config_path, cfg)
    cfg.CUDA = device.type == 'cuda'
    cfg.LOCAL_RANK = -1
    # Build the enabled model and a structurally identical force-zero model.
    enabled = build_tbsi_track(cfg, training=False).to(device).eval()
    zero_cfg = copy.deepcopy(cfg)
    zero_cfg.MODEL.BACKBONE.POST_TBSI_RESIDUAL.FORCE_ZERO = True
    zero = build_tbsi_track(zero_cfg, training=False).to(device).eval()
    if enabled.backbone.post_tbsi_residual is None:
        raise RuntimeError('post-TBSI residual was not constructed')
    with torch.no_grad():
        enabled.backbone.post_tbsi_residual.gamma_rgb.fill_(0.1)
        enabled.backbone.post_tbsi_residual.gamma_tir.fill_(0.1)
    b = 1
    torch.manual_seed(43)
    z = [torch.randn(b, 3, 128, 128, device=device), torch.randn(b, 3, 128, 128, device=device)]
    x = [torch.randn(b, 3, 256, 256, device=device), torch.randn(b, 3, 256, 256, device=device)]
    before = state_digest(zero.backbone)
    with torch.no_grad():
        out_enabled, _ = enabled.backbone(z, x)
        out_zero, _ = zero.backbone(z, x)
    after = state_digest(zero.backbone)
    diff = (out_enabled - out_zero).abs().max().item()
    result = {
        'device': str(device),
        'enabled_output_shape': list(out_enabled.shape),
        'zero_output_shape': list(out_zero.shape),
        'force_zero_vs_enabled_max_abs_diff': diff,
        'force_zero_state_digest_before': before,
        'force_zero_state_digest_after': after,
        'non_adapter_state_unchanged': before == after,
        'finite': bool(torch.isfinite(out_enabled).all() and torch.isfinite(out_zero).all()),
    }
    # C0 is represented by the force-zero path; exact equality is checked separately
    # against a disabled model below to avoid treating enabled random weights as C0.
    c0_cfg = copy.deepcopy(cfg)
    c0_cfg.MODEL.BACKBONE.POST_TBSI_RESIDUAL.ENABLE = False
    c0 = build_tbsi_track(c0_cfg, training=False).to(device).eval()
    with torch.no_grad():
        out_c0, _ = c0.backbone(z, x)
    result['gamma0_vs_c0_max_abs_diff'] = (out_zero - out_c0).abs().max().item()
    result['gamma0_parity'] = result['gamma0_vs_c0_max_abs_diff'] == 0.0
    result['passed'] = bool(result['finite'] and result['non_adapter_state_unchanged']
                            and result['gamma0_parity'] and diff > 0.0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/tbsi_track/G0_FFM_CSPATIAL_seed42.yaml')
    parser.add_argument('--output', default='output/experiments/g0_ffm_20261005/p0_result.json')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = {'config': args.config, 'module': module_checks(device)}
    if result['module']['passed']:
        result['tracker'] = tracker_checks(args.config, device)
    else:
        result['tracker'] = {'passed': False, 'skipped': True}
    result['passed'] = bool(result['module']['passed'] and result['tracker']['passed'])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
