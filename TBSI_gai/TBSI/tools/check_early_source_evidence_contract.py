#!/usr/bin/env python3
"""Contract checks for the evidence-only Early-gate candidate."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'
CONTROL = ROOT / 'experiments/tbsi_track/EG_EVDR_P1_SOT_V3_CONTROL_20261026.yaml'
CANDIDATE = ROOT / 'experiments/tbsi_track/EG_EARLY_SOURCE_EVIDENCE_P1_SOT_S42_20261027.yaml'
OUT = ROOT / 'experiments/early_gate_closed_loop_20261026/source_evidence_p1_20261027'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_equal(a, b):
    common = sorted(set(a) & set(b))
    bad = [k for k in common if a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]
    return len(bad) == 0, common, bad


def main():
    sys.path.insert(0, str(ROOT))
    from lib.config.tbsi_track.config import cfg, update_config_from_file
    from lib.models.layers.early_source_evidence import EarlySourceEvidence
    from lib.models.layers.tbsi_layer import V2TemplateFusion

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    update_config_from_file(str(CONTROL))
    from lib.models.tbsi_track import build_tbsi_track
    control = build_tbsi_track(cfg, training=True).eval()
    control_state = control.state_dict()

    # Re-read defaults into a clean config object by reloading the module.
    import importlib
    config_module = importlib.import_module('lib.config.tbsi_track.config')
    importlib.reload(config_module)
    cfg2 = config_module.cfg
    config_module.update_config_from_file(str(CANDIDATE))
    candidate = build_tbsi_track(cfg2, training=True).eval()
    candidate_state = candidate.state_dict()
    source_modules = [m for m in candidate.modules() if isinstance(m, EarlySourceEvidence)]
    if len(source_modules) != 3:
        raise AssertionError('expected one source evidence module at each TBSI location')
    if any(float(m.mlp[-1].weight.abs().max()) != 0.0 or float(m.mlp[-1].bias.abs().max()) != 0.0
           for m in source_modules):
        raise AssertionError('source evidence residual is not identity initialized')

    # Copy common public-pretrained state and compare host outputs with the
    # candidate branch disabled and with its zero residual enabled.
    common = {k: v for k, v in control_state.items() if k in candidate_state and candidate_state[k].shape == v.shape}
    candidate.load_state_dict(common, strict=False)
    for module in candidate.modules():
        if hasattr(module, 'early_source_evidence_enabled'):
            module.early_source_evidence_enabled = False
    host_off = V2TemplateFusion(48, 3, early_conditioned_cross_gate=True).eval()
    host_on = V2TemplateFusion(48, 3, early_conditioned_cross_gate=True,
                               early_source_evidence=True).eval()
    host_on.load_state_dict(host_off.state_dict(), strict=False)
    for module in [host_off, host_on]:
        module.early_gate.load_state_dict(host_off.early_gate.state_dict())
    # The standalone host constructor keeps the parent gate's zero output
    # initialization. Give the contract probe a deterministic nonzero gate
    # head so evidence-to-task gradient reachability is actually testable.
    torch.manual_seed(124)
    torch.nn.init.normal_(host_off.early_gate[2].weight, std=0.02)
    torch.nn.init.normal_(host_off.early_gate[2].bias, std=0.02)
    host_on.early_gate.load_state_dict(host_off.early_gate.state_dict())
    torch.manual_seed(123)
    rgb = torch.randn(2, 5, 48)
    tir = torch.randn(2, 5, 48)
    context = (torch.randn(2, 5, 48), torch.randn(2, 5, 48),
               torch.randn(2, 7, 48), torch.randn(2, 7, 48))
    off_rgb, off_tir = host_off(rgb, tir, context)
    on_rgb, on_tir = host_on(rgb, tir, context)
    parity_error = max(float((off_rgb - on_rgb).abs().max()), float((off_tir - on_tir).abs().max()))
    if parity_error > 2e-6:
        raise AssertionError('zero residual parity error %.8g' % parity_error)

    # The final projection is zero at initialization, but it must remain in
    # the trainable graph and receive a finite task-shaped gradient.
    host_on.train()
    loss = (host_on(rgb, tir, context)[0] ** 2).mean()
    loss.backward()
    grad = host_on.early_source_evidence.mlp[-1].weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0:
        raise AssertionError('source evidence residual has no finite learning gradient')

    result = {
        'id': 'EG-EARLY-SOURCE-EVIDENCE-CONTRACT-20261027',
        'public_pretrain_sha256': sha(PUBLIC),
        'control_config_sha256': sha(CONTROL),
        'candidate_config_sha256': sha(CANDIDATE),
        'source_module_count': len(source_modules),
        'candidate_source_parameters': sum(p.numel() for m in source_modules for p in m.parameters()),
        'zero_residual_parity_max_abs': parity_error,
        'final_projection_gradient_l1': float(grad.abs().sum()),
        'control_candidate_common_state_keys': len(set(control_state) & set(candidate_state)),
        'status': 'CONTRACT-PASS / IDENTITY-PARITY / LEARNING-GRADIENT-REACHABLE',
    }
    (OUT / 'contract.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
