#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.config.tbsi_track.config import cfg as default_cfg, update_config_from_file
from lib.models.layers.local_cross_modal_correspondence import LocalCrossModalCorrespondence
from lib.models.tbsi_track.tbsi_track import build_tbsi_track


def load_model(name):
    cfg = copy.deepcopy(default_cfg)
    update_config_from_file(str(ROOT / 'experiments/tbsi_track' / (name + '.yaml')), cfg)
    return build_tbsi_track(cfg, training=False).eval().cuda()


def main():
    torch.manual_seed(42)
    device = torch.device('cuda')
    module = LocalCrossModalCorrespondence(dim=32, hidden_dim=8,
                                           relevance_dim=8, max_delta=0.05).to(device)
    rgb = torch.randn(2, 256, 32, device=device, requires_grad=True)
    tir = torch.randn(2, 256, 32, device=device, requires_grad=True)
    rgb_template = torch.randn(2, 64, 32, device=device)
    tir_template = torch.randn(2, 64, 32, device=device)
    masks = (torch.ones(2, 64, device=device), torch.ones(2, 64, device=device))

    rgb0, tir0 = module(rgb, tir, rgb_template, tir_template, masks)
    parity = max(float((rgb0 - rgb).abs().max()), float((tir0 - tir).abs().max()))
    (rgb0.square().mean() + tir0.square().mean()).backward()
    first_grad = float(module.rgb_from_tir.output.weight.grad.abs().sum() +
                       module.tir_from_rgb.output.weight.grad.abs().sum())
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    rgb1, tir1 = module(rgb, tir, rgb_template, tir_template, masks)
    (rgb1.square().mean() + tir1.square().mean()).backward()
    upstream_grad = sum(float(parameter.grad.abs().sum()) for name, parameter in module.named_parameters()
                        if any(key in name for key in ('query.', 'key.', 'value.', 'relevance.'))
                        and parameter.grad is not None)

    control = load_model('M1_LOCAL_CROSS_MODAL_CORRESPONDENCE_CONTROL_QUICK3_20261024')
    candidate = load_model('M1_LOCAL_CROSS_MODAL_CORRESPONDENCE_QUICK3_20261024')
    with torch.no_grad():
        template = [torch.randn(1, 3, 128, 128, device=device),
                    torch.randn(1, 3, 128, 128, device=device)]
        search = [torch.randn(1, 3, 256, 256, device=device),
                  torch.randn(1, 3, 256, 256, device=device)]
        fg = torch.ones(1, 64, device=device)
        outputs_control = control(template=template, search=search,
                                  template_fg_masks=(fg, fg))
        outputs_candidate = candidate(template=template, search=search,
                                      template_fg_masks=(fg, fg))
        model_parity = max(float((outputs_control[key] - outputs_candidate[key]).abs().max())
                           for key in ('score_map', 'pred_boxes', 'size_map', 'offset_map'))

    report = {
        'device': torch.cuda.get_device_name(0),
        'module_identity_max_abs_diff': parity,
        'first_step_output_projection_grad_l1': first_grad,
        'multi_step_upstream_grad_l1': upstream_grad,
        'full_model_zero_delta_max_abs_diff': model_parity,
        'status': 'PASS' if parity == 0.0 and first_grad > 0 and upstream_grad > 0 and model_parity == 0.0 else 'FAIL',
    }
    out = ROOT / 'experiments/snapshots/m1_local_cross_modal_correspondence_20261024/preflight.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
