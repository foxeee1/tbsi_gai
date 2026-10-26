#!/usr/bin/env python3
"""P0 audit for the bounded P2 dispatch-only and soft-prior candidates."""
import copy
import os

import torch

from lib.config.tbsi_track import config as config_module
from lib.models.tbsi_track import build_tbsi_track
from lib.models.layers.masked_template_hmoe import MaskedTemplateHMoE
from lib.models.layers.tbsi_layer import template_box_to_token_weights


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CKPT = os.path.join(ROOT, 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar')
CONFIGS = {
    'disp_only': 'experiments/tbsi_track/TEMPLATE_EXPERT_DISPONLY_P2_seed42.yaml',
    'soft': 'experiments/tbsi_track/TEMPLATE_EXPERT_SOFT_P2_seed42.yaml',
}


def make_cfg(path):
    cfg = copy.deepcopy(config_module.cfg)
    config_module.update_config_from_file(os.path.join(ROOT, path), base_cfg=cfg)
    return cfg


def load_model(path, device):
    cfg = make_cfg(path)
    model = build_tbsi_track(cfg, training=True).to(device)
    model.eval()
    return model


def expert_module(model):
    fusion = model.backbone.tbsi_layers[2].v2_template_fusion
    return fusion, fusion.template_expert


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isfile(CKPT):
        raise FileNotFoundError(CKPT)
    print('P2 P0 device:', device)
    print('P2 P0 checkpoint:', CKPT)

    # Match the training entry's seed-42 initialization contract for both candidates.
    models = {}
    for label, path in CONFIGS.items():
        torch.manual_seed(42)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(42)
        models[label] = load_model(path, device)
    states = []
    for label, model in models.items():
        fusion, expert = expert_module(model)
        assert fusion.expert_mode == ('target_disp_only' if label == 'disp_only' else 'target_soft')
        assert expert.prior_mode == ('disp_only' if label == 'disp_only' else 'soft')
        assert torch.equal(fusion.expert_gamma.detach(), torch.zeros_like(fusion.expert_gamma))
        states.append({k: v.detach().cpu() for k, v in expert.state_dict().items()})
        print(label, 'prior_mode=', expert.prior_mode,
              'params=', sum(p.numel() for p in fusion.parameters()))
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), 'candidate init mismatch: ' + key
    print('candidate expert initialization: identical')

    torch.manual_seed(3407)
    x = torch.randn(2, 128, 768, device=device)
    boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.05, 0.05, 0.15, 0.15]], device=device)
    mask = template_box_to_token_weights(boxes, grid_size=8)
    print('mask fg ratio mean/median/min/max=',
          float(mask.mean()), float(mask.median()), float(mask.min()), float(mask.max()))

    # Same expert weights and input: quantify the mechanical hard-writeback effect.
    raw_norm = None
    hard_norm = None
    for mode in ('hard', 'disp_only', 'soft'):
        module = MaskedTemplateHMoE(dim=768, experts=4, slots=4, hid_dim=4, prior_mode=mode).to(device)
        module.load_state_dict(models['disp_only'].backbone.tbsi_layers[2].v2_template_fusion.template_expert.state_dict())
        module.eval()
        with torch.no_grad():
            out = module(x, torch.cat([mask, mask], dim=1))
        value = float(out.float().norm(dim=-1).mean())
        print('mode=', mode, 'output_norm=', value)
        if mode == 'hard':
            hard_norm = value
        if mode == 'disp_only':
            raw_norm = value
    if raw_norm is not None and hard_norm is not None:
        print('hard_writeback_ratio=', hard_norm / (raw_norm + 1e-12))

    # Zero-residual full-model parity and finite/nonzero candidate gradients.
    template = [torch.randn(1, 3, 128, 128, device=device), torch.randn(1, 3, 128, 128, device=device)]
    search = [torch.randn(1, 3, 256, 256, device=device), torch.randn(1, 3, 256, 256, device=device)]
    weights = template_box_to_token_weights(torch.tensor([[0.25, 0.25, 0.5, 0.5]], device=device))
    outputs = []
    for label, model in models.items():
        with torch.no_grad():
            outputs.append(model(template=template, search=search,
                                 template_fg_masks=(weights, weights))['pred_boxes'])
    assert torch.equal(outputs[0], outputs[1]), 'zero-gamma candidate parity failed'
    print('zero-gamma candidate parity: exact')

    for label, model in models.items():
        model.train()
        fusion, _ = expert_module(model)
        with torch.no_grad():
            fusion.expert_gamma.fill_(0.01)
        pred = model(template=template, search=search,
                     template_fg_masks=(weights, weights))['pred_boxes']
        loss = pred.float().square().mean()
        loss.backward()
        grads = [p.grad for p in fusion.template_expert.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads)
        assert any(g.abs().sum().item() > 0 for g in grads)
        print(label, 'gradient finite/nonzero: PASS', 'loss=', float(loss))
    print('P2 P0 PASS')


if __name__ == '__main__':
    main()
