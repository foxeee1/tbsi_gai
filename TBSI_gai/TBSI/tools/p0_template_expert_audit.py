#!/usr/bin/env python3
"""P0 audit for the C/G/T template-expert continuation from M1."""
import argparse
import copy
import os
import sys

import torch

from lib.config.tbsi_track import config as config_module
from lib.models.tbsi_track import build_tbsi_track
from lib.models.layers.tbsi_layer import template_box_to_token_weights


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CKPT = os.path.join(ROOT, 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar')
CONFIGS = {
    'C': 'experiments/tbsi_track/TEMPLATE_EXPERT_C_P1_seed42.yaml',
    'G': 'experiments/tbsi_track/TEMPLATE_EXPERT_G_P1_seed42.yaml',
    'T': 'experiments/tbsi_track/TEMPLATE_EXPERT_T_P1_seed42.yaml',
}


def make_cfg(path):
    cfg = copy.deepcopy(config_module.cfg)
    config_module.update_config_from_file(os.path.join(ROOT, path), base_cfg=cfg)
    return cfg


def load_model(label, device):
    cfg = make_cfg(CONFIGS[label])
    model = build_tbsi_track(cfg, training=True).to(device)
    model.eval()
    return cfg, model


def expert_params(model):
    return [(n, p) for n, p in model.named_parameters() if 'template_expert' in n or 'expert_gamma' in n]


def forward(model, device, mask=None):
    torch.manual_seed(3407)
    template = [torch.randn(1, 3, 128, 128, device=device),
                torch.randn(1, 3, 128, 128, device=device)]
    search = [torch.randn(1, 3, 256, 256, device=device),
              torch.randn(1, 3, 256, 256, device=device)]
    with torch.no_grad():
        out = model(template=template, search=search, template_fg_masks=mask)
    return out['pred_boxes'].detach().float().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    if not os.path.exists(CKPT):
        raise FileNotFoundError(CKPT)
    print('P0 device:', device)
    print('P0 checkpoint:', CKPT)

    outputs = {}
    counts = {}
    for label in ('C', 'G', 'T'):
        cfg, model = load_model(label, device)
        params = expert_params(model)
        counts[label] = sum(p.numel() for _, p in params)
        layers = [getattr(layer, 'v2_template_fusion', None) for layer in model.backbone.tbsi_layers]
        active = [i for i, fusion in enumerate(layers) if fusion is not None and hasattr(fusion, 'template_expert')]
        print(label, 'expert_layer_indices=', active, 'expert_params=', counts[label])
        if label == 'C' and counts[label] != 0:
            raise AssertionError('C parameter count is not parent-compatible')
        if label == 'C' and params:
            raise AssertionError('C unexpectedly contains expert parameters')
        if label in ('G', 'T'):
            if active != [2]:
                raise AssertionError('expert must be enabled only at tbsi layer index 2')
            gamma = layers[2].expert_gamma
            if not torch.equal(gamma.detach().cpu(), torch.zeros_like(gamma.detach().cpu())):
                raise AssertionError('{} gamma is not zero initialized'.format(label))
        mask = None
        if label == 'T':
            boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5]], device=device)
            weights = template_box_to_token_weights(boxes)
            mask = (weights, weights)
        outputs[label] = forward(model, device, mask)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if counts['G'] != counts['T'] or counts['G'] <= 0:
        raise AssertionError('G/T expert parameter structures differ')
    for label in ('G', 'T'):
        delta = (outputs[label] - outputs['C']).abs().max().item()
        print('zero-residual parity C vs {} max_abs={:.9g}'.format(label, delta))
        if delta > 1e-6:
            raise AssertionError('{} zero-residual output is not C-parity'.format(label))

    for label in ('G', 'T'):
        cfg, model = load_model(label, device)
        model.train()
        fusion = model.backbone.tbsi_layers[2].v2_template_fusion
        with torch.no_grad():
            fusion.expert_gamma.fill_(0.01)
        mask = None
        if label == 'T':
            boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5]], device=device)
            weights = template_box_to_token_weights(boxes)
            mask = (weights, weights)
        torch.manual_seed(3407)
        template = [torch.randn(1, 3, 128, 128, device=device), torch.randn(1, 3, 128, 128, device=device)]
        search = [torch.randn(1, 3, 256, 256, device=device), torch.randn(1, 3, 256, 256, device=device)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        pred = model(template=template, search=search, template_fg_masks=mask)['pred_boxes']
        loss = pred.float().square().mean()
        loss.backward()
        grads = [p.grad for n, p in expert_params(model) if p.grad is not None]
        finite = all(torch.isfinite(g).all().item() for g in grads)
        nonzero = any(g.abs().sum().item() > 0 for g in grads)
        before = [p.detach().clone() for _, p in expert_params(model)]
        optimizer.step()
        changed = any(not torch.equal(a, b.detach()) for a, (_, b) in zip(before, expert_params(model)))
        print('{} gradient finite={} nonzero={} optimizer_changed={} loss={:.6g}'.format(
            label, finite, nonzero, changed, loss.item()))
        if not (grads and finite and nonzero and changed):
            raise AssertionError('{} gradient/update smoke failed'.format(label))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('P0 PASS')


if __name__ == '__main__':
    main()
