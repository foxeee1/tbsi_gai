#!/usr/bin/env python3
"""Implementation contract checks for the zero-init MSHA role-pair bias."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.config.tbsi_track.config import cfg as default_cfg
from lib.config.tbsi_track.config import update_config_from_file
from lib.models.layers.tbsi_layer import template_box_to_token_weights
from lib.models.tbsi_track.tbsi_track import build_tbsi_track


CHECKPOINT = PROJECT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
EXPECTED_CHECKPOINT_SHA256 = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
EXPECTED_CONFIG_SHA256 = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_cfg(path):
    cfg = copy.deepcopy(default_cfg)
    update_config_from_file(str(path), cfg)
    cfg.MODEL.INIT_CHECKPOINT = str(CHECKPOINT)
    return cfg


def run(model, template, search, masks):
    with torch.no_grad():
        out = model(template=template, search=search, template_fg_masks=masks)
    return out['score_map'].detach(), out['pred_boxes'].detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for M1 role-pair preflight')
    protected_config = PROJECT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'
    if sha256(protected_config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError('protected M1 config hash mismatch')
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError('protected M1 checkpoint hash mismatch')

    candidate_cfg = load_cfg(args.candidate_config)
    if not candidate_cfg.MODEL.BACKBONE.MSHA_ROLE_PAIR_BIAS_ENABLE:
        raise ValueError('candidate must enable MSHA_ROLE_PAIR_BIAS')
    if candidate_cfg.MODEL.BACKBONE.MSHA_ROLE_PAIR_BIAS_LOC != 9:
        raise ValueError('candidate location must remain frozen at TBSI loc9')
    if not candidate_cfg.TRAIN.MSHA_ROLE_PAIR_BIAS_ONLY:
        raise ValueError('candidate must freeze all but role-pair gains')
    parent_cfg = copy.deepcopy(candidate_cfg)
    parent_cfg.MODEL.BACKBONE.MSHA_ROLE_PAIR_BIAS_ENABLE = False
    parent_cfg.TRAIN.MSHA_ROLE_PAIR_BIAS_ONLY = False

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    template = (torch.randn(1, 3, 128, 128, device='cuda'),
                torch.randn(1, 3, 128, 128, device='cuda'))
    search = (torch.randn(1, 3, 256, 256, device='cuda'),
              torch.randn(1, 3, 256, 256, device='cuda'))
    box = torch.tensor([[0.25, 0.25, 0.50, 0.50]], device='cuda')
    weights = template_box_to_token_weights(box, grid_size=8)
    masks = (weights, weights)

    parent = build_tbsi_track(parent_cfg, training=False).cuda().eval()
    candidate = build_tbsi_track(candidate_cfg, training=False).cuda().eval()
    parent_score, parent_box = run(parent, template, search, masks)
    zero_score, zero_box = run(candidate, template, search, masks)
    zero_score_diff = float((parent_score - zero_score).abs().max())
    zero_box_diff = float((parent_box - zero_box).abs().max())
    if zero_score_diff > 5e-5 or zero_box_diff > 5e-5:
        raise RuntimeError('zero role-pair gain does not preserve parent output')

    fusion = next(layer.v2_template_fusion for layer in candidate.backbone.tbsi_layers
                  if getattr(layer, 'v2_template_fusion', None) is not None and
                  getattr(layer.v2_template_fusion, 'role_pair_bias', False))
    with torch.no_grad():
        fusion.role_pair_bias_gain.fill_(0.5)
    active_score, active_box = run(candidate, template, search, masks)
    active_score_diff = float((zero_score - active_score).abs().max())
    active_box_diff = float((zero_box - active_box).abs().max())
    if active_score_diff <= 1e-8:
        raise RuntimeError('nonzero role-pair gain did not change model output')
    if not torch.isfinite(active_score).all() or not torch.isfinite(active_box).all():
        raise RuntimeError('active role-pair output is non-finite')

    train_cfg = load_cfg(args.candidate_config)
    train_model = build_tbsi_track(train_cfg, training=True).cuda()
    train_model.eval()
    trainable = [(n, p) for n, p in train_model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for _, p in trainable)
    if len(trainable) != 1 or trainable_count != 12 or not trainable[0][0].endswith('role_pair_bias_gain'):
        raise RuntimeError('expected exactly 12 trainable role-pair gain values')
    frozen_params = [(n, p, p.detach().clone()) for n, p in train_model.named_parameters()
                     if not p.requires_grad]
    frozen_bn = [(m, m.running_mean.detach().clone(), m.running_var.detach().clone())
                 for m in train_model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    train_model.zero_grad(set_to_none=True)
    out = train_model(template=template, search=search, template_fg_masks=masks)
    loss = out['score_map'].float().square().mean() + out['pred_boxes'].float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError('preflight loss is non-finite')
    loss.backward()
    gain = trainable[0][1]
    grad = gain.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0:
        raise RuntimeError('role-pair gains have no finite nonzero gradient')
    before = gain.detach().clone()
    optimizer = torch.optim.SGD([gain], lr=1e-2)
    optimizer.step()
    update = float((gain.detach() - before).abs().max())
    if update <= 0:
        raise RuntimeError('optimizer did not update role-pair gain')
    changed_frozen = [n for n, p, old in frozen_params if not torch.equal(p.detach(), old)]
    changed_bn = [i for i, (m, mean, var) in enumerate(frozen_bn)
                  if not torch.equal(mean, m.running_mean) or not torch.equal(var, m.running_var)]
    if changed_frozen or changed_bn:
        raise RuntimeError('frozen parent parameter or BatchNorm state changed')
    state = copy.deepcopy(train_model.state_dict())
    result = {
        'status': 'IMPLEMENTATION-PREFLIGHT-PASSED',
        'candidate_config': str(Path(args.candidate_config).resolve()),
        'candidate_config_sha256': sha256(args.candidate_config),
        'protected_config_sha256': sha256(protected_config),
        'protected_checkpoint_sha256': sha256(CHECKPOINT),
        'snapshot_sha256': sha256(PROJECT / 'experiments/snapshots/m1_msha_role_pair_bias_20261023_prechange.tar.gz'),
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'seed': 42, 'feature_location': 9, 'trainable_parameter_count': trainable_count,
        'trainable_parameter_name': trainable[0][0], 'zero_gate_max_score_diff': zero_score_diff,
        'zero_gate_max_box_diff': zero_box_diff, 'active_gate_max_score_diff': active_score_diff,
        'active_gate_max_box_diff': active_box_diff, 'gain_gradient_l1': float(grad.abs().sum()),
        'gain_update_max_abs': update, 'frozen_parameters_changed': changed_frozen,
        'frozen_batchnorm_changed': changed_bn, 'state_dict_roundtrip_keys': len(state),
        'input_contract': {'template': [list(x.shape) for x in template],
                           'search': [list(x.shape) for x in search],
                           'foreground_mask': list(weights.shape),
                           'score_map': list(active_score.shape), 'pred_boxes': list(active_box.shape)},
        'notes': ['Synthetic inputs test implementation only.',
                  'No tracking utility or dataset fit claim is made.'],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
