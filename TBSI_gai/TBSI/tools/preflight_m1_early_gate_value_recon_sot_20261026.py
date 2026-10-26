#!/usr/bin/env python3
"""SOT-origin contract checks for the learnable Early-gate Value adapter."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / 'experiments/tbsi_track/M1_EARLY_GATE_VALUE_RECON_SOT_CONTRACT_20261026.yaml'
SOT_PATH = PROJECT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'

import sys
sys.path.insert(0, str(PROJECT))
from lib.config.tbsi_track.config import cfg as default_cfg
from lib.config.tbsi_track.config import update_config_from_file
from lib.models.layers.tbsi_layer import template_box_to_token_weights
from lib.models.tbsi_track.tbsi_track import build_tbsi_track


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def load_cfg(reconstruction=True, train_only=False):
    cfg = copy.deepcopy(default_cfg)
    update_config_from_file(str(CONFIG_PATH), cfg)
    cfg.MODEL.BACKBONE.EARLY_TEMPLATE_VALUE_RECONSTRUCTION = reconstruction
    cfg.TRAIN.EARLY_TEMPLATE_VALUE_RECONSTRUCTION_ONLY = train_only
    cfg.MODEL.INIT_CHECKPOINT = ''
    return cfg


def inputs(batch=2, device='cuda'):
    # The RGBT backbone consumes two modality tensors, not a concatenated
    # six-channel tensor.  The tracker wrapper concatenates them internally.
    template = (torch.randn(batch, 3, 128, 128, device=device),
                torch.randn(batch, 3, 128, 128, device=device))
    search = (torch.randn(batch, 3, 256, 256, device=device),
              torch.randn(batch, 3, 256, 256, device=device))
    box = torch.tensor([[0.25, 0.25, 0.50, 0.50]], device=device).repeat(batch, 1)
    weights = template_box_to_token_weights(box, grid_size=8)
    return template, search, (weights, weights)


def forward(model, data):
    template, search, masks = data
    return model(template=template, search=search, template_fg_masks=masks)


def adapters(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if '.early_value_reconstruction.' in name]


def make_teacher(model):
    generator = torch.Generator(device='cuda')
    generator.manual_seed(20261026)
    with torch.no_grad():
        for layer in model.backbone.tbsi_layers:
            fusion = getattr(layer, 'v2_template_fusion', None)
            module = getattr(fusion, 'early_value_reconstruction', None) if fusion else None
            if module is None:
                continue
            for parameter in (module.up_rgb.weight, module.up_tir.weight):
                parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                             device=parameter.device,
                                             dtype=parameter.dtype) * 0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this preflight')
    seed()
    device = 'cuda'
    report = {
        'config': str(CONFIG_PATH),
        'config_sha256': sha256(CONFIG_PATH),
        'sot_checkpoint': str(SOT_PATH),
        'sot_sha256': sha256(SOT_PATH),
        'seed': 42,
        'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
    }

    # Parent and candidate both originate directly from SOT. Re-seeding makes
    # the zero-initialized Early gate identical before the extra adapter.
    candidate_cfg = load_cfg(reconstruction=True, train_only=False)
    seed()
    candidate = build_tbsi_track(candidate_cfg, training=True).to(device).eval()
    parent_cfg = load_cfg(reconstruction=False, train_only=False)
    seed()
    parent = build_tbsi_track(parent_cfg, training=True).to(device).eval()
    # The SOT checkpoint predates these TBSI fusion parameters, so they are
    # randomly initialized. Copy the candidate's common state into the parent
    # to make the contract a true single-variable comparison.
    parent.load_state_dict(candidate.state_dict(), strict=False)
    common_state_diffs = []
    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    for name in sorted(set(parent_state).intersection(candidate_state)):
        if parent_state[name].shape == candidate_state[name].shape:
            delta = float((parent_state[name].float() - candidate_state[name].float()).abs().max())
            if delta > 1e-7:
                common_state_diffs.append((name, delta))
    report['common_state_diff_count'] = len(common_state_diffs)
    report['common_state_diff_max'] = max((value for _, value in common_state_diffs), default=0.0)
    report['common_state_diff_examples'] = common_state_diffs[:10]
    data = inputs(batch=2, device=device)
    with torch.no_grad():
        parent_out = forward(parent, data)
        candidate_out = forward(candidate, data)
    score_diff = float((parent_out['score_map'] - candidate_out['score_map']).abs().max())
    box_diff = float((parent_out['pred_boxes'] - candidate_out['pred_boxes']).abs().max())
    report['zero_init_max_score_diff'] = score_diff
    report['zero_init_max_box_diff'] = box_diff
    # The zero residual still traverses a transpose/reshape/add-zero path;
    # accept bounded FP32 rounding while requiring common parameters identical.
    if score_diff > 1e-4 or box_diff > 1e-4:
        print(json.dumps({
            'zero_init_max_score_diff': score_diff,
            'zero_init_max_box_diff': box_diff,
            'common_state_diff_count': report['common_state_diff_count'],
            'common_state_diff_max': report['common_state_diff_max'],
            'common_state_diff_examples': report['common_state_diff_examples'],
            'parent_score_range': [float(parent_out['score_map'].min()),
                                   float(parent_out['score_map'].max())],
            'candidate_score_range': [float(candidate_out['score_map'].min()),
                                      float(candidate_out['score_map'].max())],
            'parent_box_range': [float(parent_out['pred_boxes'].min()),
                                 float(parent_out['pred_boxes'].max())],
            'candidate_box_range': [float(candidate_out['pred_boxes'].min()),
                                    float(candidate_out['pred_boxes'].max())],
        }, indent=2))
        raise RuntimeError('zero-initialized reconstruction is not parent identity')
    del parent, parent_out, candidate_out
    torch.cuda.empty_cache()

    # Compare a fresh zero and nonzero candidate with the same SOT-origin state.
    del candidate
    torch.cuda.empty_cache()
    seed()
    zero_candidate = build_tbsi_track(load_cfg(reconstruction=True, train_only=False), training=True).to(device).eval()
    with torch.no_grad():
        zero_out = forward(zero_candidate, data)
    seed()
    active_candidate = build_tbsi_track(load_cfg(reconstruction=True, train_only=False), training=True).to(device).eval()
    active_candidate.load_state_dict(zero_candidate.state_dict(), strict=False)
    make_teacher(active_candidate)
    with torch.no_grad():
        active_out = forward(active_candidate, data)
    active_score_diff = float((zero_out['score_map'] - active_out['score_map']).abs().max())
    active_box_diff = float((zero_out['pred_boxes'] - active_out['pred_boxes']).abs().max())
    report['active_probe_max_score_diff'] = active_score_diff
    report['active_probe_max_box_diff'] = active_box_diff
    if active_score_diff <= 1e-8 or not torch.isfinite(active_out['score_map']).all():
        raise RuntimeError('nonzero reconstruction probe did not affect finite output')
    del zero_candidate, zero_out, active_candidate, active_out
    torch.cuda.empty_cache()

    # Train only the new adapter. Verify gradients, one optimizer update, and
    # frozen parent parameters/BN state. The first backward may leave down
    # gradients at zero because the up projection starts at zero.
    seed()
    train_model = build_tbsi_track(load_cfg(reconstruction=True, train_only=True), training=True).to(device)
    train_model.eval()
    trainable = adapters(train_model)
    if not trainable:
        raise RuntimeError('no reconstruction parameters were selected')
    non_adapter = [(n, p, p.detach().clone()) for n, p in train_model.named_parameters()
                   if p.requires_grad is False]
    frozen_bn = [(m, m.running_mean.detach().clone(), m.running_var.detach().clone())
                 for m in train_model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = {n: p.detach().clone() for n, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-3)
    data = inputs(batch=2, device=device)
    out = forward(train_model, data)
    loss = out['score_map'].float().square().mean() + out['pred_boxes'].float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError('single-batch loss is non-finite')
    loss.backward()
    up_grads = [float(p.grad.abs().sum()) for n, p in trainable if '.up_' in n and p.grad is not None]
    if not up_grads or max(up_grads) <= 0 or not all(np.isfinite(up_grads)):
        raise RuntimeError('up projection did not receive a finite nonzero gradient')
    optimizer.step()
    updates = {n: float((p.detach() - before[n]).abs().max()) for n, p in trainable}
    if max(updates.values()) <= 0:
        raise RuntimeError('optimizer did not update reconstruction parameters')
    changed_frozen = [n for n, p, old in non_adapter if not torch.equal(p.detach(), old)]
    changed_bn = [i for i, (m, mean, var) in enumerate(frozen_bn)
                  if not torch.equal(mean, m.running_mean) or not torch.equal(var, m.running_var)]
    if changed_frozen or changed_bn:
        raise RuntimeError('frozen parent or BN state changed')
    report['trainable_parameter_count'] = sum(p.numel() for _, p in trainable)
    report['trainable_parameter_tensors'] = len(trainable)
    report['single_batch_loss'] = float(loss.detach())
    report['up_gradient_l1_max'] = max(up_grads)
    report['max_parameter_update'] = max(updates.values())
    report['frozen_parameter_changes'] = changed_frozen
    report['frozen_batchnorm_changes'] = changed_bn

    # 16-sample repeated-input fit: a nonzero teacher target is generated once;
    # eight B=2 updates count as 16 sample presentations and test learnability.
    seed()
    teacher = build_tbsi_track(load_cfg(reconstruction=True, train_only=False), training=True).to(device).eval()
    make_teacher(teacher)
    fit_data = inputs(batch=2, device=device)
    with torch.no_grad():
        target = forward(teacher, fit_data)
        target_score = target['score_map'].detach()
        target_box = target['pred_boxes'].detach()
    fit_model = build_tbsi_track(load_cfg(reconstruction=True, train_only=True), training=True).to(device)
    teacher_state = teacher.state_dict()
    fit_state = fit_model.state_dict()
    for name in fit_state:
        if '.early_value_reconstruction.' not in name and name in teacher_state:
            fit_state[name].copy_(teacher_state[name])
    fit_model.load_state_dict(fit_state, strict=False)
    fit_model.eval()
    fit_params = [p for _, p in adapters(fit_model)]
    fit_optimizer = torch.optim.AdamW(fit_params, lr=2e-3)
    def fit_loss():
        pred = forward(fit_model, fit_data)
        return ((pred['score_map'] - target_score).float().square().mean() +
                (pred['pred_boxes'] - target_box).float().square().mean())
    with torch.no_grad():
        initial_fit_loss = float(fit_loss())
    fit_losses = []
    for _ in range(8):
        fit_optimizer.zero_grad(set_to_none=True)
        current = fit_loss()
        if not torch.isfinite(current):
            raise RuntimeError('16-sample fit produced non-finite loss')
        current.backward()
        torch.nn.utils.clip_grad_norm_(fit_params, 1.0)
        fit_optimizer.step()
        fit_losses.append(float(current.detach()))
    final_fit_loss = float(fit_loss().detach())
    report['fit_samples_presented'] = 16
    report['fit_steps'] = 8
    report['fit_initial_loss'] = initial_fit_loss
    report['fit_logged_losses'] = fit_losses
    report['fit_final_loss'] = final_fit_loss
    report['fit_relative_reduction'] = ((initial_fit_loss - final_fit_loss) /
                                        max(initial_fit_loss, 1e-12))
    if not final_fit_loss < initial_fit_loss:
        raise RuntimeError('16-sample fit did not reduce loss')
    report['status'] = 'IMPLEMENTATION-PREFLIGHT-PASSED'
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
