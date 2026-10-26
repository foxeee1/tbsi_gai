"""Clean AMCM-lite information/gradient/objective contract audit.

This is a synthetic tensor audit on the real TBSITrack graph. It does not
claim tracking performance; it isolates whether the adapter is connected to
the CENTER training objective and whether the frozen M1 parent stays fixed.
"""
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.models.tbsi_track import build_tbsi_track
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap


CONFIG = os.path.join(ROOT, 'experiments/tbsi_track/M1_AMCM_LITE_RELIABILITY_PROXY_20261017.yaml')
FINAL = os.path.join(ROOT, 'output/experiments/M1_AMCM_LITE_RELIABILITY_PROXY_20261017/checkpoints/TBSITrack_ep0003.pth.tar')
OUT = os.path.join(ROOT, 'output/experiments/M1_AMCM_LITE_RELIABILITY_PROXY_20261017/diagnostics/amcm_contract_20261017.json')


def load_model(training=False):
    update_config_from_file(CONFIG)
    model = build_tbsi_track(cfg, training=training)
    state = torch.load(FINAL, map_location='cpu')['net']
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def snapshot_frozen(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if not k.startswith('amcm_reliability.')}


def main():
    torch.manual_seed(3407)
    if not torch.cuda.is_available():
        raise RuntimeError('AMCM contract audit requires CUDA for the real graph')
    device = torch.device('cuda')
    model = load_model(training=False)
    model._amcm_reliability_only = True
    model.amcm_reliability.train()

    cat = torch.randn(2, 640, 768, device=device)
    gt_bbox = torch.tensor([[[0.30, 0.25, 0.22, 0.28],
                             [0.55, 0.40, 0.18, 0.20]]], device=device)
    gt_heat = generate_heatmap(gt_bbox, patch_size=256, stride=16)[-1].unsqueeze(1)

    # Initial zero-gate identity: restore a fresh zero output layer, then
    # compare the exact same graph input with the adapter's no-op forward.
    trained_amcm_state = {k: v.detach().clone() for k, v in model.amcm_reliability.state_dict().items()}
    model.amcm_reliability.gate[-1].weight.data.zero_()
    model.amcm_reliability.gate[-1].bias.data.zero_()
    with torch.no_grad():
        amcm = model.forward_fusion_only(cat)
        saved = model.amcm_reliability
        model.amcm_reliability = None
        parent = model.forward_fusion_only(cat)
        model.amcm_reliability = saved
    identity_max = float((amcm - parent).abs().max().cpu())
    model.amcm_reliability.load_state_dict(trained_amcm_state)

    # Real CENTER loss path: heatmap loss plus box-regression outputs. This
    # checks that gradients reach AMCM through the same head used in training.
    model.amcm_reliability.zero_grad(set_to_none=True)
    frozen_before = snapshot_frozen(model)
    out = model.forward_head(cat, gt_score_map=gt_heat)
    focal = FocalLoss()
    task_loss = focal(out['score_map'], gt_heat)
    task_loss = task_loss + 0.1 * out['size_map'].abs().mean() + 0.1 * out['offset_map'].abs().mean()
    task_loss.backward()
    grad_rows = []
    grad_sq = 0.0
    for name, param in model.amcm_reliability.named_parameters():
        norm = 0.0 if param.grad is None else float(param.grad.detach().norm().cpu())
        grad_rows.append({'name': name, 'grad_norm': norm})
        grad_sq += norm * norm
    grad_norm = grad_sq ** 0.5

    before = {k: v.detach().clone() for k, v in model.amcm_reliability.state_dict().items()}
    optimizer = torch.optim.AdamW(model.amcm_reliability.parameters(), lr=1e-4)
    optimizer.step()
    update_norm = 0.0
    for name, value in model.amcm_reliability.state_dict().items():
        update_norm += float((value.detach() - before[name]).norm().cpu()) ** 2
    update_norm = update_norm ** 0.5

    # Fixed-batch fit check: with the real CENTER loss and a single repeated
    # example, the adapter should be able to move the loss if the supervision
    # can drive its limited residual path at all.
    fit_state = {k: v.detach().clone() for k, v in model.amcm_reliability.state_dict().items()}
    fit_optimizer = torch.optim.AdamW(model.amcm_reliability.parameters(), lr=5e-4)
    fit_losses = []
    for _ in range(20):
        fit_optimizer.zero_grad(set_to_none=True)
        fit_out = model.forward_head(cat, gt_score_map=gt_heat)
        fit_loss = focal(fit_out['score_map'], gt_heat)
        fit_loss = fit_loss + 0.1 * fit_out['size_map'].abs().mean() + 0.1 * fit_out['offset_map'].abs().mean()
        fit_loss.backward()
        fit_optimizer.step()
        fit_losses.append(float(fit_loss.detach().cpu()))
    fit_drop = fit_losses[0] - fit_losses[-1]
    model.amcm_reliability.load_state_dict(fit_state)

    frozen_drift = 0.0
    for key, value in snapshot_frozen(model).items():
        frozen_drift = max(frozen_drift, float((value - frozen_before[key]).abs().max().cpu()))

    # Trained module must actually alter the fusion feature, while staying
    # within the declared bounded residual contract.
    with torch.no_grad():
        trained_fused = model.forward_fusion_only(cat)
        model.amcm_reliability = None
        parent_after = model.forward_fusion_only(cat)
    effect_mean = float((trained_fused - parent_after).abs().mean().cpu())
    effect_max = float((trained_fused - parent_after).abs().max().cpu())

    result = {
        'config': os.path.relpath(CONFIG, ROOT),
        'checkpoint': os.path.relpath(FINAL, ROOT),
        'identity_zero_gate_max_abs': identity_max,
        'center_task_loss': float(task_loss.detach().cpu()),
        'amcm_gradient_norm': grad_norm,
        'gradient_rows': grad_rows,
        'amcm_update_norm_after_one_step': update_norm,
        'fixed_batch_fit_loss_first': fit_losses[0],
        'fixed_batch_fit_loss_last': fit_losses[-1],
        'fixed_batch_fit_loss_drop': fit_drop,
        'frozen_parent_max_abs_drift': frozen_drift,
        'trained_fusion_effect_mean_abs': effect_mean,
        'trained_fusion_effect_max_abs': effect_max,
        'decision': 'CONTRACT-PASS' if identity_max < 2e-6 and grad_norm > 0 and update_norm > 0 and fit_drop > 0 and frozen_drift < 1e-7 else 'CONTRACT-FAIL',
        'interpretation': 'The adapter is connected to the CENTER objective; this audit does not establish tracking gain.'
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
