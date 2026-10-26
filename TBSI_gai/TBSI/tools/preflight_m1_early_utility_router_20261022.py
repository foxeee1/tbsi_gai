import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent.parent
sys.path.insert(0, str(PROJECT))

from lib.config.tbsi_track.config import cfg as default_cfg
from lib.config.tbsi_track.config import update_config_from_file
from lib.models.layers.tbsi_layer import template_box_to_token_weights
from lib.models.tbsi_track.tbsi_track import build_tbsi_track


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_cfg(path):
    config = copy.deepcopy(default_cfg)
    update_config_from_file(str(path), config)
    return config


def run_model(config, template, search, masks):
    model = build_tbsi_track(config, training=False).cuda().eval()
    with torch.no_grad():
        output = model(template=template, search=search, template_fg_masks=masks)
    result = {
        'score_map': output['score_map'].detach().cpu(),
        'pred_boxes': output['pred_boxes'].detach().cpu(),
        'utility_logits': output.get('early_utility_logits'),
    }
    if result['utility_logits'] is not None:
        result['utility_logits'] = result['utility_logits'].detach().cpu()
    del output, model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-config', required=True)
    parser.add_argument('--parent-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    candidate_path = Path(args.candidate_config).resolve()
    parent_path = Path(args.parent_config).resolve()
    output_path = Path(args.output).resolve()
    checkpoint = Path(
        '/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/'
        'TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar')
    expected_checkpoint_hash = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the M1 preflight')
    if sha256(checkpoint) != expected_checkpoint_hash:
        raise RuntimeError('Protected M1 checkpoint hash mismatch')

    candidate_cfg, parent_cfg = load_cfg(candidate_path), load_cfg(parent_path)
    if not candidate_cfg.MODEL.BACKBONE.EARLY_UTILITY_ROUTER_ENABLE:
        raise ValueError('candidate config must enable EARLY_UTILITY_ROUTER')
    if parent_cfg.MODEL.BACKBONE.EARLY_UTILITY_ROUTER_ENABLE:
        raise ValueError('parent config must keep EARLY_UTILITY_ROUTER disabled')
    if candidate_cfg.MODEL.INIT_CHECKPOINT != str(checkpoint):
        raise ValueError('candidate must initialize from the registered M1 checkpoint')

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    template = (torch.randn(1, 3, 128, 128, device='cuda'),
                torch.randn(1, 3, 128, 128, device='cuda'))
    search = (torch.randn(1, 3, 256, 256, device='cuda'),
              torch.randn(1, 3, 256, 256, device='cuda'))
    box = torch.tensor([[0.25, 0.25, 0.50, 0.50]], device='cuda')
    weights = template_box_to_token_weights(box, grid_size=8)
    masks = (weights, weights)

    old_fg_mode = os.environ.get('TBSI_MSHA_FG_CROSS_MODE')
    os.environ['TBSI_MSHA_FG_CROSS_MODE'] = 'off'
    parent_result = run_model(parent_cfg, template, search, masks)
    candidate_result = run_model(candidate_cfg, template, search, masks)
    if old_fg_mode is None:
        os.environ.pop('TBSI_MSHA_FG_CROSS_MODE', None)
    else:
        os.environ['TBSI_MSHA_FG_CROSS_MODE'] = old_fg_mode

    if candidate_result['utility_logits'] is None:
        raise RuntimeError('candidate forward did not return utility logits')
    max_score_diff = float((parent_result['score_map'] - candidate_result['score_map']).abs().max())
    max_box_diff = float((parent_result['pred_boxes'] - candidate_result['pred_boxes']).abs().max())
    if max_score_diff > 5e-5 or max_box_diff > 5e-5:
        raise RuntimeError('zero-initialized candidate does not preserve M1 output')
    if float(candidate_result['utility_logits'].abs().max()) != 0.0:
        raise RuntimeError('zero-initialized router did not return uniform logits')

    train_cfg = load_cfg(candidate_path)
    train_model = build_tbsi_track(train_cfg, training=True).cuda()
    train_model.train()
    router = train_model.backbone.early_utility_router
    if not router.training or train_model.backbone.blocks[0].training:
        raise RuntimeError('router-only train/eval mode contract failed')
    trainable = [name for name, parameter in train_model.named_parameters()
                 if parameter.requires_grad]
    if not trainable or any(not name.startswith('backbone.early_utility_router.')
                            for name in trainable):
        raise RuntimeError('non-router parameters are trainable')
    bn_before = [(module, module.running_mean.detach().clone(), module.running_var.detach().clone())
                 for module in train_model.modules()
                 if isinstance(module, torch.nn.BatchNorm2d)]

    train_model.zero_grad(set_to_none=True)
    first = train_model(template=template, search=search, template_fg_masks=masks)
    logits = first['early_utility_logits']
    teacher_rgb, teacher_tir = first['early_utility_teacher_score_maps']
    if logits.shape != (1, 2) or teacher_rgb.shape != teacher_tir.shape:
        raise RuntimeError('router or counterfactual teacher shape contract failed')
    loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0], device='cuda'))
    loss = loss + 1e-4 * first['score_map'].float().mean()
    if not torch.isfinite(loss):
        raise RuntimeError('preflight loss is not finite')
    loss.backward()
    final_weight = router.predictor[-1].weight
    first_grad = float(final_weight.grad.abs().sum()) if final_weight.grad is not None else 0.0
    if not torch.isfinite(final_weight.grad).all() or first_grad <= 0:
        raise RuntimeError('router final layer has no finite nonzero gradient')
    with torch.no_grad():
        final_weight.add_(1e-3 * final_weight.grad)
    train_model.zero_grad(set_to_none=True)
    second = train_model(template=template, search=search, template_fg_masks=masks)
    second_loss = torch.nn.functional.cross_entropy(
        second['early_utility_logits'].float(), torch.tensor([0], device='cuda'))
    second_loss.backward()
    projection_grad = router.descriptor_projection[1].weight.grad
    projection_grad_sum = float(projection_grad.abs().sum()) if projection_grad is not None else 0.0
    if projection_grad is None or not torch.isfinite(projection_grad).all() or projection_grad_sum <= 0:
        raise RuntimeError('router descriptor projection is not trainable after output-layer warm start')
    bn_changed = any(not torch.equal(mean, module.running_mean) or
                     not torch.equal(var, module.running_var)
                     for module, mean, var in bn_before)
    if bn_changed:
        raise RuntimeError('frozen M1 BatchNorm state changed during router-only preflight')

    router_state = copy.deepcopy(router.state_dict())
    router.load_state_dict(router_state, strict=True)
    trainable_params = sum(parameter.numel() for parameter in router.parameters())
    result = {
        'status': 'PREFLIGHT-PASSED',
        'candidate_config': str(candidate_path),
        'candidate_config_sha256': sha256(candidate_path),
        'parent_config': str(parent_path),
        'parent_config_sha256': sha256(parent_path),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'seed': 42,
        'input_template_shapes': [list(tensor.shape) for tensor in template],
        'input_search_shapes': [list(tensor.shape) for tensor in search],
        'score_map_shape': list(candidate_result['score_map'].shape),
        'pred_boxes_shape': list(candidate_result['pred_boxes'].shape),
        'utility_logits_shape': list(candidate_result['utility_logits'].shape),
        'max_abs_parent_candidate_score_diff': max_score_diff,
        'max_abs_parent_candidate_box_diff': max_box_diff,
        'zero_init_utility_logits_max_abs': float(candidate_result['utility_logits'].abs().max()),
        'router_trainable_parameters': trainable_params,
        'trainable_names': trainable,
        'first_step_final_layer_grad_l1': first_grad,
        'second_step_descriptor_projection_grad_l1': projection_grad_sum,
        'bn_running_stats_changed': bn_changed,
        'counterfactual_teacher_shapes': [list(teacher_rgb.shape), list(teacher_tir.shape)],
        'notes': [
            'Synthetic random-image inputs validate implementation contracts only.',
            'This does not satisfy the preregistered calibration utility-ranking gate.',
            'No optimizer step or checkpoint was persisted.'
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
