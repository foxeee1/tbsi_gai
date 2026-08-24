import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tracking'))
import _init_paths  # noqa: F401
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.models.tbsi_track import build_tbsi_track
from lib.train.base_functions import build_dataloaders, update_settings
from lib.train.admin.settings import Settings


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_net(config, checkpoint):
    update_config_from_file('experiments/tbsi_track/%s.yaml' % config)
    settings = Settings()
    settings.script_name = 'tbsi_track'
    settings.config_name = config
    settings.cfg_file = os.path.abspath('experiments/tbsi_track/%s.yaml' % config)
    settings.save_dir = os.path.abspath('output')
    settings.local_rank = -1
    settings.use_lmdb = 0
    update_settings(settings, cfg)
    loader_train, _ = build_dataloaders(cfg, settings)
    net = build_tbsi_track(cfg, training=False).cuda()
    state = torch.load(checkpoint, map_location='cpu')['net']
    state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('load_state_missing=%d unexpected=%d' % (len(missing), len(unexpected)))
    return net, loader_train


def ctvm_modules(net):
    return [m for m in net.modules() if hasattr(m, 'ctvm_gamma')]


def feature_tensor(out):
    feat = out.get('backbone_feat', out)
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]
    return feat.float()


def first_sample(data):
    """Keep one deterministic sample while preserving [search_count, batch, ...]."""
    for modality in ('visible', 'infrared'):
        branch = data[modality]
        for key, value in list(branch.items()):
            if torch.is_tensor(value) and value.dim() >= 2:
                branch[key] = value[:, :1].contiguous()
    return data


def paired_iou(out, data):
    pred = out['pred_boxes'][:, 0].float()
    gt = data['visible']['search_anno'][-1].float().to(pred.device)
    pc = torch.stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2,
                      pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2], dim=-1)
    gc = torch.stack([gt[:, 0], gt[:, 1], gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]], dim=-1)
    tl = torch.maximum(pc[:, :2], gc[:, :2])
    br = torch.minimum(pc[:, 2:], gc[:, 2:])
    inter = (br - tl).clamp_min(0).prod(-1)
    area_p = (pc[:, 2:] - pc[:, :2]).clamp_min(0).prod(-1)
    area_g = (gc[:, 2:] - gc[:, :2]).clamp_min(0).prod(-1)
    return (inter / (area_p + area_g - inter + 1e-6)).mean().item()


def run_forward(net, data, gamma_scale=1.0, uniform=False, backward=False):
    net.zero_grad(set_to_none=True)
    net.train(mode=backward)
    modules = ctvm_modules(net)
    saved = [(m.ctvm_gamma.detach().clone(), m.ctvm_force_uniform) for m in modules]
    for m in modules:
        m.ctvm_gamma.data.mul_(gamma_scale)
        m.ctvm_force_uniform = uniform
    tv = data['visible']['template_images'][0].view(-1, *data['visible']['template_images'].shape[2:]).cuda(non_blocking=True)
    ti = data['infrared']['template_images'][0].view(-1, *data['infrared']['template_images'].shape[2:]).cuda(non_blocking=True)
    sv = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:]).cuda(non_blocking=True)
    si = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:]).cuda(non_blocking=True)
    out = net(template=[tv, ti], search=[sv, si], return_last_attn=False)
    if backward:
        loss = out['pred_boxes'].float().square().mean() + out['score_map'].float().square().mean()
        loss.backward()
    feat = feature_tensor(out).detach()
    updates, values, entropies = [], [], []
    grads = []
    for m in modules:
        if m.last_ctvm_update is not None:
            updates.append(m.last_ctvm_update.float())
            values.append(m.last_ctvm_value.float())
            p = m.last_ctvm_relation.float().clamp_min(1e-12)
            entropies.append(float((-(p * p.log()).sum(-1) / np.log(p.shape[-1])).mean().item()))
        if backward:
            grads.append({
                'norm_w': float(m.ctvm_norm.weight.grad.norm().item()) if m.ctvm_norm.weight.grad is not None else 0.0,
                'norm_b': float(m.ctvm_norm.bias.grad.norm().item()) if m.ctvm_norm.bias.grad is not None else 0.0,
                'gamma': float(m.ctvm_gamma.grad.norm().item()) if m.ctvm_gamma.grad is not None else 0.0,
            })
    for m, (gamma, uniform_old) in zip(modules, saved):
        m.ctvm_gamma.data.copy_(gamma)
        m.ctvm_force_uniform = uniform_old
    return out, feat, updates, values, entropies, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--configs', nargs='+', default=['v1.2.1-ctvm-c1-rpp-3ep', 'v1.2.2-ctvm-c2-rpp-3ep'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seed_all(args.seed)
    results = {}
    for config in args.configs:
        net, loader = load_net(config, args.checkpoint)
        data = first_sample(next(iter(loader)))
        off, feat_off, _, _, _, _ = run_forward(net, data, gamma_scale=0.0)
        on, feat_on, updates, values, ent, grads = run_forward(net, data, gamma_scale=1.0, backward=True)
        uniform, feat_uniform, _, _, ent_uniform, _ = run_forward(net, data, gamma_scale=1.0, uniform=True)
        update_ratio = float(np.mean([u.norm().item() / (v.norm().item() + 1e-6) for u, v in zip(updates, values)]))
        feature_ratio = (feat_on - feat_off).norm().item() / (feat_off.norm().item() + 1e-6)
        cosine = torch.nn.functional.cosine_similarity(feat_on.flatten(1), feat_off.flatten(1), dim=1).mean().item()
        uniform_ratio = (feat_on - feat_uniform).norm().item() / (feat_on.norm().item() + 1e-6)
        results[config] = {
            'checkpoint_role': 'shared_old_ctvm_auxiliary_baseline',
            'gradient': grads,
            'R_V': update_ratio,
            'R_Z': feature_ratio,
            'D_Z': 1.0 - cosine,
            'uniform_relation_feature_delta': uniform_ratio,
            'relation_normalized_entropy': float(np.mean(ent)),
            'uniform_relation_normalized_entropy': float(np.mean(ent_uniform)),
            'IoU_off': paired_iou(off, data),
            'IoU_on': paired_iou(on, data),
            'delta_IoU': paired_iou(on, data) - paired_iou(off, data),
            'IoU_uniform': paired_iou(uniform, data),
        }
        del net, loader, data
        torch.cuda.empty_cache()
    with open(args.out + '/ctvm_final_diagnostic.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
