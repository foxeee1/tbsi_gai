"""Synthetic forward-only comparison for the expected-score MR refinement.

No dataset, backward, optimizer or learned accuracy evaluation. Explicit MR
checkpoint is only a read-only timing reference, never training initialization.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import torch
from benchmark_mr_lr_joint import measure, state_hash
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.models.tbsi_track.tbsi_track import build_tbsi_track
from lib.models.layers.tbsi_layer import V2TemplateFusion
from lib.models.layers.low_rank_joint_mass_reallocator import LowRankJointMassReallocator
from lib.models.layers.expected_score_mass_reallocator import ExpectedScoreMassReallocator


def numerical_contract(device):
    """Analytic operator checks, without a task utility or learning claim."""
    candidate = ExpectedScoreMassReallocator().to(device)
    parent = LowRankJointMassReallocator().to(device)
    torch.manual_seed(921)
    prob = torch.randn(1, 12, 64, 128, device=device).softmax(-1)
    p, sp = prob[..., 64:], prob[..., :64]
    s = sp.sum(-1, keepdim=True)
    m = p.sum(-1, keepdim=True)
    zero = torch.zeros(1, 64, 64, device=device)
    a0, scale0 = candidate.allocate(p, s, zero)
    err0 = max((a0 - p).abs().max().item(), (scale0 - 1).abs().max().item())
    d = torch.randn_like(zero).clamp(-4, 4)
    cr, sc = candidate.allocate(p, s, d)
    mu = (p * d[:, None]).sum(-1, keepdim=True) / m
    expected_mass = torch.sigmoid(torch.log(m / s) + mu)
    budget_error = (cr.sum(-1, keepdim=True) - expected_mass).abs().max().item()
    joint, _ = parent.allocate(p, s, d)
    conditional_error = (cr / cr.sum(-1, keepdim=True) -
                         joint / joint.sum(-1, keepdim=True)).abs().max().item()
    normalization_error = (s * sc + cr.sum(-1, keepdim=True) - 1).abs().max().item()
    dropout = torch.nn.Dropout(.2).train()
    dropped = dropout(prob)
    drop_cross, drop_self = candidate.allocate(p, s, d, dropped[..., 64:])
    mask = dropped / prob
    actual = torch.cat((dropped[..., :64] * drop_self, drop_cross), -1)
    reference = torch.cat((sp * sc, cr), -1) * mask
    drop_error = (actual - reference).abs().max().item()
    zc, zs = candidate.allocate(torch.zeros_like(p), torch.ones_like(s), d)
    zero_support = bool(torch.count_nonzero(zc).item() == 0 and torch.equal(zs, torch.ones_like(s)))
    # Matched constant mean, varying dispersion: no automatic budget uplift.
    flat_p = torch.full((1, 1, 1, 2), .25, device=device)
    flat_s = torch.full((1, 1, 1, 1), .5, device=device)
    mixed = torch.tensor([[[-2., 2.]]], device=device)
    ec, _ = candidate.allocate(flat_p, flat_s, mixed)
    jc, _ = parent.allocate(flat_p, flat_s, mixed)
    neg, _ = candidate.allocate(flat_p, flat_s, torch.full_like(mixed, -2))
    pos, _ = candidate.allocate(flat_p, flat_s, torch.full_like(mixed, 2))
    errors = dict(zero_error=err0, budget_formula_max_abs=budget_error,
                  conditional_distribution_max_abs=conditional_error,
                  normalization_max_abs=normalization_error, dropout_reference_max_abs=drop_error)
    if max(errors.values()) > 2e-6 or not zero_support:
        raise RuntimeError('numeric contract: ' + str(errors))
    if abs(ec.sum().item() - .5) > 1e-6 or not neg.sum() < .5 < pos.sum():
        raise RuntimeError('signed acceptance or dispersion contract failed')
    return dict(errors, zero_cross_support=zero_support,
                balanced_scores_cross_mass_expected=ec.sum().item(),
                balanced_scores_cross_mass_joint=jc.sum().item(),
                constant_negative_cross_mass=neg.sum().item(),
                constant_positive_cross_mass=pos.sum().item(),
                backward=False, optimizer_steps=0, task_utility='unmeasured')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-config', required=True)
    parser.add_argument('--reference-checkpoint', required=True)
    parser.add_argument('--before-output', required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--reverse', action='store_true')
    opt = parser.parse_args()
    out = Path(opt.output).resolve()
    if any('/autodl-fs' in str(p) for p in (ROOT, out, Path.cwd().resolve())):
        raise RuntimeError('project disk boundary')
    if opt.iterations < 1:
        raise ValueError('iterations must be positive')
    out.mkdir(parents=True, exist_ok=False)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if active:
        raise RuntimeError('another CUDA process is active: ' + active)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = 'cuda'
    update_config_from_file(opt.reference_config)
    base = copy.deepcopy(cfg)
    base.MODEL.BACKBONE.CROSS_MASS_PRESERVING_RELATION = False
    base.MODEL.BACKBONE.MASS_REDIST_ENABLE = True
    base.MODEL.INIT_CHECKPOINT = ''
    weights = torch.load(opt.reference_checkpoint, map_location='cpu')['net']
    reference = torch.load(opt.before_output, map_location='cpu')
    z = list(reference['z'].to(device).split(3, dim=1))
    x = list(reference['x'].to(device).split(3, dim=1))
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = {'environment': {'python': sys.version, 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'precision': 'FP32', 'TF32': False, 'threads': 1,
              'input': 'B1 fixed Gaussian RGB/TIR template128/search256',
              'dataset': 'NA', 'optimizer_steps': 0, 'backward': False},
              'checkpoint': {'path': str(Path(opt.reference_checkpoint).resolve()), 'sha256': sha(opt.reference_checkpoint)},
              'config': {'path': str(Path(opt.reference_config).resolve()), 'sha256': sha(opt.reference_config)},
              'contracts': numerical_contract(device), 'model': {}, 'fusion': {}}
    modes = ['pair', 'lr_joint', 'lr_expected', 'b0']
    if opt.reverse:
        modes.reverse()
    identities = {}
    for mode in modes:
        local = copy.deepcopy(base)
        if mode == 'b0':
            local.MODEL.BACKBONE.MASS_REDIST_ENABLE = False
            local.MODEL.BACKBONE.EARLY_CONDITIONED_CROSS_GATE = False
        else:
            local.MODEL.BACKBONE.MASS_REDIST_MODE = mode
        torch.manual_seed(42)
        model = build_tbsi_track(local, training=False)
        target = model.state_dict()
        shared = {k: v for k, v in weights.items() if k in target and target[k].shape == v.shape
                  and not (mode.startswith('lr') and 'mass_reallocator_' in k)}
        missing, unexpected = model.load_state_dict(shared, strict=False)
        if unexpected or (mode == 'pair' and missing) or (mode.startswith('lr') and any('mass_reallocator_' not in k for k in missing)):
            raise RuntimeError('unexpected loading mismatch')
        for module in model.modules():
            if isinstance(module, V2TemplateFusion) and hasattr(module, 'early_gate_scale'):
                module.early_gate_scale.data.zero_()
                module.early_gate_scale.requires_grad_(False)
        loaded = model.state_dict()
        if state_hash({k: loaded[k] for k in shared}) != state_hash(shared):
            raise RuntimeError('shared state changed')
        model = model.to(device).eval()
        fusions = [m for m in model.modules() if isinstance(m, V2TemplateFusion)]
        captured = {}
        def hook(module, args, kwargs):
            if not captured:
                captured.update(args=args, kwargs=kwargs)
        handle = fusions[0].register_forward_pre_hook(hook, with_kwargs=True)
        with torch.inference_mode():
            prediction = model(z, x)
        handle.remove()
        identities[mode] = prediction['pred_boxes'].detach().cpu()
        details = {'parameters': sum(p.numel() for p in model.parameters()),
                   'missing_keys': missing, 'shared_state_sha256': state_hash(shared)}
        if mode == 'pair':
            errors = {k: (prediction[k].cpu() - v).abs().max().item() for k, v in reference['out'].items()}
            if any(v != 0 for v in errors.values()):
                raise RuntimeError('original MR changed: ' + str(errors))
            details['pre_edit_output_max_abs'] = errors
        if mode.startswith('lr'):
            # Same nonzero query weights in both low-rank modes, no fast zero shortcut.
            torch.manual_seed(927)
            for module in model.modules():
                if isinstance(module, LowRankJointMassReallocator):
                    torch.nn.init.normal_(module.receiver.weight, std=.02)
            details['allocator_state_sha256'] = state_hash({k: v for k, v in model.state_dict().items() if 'mass_reallocator_' in k})
            with torch.inference_mode():
                changed = model(z, x)
            if not all(torch.isfinite(v).all() for v in changed.values() if isinstance(v, torch.Tensor)):
                raise RuntimeError('nonfinite active output')
            details['nonzero_box_change_max_abs'] = (changed['pred_boxes'] - prediction['pred_boxes']).abs().max().item()
        print('MEASURE model', mode, flush=True)
        details.update(measure(lambda: model(z, x), opt.iterations))
        result['model'][mode] = details
        print('MEASURE fusion', mode, flush=True)
        result['fusion'][mode] = measure(lambda: fusions[0](*captured['args'], **captured['kwargs']), opt.iterations)
        if mode.startswith('lr'):
            del changed
        del model, fusions, captured, prediction, target, loaded, shared
        gc.collect()
        torch.cuda.empty_cache()
    result['zero_identity_box_max_abs'] = {mode: (identities[mode] - identities['b0']).abs().max().item()
                                           for mode in ('lr_joint', 'lr_expected')}
    if max(result['zero_identity_box_max_abs'].values()) > 1e-5:
        raise RuntimeError('zero candidate differs from common-state B0 structure')
    if result['model']['lr_joint']['allocator_state_sha256'] != result['model']['lr_expected']['allocator_state_sha256']:
        raise RuntimeError('allocator parameters differ between budget modes')
    (out / 'benchmark.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {m: v['median_wall_ms'] for m, v in result[k].items()} for k in ('model', 'fusion')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
