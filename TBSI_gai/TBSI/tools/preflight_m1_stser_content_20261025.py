"""Operator-level checks for the matched content experiment; run explicitly later."""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.models.layers.search_conditioned_t2s_router import SearchConditionedT2SRouter


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(device, amp):
    torch.manual_seed(42)
    base = SearchConditionedT2SRouter(64, num_heads=4,
                                     mode='stser_competition_content',
                                     content_strength=0.0).to(device)
    candidate = copy.deepcopy(base)
    candidate.content_strength = 1.0
    require(all(torch.equal(a, b) for a, b in zip(base.state_dict().values(),
                                                   candidate.state_dict().values())),
            'paired initial state differs')
    source = [torch.randn(2, n, 64, device=device) for n in (64, 64, 256, 256)]
    mask = torch.zeros(2, 64, device=device)
    mask[:, 18:22] = 1
    mask[:, 26:30] = 1
    masks = (mask, mask.clone())
    use_amp = amp and device.type == 'cuda'
    with torch.cuda.amp.autocast(enabled=use_amp):
        zero = base(*source, template_fg_masks=masks)
        old = copy.deepcopy(base)
        old.mode = 'stser_competition'
        old_output = old(*source)
    for a, b in zip(zero, old_output):
        require(torch.equal(a, b), 'strength=0 failed exact legacy-path parity')
    require(all(a.shape == (2, 256, 64) and torch.isfinite(a).all() for a in zero),
            'invalid output shape or finite contract')

    # Gamma is initially zero. Set equal nonzero values to expose the message
    # path, then inspect independent content and template-value gradients.
    with torch.no_grad():
        base.gamma.fill_(0.1)
        candidate.gamma.fill_(0.1)
    with torch.cuda.amp.autocast(enabled=use_amp):
        control_out = base(*source, template_fg_masks=masks)
        candidate_out = candidate(*source, template_fg_masks=masks)
        output_delta = sum((a - b).float().square().mean() for a, b in
                           zip(candidate_out, control_out))
        # Use a non-cancelling scalar for the gradient contract. The paired
        # difference remains an output-change assertion, but its derivative
        # can vanish on symmetric random tensors.
        probe_loss = candidate_out[0].float().square().mean()
    require(float(output_delta.detach()) > 1e-12, 'content path did not change output')
    probe_loss.backward()
    require(candidate.content_logit.grad is not None and
            torch.isfinite(candidate.content_logit.grad).all() and
            float(candidate.content_logit.grad.abs().sum()) > 0,
            'content coefficient has no finite gradient')
    require(candidate.v_tir.weight.grad is not None and
            float(candidate.v_tir.weight.grad.abs().sum()) > 0,
            'template value path has no gradient')
    require(candidate.last_stats['rgb_content_source_shift_l1'] > 0 and
            candidate.last_stats['rgb_content_valid_reference_fraction'] > 0,
            'content source redistribution is inactive')
    return {'device': str(device), 'amp': use_amp,
            'control_dtype': str(control_out[0].dtype),
            'content_output_mse': float(output_delta.detach()),
            'content_grad_l1': float(candidate.content_logit.grad.abs().sum()),
            'source_shift_l1': candidate.last_stats['rgb_content_source_shift_l1']}


def full_model_contract(scope, device):
    from lib.config.tbsi_track.config import cfg, update_config_from_file
    from lib.models.tbsi_track import build_tbsi_track

    root = Path(__file__).resolve().parents[1]
    name = 'M1_STSER_CONTENT_{}_CONTROL_20261025'.format(scope.upper())
    update_config_from_file(str(root / 'experiments/tbsi_track/{}.yaml'.format(name)))
    path = root / cfg.MODEL.INIT_CHECKPOINT
    state = torch.load(str(path), map_location='cpu')['net']
    model = build_tbsi_track(cfg, training=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    routers = [m for m in model.modules() if isinstance(m, SearchConditionedT2SRouter)]
    require(len(routers) == 1, 'expected one content STSER layer')
    router = routers[0]
    template = [torch.randn(1, 3, 128, 128, device=device) for _ in range(2)]
    search = [torch.randn(1, 3, 256, 256, device=device) for _ in range(2)]
    mask = torch.zeros(1, 64, device=device)
    mask[:, 18:22] = 1
    masks = (mask, mask.clone())
    with torch.no_grad():
        baseline = model(template=template, search=search, template_fg_masks=masks)
        require(torch.isfinite(baseline['pred_boxes']).all(), 'full model output is non-finite')
        router.content_strength = 1.0
        identity = model(template=template, search=search, template_fg_masks=masks)
        require(torch.equal(baseline['pred_boxes'], identity['pred_boxes']),
                'zero-gamma full-model identity failed')
        router.gamma.fill_(0.1)
        active = model(template=template, search=search, template_fg_masks=masks)
        require(torch.isfinite(active['pred_boxes']).all() and
                router.last_stats['rgb_content_source_shift_l1'] > 0,
                'full model content path is inactive or non-finite')
    return {'scope': scope, 'strict_state_keys': len(state),
            'pred_boxes_shape': list(active['pred_boxes'].shape),
            'identity_equal': True,
            'source_shift_l1': router.last_stats['rgb_content_source_shift_l1']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--scope', choices=('quick3', 'full15'), required=True)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    device = torch.device(args.device)
    results = [run(device, False)]
    if device.type == 'cuda':
        results.append(run(device, True))
    model_result = full_model_contract(args.scope, device)
    print(json.dumps({'status': 'operator_and_full_model_preflight',
                      'runs': results, 'full_model': model_result}, indent=2))


if __name__ == '__main__':
    main()
