#!/usr/bin/env python3
"""Low-cost contract checks for the bounded Early-gate token residual."""
import math
import yaml
import torch

from lib.models.layers.tbsi_layer import V2TemplateFusion


ROOT = '/root/autodl-tmp/TBSI_gai/TBSI'
CONTROL = ROOT + '/experiments/tbsi_track/M1_EARLY_GATE_TOKEN_REL_QUICK3_CONTROL_RETRY1_20261026.yaml'
CANDIDATE = ROOT + '/experiments/tbsi_track/M1_EARLY_GATE_TOKEN_REL_QUICK3_CANDIDATE_RETRY1_20261026.yaml'


def main():
    with open(CONTROL, 'r') as stream:
        control_cfg = yaml.safe_load(stream)
    with open(CANDIDATE, 'r') as stream:
        candidate_cfg = yaml.safe_load(stream)
    if control_cfg != candidate_cfg:
        raise AssertionError('control and candidate YAML behavior must match exactly')
    if not control_cfg['MODEL']['BACKBONE']['EARLY_CONDITIONED_CROSS_GATE']:
        raise AssertionError('both arms must enable Early-conditioned MSHA gate')
    if control_cfg['DATA']['TRAIN']['SAMPLE_PER_EPOCH'] != 512:
        raise AssertionError('quick3 train sample count mismatch')
    if control_cfg['TRAIN']['EPOCH'] != 3 or control_cfg['TRAIN']['GRAD_ACCUM_STEPS'] != 8:
        raise AssertionError('quick3 progress/accumulation contract mismatch')
    if control_cfg['TRAIN']['TBSI_DROP_PATH'] != [0.0, 0.0, 0.0]:
        raise AssertionError('one drop-path entry per TBSI fusion location is required')

    torch.manual_seed(42)
    bsz, n_template, n_search, dim = 4, 64, 256, 32
    early = tuple(torch.randn(bsz, n, dim) for n in
                  (n_template, n_template, n_search, n_search))
    rgb_scale, tir_scale, stats = V2TemplateFusion._early_gate_token_reliability(early)
    for name, scale in (('rgb', rgb_scale), ('tir', tir_scale)):
        if scale.shape != (bsz, n_template) or not torch.isfinite(scale).all():
            raise AssertionError('{} scale shape/finiteness failure'.format(name))
        if float(scale.min()) < 0.9 - 1e-6 or float(scale.max()) > 1.1 + 1e-6:
            raise AssertionError('{} scale outside [0.9,1.1]'.format(name))
        if not torch.allclose(scale.mean(dim=1), torch.ones(bsz), atol=1e-6, rtol=0):
            raise AssertionError('{} token scales do not preserve mean one'.format(name))
    if not all(math.isfinite(value) for value in stats.values()):
        raise AssertionError('non-finite reliability diagnostics')
    if max(stats['rgb_residual_std'], stats['tir_residual_std']) <= 0.0:
        raise AssertionError('synthetic reliability signal unexpectedly collapsed to zero')
    print('PREFLIGHT_PASS: paired configs identical; early gate enabled; '
          'token residual finite, per-sample mean=1, bounded [0.9,1.1]')
    print('synthetic_stats={}'.format(stats))


if __name__ == '__main__':
    main()
