"""Analyze saved STSER assets only; never construct a model or run a tracker."""
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Checkpoint metadata references lib.train classes.
NAME = 'M1_STSER_SOT_FULL15_COMPETITION_20261025'
RUN = ROOT / 'output/experiments' / NAME
OUT = RUN / 'attribution_assets_only_20261025'


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def entropy_upper_bound(row, modality):
    # H(E[p]) >= E[H(p)]; logs contain token means, not per-token p.
    probabilities = [float(row[modality + '_' + key])
                     for key in ('shared', 'private', 'null')]
    return -sum(p * math.log(max(p, 1e-300)) for p in probabilities)


def config_diff(left, right, prefix=''):
    result = []
    for key in sorted(set(left) | set(right)):
        path = prefix + key
        if key not in left or key not in right:
            result.append({'path': path, 'm1': left.get(key, '<not explicit>'),
                           'candidate': right.get(key, '<not explicit>')})
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            result.extend(config_diff(left[key], right[key], path + '.'))
        elif left[key] != right[key]:
            result.append({'path': path, 'm1': left[key], 'candidate': right[key]})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flow = read_csv(RUN / 'logs/stser_flow.csv')
    grad = read_csv(RUN / 'logs/stser_gradient_flow.csv')
    epoch_rows = []
    for epoch in sorted({int(r['epoch']) for r in flow}):
        samples = [r for r in flow if int(r['epoch']) == epoch]
        result = {'epoch': epoch, 'sampled_batches': len(samples)}
        for key in ('rgb_shared', 'rgb_private', 'rgb_null', 'tir_shared',
                    'tir_private', 'tir_null', 'rgb_relative_delta',
                    'tir_relative_delta', 'gamma_rgb', 'gamma_tir'):
            result[key] = statistics.mean(float(r[key]) for r in samples)
        for modality in ('rgb', 'tir'):
            result[modality + '_entropy_upper_bound_nats'] = statistics.mean(
                entropy_upper_bound(r, modality) for r in samples)
            # For p=softmax(logits/tau), E[tr J] is bounded above by
            # (1 - ||E[p]||^2)/tau, by Jensen. This bounds the softmax stage
            # only; it is not the full network's sensitivity to competitors.
            result[modality + '_router_jacobian_trace_upper_bound'] = statistics.mean(
                max(0.0, 1.0 - sum(float(r[modality + '_' + key]) ** 2
                                   for key in ('shared', 'private', 'null'))) / .07
                for r in samples)
        for group in ('router', 'experts', 't2s_attention', 'outer'):
            selected = [r for r in grad
                        if int(r['epoch']) == epoch and r['group'] == group]
            result[group + '_grad_mean_post_clip'] = statistics.mean(
                float(r['grad_norm']) for r in selected)
            result[group + '_nonzero_tensors_min'] = min(
                int(r['nonzero_tensors']) for r in selected)
        epoch_rows.append(result)
    with (OUT / 'epoch_diagnostics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)

    with (ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml').open() as handle:
        parent_config = yaml.safe_load(handle)
    with (ROOT / 'experiments/tbsi_track' / (NAME + '.yaml')).open() as handle:
        candidate_config = yaml.safe_load(handle)

    # Loading tensors on CPU is a checkpoint audit, not model execution.
    pretrain = torch.load(str(ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'),
                          map_location='cpu')
    pretrain_keys = set(pretrain['net'])
    pretrain_audit = {
        'native_v2_entries': sum('.v2_template_fusion.' in k for k in pretrain_keys),
        'stser_entries': sum('.search_conditioned_t2s.' in k for k in pretrain_keys),
    }
    del pretrain
    gc.collect()
    checkpoint = torch.load(str(RUN / 'checkpoints/TBSITrack_ep0015.pth.tar'),
                            map_location='cpu')
    state = checkpoint['net']
    new_state = {k: v for k, v in state.items() if '.search_conditioned_t2s.' in k}
    groups = []
    for group in checkpoint['optimizer']['param_groups']:
        metadata = {k: v for k, v in group.items() if k != 'params'}
        metadata['parameter_tensor_count'] = len(group['params'])
        groups.append(metadata)
    steps = Counter(float(value['step'])
                    for value in checkpoint['optimizer']['state'].values()
                    if 'step' in value)
    checkpoint_audit = {
        'epoch': checkpoint['epoch'],
        'stser_parameter_count': sum(v.numel() for v in new_state.values()),
        'stser_state_entries': len(new_state),
        'native_v2_entries_missing_from_sot': sum(
            '.v2_template_fusion.' in k and k not in pretrain_keys for k in state),
        'gamma': {k: v.tolist() for k, v in new_state.items() if k.endswith('gamma')},
        'optimizer_param_groups': groups,
        'optimizer_step_histogram': dict(steps),
    }
    del checkpoint, state, new_state
    gc.collect()
    inputs = [RUN / 'logs/stser_flow.csv', RUN / 'logs/stser_gradient_flow.csv',
              ROOT / 'lib/models/layers/search_conditioned_t2s_router.py',
              ROOT / 'lib/train/trainers/diagnostics.py',
              ROOT / 'lib/train/trainers/ltr_trainer.py',
              ROOT / 'experiments/tbsi_track' / (NAME + '.yaml')]
    summary = {
        'execution': 'saved CSV/config/CPU checkpoint analysis only; no model forward, training or tracking',
        'analysis_python': platform.python_version(), 'analysis_torch': torch.__version__,
        'flow_rows': len(flow), 'gradient_rows': len(grad),
        'config_diff_explicit_fields': config_diff(parent_config, candidate_config),
        'pretrain_audit': pretrain_audit, 'checkpoint_audit': checkpoint_audit,
        'training_progress': {
            'nominal_full_update_opportunities': 15 * (59904 // 16 // 8),
            'observed_adam_steps_histogram': dict(steps),
            'nominal_quick_update_opportunities': 3 * (10112 // 16 // 8),
            'note': 'Four fewer observed Adam steps; AMP skips are plausible but unverified.',
        },
        'dominance': {},
        'epoch_diagnostics': epoch_rows,
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in inputs},
        'limitations': [
            'route probabilities are batch/token means; entropy is an upper bound',
            'gradients were recorded after global clipping and rounded to eight decimals',
            'step column is minibatch index within epoch, not optimizer update index',
            'no valid protected M1 per-sequence predictions available for pairing',
            'no same-state intervention or independent seed executed in this analysis',
        ],
    }
    for key in ('rgb_shared', 'tir_private'):
        for threshold in (0.99, 0.999):
            selected = [r for r in flow if float(r[key]) >= threshold]
            summary['dominance'][key + '_ge_' + str(threshold)] = {
                'sampled_batches': len(selected),
                'first_epoch_minibatch': ([int(selected[0]['epoch']), int(selected[0]['step'])]
                                         if selected else None)}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4), constrained_layout=True)
        xs = [r['epoch'] for r in epoch_rows]
        for modality, color in [('rgb', '#b34a36'), ('tir', '#267e9e')]:
            axes[0].semilogy(xs, [max(r[modality + '_entropy_upper_bound_nats'], 1e-8)
                                 for r in epoch_rows], color=color, label=modality.upper())
            axes[1].semilogy(xs, [r[modality + '_relative_delta'] for r in epoch_rows],
                             color=color, label=modality.upper())
        axes[0].set(ylabel='H(mean route), upper bound [nats]', title='Routing loses variability')
        axes[1].set(ylabel='||output - input|| / ||input||', title='Residual writes remain small')
        for group, color in [('router', '#8062ad'), ('experts', '#438e71'),
                             ('t2s_attention', '#c38a35')]:
            axes[2].semilogy(xs, [max(r[group + '_grad_mean_post_clip'], 1e-10)
                                 for r in epoch_rows], color=color, label=group)
        axes[2].set(ylabel='Post-clip group gradient norm', title='Gradient exists; utility unproven')
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=8)
        fig.savefig(str(OUT / 'diagnostics.png'), dpi=180)
        fig.savefig(str(OUT / 'diagnostics.svg'))
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({k: v for k, v in summary.items() if k != 'epoch_diagnostics'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
