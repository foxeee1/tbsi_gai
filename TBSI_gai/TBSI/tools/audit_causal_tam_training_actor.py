#!/usr/bin/env python
"""Three real optimizer-step audit for the trace-backed TAM actor."""
import argparse
import hashlib
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.tbsi_track import config as config_module
from lib.models.tbsi_track import build_tbsi_track
from lib.train.actors.causal_tam_trace_actor import CausalTAMTraceActor
from lib.train.data.trace_temporal_dataset import TraceTemporalDataset


def json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def digest_non_tam_state(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith('causal_tam.'):
            continue
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def pick_one_per_sequence(dataset):
    selected = []
    seen = set()
    for index, (sequence_id, _, _) in enumerate(dataset.rows):
        if sequence_id not in seen:
            selected.append(index)
            seen.add(sequence_id)
    if len(selected) < 3:
        raise RuntimeError('need at least three distinct trace sequences')
    return selected[:3]


def finite_grad(parameter):
    return (parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            and parameter.grad.detach().abs().sum().item() > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--config', default='experiments/tbsi_track/G0_CAUSAL_TAM_PILOT_seed42.yaml')
    parser.add_argument('--checkpoint', default=(
        'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/'
        'TBSITrack_ep0015.pth.tar'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config_module.update_config_from_file(args.config)
    config_module.cfg.MODEL.INIT_CHECKPOINT = os.path.abspath(args.checkpoint)
    config_module.cfg.MODEL.CAUSAL_TAM.ENABLE = True
    config_module.cfg.MODEL.CAUSAL_TAM.FORCE_ZERO = False
    model = build_tbsi_track(config_module.cfg, training=False).to(args.device)
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('causal_tam.'))
    model.causal_tam.gamma.data.zero_()

    dataset = TraceTemporalDataset(args.trace_dir)
    selected = pick_one_per_sequence(dataset)
    loader = DataLoader(Subset(dataset, selected), batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=False)
    optimizer = torch.optim.AdamW(model.causal_tam.parameters(), lr=1e-4,
                                  weight_decay=1e-4)
    actor = CausalTAMTraceActor(model, optimizer, loss_weights={
        'giou': float(config_module.cfg.TRAIN.GIOU_WEIGHT),
        'l1': float(config_module.cfg.TRAIN.L1_WEIGHT),
        'focal': 1.0,
    }, crop_size=int(config_module.cfg.DATA.SEARCH.SIZE),
       stride=int(config_module.cfg.MODEL.BACKBONE.STRIDE))

    before_digest = digest_non_tam_state(model)
    records = []
    all_finite = True
    causal_ok = True
    seq_ok = True
    shape_ok = True
    ratio_ok = True
    gate_ok = True
    branch_ok_after_open = True
    optimizer_steps = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= 3:
            break
        cur_shape = list(batch['current_raw'].shape)
        hist_shape = list(batch['history_raw'].shape)
        shape_valid = cur_shape == [1, 256, 768] and hist_shape == [1, 256, 768]
        shape_ok = shape_ok and shape_valid
        ids_valid = bool((batch['history_frame_id'] < batch['frame_id']).all().item())
        causal_ok = causal_ok and ids_valid
        same_seq = float(all(a == b for a, b in zip(
            batch['sequence_id'], batch['history_sequence_id'])))
        seq_ok = seq_ok and same_seq == 1.0

        metrics = actor.step(batch)
        optimizer_steps += 1
        grad_values_finite = all(torch.isfinite(torch.tensor(metrics[key]))
                                 for key in ('gamma_grad', 'q_grad', 'k_grad', 'v_grad', 'out_grad'))
        all_finite = all_finite and metrics['finite'] and grad_values_finite and torch.isfinite(torch.tensor(metrics['loss']))
        gate_ok = gate_ok and metrics['gamma_grad'] is not None and abs(metrics['gamma_grad']) > 0
        if batch_index > 0:
            branch_ok_after_open = branch_ok_after_open and all(
                metrics[key] > 0 for key in ('q_grad', 'k_grad', 'v_grad', 'out_grad'))
        ratio_ok = ratio_ok and metrics['actual_ratio_max'] <= 0.10 + 1e-6
        stats = metrics.pop('tam_stats')
        records.append({
            'batch_index': batch_index,
            'sequence_ids': list(batch['sequence_id']),
            'frame_ids': batch['frame_id'].tolist(),
            'history_frame_ids': batch['history_frame_id'].tolist(),
            'same_sequence_rate': same_seq,
            'causal_violation_count': 0 if ids_valid else 1,
            'current_raw_shape': cur_shape,
            'history_raw_shape': hist_shape,
            'coverage_mean': stats.get('coverage'),
            'query_usable_fraction': stats.get('query_usable_fraction'),
            'actual_ratio_mean': stats.get('actual_ratio_mean'),
            'actual_ratio_p95': metrics['actual_ratio_p95'],
            'actual_ratio_max': metrics['actual_ratio_max'],
            'loss': metrics['loss'],
            'loss_giou': metrics['loss_giou'],
            'loss_l1': metrics['loss_l1'],
            'loss_focal': metrics['loss_focal'],
            'mean_iou': metrics['mean_iou'],
            'gamma_before': metrics['gamma_before'],
            'gamma_after': metrics['gamma_after'],
            'gamma_grad': metrics['gamma_grad'],
            'q_grad': metrics['q_grad'],
            'k_grad': metrics['k_grad'],
            'v_grad': metrics['v_grad'],
            'out_grad': metrics['out_grad'],
            'finite': metrics['finite'] and bool(grad_values_finite),
        })

    after_digest = digest_non_tam_state(model)
    non_tam_state_unchanged = before_digest == after_digest
    head_frozen_eval = (not model.box_head.training and
                        all(not p.requires_grad for p in model.box_head.parameters()))
    passed = (len(records) == 3 and optimizer_steps == 3 and all_finite and
              causal_ok and seq_ok and shape_ok and ratio_ok and gate_ok and
              branch_ok_after_open and non_tam_state_unchanged and head_frozen_eval)
    result = {
        'decision': 'TRACE_BACKED_ACTOR_PASS' if passed else 'STOP_TRACE_BACKED_ACTOR',
        'batch_count': len(records),
        'optimizer_steps': optimizer_steps,
        'records': records,
        'checks': {
            'same_sequence_rate_one': seq_ok,
            'causal_order': causal_ok,
            'raw_shape': shape_ok,
            'finite_loss_and_gradients': all_finite,
            'gamma_gradient_nonzero': gate_ok,
            'qkv_out_gradients_nonzero_after_gate_open': branch_ok_after_open,
            'actual_ratio_max_at_most_0_10': ratio_ok,
            'non_tam_parameters_and_buffers_unchanged': non_tam_state_unchanged,
        },
        'non_tam_state_sha256_before': before_digest,
        'non_tam_state_sha256_after': after_digest,
        'optimizer_scope': 'model.causal_tam.parameters() only',
        'model_training_mode': False,
        'box_head_frozen_eval': head_frozen_eval,
        'tracking_effect_claim': False,
        'trace_dir': os.path.abspath(args.trace_dir),
        'checkpoint': os.path.abspath(args.checkpoint),
        'device': str(args.device),
        'seed': int(args.seed),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result = json_safe(result)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == '__main__':
    sys.exit(main())
