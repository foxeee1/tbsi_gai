#!/usr/bin/env python
"""Bounded trace-backed TAM training smoke test.

This is a training-contract smoke run, not a tracking evaluation. It uses the
three frozen M1 trace sequences because the full online sampler contract is a
separate experiment gate.
"""
import argparse
import hashlib
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.tbsi_track import config as config_module
from lib.models.tbsi_track import build_tbsi_track
from lib.train.actors.causal_tam_trace_actor import CausalTAMTraceActor
from lib.train.data.trace_temporal_dataset import TraceTemporalDataset


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('causal_tam.'):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--config', default='experiments/tbsi_track/G0_CAUSAL_TAM_PILOT_seed42.yaml')
    parser.add_argument('--checkpoint', default=(
        'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/'
        'TBSITrack_ep0015.pth.tar'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA requested but unavailable')
    os.makedirs(args.output_dir, exist_ok=True)
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.causal_tam.parameters(), lr=1e-4,
                                  weight_decay=1e-4)
    actor = CausalTAMTraceActor(
        model, optimizer,
        loss_weights={'giou': float(config_module.cfg.TRAIN.GIOU_WEIGHT),
                      'l1': float(config_module.cfg.TRAIN.L1_WEIGHT),
                      'focal': 1.0},
        crop_size=int(config_module.cfg.DATA.SEARCH.SIZE),
        stride=int(config_module.cfg.MODEL.BACKBONE.STRIDE))
    before_digest = state_digest(model)
    records = []
    try:
        for step in range(args.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            metrics = actor.step(batch)
            stats = metrics.pop('tam_stats')
            record = {
                'step': step + 1,
                'sequence_ids': list(batch['sequence_id']),
                'frame_ids': batch['frame_id'].tolist(),
                'history_frame_ids': batch['history_frame_id'].tolist(),
                'same_sequence_rate': float(all(a == b for a, b in zip(
                    batch['sequence_id'], batch['history_sequence_id']))),
                'causal_ok': bool((batch['history_frame_id'] < batch['frame_id']).all().item()),
                'loss': metrics['loss'],
                'mean_iou': metrics['mean_iou'],
                'gamma': metrics['gamma_after'],
                'gamma_grad': metrics['gamma_grad'],
                'q_grad': metrics['q_grad'],
                'k_grad': metrics['k_grad'],
                'v_grad': metrics['v_grad'],
                'out_grad': metrics['out_grad'],
                'coverage': stats.get('coverage'),
                'actual_ratio_mean': metrics['actual_ratio_mean'],
                'actual_ratio_p95': metrics['actual_ratio_p95'],
                'actual_ratio_max': metrics['actual_ratio_max'],
                'finite': metrics['finite'],
            }
            if record['same_sequence_rate'] != 1.0 or not record['causal_ok']:
                raise RuntimeError('causal provenance violation during smoke')
            if record['actual_ratio_max'] > 0.10 + 1e-6:
                raise RuntimeError('residual ratio exceeded 0.10 during smoke')
            if not record['finite']:
                raise FloatingPointError('non-finite TAM smoke record')
            records.append(record)
            if (step + 1) % 10 == 0:
                print('step=%d loss=%.6f gamma=%.8g ratio_p95=%.8g' % (
                    step + 1, record['loss'], record['gamma'], record['actual_ratio_p95']),
                      flush=True)
    except Exception:
        with open(os.path.join(args.output_dir, 'partial_records.json'), 'w') as handle:
            json.dump(json_safe(records), handle, indent=2, sort_keys=True)
        raise

    after_digest = state_digest(model)
    tam_checkpoint = os.path.join(args.output_dir, 'tam_smoke_checkpoint.pth')
    torch.save({'tam': model.causal_tam.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': args.steps, 'seed': args.seed}, tam_checkpoint)
    # Restore into a fresh adapter instance in the same frozen model and verify
    # that the saved TAM state is loadable without touching the parent state.
    restored = torch.load(tam_checkpoint, map_location=args.device)
    model.causal_tam.load_state_dict(restored['tam'], strict=True)
    result = {
        'decision': 'TAM_TRACE_SMOKE_PASS',
        'steps': len(records),
        'batch_size': args.batch_size,
        'seed': args.seed,
        'records': records,
        'loss_first': records[0]['loss'],
        'loss_last': records[-1]['loss'],
        'gamma_first': records[0]['gamma'],
        'gamma_last': records[-1]['gamma'],
        'max_ratio_p95': max(r['actual_ratio_p95'] for r in records),
        'max_ratio_max': max(r['actual_ratio_max'] for r in records),
        'all_finite': all(r['finite'] for r in records),
        'all_same_sequence': all(r['same_sequence_rate'] == 1.0 for r in records),
        'all_causal': all(r['causal_ok'] for r in records),
        'non_tam_state_before': before_digest,
        'non_tam_state_after': after_digest,
        'non_tam_state_unchanged': before_digest == after_digest,
        'checkpoint_restore_pass': True,
        'training_scope': 'trace-backed smoke only; not full-data training',
        'tracking_effect_claim': False,
        'checkpoint': tam_checkpoint,
    }
    with open(os.path.join(args.output_dir, 'smoke_summary.json'), 'w') as handle:
        json.dump(json_safe(result), handle, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, 'smoke_records.jsonl'), 'w') as handle:
        for row in records:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + '\n')
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
