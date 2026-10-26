#!/usr/bin/env python
"""Fixed-crop, single-frame counterfactual for an existing causal TAM adapter."""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib.config.tbsi_track import config as config_module
from lib.models.tbsi_track import build_tbsi_track
from lib.train.actors.causal_tam_trace_actor import image_box_to_normalized_crop
from lib.utils.box_ops import box_cxcywh_to_xyxy


def iou_xyxy(a, b):
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(dim=-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(dim=-1)
    return inter / (area_a + area_b - inter).clamp_min(1e-8)


def read_rows(trace_dir, sequences):
    rows = []
    for seq in sequences:
        path = os.path.join(trace_dir, seq, 'frames.jsonl')
        with open(path) as stream:
            seq_rows = [json.loads(line) for line in stream if line.strip()]
        by_frame = {int(row['frame_id']): row for row in seq_rows}
        valid = [r for r in seq_rows if r.get('has_history') and r.get('gt_valid')]
        if len(valid) < 16:
            raise RuntimeError('fewer than 16 valid causal rows for ' + seq)
        indices = np.linspace(0, len(valid) - 1, 16).round().astype(int)
        for idx in sorted(set(indices.tolist())):
            row = valid[idx]
            prev = by_frame.get(int(row['history_frame_id']))
            if prev is None or prev['sequence_id'] != seq:
                raise RuntimeError('missing same-sequence raw history for ' + seq)
            row['_history_row'] = prev
            rows.append(row)
    return rows


def load_raw(trace_dir, row, device):
    path = os.path.join(trace_dir, row['sequence_id'], row['raw_head_file'])
    arr = np.load(path, allow_pickle=False)
    if arr.shape != (1, 256, 768) or not np.isfinite(arr).all():
        raise RuntimeError('invalid trace raw feature: ' + path)
    return torch.from_numpy(arr.copy()).to(device=device, dtype=torch.float32)


def evaluate(model, tam_state, rows, trace_dir, device):
    records = []
    for row in rows:
        hist = row['_history_row']
        current = load_raw(trace_dir, row, device)
        history = load_raw(trace_dir, hist, device)
        cur_rect = torch.tensor([row['current_crop_xywh']], device=device)
        hist_rect = torch.tensor([hist['current_crop_xywh']], device=device)
        frame = [int(row['frame_id'])]
        old_frame = [int(row['history_frame_id'])]
        seq = [row['sequence_id']]
        gt = image_box_to_normalized_crop(
            torch.tensor([row['gt_box']], device=device, dtype=torch.float32),
            torch.tensor([row['current_crop_transform']], device=device,
                         dtype=torch.float32), crop_size=256)
        gt_xyxy = torch.cat((gt[:, :2], gt[:, :2] + gt[:, 2:]), dim=-1)

        with torch.no_grad():
            base_map = current.transpose(1, 2).reshape(1, 768, 16, 16)
            base = model.box_head(base_map, None)[1].reshape(1, 4)

            model.causal_tam.load_state_dict(tam_state, strict=True)
            model.causal_tam.gamma.data.copy_(tam_state['gamma'])
            aligned, stats = model.causal_tam(
                current, history, cur_rect, hist_rect, frame, old_frame,
                seq, seq, return_stats=True)
            aligned_map = aligned.transpose(1, 2).reshape(1, 768, 16, 16)
            aligned_box = model.box_head(aligned_map, None)[1].reshape(1, 4)

            model.causal_tam.gamma.zero_()
            zero, _ = model.causal_tam(
                current, history, cur_rect, hist_rect, frame, old_frame,
                seq, seq, return_stats=True)
            zero_box = model.box_head(zero.transpose(1, 2).reshape(1, 768, 16, 16), None)[1].reshape(1, 4)

            # Deliberately omit crop motion by assigning the history the current crop.
            model.causal_tam.load_state_dict(tam_state, strict=True)
            unaligned, unaligned_stats = model.causal_tam(
                current, history, cur_rect, cur_rect, frame, old_frame,
                seq, seq, return_stats=True)
            unaligned_box = model.box_head(
                unaligned.transpose(1, 2).reshape(1, 768, 16, 16), None)[1].reshape(1, 4)

        base_iou = float(iou_xyxy(box_cxcywh_to_xyxy(base), gt_xyxy).item())
        aligned_iou = float(iou_xyxy(box_cxcywh_to_xyxy(aligned_box), gt_xyxy).item())
        unaligned_iou = float(iou_xyxy(box_cxcywh_to_xyxy(unaligned_box), gt_xyxy).item())
        zero_diff = float((zero_box - base).abs().max().item())
        records.append({
            'sequence_id': row['sequence_id'], 'frame_id': int(row['frame_id']),
            'history_frame_id': int(row['history_frame_id']),
            'coverage': stats['coverage'], 'aligned_residual_ratio': stats['message_ratio'],
            'unaligned_coverage': unaligned_stats['coverage'],
            'base_iou': base_iou, 'tam_aligned_iou': aligned_iou,
            'tam_unaligned_iou': unaligned_iou,
            'aligned_delta_iou': aligned_iou - base_iou,
            'unaligned_delta_iou': unaligned_iou - base_iou,
            'gamma_zero_box_max_abs_diff': zero_diff,
            'base_box_crop_cxcywh': base[0].cpu().tolist(),
            'tam_box_crop_cxcywh': aligned_box[0].cpu().tolist(),
        })
    return records


def summarize(records):
    delta = np.asarray([r['aligned_delta_iou'] for r in records], dtype=np.float64)
    per_seq = {}
    for row in records:
        per_seq.setdefault(row['sequence_id'], []).append(row['aligned_delta_iou'])
    return {
        'events': len(records), 'mean_delta_iou': float(delta.mean()),
        'median_delta_iou': float(np.median(delta)),
        'win_tie_loss': [int((delta > 1e-9).sum()), int((np.abs(delta) <= 1e-9).sum()),
                         int((delta < -1e-9).sum())],
        'per_sequence_mean_delta_iou': {k: float(np.mean(v)) for k, v in per_seq.items()},
        'mean_base_iou': float(np.mean([r['base_iou'] for r in records])),
        'mean_tam_aligned_iou': float(np.mean([r['tam_aligned_iou'] for r in records])),
        'mean_tam_unaligned_delta_iou': float(np.mean([r['unaligned_delta_iou'] for r in records])),
        'mean_coverage': float(np.mean([r['coverage'] for r in records])),
        'mean_residual_ratio': float(np.mean([r['aligned_residual_ratio'] for r in records])),
        'max_gamma0_box_diff': float(max(r['gamma_zero_box_max_abs_diff'] for r in records)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--m1-checkpoint', required=True)
    ap.add_argument('--tam-checkpoint', required=True)
    ap.add_argument('--sequences', nargs=2, required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this registered run')
    device = torch.device(args.device)
    rows = read_rows(args.trace_dir, args.sequences)
    config_module.update_config_from_file(args.config)
    config_module.cfg.MODEL.INIT_CHECKPOINT = os.path.abspath(args.m1_checkpoint)
    config_module.cfg.MODEL.CAUSAL_TAM.ENABLE = True
    config_module.cfg.MODEL.CAUSAL_TAM.FORCE_ZERO = False
    model = build_tbsi_track(config_module.cfg, training=False).to(device).eval()
    checkpoint = torch.load(args.m1_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['net'], strict=False)
    tam_blob = torch.load(args.tam_checkpoint, map_location='cpu')
    if int(tam_blob.get('step', -1)) != 100:
        raise RuntimeError('expected the pre-existing 100-step actor audit checkpoint')
    tam_state = tam_blob['tam']
    model.causal_tam.load_state_dict(tam_state, strict=True)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
              if not k.startswith('causal_tam.')}
    records = evaluate(model, tam_state, rows, args.trace_dir, device)
    after = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if not k.startswith('causal_tam.')}
    state_unchanged = all(torch.equal(before[k], after[k]) for k in before)
    result = {
        'decision': 'TAM_FIXED_CROP_COUNTERFACTUAL_COMPLETE',
        'claim_scope': 'single-frame fixed-state/head diagnostic only; no closed-loop or benchmark claim',
        'training_checkpoint': os.path.abspath(args.tam_checkpoint),
        'training_steps': 100, 'adapter_training_sequence': 'abreastinnerboy',
        'evaluation_sequences': args.sequences, 'events_per_sequence': 16,
        'gt_use': 'posthoc scoring only; crops and raw features are frozen M1 trace artifacts',
        'metrics': summarize(records), 'non_tam_state_unchanged': state_unchanged,
        'records': records,
    }
    if not state_unchanged:
        raise RuntimeError('non-TAM model state changed during read-only evaluation')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result['metrics'], indent=2, sort_keys=True))
    print('DECISION', result['decision'])


if __name__ == '__main__':
    main()
