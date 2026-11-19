#!/usr/bin/env python3
"""G0 frozen M1 audit for local cross-modal search correspondence.

GT boxes are used only to label target/background tokens after inference. No
GT-derived value enters the model forward. The sequence/event selection is
fixed to the same 24-sequence, two-event development contract used by the
frequency audits.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from tools.template_route_probe_v1 import event_frames, read_frame, select_sequences
from lib.train.data.processing_utils import sample_target


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(v) for v in value)
    return copy.deepcopy(value)


def token_mask(box, state, resize_factor, search_size=256, grid=16):
    cx = float(state[0]) + float(state[2]) / 2.0
    cy = float(state[1]) + float(state[3]) / 2.0
    side = float(search_size) / float(resize_factor)
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    x = (float(box[0]) - x0) * resize_factor
    y = (float(box[1]) - y0) * resize_factor
    w = float(box[2]) * resize_factor
    h = float(box[3]) * resize_factor
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
    px0, py0 = xx * (search_size / grid), yy * (search_size / grid)
    px1, py1 = px0 + search_size / grid, py0 + search_size / grid
    return (px1 > x) & (px0 < x + w) & (py1 > y) & (py0 < y + h)


def cosine_map(a, b):
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return torch.einsum('bnc,bmc->bnm', a, b)


def local_best(values, grid=16):
    b, n, m = values.shape
    if n != grid * grid or m != grid * grid:
        raise ValueError('local_best expects square token grids')
    indices = torch.arange(n, device=values.device).reshape(grid, grid)
    candidates = []
    for dh, dw in [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]:
        shifted = torch.full_like(indices, -1)
        src_h = slice(max(0, -dh), min(grid, grid - dh))
        src_w = slice(max(0, -dw), min(grid, grid - dw))
        dst_h = slice(max(0, dh), min(grid, grid + dh))
        dst_w = slice(max(0, dw), min(grid, grid + dw))
        shifted[dst_h, dst_w] = indices[src_h, src_w]
        valid = shifted >= 0
        gathered = torch.full((b, n), -1.0, device=values.device, dtype=values.dtype)
        flat = shifted.reshape(-1)
        gathered[:, valid.reshape(-1)] = values[:, valid.reshape(-1), flat[valid.reshape(-1)]]
        candidates.append(gathered)
    return torch.stack(candidates, dim=0).max(dim=0).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sequence-list', default=str(ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'))
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    params = parameters(args.config); params.debug = 0; params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=args.sequence_list).get_sequence_list()
    sequences = select_sequences(dataset, 24)
    (out / 'selected_sequences.txt').write_text('\n'.join(s.name for s in sequences) + '\n')
    tracker = TBSITrack(params, 'lasher_train')
    captured = {}
    original_forward = tracker.network.forward

    def capture_forward(*forward_args, **forward_kwargs):
        result = original_forward(*forward_args, **forward_kwargs)
        feat = result.get('backbone_feat')
        captured['feat'] = (feat[-1] if isinstance(feat, list) else feat).detach().float()
        return result

    tracker.network.forward = capture_forward
    rows = []
    for sequence_index, seq in enumerate(sequences):
        print('[{}/{}] {}'.format(sequence_index + 1, len(sequences), seq.name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        events = set(event_frames(seq))
        for frame_num in range(1, len(seq.frames[0])):
            state_before = list(tracker.state)
            image = read_frame(seq, frame_num)
            if frame_num not in events:
                tracker.track(image, {'sequence_name': seq.name, 'frame_num': frame_num})
                continue
            _, resize_factor, _ = sample_target(image, state_before, params.search_factor,
                                                output_sz=params.search_size)
            tracker.track(image, {'sequence_name': seq.name, 'frame_num': frame_num,
                                  'gt_bbox': seq.ground_truth_rect[frame_num]})
            feat = captured['feat']
            if feat.shape[1] != 640 or feat.shape[2] != 768:
                raise RuntimeError('unexpected final token shape {}'.format(tuple(feat.shape)))
            rgb_template, rgb_search = feat[:, :64], feat[:, 64:320]
            tir_template, tir_search = feat[:, 320:384], feat[:, 384:640]
            cross = cosine_map(rgb_search, tir_search)[0]
            reverse = cross.transpose(0, 1)
            same = torch.diagonal(cross).detach().cpu().numpy()
            same_reverse = torch.diagonal(reverse).detach().cpu().numpy()
            best = local_best(cross.unsqueeze(0))[0].detach().cpu().numpy()
            best_reverse = local_best(reverse.unsqueeze(0))[0].detach().cpu().numpy()
            mask = token_mask(seq.ground_truth_rect[frame_num], state_before, resize_factor).reshape(-1)
            target_rgb = F.normalize(rgb_template.mean(1), dim=-1)
            target_tir = F.normalize(tir_template.mean(1), dim=-1)
            sim_rgb = torch.einsum('bc,bnc->bn', target_rgb, F.normalize(rgb_search, dim=-1))[0].cpu().numpy()
            sim_tir = torch.einsum('bc,bnc->bn', target_tir, F.normalize(tir_search, dim=-1))[0].cpu().numpy()
            bg = ~mask
            rows.append({'sequence': seq.name, 'frame': frame_num, 'target_tokens': int(mask.sum()),
                         'target_local_rescue_rgb_to_tir': float((best[mask] - same[mask]).mean()),
                         'background_local_rescue_rgb_to_tir': float((best[bg] - same[bg]).mean()) if bg.any() else 0.0,
                         'target_local_rescue_rate_rgb_to_tir': float((best[mask] > same[mask]).mean()),
                         'target_local_rescue_tir_to_rgb': float((best_reverse[mask] - same_reverse[mask]).mean()),
                         'background_local_rescue_tir_to_rgb': float((best_reverse[bg] - same_reverse[bg]).mean()) if bg.any() else 0.0,
                         'target_template_similarity_rgb': float(sim_rgb[mask].mean()),
                         'background_template_similarity_rgb': float(sim_rgb[bg].mean()) if bg.any() else 0.0,
                         'target_template_similarity_tir': float(sim_tir[mask].mean()),
                         'background_template_similarity_tir': float(sim_tir[bg].mean()) if bg.any() else 0.0})
    with (out / 'event_rows.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    usable = [r for r in rows if r['target_tokens'] > 0]
    def mean(key): return float(np.mean([r[key] for r in usable]))
    summary = {
        'events': len(rows), 'usable_target_events': len(usable),
        'target_token_coverage': float(len(usable) / max(len(rows), 1)),
        'sequences': len(sequences), 'token_shape': [640, 768],
        'target_local_rescue_rgb_to_tir': mean('target_local_rescue_rgb_to_tir'),
        'background_local_rescue_rgb_to_tir': mean('background_local_rescue_rgb_to_tir'),
        'target_local_rescue_rate_rgb_to_tir': mean('target_local_rescue_rate_rgb_to_tir'),
        'target_local_rescue_tir_to_rgb': mean('target_local_rescue_tir_to_rgb'),
        'background_local_rescue_tir_to_rgb': mean('background_local_rescue_tir_to_rgb'),
        'target_similarity_margin_rgb': mean('target_template_similarity_rgb') - mean('background_template_similarity_rgb'),
        'target_similarity_margin_tir': mean('target_template_similarity_tir') - mean('background_template_similarity_tir'),
        'gate_definition': {
            'target_rescue_rate_gt': 0.50,
            'target_minus_background_rescue_gt': 0.01,
            'target_similarity_margin_gt': 0.0,
            'target_token_coverage_gt': 0.80,
            'diagnostic_only': True,
        },
    }
    summary['status'] = 'PASS' if (
        summary['target_token_coverage'] > 0.80 and
        summary['target_local_rescue_rate_rgb_to_tir'] > 0.50 and
        summary['target_local_rescue_rgb_to_tir'] - summary['background_local_rescue_rgb_to_tir'] > 0.01 and
        summary['target_similarity_margin_rgb'] > 0.0 and
        summary['target_similarity_margin_tir'] > 0.0) else 'STOP'
    summary['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
