#!/usr/bin/env python3
"""Collect compact per-frame M1 joint-template MHSA statistics."""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from lib.test.evaluation.lasherdataset import LasHeRDataset  # noqa: E402
from lib.test.parameter.tbsi_track import parameters  # noqa: E402
from lib.test.tracker.tbsi_track import TBSITrack  # noqa: E402


def read_frame(seq, frame_num):
    visible = cv2.cvtColor(cv2.imread(seq.frames[0][frame_num]), cv2.COLOR_BGR2RGB)
    infrared = cv2.cvtColor(cv2.imread(seq.frames[1][frame_num]), cv2.COLOR_BGR2RGB)
    return np.concatenate([visible, infrared], axis=2)


def scalar(value):
    return float(value.detach().cpu().item())


def collect_stats(attn_prob, rgb_template, tir_template, fg_masks, **_ignored):
    attn = attn_prob
    rgb = rgb_template
    tir = tir_template
    eps = 1e-8
    bsz, heads, length, _ = attn.shape
    n_rgb = rgb.shape[1]
    n_tir = tir.shape[1]
    if bsz != 1 or n_rgb != n_tir or length != n_rgb + n_tir:
        raise ValueError('expected batch-1 equal RGB/TIR template lengths')
    entropy = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(dim=-1)
    entropy = entropy.mean() / np.log(length)
    rgb_q, tir_q = attn[:, :, :n_rgb], attn[:, :, n_rgb:]
    cross_rt = rgb_q[:, :, :, n_rgb:].mean()
    cross_tr = tir_q[:, :, :, :n_rgb].mean()
    self_r = rgb_q[:, :, :, :n_rgb].mean()
    self_t = tir_q[:, :, :, n_rgb:].mean()
    rgb_norm = rgb.norm(dim=-1).clamp_min(eps)
    tir_norm = tir.norm(dim=-1).clamp_min(eps)
    cosine = ((rgb * tir).sum(dim=-1) / (rgb_norm * tir_norm)).mean()
    result = {
        'entropy_all': scalar(entropy),
        'entropy_rgb_query': scalar((-(rgb_q.clamp_min(eps) * rgb_q.clamp_min(eps).log()).sum(-1).mean() / np.log(length))),
        'entropy_tir_query': scalar((-(tir_q.clamp_min(eps) * tir_q.clamp_min(eps).log()).sum(-1).mean() / np.log(length))),
        'cross_mass_rgb_to_tir': scalar(cross_rt), 'cross_mass_tir_to_rgb': scalar(cross_tr),
        'self_mass_rgb': scalar(self_r), 'self_mass_tir': scalar(self_t),
        'cross_modal_cosine': scalar(cosine), 'template_disagreement': scalar(1.0 - cosine),
    }
    if fg_masks is None:
        result.update({'fg_mass_rgb': 'NA', 'fg_mass_tir': 'NA', 'bg_mass_rgb': 'NA', 'bg_mass_tir': 'NA'})
    else:
        mask_r, mask_t = fg_masks
        mask_r = mask_r.to(attn.device).float().view(1, 1, 1, n_rgb)
        mask_t = mask_t.to(attn.device).float().view(1, 1, 1, n_tir)
        key_fg_r = torch.cat([mask_r, mask_t], dim=-1)
        key_fg_t = key_fg_r
        fg_r = (rgb_q * key_fg_r).sum(-1).mean()
        fg_t = (tir_q * key_fg_t).sum(-1).mean()
        result.update({'fg_mass_rgb': scalar(fg_r), 'fg_mass_tir': scalar(fg_t),
                       'bg_mass_rgb': scalar(1.0 - fg_r), 'bg_mass_tir': scalar(1.0 - fg_t)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    parser.add_argument('--checkpoint', default=str(ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'))
    parser.add_argument('--output', default=str(ROOT / 'output/experiments/mhsa_failure_analysis_v1/attention_stats'))
    parser.add_argument('--sequence-list', default=None)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames-per-sequence', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing to reuse non-empty output directory: {}'.format(output))
    output.mkdir(parents=True, exist_ok=True)
    dataset = LasHeRDataset(split='testingset', sequence_list_file=args.sequence_list).get_sequence_list() if args.sequence_list else LasHeRDataset(split='testingset').get_sequence_list()
    sequences = list(dataset[:args.max_sequences] if args.max_sequences else dataset)
    params = parameters(args.config)
    params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_test')
    rows = []
    for seq_index, seq in enumerate(sequences):
        print('[{}/{}] {}'.format(seq_index + 1, len(sequences), seq.name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        pending = {}
        for layer_index, layer in enumerate(tracker.network.backbone.tbsi_layers):
            fusion = getattr(layer, 'v2_template_fusion', None)
            if fusion is None:
                continue
            fusion.analysis_enabled = True
            fusion.analysis_collector = lambda layer_index=layer_index, **kwargs: pending.__setitem__(layer_index, collect_stats(**kwargs))
        frame_limit = len(seq.frames[0])
        if args.max_frames_per_sequence > 0:
            frame_limit = min(frame_limit, args.max_frames_per_sequence + 1)
        for frame_num in range(1, frame_limit):
            pending.clear()
            tracker.track(read_frame(seq, frame_num), {'sequence_name': seq.name, 'frame_num': frame_num})
            for layer_index, stats in sorted(pending.items()):
                row = {'seq': seq.name, 'frame': frame_num, 'layer': layer_index}
                row.update(stats)
                rows.append(row)
    if not rows:
        raise RuntimeError('no attention statistics collected')
    fields = list(rows[0])
    with open(output / 'mhsa_layer_stats.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with open(output / 'collection_manifest.json', 'w') as handle:
        json.dump({'experiment': 'mhsa_failure_analysis_v1', 'stage': 'B', 'config': args.config,
                   'checkpoint': str(checkpoint), 'seed': args.seed, 'sequences': len(sequences),
                   'rows': len(rows), 'max_frames_per_sequence': args.max_frames_per_sequence,
                   'sampling': 'deterministic_prefix', 'fp16': False, 'analysis_gate': True}, handle, indent=2)
    print(json.dumps({'sequences': len(sequences), 'rows': len(rows)}))


if __name__ == '__main__':
    main()
