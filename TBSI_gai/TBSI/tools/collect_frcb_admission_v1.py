#!/usr/bin/env python3
"""Read-only FRCB admission diagnostics on fixed LasHeR train events."""
import argparse
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


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_frame(seq, frame):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][frame]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][frame]), cv2.COLOR_BGR2RGB)
    return np.concatenate([rgb, tir], axis=2)


def iou(a, b):
    ax, ay, aw, ah = [float(x) for x in a]; bx, by, bw, bh = [float(x) for x in b]
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def bands(tokens):
    b, n, c = tokens.shape
    if n != 64: raise ValueError('expected 64 template tokens, got {}'.format(n))
    x = tokens.transpose(1, 2).contiguous().reshape(b, c, 8, 8)
    a, d = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    e, f = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return {'LL': .5 * (a + d + e + f), 'LH': .5 * (a - d + e - f),
            'HL': .5 * (a + d - e - f), 'HH': .5 * (a - d - e + f)}


def token_band_values(tokens):
    result = {}
    for name, value in bands(tokens).items():
        # Haar bands are 4x4; restore the 8x8 token grid before applying the
        # template foreground mask and local +/-1 token correspondence test.
        value = F.interpolate(value.float(), size=(8, 8), mode='nearest')
        result[name] = value.permute(0, 2, 3, 1).reshape(value.shape[0], 64, value.shape[1])
    return result


def cosine(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-6)


def aggregate_frequency(rgb, tir, fg):
    rb, tb = token_band_values(rgb), token_band_values(tir)
    fg = fg.to(rgb.device).view(1, 64).bool()
    output = {}
    for name in ('LL', 'LH', 'HL', 'HH'):
        r, t = rb[name].float(), tb[name].float()
        same = cosine(r, t)
        local = torch.stack([cosine(r, t.roll(shift, dims=1)) for shift in (-1, 0, 1)]).max(0).values
        discrepancy = (r - t).norm(dim=-1) / (r.norm(dim=-1) + t.norm(dim=-1) + 1e-6)
        for label, mask in (('fg', fg), ('bg', ~fg)):
            denom = mask.float().sum().clamp_min(1.0)
            output['{}_{}_energy_rgb'.format(label, name)] = float(r.square().sum(-1)[mask].sum() / denom)
            output['{}_{}_energy_tir'.format(label, name)] = float(t.square().sum(-1)[mask].sum() / denom)
            output['{}_{}_coherence'.format(label, name)] = float(same[mask].sum() / denom)
            output['{}_{}_aligned_coherence'.format(label, name)] = float(local[mask].sum() / denom)
            output['{}_{}_discrepancy'.format(label, name)] = float(discrepancy[mask].sum() / denom)
    return output


def install_collectors(tracker, event_frames, collected):
    handles = []
    fg = tracker.c1_template_mask.detach().cpu()
    for index, layer in enumerate(tracker.network.backbone.tbsi_layers):
        fusion = getattr(layer, 'v2_template_fusion', None)
        if fusion is None: continue
        fusion.analysis_enabled = True
        def collector(attn_prob, rgb_template, tir_template, layer_index=index, **kwargs):
            if not event_frames[0]: return
            p = attn_prob.float(); n = rgb_template.shape[1]
            values = aggregate_frequency(rgb_template, tir_template, fg)
            values['r_to_t_mass'] = float(p[:, :, :n, n:].sum(-1).mean())
            values['t_to_r_mass'] = float(p[:, :, n:, :n].sum(-1).mean())
            values['attn_entropy'] = float((-(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(-1)).mean())
            collected[layer_index] = values
        fusion.analysis_collector = collector
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--only-sequence', default='')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'; seed_all(args.seed)
    checkpoint, event_file, output = map(lambda x: Path(x).resolve(), (args.checkpoint, args.events, args.output))
    if not checkpoint.is_file(): raise FileNotFoundError(str(checkpoint))
    if output.exists() and any(output.iterdir()): raise RuntimeError('refusing non-empty output {}'.format(output))
    events = list(csv.DictReader(event_file.open()))
    by_seq = {}
    for row in events: by_seq.setdefault(row['sequence'], {})[int(row['frame'])] = row
    if args.only_sequence:
        if args.only_sequence not in by_seq:
            raise ValueError('unknown sequence {}'.format(args.only_sequence))
        by_seq = {args.only_sequence: by_seq[args.only_sequence]}
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(event_file.parent / 'sequence_list.txt')).get_sequence_list()
    seq_map = {s.name: s for s in dataset}
    params = parameters(args.config); params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_train_diagnostic')
    rows = []
    for seq_name in sorted(by_seq):
        seq = seq_map[seq_name]; print('[SEQ] {}'.format(seq_name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        previous = {}; skip_until = 0
        for frame in range(1, len(seq.frames[0])):
            if frame <= skip_until:
                continue
            info = {'previous_output': previous, 'sequence_name': seq.name, 'frame_num': frame,
                    'gt_bbox': seq.ground_truth_rect[frame]}
            is_event = frame in by_seq[seq_name]
            collected = {}; event_flag = [is_event]
            if is_event: install_collectors(tracker, event_flag, collected)
            result = tracker.track(read_frame(seq, frame), info)
            pred = result['target_bbox']
            previous = result
            if is_event:
                event = by_seq[seq_name][frame]
                current_iou = iou(pred, seq.ground_truth_rect[frame])
                future = {}
                for horizon in range(1, 6):
                    if frame + horizon >= len(seq.frames[0]): break
                    future_info = {'previous_output': previous, 'sequence_name': seq.name,
                                   'frame_num': frame + horizon, 'gt_bbox': seq.ground_truth_rect[frame + horizon]}
                    future_result = tracker.track(read_frame(seq, frame + horizon), future_info)
                    future[horizon] = iou(future_result['target_bbox'], seq.ground_truth_rect[frame + horizon])
                    previous = future_result
                for layer, values in sorted(collected.items()):
                    for direction in ('R2T', 'T2R'):
                        for band in ('LL', 'LH', 'HL', 'HH'):
                            row = {'event_id': event['event_id'], 'sequence': seq.name, 'frame': frame,
                                   'layer': layer, 'direction': direction, 'band': band,
                                   'current_iou': current_iou}
                            for h in range(1, 6): row['next_{}_iou'.format(h)] = future.get(h, 'NA')
                            for key, value in values.items(): row[key] = value
                            rows.append(row)
                skip_until = frame + max(future.keys(), default=0)
                continue
    if not rows: raise RuntimeError('no rows collected')
    output.mkdir(parents=True)
    with (output / 'event_frequency_rows.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {'events': len(events), 'rows': len(rows), 'sequences': len(by_seq),
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'event_manifest_sha256': hashlib.sha256(event_file.read_bytes()).hexdigest(),
                'read_only': True, 'future_horizon': 5}
    (output / 'collection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
