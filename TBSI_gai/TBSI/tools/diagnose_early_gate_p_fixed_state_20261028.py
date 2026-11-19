#!/usr/bin/env python3
"""Query-level fixed-state diagnostics for the Early gate P checkpoint."""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TBSI_INFER_FP16'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch

from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack


def read_frame(seq, frame_num):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][frame_num]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][frame_num]), cv2.COLOR_BGR2RGB)
    return np.concatenate([rgb, tir], axis=2)


def scalar(x):
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


def query_rows(payload, layer, frame, output, topk):
    attn = payload['attention'][0].float()
    gate = payload['gate'][0].float()
    msg = {key: payload[key][0].float() for key in ('msg_rr', 'msg_rt', 'msg_tt', 'msg_tr')}
    n = attn.shape[-1] // 2
    rows = []
    for direction, q0, q1, s0, s1, source_gate_index, self_key, cross_key in (
        ('RGB_receiver_TIR_source', 0, n, n, 2*n, 1, 'msg_rr', 'msg_rt'),
        ('TIR_receiver_RGB_source', n, 2*n, 0, n, 0, 'msg_tt', 'msg_tr')):
        cross = attn[:, q0:q1, s0:s1]
        self_attn = attn[:, q0:q1, q0:q1]
        mass = cross.sum(-1)
        cond = cross / mass.unsqueeze(-1).clamp_min(1e-12)
        entropy = -(cond.clamp_min(1e-12) * cond.clamp_min(1e-12).log()).sum(-1)
        mean_cond = cond.mean(0)
        vals, inds = mean_cond.topk(min(topk, mean_cond.shape[-1]), dim=-1)
        self_norm = msg[self_key].norm(dim=-1).mean(0)
        cross_norm = msg[cross_key].norm(dim=-1).mean(0)
        # Early evidence/gate has one row per spatial token per modality,
        # while the joint attention query axis is concatenated RGB|TIR.
        source_gate = gate[:q1 - q0, source_gate_index]
        for query_index in range(q1 - q0):
            rows.append({
                'layer': int(layer), 'frame': int(frame), 'direction': direction,
                'query_index': query_index,
                'cross_mass': scalar(mass[:, query_index].mean()),
                'cross_entropy': scalar(entropy[:, query_index].mean()),
                'cross_entropy_normalized': scalar(entropy[:, query_index].mean() / np.log(max(2, s1-s0))),
                'self_message_norm': scalar(self_norm[query_index]),
                'cross_message_norm': scalar(cross_norm[query_index]),
                'source_gate': scalar(source_gate[query_index]),
                'topk_source_indices': ','.join(str(int(v)) for v in inds[query_index].tolist()),
                'topk_source_probs': ','.join('%.8g' % float(v) for v in vals[query_index].tolist()),
                'center_peak_score': output['center_peak_score'],
                'center_peak_index': output['center_peak_index'],
                'offset_x': output['offset_x'], 'offset_y': output['offset_y'],
                'pred_center_x_norm': output['pred_center_x_norm'],
                'pred_center_y_norm': output['pred_center_y_norm'],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='EG_EARLY_GATE_FIXED_SOT_QUICK_S42_20261028')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequence-list', required=True)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames-per-sequence', type=int, default=0)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing non-empty diagnostic output: %s' % output)
    output.mkdir(parents=True, exist_ok=True)
    sequences = LasHeRDataset(split='testingset', sequence_list_file=args.sequence_list).get_sequence_list()
    if args.max_sequences:
        sequences = sequences[:args.max_sequences]
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(Path(args.checkpoint).resolve())
    params = parameters(args.config)
    params.debug = 0; params.save_all_boxes = False
    tracker = TBSITrack(params, 'lasher_test')
    pending = {}
    current_output = {}
    original_forward = tracker.network.forward
    def wrapped_forward(*forward_args, **forward_kwargs):
        result = original_forward(*forward_args, **forward_kwargs)
        score = result['score_map'].detach().float()
        size = result['size_map'].detach().float()
        offset = result['offset_map'].detach().float()
        peak_score, peak = score.flatten(1).max(dim=1)
        peak = int(peak[0])
        y, x = divmod(peak, score.shape[-1])
        current_output.clear()
        current_output.update({
            'center_peak_score': scalar(peak_score[0]), 'center_peak_index': peak,
            'offset_x': scalar(offset[0, 0, y, x]), 'offset_y': scalar(offset[0, 1, y, x]),
            'pred_center_x_norm': float((x + offset[0, 0, y, x]).cpu() / score.shape[-1]),
            'pred_center_y_norm': float((y + offset[0, 1, y, x]).cpu() / score.shape[-2]),
        })
        return result
    tracker.network.forward = wrapped_forward
    layer_ids = getattr(params.cfg.MODEL.BACKBONE, 'TBSI_LOC', [3, 6, 9])
    for index, layer in enumerate(tracker.network.backbone.tbsi_layers):
        fusion = getattr(layer, 'v2_template_fusion', None)
        if fusion is None:
            continue
        fusion.analysis_enabled = True
        fusion.analysis_collector = lambda layer_index=index, **payload: pending.__setitem__(layer_index, payload)
    fields = ['sequence', 'layer', 'frame', 'direction', 'query_index', 'cross_mass', 'cross_entropy',
              'cross_entropy_normalized', 'self_message_norm', 'cross_message_norm',
              'source_gate', 'topk_source_indices', 'topk_source_probs', 'center_peak_score',
              'center_peak_index', 'offset_x', 'offset_y', 'pred_center_x_norm', 'pred_center_y_norm']
    with (output / 'query_level.jsonl').open('w') as handle, (output / 'query_level.csv').open('w', newline='') as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fields); writer.writeheader()
        total = 0
        for seq_index, seq in enumerate(sequences):
            print('[%d/%d] %s' % (seq_index + 1, len(sequences), seq.name), flush=True)
            tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
            limit = len(seq.frames[0])
            if args.max_frames_per_sequence:
                limit = min(limit, args.max_frames_per_sequence + 1)
            for frame_num in range(1, limit):
                pending.clear(); current_output.clear()
                tracker.track(read_frame(seq, frame_num), {'sequence_name': seq.name, 'frame_num': frame_num})
                for layer_index, payload in sorted(pending.items()):
                    rows = query_rows(payload, layer_ids[layer_index], frame_num, current_output, args.topk)
                    for row in rows:
                        row['sequence'] = seq.name
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                        writer.writerow(row)
                        total += 1
                handle.flush(); csv_handle.flush()
    (output / 'manifest.json').write_text(json.dumps({
        'checkpoint': str(Path(args.checkpoint).resolve()), 'config': args.config,
        'sequence_list': str(Path(args.sequence_list).resolve()), 'sequences': len(sequences),
        'rows': total, 'fp16': False, 'threads': 0, 'template_update': False,
        'fields': fields, 'diagnostic_type': 'fixed-state query-level readout; no causal claim'},
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'sequences': len(sequences), 'rows': total}))


if __name__ == '__main__':
    main()
