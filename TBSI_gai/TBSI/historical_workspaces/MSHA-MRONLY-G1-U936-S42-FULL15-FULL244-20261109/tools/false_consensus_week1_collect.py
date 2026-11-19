#!/usr/bin/env python3
"""Collect pre-TBSI RGB/TIR candidate evidence on LasHeR train sequences.

This phase does not alter model computation, train weights, or expose any
intervention outcome in the review artifacts.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target


def read_pair(seq, index):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][index]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][index]), cv2.COLOR_BGR2RGB)
    return rgb, tir, np.concatenate([rgb, tir], axis=2)


def xywh_iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ix = max(0., min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0., min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.


def token_box_in_image(token_x, token_y, state, resize_factor, search_size):
    cx = float(state[0]) + float(state[2]) / 2.
    cy = float(state[1]) + float(state[3]) / 2.
    crop_side = float(search_size) / float(resize_factor)
    x0, y0 = cx - crop_side / 2., cy - crop_side / 2.
    scale = 16. / float(resize_factor)
    return [x0 + token_x * scale, y0 + token_y * scale, scale, scale]


def components(response, quantile, min_tokens, max_tokens):
    threshold = float(np.quantile(response, quantile))
    mask = (response >= threshold).astype(np.uint8)
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_tokens <= area <= max_tokens:
            x, y, w, h = [int(v) for v in stats[label, :4]]
            out.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
                        'cx': float(cents[label, 0]), 'cy': float(cents[label, 1]),
                        'threshold': threshold})
    if not out:
        y, x = np.unravel_index(int(np.argmax(response)), response.shape)
        out.append({'x': int(x), 'y': int(y), 'w': 1, 'h': 1, 'area': 1,
                    'cx': float(x) + .5, 'cy': float(y) + .5, 'threshold': threshold})
    return out


def topk_cosine(template, search, foreground, topk):
    template = template.float()
    search = search.float()
    foreground = foreground.bool()
    if int(foreground.sum()) == 0:
        foreground = torch.ones(template.shape[0], dtype=torch.bool)
    z = template[foreground]
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = search / search.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sim = z @ x.t()
    k = min(int(topk), sim.shape[0])
    return sim.topk(k, dim=0).values.mean(dim=0).reshape(16, 16).cpu().numpy()


def draw_review(rgb, tir, rgb_map, tir_map, gt, pred, candidates, path):
    def panel(image, response, title, color):
        image = image.copy()
        norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(cv2.resize(norm, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST), color)
        image = cv2.addWeighted(image, .72, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), .28, 0)
        cv2.rectangle(image, (int(gt[0]), int(gt[1])), (int(gt[0] + gt[2]), int(gt[1] + gt[3])), (0, 255, 0), 2)
        cv2.rectangle(image, (int(pred[0]), int(pred[1])), (int(pred[0] + pred[2]), int(pred[1] + pred[3])), (255, 255, 0), 2)
        for c in candidates:
            b = c['box']
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), (255, 0, 255), 2)
        cv2.putText(image, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        return image
    left = panel(rgb, rgb_map, 'RGB response; green=GT yellow=baseline magenta=candidate', cv2.COLORMAP_JET)
    right = panel(tir, tir_map, 'TIR response; no intervention outcome shown', cv2.COLORMAP_HOT)
    sheet = np.concatenate([left, right], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='v1.0.3-单卡4090D-24G-等效全局128-seed3407')
    ap.add_argument('--output', default='output/experiments/false_consensus_week1')
    ap.add_argument('--sequence-count', type=int, default=30)
    ap.add_argument('--points', type=int, default=6)
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--quantile', type=float, default=.90)
    args = ap.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / 'review_images').mkdir(exist_ok=True)
    (out / 'feature_records').mkdir(exist_ok=True)
    params = parameters(args.config); params.debug = 0; params.save_all_boxes = False
    tracker = TBSITrack(params, 'lasher_train')
    list_file = ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(list_file)).get_sequence_list()
    sequences = list(dataset[:args.sequence_count])
    capture = {}
    layer = tracker.network.backbone.tbsi_layers[args.layer]

    def pre_hook(module, inputs):
        capture['x_v'] = inputs[0].detach().float().cpu()
        capture['x_i'] = inputs[1].detach().float().cpu()
        capture['lens_z'] = int(inputs[2])
    handle = layer.register_forward_pre_hook(pre_hook)
    records = []
    event_id = 0
    try:
        for si, seq in enumerate(sequences):
            print('[%d/%d] %s (%d frames)' % (si + 1, len(sequences), seq.name, len(seq.frames[0])), flush=True)
            _, _, init = read_pair(seq, 0)
            tracker.initialize(init, {'init_bbox': seq.ground_truth_rect[0].tolist()})
            last = len(seq.frames[0]) - 1
            starts = sorted(set(np.linspace(1, max(1, last), args.points).round().astype(int).tolist()))
            selected = set(starts)
            for frame_id in range(1, len(seq.frames[0])):
                rgb, tir, image = read_pair(seq, frame_id)
                state_before = list(tracker.state)
                out_box = tracker.track(image, {'sequence_name': seq.name, 'frame_num': frame_id})
                if frame_id not in selected:
                    continue
                x_v, x_i = capture['x_v'][0], capture['x_i'][0]
                lens_z = capture['lens_z']
                template_fg = tracker.c1_template_mask.cpu() if hasattr(tracker, 'c1_template_mask') else torch.ones(lens_z, dtype=torch.bool)
                rgb_map = topk_cosine(x_v[:lens_z], x_v[lens_z:], template_fg, args.topk)
                tir_map = topk_cosine(x_i[:lens_z], x_i[lens_z:], template_fg, args.topk)
                _, resize_factor, _ = sample_target(image, state_before, params.search_factor, output_sz=params.search_size)
                gt = [float(v) for v in seq.ground_truth_rect[frame_id]]
                pred = [float(v) for v in out_box['target_bbox']]
                rgb_comps = components(rgb_map, args.quantile, 1, 64)
                tir_comps = components(tir_map, args.quantile, 1, 64)
                proposals = []
                for modality, comps in (('RGB', rgb_comps), ('TIR', tir_comps)):
                    for comp in comps:
                        box = token_box_in_image(comp['cx'] - .5, comp['cy'] - .5, state_before, resize_factor, params.search_size)
                        proposals.append({'modality': modality, 'box': box, 'score': float((rgb_map if modality == 'RGB' else tir_map)[int(min(15, comp['cy'])), int(min(15, comp['cx']))]), 'tokens': comp['area']})
                for r in rgb_comps:
                    for t in tir_comps:
                        rb = token_box_in_image(r['cx'] - .5, r['cy'] - .5, state_before, resize_factor, params.search_size)
                        tb = token_box_in_image(t['cx'] - .5, t['cy'] - .5, state_before, resize_factor, params.search_size)
                        rc = np.array([rb[0] + rb[2]/2, rb[1] + rb[3]/2])
                        tc = np.array([tb[0] + tb[2]/2, tb[1] + tb[3]/2])
                        distance = float(np.linalg.norm(rc - tc))
                        outside = max(xywh_iou(rb, gt), xywh_iou(tb, gt)) <= .10
                        if distance <= 64.0 and outside:
                            event_id += 1
                            candidate = {'event_id': 'FC-%05d' % event_id, 'seq': seq.name, 'frame': frame_id, 'rgb_box': rb, 'tir_box': tb, 'rgb_score': float(rgb_map[int(min(15, r['cy'])), int(min(15, r['cx']))]), 'tir_score': float(tir_map[int(min(15, t['cy'])), int(min(15, t['cx']))]), 'center_distance': distance, 'gt_box': gt, 'baseline_box': pred, 'auto_type': 'P_candidate'}
                            records.append(candidate)
                            review_candidates = [{'box': rb}, {'box': tb}]
                            draw_review(rgb, tir, rgb_map, tir_map, gt, pred, review_candidates, out / 'review_images' / (candidate['event_id'] + '.jpg'))
                np.savez_compressed(out / 'feature_records' / ('%s_%06d.npz' % (seq.name.replace('/', '_'), frame_id)), rgb_response=rgb_map, tir_response=tir_map, gt_box=np.asarray(gt), baseline_box=np.asarray(pred), state_before=np.asarray(state_before), resize_factor=np.asarray(resize_factor))
    finally:
        handle.remove()
    fields = ['event_id', 'seq', 'frame', 'rgb_box', 'tir_box', 'rgb_score', 'tir_score', 'center_distance', 'gt_box', 'baseline_box', 'auto_type']
    with open(out / 'candidate_events.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records:
            w.writerow({k: json.dumps(r[k], ensure_ascii=True) if isinstance(r[k], list) else r[k] for k in fields})
    with open(out / 'annotations.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['event_id', 'seq', 'frame', 'event_type', 'distractor_box_rgb', 'distractor_box_tir', 'same_physical_object', 'visible_in_both', 'alignment_ok', 'reviewer', 'comment'])
        w.writeheader()
        for r in records:
            w.writerow({'event_id': r['event_id'], 'seq': r['seq'], 'frame': r['frame'], 'event_type': 'uncertain', 'distractor_box_rgb': json.dumps(r['rgb_box']), 'distractor_box_tir': json.dumps(r['tir_box']), 'same_physical_object': '', 'visible_in_both': '', 'alignment_ok': '', 'reviewer': '', 'comment': ''})
    meta = {'experiment': 'false_consensus_week1', 'phase': 'candidate_discovery_and_blind_review', 'config': args.config, 'checkpoint': params.checkpoint, 'checkpoint_sha256': hashlib.sha256(Path(params.checkpoint).read_bytes()).hexdigest(), 'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(), 'sequence_count': len(sequences), 'sequences': [s.name for s in sequences], 'points_per_sequence': args.points, 'candidate_count': len(records), 'layer_index': args.layer, 'response_definition': 'TopKMean cosine over initial template foreground tokens before first TBSI layer', 'intervention_run': False}
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=True))
    print(json.dumps({'sequences': len(sequences), 'frames': len(sequences) * args.points, 'candidate_events': len(records), 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
