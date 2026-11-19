#!/usr/bin/env python3
"""Select and export one fixed G0-Hard cohort from frozen M1 tracking.

Selection uses only M1 prediction, GT geometry, and the actual search crop.
Cross-modal feature scores are never consulted during selection.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset  # noqa: E402
from lib.test.parameter.tbsi_track import parameters  # noqa: E402
from lib.test.tracker.tbsi_track import TBSITrack  # noqa: E402
from lib.train.data.processing_utils import sample_target  # noqa: E402
from tools.g0_post_tbsi_complement import token_mask, crop_center_token  # noqa: E402
from tools.template_route_probe_v1 import read_frame  # noqa: E402


def iou(a, b):
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sequence-list', default=str(ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'))
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-per-group', type=int, default=48)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    checkpoint = Path(args.checkpoint).resolve()
    sequence_list = Path(args.sequence_list).resolve()
    out = Path(args.output).resolve()
    if not checkpoint.is_file() or not sequence_list.is_file():
        raise FileNotFoundError('checkpoint or sequence list missing')
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(sequence_list)).get_sequence_list()
    params = parameters(args.config)
    params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_train')
    tracker.diagnostic = True
    tracker.diagnostic_tbsi_index = 0
    tracker.diagnostic_post_block = 5
    groups = ('rescue_failure', 'offcenter', 'easy_centered')
    counts = {group: 0 for group in groups}
    rows = []
    excluded_crop_out = []
    seen_per_sequence = {group: set() for group in groups}
    parity = []
    for seq_index, seq in enumerate(dataset):
        print('[{}/{}] {}'.format(seq_index + 1, len(dataset), seq.name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        n = min(len(seq.frames[0]), len(seq.frames[1]), len(seq.ground_truth_rect))
        for frame_num in range(1, n):
            state_before = list(tracker.state)
            image = read_frame(seq, frame_num)
            _, resize_factor, _ = sample_target(image, state_before, params.search_factor,
                                                output_sz=params.search_size)
            result = tracker.track(image, {'sequence_name': seq.name, 'frame_num': frame_num,
                                           'gt_bbox': seq.ground_truth_rect[frame_num]})
            gt = seq.ground_truth_rect[frame_num]
            mask = token_mask(gt, state_before, resize_factor, search_size=params.search_size)
            in_crop = bool(mask.any())
            gt_cx = float(gt[0] + gt[2] / 2.0)
            gt_cy = float(gt[1] + gt[3] / 2.0)
            state_cx = float(state_before[0] + state_before[2] / 2.0)
            state_cy = float(state_before[1] + state_before[3] / 2.0)
            center_distance = float(np.hypot(gt_cx - state_cx, gt_cy - state_cy) /
                                   (params.search_size / 16.0))
            event_iou = iou(result['target_bbox'], gt)
            eligible = []
            if not in_crop:
                excluded_crop_out.append({'sequence': seq.name, 'frame': frame_num,
                                          'iou': event_iou, 'center_distance_tokens': center_distance})
            else:
                if event_iou < 0.5:
                    eligible.append('rescue_failure')
                if center_distance > 2.0:
                    eligible.append('offcenter')
                if event_iou >= 0.5 and center_distance <= 2.0:
                    eligible.append('easy_centered')
            selected_groups = [group for group in eligible
                               if counts[group] < args.max_per_group and seq.name not in seen_per_sequence[group]]
            if not selected_groups:
                continue
            features = tracker._last_diagnostic_features
            required = ('post_tbsi_rgb', 'post_tbsi_tir', 'post_block_rgb', 'post_block_tir')
            if any(key not in features for key in required):
                raise RuntimeError('missing diagnostic features: {}'.format(required))
            event_id = 'hard_%04d' % len(rows)
            arrays = {key: features[key][0].float().cpu().numpy() for key in required}
            np.savez_compressed(out / (event_id + '.npz'), mask=mask.astype(np.uint8), **arrays)
            row = {'event_id': event_id, 'sequence': seq.name, 'frame': frame_num,
                   'groups': selected_groups, 'all_eligible_groups': eligible,
                   'state_before': [float(x) for x in state_before],
                   'pred_box': [float(x) for x in result['target_bbox']],
                   'gt_box': [float(x) for x in gt], 'm1_iou': event_iou,
                   'resize_factor': float(resize_factor),
                   'center_distance_tokens': center_distance, 'target_tokens': int(mask.sum()),
                   'target_in_crop': in_crop, 'token_grid': [16, 16]}
            rows.append(row)
            for group in selected_groups:
                counts[group] += 1
                seen_per_sequence[group].add(seq.name)
        if all(counts[group] >= args.max_per_group for group in groups):
            print('all groups reached fixed cap; stopping scan', flush=True)
            break
    (out / 'selected_sequences.txt').write_text('\n'.join(sorted(set(r['sequence'] for r in rows))) + '\n')
    with (out / 'events.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    selection = {
        'selection_rule': {
            'source': 'frozen M1 closed-loop prediction and GT/crop geometry only',
            'rescue_failure': 'IoU < 0.5 and target token mask non-empty',
            'offcenter': 'center distance > 2 tokens and target token mask non-empty',
            'easy_centered': 'IoU >= 0.5 and center distance <= 2 tokens and target token mask non-empty',
            'crop_out': 'target token mask empty; counted separately and never assigned to main groups',
            'per_sequence_first_event_per_group': True,
            'max_per_group': args.max_per_group,
            'selection_does_not_use_cross_modal_scores': True,
        },
        'counts': counts, 'selected_events': len(rows),
        'selected_sequences': len(set(r['sequence'] for r in rows)),
        'crop_out_events': len(excluded_crop_out),
        'crop_out_by_sequence': len(set(r['sequence'] for r in excluded_crop_out)),
        'excluded_crop_out': excluded_crop_out,
        'checkpoint_sha256': sha256(checkpoint), 'config': args.config,
        'sequence_list': str(sequence_list), 'seed': args.seed,
        'decision_for_export': 'READY_FOR_HARD_FEATURE_ANALYSIS' if rows else 'NO_VALID_EVENTS',
    }
    (out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
