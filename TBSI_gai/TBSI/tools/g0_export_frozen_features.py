#!/usr/bin/env python3
"""Export frozen M1 early and location-9 search features on fixed train events."""
import argparse
import copy
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
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from tools.template_route_probe_v1 import event_frames, read_frame, select_sequences


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def crop_transform(state, resize_factor, search_size):
    cx = float(state[0]) + float(state[2]) / 2.0
    cy = float(state[1]) + float(state[3]) / 2.0
    side = float(search_size) / float(resize_factor)
    return {'crop_x': cx - side / 2.0, 'crop_y': cy - side / 2.0,
            'resize_factor': float(resize_factor), 'search_size': int(search_size)}


def make_tracker(args, checkpoint):
    params = parameters(args.config)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'lasher_train')
    tracker.diagnostic = True
    tracker.diagnostic_early_block = args.early_block
    tracker.diagnostic_late_block = args.late_block
    return tracker, params


def run_parity(args, sequences, checkpoint):
    baseline, params = make_tracker(args, checkpoint)
    diagnostic, _ = make_tracker(args, checkpoint)
    diagnostic.diagnostic = True
    rows = []
    for seq in sequences[:args.parity_sequences]:
        baseline.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        diagnostic.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        events = set(event_frames(seq))
        for frame_num in range(1, len(seq.frames[0])):
            image = read_frame(seq, frame_num)
            info = {'sequence_name': seq.name, 'frame_num': frame_num,
                    'gt_bbox': seq.ground_truth_rect[frame_num]}
            a = baseline.track(image, info)['target_bbox']
            b = diagnostic.track(image, info)['target_bbox']
            if frame_num in events:
                rows.append({'sequence': seq.name, 'frame': frame_num,
                             'max_abs_box_diff': float(np.max(np.abs(np.asarray(a) - np.asarray(b))))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='TBSI_E_MHSA_FULL_4090D_seed42')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sequence-list', default=str(ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'))
    ap.add_argument('--output', required=True)
    ap.add_argument('--sequences', type=int, default=24)
    ap.add_argument('--events-per-sequence', type=int, default=2)
    ap.add_argument('--parity-sequences', type=int, default=3)
    ap.add_argument('--early-block', type=int, default=2)
    ap.add_argument('--late-block', type=int, default=9)
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
    sequences = select_sequences(dataset, args.sequences)
    if args.events_per_sequence != 2:
        raise ValueError('G0 contract requires exactly two events per sequence')
    (out / 'selected_sequences.txt').write_text('\n'.join(s.name for s in sequences) + '\n')
    parity = run_parity(args, sequences, checkpoint)
    (out / 'parity.json').write_text(json.dumps({
        'events': parity, 'max_abs_box_diff': max([r['max_abs_box_diff'] for r in parity] or [0.0]),
        'passed': all(r['max_abs_box_diff'] == 0.0 for r in parity),
        'parity_sequences': args.parity_sequences}, indent=2) + '\n')
    if not all(r['max_abs_box_diff'] == 0.0 for r in parity):
        raise RuntimeError('diagnostic export parity failed; no feature export accepted')

    tracker, params = make_tracker(args, checkpoint)
    metadata = []
    event_index = 0
    for seq_index, seq in enumerate(sequences):
        print('[{}/{}] {}'.format(seq_index + 1, len(sequences), seq.name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        events = set(event_frames(seq))
        for frame_num in range(1, len(seq.frames[0])):
            state_before = list(tracker.state)
            image = read_frame(seq, frame_num)
            resize_factor = sample_target(image, state_before, params.search_factor,
                                          output_sz=params.search_size)[1]
            info = {'sequence_name': seq.name, 'frame_num': frame_num,
                    'gt_bbox': seq.ground_truth_rect[frame_num]}
            result = tracker.track(image, info)
            if frame_num not in events:
                continue
            features = getattr(tracker, '_last_diagnostic_features', None)
            if features is None:
                raise RuntimeError('diagnostic feature payload missing')
            required = ('early_rgb', 'early_tir', 'late_input_rgb', 'late_input_tir',
                        'late_output_rgb', 'late_output_tir')
            if any(k not in features for k in required):
                raise RuntimeError('missing diagnostic feature keys: {}'.format(required))
            event_id = 'G0-%04d' % event_index
            arrays = {k: features[k][0].float().cpu().numpy() for k in required}
            np.savez_compressed(out / (event_id + '.npz'), **arrays)
            metadata.append({'event_id': event_id, 'sequence': seq.name, 'frame': int(frame_num),
                             'state_before': [float(x) for x in state_before],
                             'pred_box': [float(x) for x in result['target_bbox']],
                             'gt_box_original': [float(x) for x in seq.ground_truth_rect[frame_num]],
                             'search_crop_transform': crop_transform(state_before, resize_factor, params.search_size),
                             'early_block': args.early_block, 'late_block': args.late_block,
                             'token_grid': [16, 16]})
            event_index += 1
    with (out / 'events.jsonl').open('w') as handle:
        for row in metadata:
            handle.write(json.dumps(row) + '\n')
    manifest = {'config': args.config, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': sha256(checkpoint), 'sequence_list': str(sequence_list),
                'events': len(metadata), 'sequences': len(sequences), 'fp16': False,
                'early_block': args.early_block, 'late_block': args.late_block,
                'early_and_late_grid': '16x16; no resolution increase claimed'}
    (out / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
