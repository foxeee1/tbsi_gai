#!/usr/bin/env python3
"""APSR-v0 U0 frozen-memory one-step counterfactual runner.

U0-GT uses GT crops for dynamic memory slots; U0-Pred uses the S0 anchor's
predicted boxes. Both modes evaluate every valid slot independently from the
same anchor state and never propagate a counterfactual prediction.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from lib.test.evaluation.datasets import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.train.data.processing_utils import sample_target


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['gt', 'pred'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--sequence-list', required=True)
    p.add_argument('--split', choices=['train', 'testingset'], default='train')
    p.add_argument('--write-interval', type=int, default=25)
    p.add_argument('--memory-size', type=int, default=4)
    p.add_argument('--max-sequences', type=int, default=0)
    p.add_argument('--output-root', default='output/experiments/apsr-u0')
    p.add_argument('--resume', action='store_true', help='reuse completed per-sequence JSONL files')
    return p.parse_args()


def read_rgbt(v_path, i_path):
    v_bgr = cv2.imread(v_path)
    i_bgr = cv2.imread(i_path)
    if v_bgr is None or i_bgr is None:
        raise RuntimeError('failed to read frame: %s / %s' % (v_path, i_path))
    v = cv2.cvtColor(v_bgr, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i_bgr, cv2.COLOR_BGR2RGB)
    return np.concatenate([v, i], axis=2)


def as_box(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().reshape(-1).tolist()
    return np.asarray(value, dtype=np.float64).reshape(4)


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return float(inter / union) if union > 0 else 0.0


def clone_state(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(clone_state(x) for x in value)
    if isinstance(value, dict):
        return {k: clone_state(v) for k, v in value.items()}
    return copy.deepcopy(value)


def make_template(tracker, image, bbox):
    bbox = as_box(bbox)
    # LasHeR contains occasional degenerate annotation/prediction boxes.
    # Keep the run total and make this deterministic; the row records the
    # original box separately in the caller's provenance if needed.
    bbox = np.nan_to_num(bbox, nan=0.0, posinf=0.0, neginf=0.0)
    bbox[2:] = np.maximum(bbox[2:], 1.0)
    patch, resize_factor, mask = sample_target(
        image, bbox.tolist(), tracker.params.template_factor,
        output_sz=tracker.params.template_size)
    return tracker.preprocessor.process(patch, mask, half=False), float(resize_factor)


def sequence_list(args):
    with open(args.sequence_list, 'r', encoding='utf-8') as f:
        names = [x.strip() for x in f if x.strip()]
    if args.max_sequences > 0:
        names = names[:args.max_sequences]
    if args.split == 'train':
        # The local LasHeR layout uses ``trainingset`` (the upstream class
        # historically assumes a directory named ``train``).
        dataset = LasHeRDataset(split='testingset', sequence_list_file=None)
        dataset.base_path = os.path.join(dataset.env_settings.lasher_path, 'trainingset')
        dataset.sequence_list = names
        seqs = dataset.get_sequence_list()
        by_name = {s.name: s for s in seqs}
        return [by_name[n] for n in names if n in by_name]
    all_seqs = get_dataset('lasher_test')
    by_name = {s.name: s for s in all_seqs}
    return [by_name[n] for n in names if n in by_name]


def run_sequence(seq, tracker, args):
    frames = [
        [read_rgbt(seq.frames[0][idx], seq.frames[1][idx]) for idx in range(len(seq.frames[0]))]
    ][0]
    gt = np.asarray(seq.ground_truth_rect, dtype=np.float64)
    tracker.initialize(frames[0], seq.init_info())
    m0 = copy.deepcopy(tracker.z_dict1)
    slots = [{'slot': 0, 'bbox': gt[0].tolist(), 'template': m0, 'source': 'initial'}]
    records = []
    anchor_state = as_box(tracker.state)
    anchor_prev_search = clone_state(tracker.prev_search)
    anchor_prev_tokens = clone_state(tracker.prev_temporal_tokens)
    anchor_frame_id = tracker.frame_id

    for t in range(0, len(frames) - 1):
        if t > 0:
            # At loop entry this is the normal S0 anchor output for frame t.
            anchor_state = as_box(tracker.state)
            anchor_prev_search = clone_state(tracker.prev_search)
            anchor_prev_tokens = clone_state(tracker.prev_temporal_tokens)
            anchor_frame_id = tracker.frame_id

        valid_slots = list(slots)
        if len(valid_slots) == args.memory_size:
            gt_next = gt[t + 1]
            slot_ious = {}
            for entry in valid_slots:
                tracker.state = anchor_state.tolist()
                tracker.prev_search = clone_state(anchor_prev_search)
                tracker.prev_temporal_tokens = clone_state(anchor_prev_tokens)
                tracker.frame_id = anchor_frame_id
                tracker.z_dict1 = entry['template']
                info = seq.frame_info(t + 1)
                info.update({'sequence_name': seq.name, 'frame_num': t + 1,
                             'gt_bbox': gt_next})
                out = tracker.track(frames[t + 1], info)
                slot_ious[str(entry['slot'])] = iou_xywh(as_box(out['target_bbox']), gt_next)

            active_iou = slot_ious.get('0', 0.0) if any(x['slot'] == 0 for x in valid_slots) else 0.0
            best_slot = max(slot_ious, key=slot_ious.get)
            records.append({
                'sequence': seq.name, 'frame': int(t),
                'active_slot': 0, 'valid_slots': [int(x['slot']) for x in valid_slots],
                'J_by_slot': slot_ious, 'J_active': active_iou,
                'J_memory_oracle': float(slot_ious[best_slot]),
                'memory_gain': float(slot_ious[best_slot] - active_iou),
                'best_slot': int(best_slot), 'positive_gain': bool(float(slot_ious[best_slot] - active_iou) > 0.0),
                'mode': 'U0-GT' if args.mode == 'gt' else 'U0-Pred',
            })

        # Restore anchor and advance exactly one normal S0 frame.
        tracker.state = anchor_state.tolist()
        tracker.prev_search = clone_state(anchor_prev_search)
        tracker.prev_temporal_tokens = clone_state(anchor_prev_tokens)
        tracker.frame_id = anchor_frame_id
        tracker.z_dict1 = m0
        info = seq.frame_info(t + 1)
        info.update({'sequence_name': seq.name, 'frame_num': t + 1,
                     'gt_bbox': gt[t + 1]})
        tracker.track(frames[t + 1], info)

        # retrieve-before-write: write only after the decision at frame t.
        write_frame = (t > 0 and t % args.write_interval == 0)
        if write_frame and len(slots) < args.memory_size:
            write_bbox = gt[t] if args.mode == 'gt' else anchor_state
            template, _ = make_template(tracker, frames[t], write_bbox)
            slots.append({'slot': len(slots), 'bbox': as_box(write_bbox).tolist(),
                          'template': template, 'source': args.mode})
        elif write_frame and len(slots) == args.memory_size:
            write_bbox = gt[t] if args.mode == 'gt' else anchor_state
            template, _ = make_template(tracker, frames[t], write_bbox)
            dynamic = slots[1:]
            dynamic.pop(0)
            next_slot = max(x['slot'] for x in slots[1:]) + 1
            # Slot ids are stable logical timestamps; report modulo physical FIFO position.
            slots = [slots[0]] + dynamic + [{'slot': next_slot, 'bbox': as_box(write_bbox).tolist(),
                                               'template': template, 'source': args.mode}]
            slots = slots[-args.memory_size:]

    return records


def main():
    args = parse_args()
    if args.memory_size != 4:
        raise SystemExit('APSR-v0 U0 requires memory-size=4')
    if not os.path.isfile(args.checkpoint):
        raise SystemExit('checkpoint not found: %s' % args.checkpoint)
    out_root = Path(args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=args.resume)
    seqs = sequence_list(args)
    if not seqs:
        raise SystemExit('no sequences resolved from %s' % args.sequence_list)

    tracker_api = Tracker('tbsi_track', args.config, args.split, args.run_id)
    params = tracker_api.get_parameters()
    params.checkpoint = os.path.abspath(args.checkpoint)
    params.debug = 0
    tracker = tracker_api.create_tracker(params)
    manifest_hash = hashlib.sha256(Path(args.sequence_list).read_bytes()).hexdigest()
    all_records = []
    if args.resume:
        for old_path in sorted(out_root.glob('*.jsonl')):
            with old_path.open('r', encoding='utf-8') as f:
                all_records.extend(json.loads(line) for line in f if line.strip())
    for seq in seqs:
        seq_path = out_root / ('%s.jsonl' % seq.name)
        if args.resume and seq_path.is_file() and seq_path.stat().st_size > 0:
            print(json.dumps({'sequence': seq.name, 'status': 'resume-skip'}, ensure_ascii=False), flush=True)
            continue
        start = time.time()
        rows = run_sequence(seq, tracker, args)
        all_records.extend(rows)
        with seq_path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'sequence': seq.name, 'frames': len(rows),
                          'seconds': round(time.time() - start, 3)}, ensure_ascii=False), flush=True)

    gains = np.asarray([r['memory_gain'] for r in all_records], dtype=np.float64)
    summary = {
        'status': 'ok', 'mode': 'U0-GT' if args.mode == 'gt' else 'U0-Pred',
        'config': args.config, 'checkpoint': os.path.abspath(args.checkpoint),
        'run_id': args.run_id, 'sequence_list': os.path.abspath(args.sequence_list),
        'sequence_manifest_sha256': manifest_hash, 'sequences': [s.name for s in seqs],
        'memory_size': args.memory_size, 'write_interval': args.write_interval,
        'counterfactual_frames': int(len(gains)),
        'mean_memory_headroom': float(gains.mean()) if len(gains) else 0.0,
        'median_memory_headroom': float(np.median(gains)) if len(gains) else 0.0,
        'positive_memory_coverage': float((gains > 0).mean()) if len(gains) else 0.0,
        'sequence_means': {
            name: float(np.mean([r['memory_gain'] for r in all_records if r['sequence'] == name]))
            for name in sorted({r['sequence'] for r in all_records})
        },
    }
    (out_root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
