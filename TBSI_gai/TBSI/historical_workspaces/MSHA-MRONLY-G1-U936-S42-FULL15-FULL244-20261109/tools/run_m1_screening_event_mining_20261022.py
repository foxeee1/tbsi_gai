#!/usr/bin/env python3
"""Resumable frozen-M1 trace and preregistered event mining on train cohorts."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

CONFIG_NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'
CHECKPOINT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_SHA256 = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA256 = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
PROTOCOL = ROOT / 'experiments/protocols/m1_screening_v1_20261022'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    os.replace(str(temporary), str(path))


def read_pair(sequence, index):
    visible = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    infrared = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if visible is None or infrared is None:
        raise IOError('failed to read paired frame {} {}'.format(sequence.name, index))
    if visible.shape[:2] != infrared.shape[:2]:
        raise RuntimeError('RGB/TIR shape mismatch for {} frame {}'.format(sequence.name, index))
    return np.concatenate((cv2.cvtColor(visible, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB)), axis=2)


def valid_trace(path, frame_count):
    if not Path(path).is_file():
        return False
    try:
        with Path(path).open() as stream:
            for expected, line in enumerate(stream, 1):
                row = json.loads(line)
                if int(row['frame_id']) != expected:
                    return False
        return expected == frame_count - 1
    except (UnboundLocalError, ValueError, KeyError, json.JSONDecodeError):
        return False


def box_iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def crop_coverage(gt, crop):
    gx, gy, gw, gh = map(float, gt)
    cx, cy, cw, ch = map(float, crop)
    if gw <= 0 or gh <= 0:
        return 0.0
    iw = max(0.0, min(gx + gw, cx + cw) - max(gx, cx))
    ih = max(0.0, min(gy + gh, cy + ch) - max(gy, cy))
    return (iw * ih) / (gw * gh)


def mine_events(trace_path, sequence):
    with Path(trace_path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    per_frame = []
    for row in rows:
        if not row.get('gt_valid') or row.get('gt_box') is None:
            continue
        iou = box_iou(row['pred_box'], row['gt_box'])
        coverage = crop_coverage(row['gt_box'], row['current_crop_xywh'])
        per_frame.append((row, iou, coverage))

    first_persistent = None
    first_crop_loss = None
    for index, (row, iou, coverage) in enumerate(per_frame):
        if coverage < 0.999 and first_crop_loss is None:
            first_crop_loss = (row, iou, coverage)
        window = per_frame[index:index + 3]
        if (first_persistent is None and len(window) == 3 and
                all(item[1] < 0.5 and item[2] >= 0.999 for item in window)):
            first_persistent = (row, iou, coverage)
    by_frame = {int(row['frame_id']): (row, iou, coverage)
                for row, iou, coverage in per_frame}
    onset = None
    if first_persistent is not None:
        row, iou, coverage = first_persistent
        frame = int(row['frame_id'])
        previous = by_frame.get(frame - 1)
        onset = {
            'sequence': sequence,
            'frame': frame,
            'pre_frame': frame - 1 if previous else None,
            'm1_iou': iou,
            'pre_iou': previous[1] if previous else None,
            'crop_coverage': coverage,
            'gt_box': row['gt_box'],
            'pred_box': row['pred_box'],
            'state_before': row['tracker_state_before'],
            'crop_xywh': row['current_crop_xywh'],
            'trace_sha256': sha256(trace_path),
            'rule': 'first 3 consecutive valid frames with IoU < 0.5 and GT crop coverage >= 0.999',
        }
        if previous:
            onset['pre_state_before'] = previous[0]['tracker_state_before']
            onset['pre_crop_xywh'] = previous[0]['current_crop_xywh']
            onset['pre_gt_box'] = previous[0]['gt_box']
    loss = None
    if first_crop_loss is not None:
        row, iou, coverage = first_crop_loss
        loss = {'frame': int(row['frame_id']), 'm1_iou': iou,
                'crop_coverage': coverage, 'gt_box': row['gt_box'],
                'pred_box': row['pred_box'], 'state_before': row['tracker_state_before'],
                'crop_xywh': row['current_crop_xywh']}
    return onset, loss, len(per_frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=('calibration', 'development', 'locked_holdout'),
                        default='calibration')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    sequence_list = PROTOCOL / '{}_sequences.txt'.format(args.cohort)
    manifest_path = PROTOCOL / '{}_manifest.json'.format(args.cohort)
    if sha256(CONFIG) != CONFIG_SHA256 or sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not sequence_list.is_file() or not manifest_path.is_file():
        raise FileNotFoundError('protocol cohort list or manifest is missing')
    names = [line.strip() for line in sequence_list.read_text().splitlines() if line.strip()]
    candidate_names = set((PROTOCOL / 'quick_train_sequences.txt').read_text().splitlines())
    if len(names) != len(set(names)) or set(names) & candidate_names:
        raise RuntimeError('cohort duplicate or candidate-training overlap')
    expected_count = {'calibration': 48, 'development': 24, 'locked_holdout': 24}[args.cohort]
    if len(names) != expected_count:
        raise RuntimeError('expected {} sequences, found {}'.format(expected_count, len(names)))

    output_dir = Path(args.output_dir).resolve()
    trace_dir = output_dir / 'trace'
    state_path = output_dir / 'run_state.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    run_id = 'M1-SCREENING-EVENT-MINING-{}-20261022'.format(args.cohort.upper())
    cohort_hash = sha256(sequence_list)
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        'run_id': run_id, 'cohort': args.cohort, 'cohort_sha256': cohort_hash,
        'config_sha256': CONFIG_SHA256, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'completed': {}, 'failures': {}, 'status': 'RUNNING'}
    expected_contract = (state.get('run_id') == run_id and
                         state.get('cohort_sha256') == cohort_hash and
                         state.get('config_sha256') == CONFIG_SHA256 and
                         state.get('checkpoint_sha256') == CHECKPOINT_SHA256)
    if not expected_contract:
        raise RuntimeError('existing run state provenance does not match this run')

    os.environ['TBSI_M1_TRACE_DIR'] = str(trace_dir)
    os.environ['TBSI_TRACE_SAVE_RAW'] = '0'
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_PERTURBATION'] = ''
    os.environ.pop('TBSI_CHECKPOINT_OVERRIDE', None)
    torch.set_num_threads(1)

    params = parameters(CONFIG_NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'lasher_test')
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(sequence_list)).get_sequence_list()
    seqmap = {seq.name: seq for seq in dataset}
    if set(seqmap) != set(names):
        raise RuntimeError('train cohort/data mismatch: missing={}, extra={}'.format(
            sorted(set(names) - set(seqmap)), sorted(set(seqmap) - set(names))))

    started = time.time()
    for index, name in enumerate(names, 1):
        sequence = seqmap[name]
        count = min(len(sequence.frames[0]), len(sequence.frames[1]), len(sequence.ground_truth_rect))
        if count < 2 or len(sequence.frames[0]) != len(sequence.frames[1]) or \
                len(sequence.frames[0]) != len(sequence.ground_truth_rect):
            raise RuntimeError('invalid paired sequence length: {}'.format(name))
        trace_path = trace_dir / name / 'frames.jsonl'
        if state['completed'].get(name, {}).get('trace_frames') == count - 1 and valid_trace(trace_path, count):
            print('skip {}/{} {} verified'.format(index, len(names), name), flush=True)
            continue
        if trace_path.exists():
            quarantine = output_dir / 'incomplete_trace'
            quarantine.mkdir(exist_ok=True)
            shutil.move(str(trace_path), str(quarantine / '{}.{}.jsonl'.format(name, int(time.time()))))
        tracker.initialize(read_pair(sequence, 0), {
            'init_bbox': sequence.ground_truth_rect[0], 'sequence_name': name})
        previous_output = {}
        for frame in range(1, count):
            previous_output = tracker.track(read_pair(sequence, frame), {
                'previous_output': previous_output,
                'sequence_name': name, 'frame_num': frame,
                'gt_bbox': sequence.ground_truth_rect[frame]})
        if not valid_trace(trace_path, count):
            raise RuntimeError('trace validation failed for {}'.format(name))
        state['completed'][name] = {'sequence': name, 'dataset_frames': count,
                                    'trace_frames': count - 1}
        state['failures'].pop(name, None)
        state['status'] = 'RUNNING'
        state['updated_unix_time'] = time.time()
        atomic_json(state_path, state)
        print('sequence {}/{} {} frames={} elapsed_s={:.1f}'.format(
            index, len(names), name, count - 1, time.time() - started), flush=True)

    if len(state['completed']) != len(names):
        raise RuntimeError('not all cohort sequences have a complete trace')
    event_rows, summaries = [], []
    for name in names:
        trace_path = trace_dir / name / 'frames.jsonl'
        onset, crop_loss, valid_frames = mine_events(trace_path, name)
        if onset is not None:
            event_rows.append(onset)
        summaries.append({'sequence': name, 'valid_frames': valid_frames,
                          'persistent_low_iou_event': onset is not None,
                          'first_persistent_event_frame': onset['frame'] if onset else None,
                          'first_crop_coverage_loss': crop_loss})
    with (output_dir / 'first_persistent_events.jsonl').open('w') as stream:
        for row in event_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output_dir / 'sequence_scan_summary.jsonl').open('w') as stream:
        for row in summaries:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    state['status'] = 'EVENT-MINING-COMPLETE'
    state['updated_unix_time'] = time.time()
    atomic_json(state_path, state)
    atomic_json(output_dir / 'run_manifest.json', {
        'run_id': run_id, 'parent': CONFIG_NAME, 'cohort': args.cohort,
        'cohort_manifest': str(manifest_path.relative_to(ROOT)),
        'cohort_manifest_sha256': sha256(manifest_path), 'sequence_list_sha256': cohort_hash,
        'config_sha256': CONFIG_SHA256, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'trace_code_sha256': sha256(Path(__file__)),
        'code_contract': 'Current working-tree tracker, validated against archived M1 on 48 frozen test events; default candidate feature gates disabled.',
        'split': 'LasHeR official trainingset', 'precision': 'FP32', 'torch_threads': 1,
        'cudnn_flags': 'left at regular tracker defaults; no deterministic override',
        'event_rule': 'first run of 3 consecutive valid frames with IoU < 0.5 and GT crop coverage >= 0.999',
        'event_count': len(event_rows), 'sequence_count': len(names),
        'candidate_train_overlap': 0, 'parent_training_exposure': 'M1 parent trained on all 977 official train sequences',
        'trace_dir': str(trace_dir), 'elapsed_seconds_this_invocation': time.time() - started,
        'status': state['status'],
    })
    print(json.dumps({'status': state['status'], 'sequences': len(names),
                      'events': len(event_rows), 'trace_frames': sum(x['trace_frames'] for x in state['completed'].values()),
                      'output_dir': str(output_dir)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
