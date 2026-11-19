#!/usr/bin/env python3
"""Frozen, resumable M1 closed-loop trace on a deterministic LasHeR train cohort."""
import argparse
import hashlib
import json
import os
import platform
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
CONFIG = ROOT / 'experiments/tbsi_track' / (CONFIG_NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / CONFIG_NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
EXPECTED_CONFIG = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
EXPECTED_CHECKPOINT = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
DATA = ROOT.parent / 'data/lasher/trainingset'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(str(temporary), str(path))


def read_pair(sequence, index):
    rgb = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    tir = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if rgb is None or tir is None or rgb.shape[:2] != tir.shape[:2]:
        raise RuntimeError('missing/misaligned modalities: {} frame {}'.format(sequence.name, index))
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)


def validate_trace(path, count):
    if not path.is_file():
        return False
    try:
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        return len(rows) == count and all(int(row['frame_id']) == index
                                          for index, row in enumerate(rows, 1))
    except (ValueError, KeyError, json.JSONDecodeError):
        return False


def make_manifest(path, count, offset, explicit_names=None, allow_short_sequences=False):
    if path.is_file():
        manifest = json.loads(path.read_text())
        expected_names = explicit_names
        actual_names = [row['name'] for row in manifest['sequences']]
        if (manifest['requested_count'] != count or
                (expected_names is None and manifest.get('rank_offset', 0) != offset) or
                (expected_names is not None and actual_names != expected_names)):
            raise RuntimeError('existing manifest count/offset differs')
        return manifest
    candidates = []
    for name in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        visible = DATA / name / 'visible'
        infrared = DATA / name / 'infrared'
        gt = DATA / name / 'init.txt'
        if not visible.is_dir() or not infrared.is_dir() or not gt.is_file():
            continue
        frame_count = sum(1 for p in visible.iterdir() if p.suffix.lower() == '.jpg')
        if ((40 <= frame_count < 150) if allow_short_sequences else
                (150 <= frame_count <= 800)):
            candidates.append((name, frame_count))
    ranked = sorted(candidates, key=lambda item: hashlib.sha256(
        ('m1-search-stage-20261009:' + item[0]).encode()).hexdigest())
    if explicit_names is not None:
        by_name = dict(candidates)
        missing = [name for name in explicit_names if name not in by_name]
        if missing or len(explicit_names) != count or len(set(explicit_names)) != count:
            raise RuntimeError('explicit sequence list invalid/missing: ' + repr(missing))
        selected = [(name, by_name[name]) for name in explicit_names]
    else:
        if len(ranked) < count + offset:
            raise RuntimeError('insufficient eligible LasHeR train sequences')
        selected = ranked[offset:offset + count]
    manifest = {
        'id': 'M1-SEARCH-STAGE-TRAIN-DIAG-20261009',
        'split': 'LasHeR trainingset', 'data_path': str(DATA),
        'selection': ('explicit pre-frozen sequence list; 40-149 RGB frames; no metric used'
                      if explicit_names is not None and allow_short_sequences else
                      'explicit pre-frozen sequence list; 150-800 RGB frames; no metric used'
                      if explicit_names is not None else
                      'SHA256 rank of fixed prefix + sequence name; 150-800 RGB frames; no metric used'),
        'requested_count': count, 'rank_offset': offset if explicit_names is None else None,
        'eligible_count': len(candidates),
        'sequences': [{'name': name, 'frames': frames} for name, frames in selected],
        'config_sha256': EXPECTED_CONFIG, 'checkpoint_sha256': EXPECTED_CHECKPOINT,
    }
    atomic_json(path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sequences', type=int, default=24)
    parser.add_argument('--rank-offset', type=int, default=0)
    parser.add_argument('--sequence-list', default=None,
                        help='optional pre-frozen newline-delimited sequence names')
    parser.add_argument('--allow-short-sequences', action='store_true',
                        help='allow 40-149 frame sequences; requires an explicit sequence list')
    args = parser.parse_args()
    if args.allow_short_sequences and not args.sequence_list:
        parser.error('--allow-short-sequences requires --sequence-list')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if sha256(CONFIG) != EXPECTED_CONFIG or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    manifest_path = output / 'manifest.json'
    if args.sequences < 1 or args.rank_offset < 0:
        raise RuntimeError('sequences must be positive and offset nonnegative')
    explicit_names = None
    if args.sequence_list:
        explicit_names = [line.strip() for line in Path(args.sequence_list).read_text().splitlines()
                          if line.strip() and not line.lstrip().startswith('#')]
    manifest = make_manifest(manifest_path, args.sequences, args.rank_offset,
                             explicit_names, args.allow_short_sequences)
    names = [row['name'] for row in manifest['sequences']]
    trace_dir = output / 'trace'
    trace_dir.mkdir(exist_ok=True)
    seq_file = trace_dir / 'trace_sequences.txt'
    if seq_file.is_file() and seq_file.read_text().splitlines() != names:
        raise RuntimeError('existing sequence list differs from manifest')
    seq_file.write_text('\n'.join(names) + '\n')
    state_file = output / 'run_state.json'
    state = json.loads(state_file.read_text()) if state_file.is_file() else {
        'manifest_sha256': sha256(manifest_path), 'completed': {}, 'status': 'RUNNING'}
    if state['manifest_sha256'] != sha256(manifest_path):
        raise RuntimeError('run state manifest mismatch')

    os.environ['TBSI_M1_TRACE_DIR'] = str(trace_dir)
    os.environ['TBSI_TRACE_SAVE_RAW'] = '0'
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_PERTURBATION'] = ''
    os.environ.pop('TBSI_CHECKPOINT_OVERRIDE', None)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters(CONFIG_NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'lasher_train')
    sequences = LasHeRDataset(split='trainingset', sequence_list_file=str(seq_file)).get_sequence_list()
    started = time.time()
    for index, sequence in enumerate(sequences, 1):
        name = sequence.name
        frame_count = len(sequence.frames[0])
        if len(sequence.frames[1]) != frame_count or len(sequence.ground_truth_rect) != frame_count:
            raise RuntimeError('image/annotation count mismatch: ' + name)
        trace_file = trace_dir / name / 'frames.jsonl'
        if validate_trace(trace_file, frame_count - 1):
            state['completed'][name] = frame_count - 1
            print('skip verified {} {}/{}'.format(name, index, len(names)), flush=True)
            continue
        if trace_file.exists():
            raise RuntimeError('incomplete trace requires inspection before resume: ' + str(trace_file))
        tracker.initialize(read_pair(sequence, 0), {
            'init_bbox': sequence.ground_truth_rect[0], 'sequence_name': name})
        for frame in range(1, frame_count):
            gt = [float(v) for v in sequence.ground_truth_rect[frame]]
            tracker.track(read_pair(sequence, frame), {
                'gt_bbox': gt, 'frame_num': frame, 'sequence_name': name})
        if not validate_trace(trace_file, frame_count - 1):
            raise RuntimeError('trace validation failed: ' + name)
        state['completed'][name] = frame_count - 1
        state['updated_unix_time'] = time.time()
        atomic_json(state_file, state)
        print('complete {} {}/{} frames={} elapsed_s={:.1f}'.format(
            name, index, len(names), frame_count - 1, time.time() - started), flush=True)
    state['status'] = 'TRACE-COMPLETE'
    state['elapsed_seconds_this_invocation'] = time.time() - started
    state['gpu'] = torch.cuda.get_device_name(0)
    state['python'] = platform.python_version()
    state['torch'] = torch.__version__
    state['cuda'] = torch.version.cuda
    atomic_json(state_file, state)
    atomic_json(trace_dir / 'manifest.json', manifest)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
