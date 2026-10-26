#!/usr/bin/env python3
"""Resumable, sequence-atomic full LasHeR test replay for M1 flow diagnostics."""
import argparse
import hashlib
import json
import os
import platform
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

EXPERIMENT_ID = 'M1-FUSION-FLOW-FULL244-DIAG-20261009'
CONFIG_NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CHECKPOINT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG = ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'
EXPECTED_CHECKPOINT = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
EXPECTED_CONFIG = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    os.replace(str(temporary), str(path))


def read_pair(sequence, index):
    visible = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    infrared = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if visible is None or infrared is None:
        raise IOError('failed to read paired frame {} {}'.format(sequence.name, index))
    if visible.shape[:2] != infrared.shape[:2]:
        raise RuntimeError('RGB/TIR image shape mismatch in {}'.format(sequence.name))
    return np.concatenate((cv2.cvtColor(visible, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB)), axis=2)


def validate_trace(path, expected_frames):
    if not path.is_file():
        return False
    count = 0
    try:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if int(row['frame_id']) != count + 1:
                    return False
                count += 1
    except (ValueError, KeyError, json.JSONDecodeError):
        return False
    return count == expected_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit-sequences', type=int, default=0,
                        help='smoke only; 0 means all manifest sequences')
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    trace_dir = output_dir / 'trace'
    state_path = output_dir / 'run_state.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT or sha256(CONFIG) != EXPECTED_CONFIG:
        raise RuntimeError('protected M1 checkpoint/config hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    names = manifest['sequences']
    if len(names) != 244 or len(set(names)) != 244:
        raise RuntimeError('full manifest must contain exactly 244 unique sequences')
    if args.limit_sequences:
        names = names[:args.limit_sequences]
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    seq_list = trace_dir / 'trace_sequences.txt'
    seq_list.write_text('\n'.join(names) + '\n')

    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        'experiment_id': EXPERIMENT_ID, 'manifest_sha256': sha256(manifest_path),
        'checkpoint_sha256': EXPECTED_CHECKPOINT, 'config_sha256': EXPECTED_CONFIG,
        'completed': {}, 'failures': {}, 'status': 'RUNNING'}
    if (state['experiment_id'] != EXPERIMENT_ID or
            state['manifest_sha256'] != sha256(manifest_path) or
            state['checkpoint_sha256'] != EXPECTED_CHECKPOINT or
            state['config_sha256'] != EXPECTED_CONFIG):
        raise RuntimeError('existing run state provenance does not match this run')

    os.environ['TBSI_M1_TRACE_DIR'] = str(trace_dir)
    os.environ['TBSI_TRACE_SAVE_RAW'] = '0'
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_PERTURBATION'] = ''
    os.environ.pop('TBSI_CHECKPOINT_OVERRIDE', None)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    params = parameters(CONFIG_NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'lasher_test')
    dataset = LasHeRDataset(split='testingset', sequence_list_file=str(seq_list)).get_sequence_list()
    sequence_map = {sequence.name: sequence for sequence in dataset}
    if set(sequence_map) != set(names):
        raise RuntimeError('manifest/data split mismatch; missing={} extra={}'.format(
            sorted(set(names) - set(sequence_map)), sorted(set(sequence_map) - set(names))))

    started = time.time()
    trace_summaries = []
    for index, name in enumerate(names, 1):
        sequence = sequence_map[name]
        frame_count = min(len(sequence.frames[0]), len(sequence.frames[1]),
                          len(sequence.ground_truth_rect))
        if (frame_count < 2 or len(sequence.frames[0]) != len(sequence.frames[1]) or
                len(sequence.frames[0]) != len(sequence.ground_truth_rect)):
            raise RuntimeError('invalid paired sequence length: {}'.format(name))
        trace_file = trace_dir / name / 'frames.jsonl'
        if state['completed'].get(name, {}).get('trace_frames') == frame_count - 1 and \
                validate_trace(trace_file, frame_count - 1):
            trace_summaries.append(state['completed'][name])
            print('skip {}/{} {} verified'.format(index, len(names), name), flush=True)
            continue
        if trace_file.exists():
            quarantine = output_dir / 'incomplete_trace'
            quarantine.mkdir(exist_ok=True)
            target = quarantine / ('{}.{}.jsonl'.format(name, int(time.time())))
            shutil.move(str(trace_file), str(target))
        first = read_pair(sequence, 0)
        tracker.initialize(first, {'init_bbox': sequence.ground_truth_rect[0],
                                   'sequence_name': name})
        for frame_index in range(1, frame_count):
            image = read_pair(sequence, frame_index)
            gt = [float(v) for v in sequence.ground_truth_rect[frame_index]]
            tracker.track(image, {'gt_bbox': gt, 'frame_num': frame_index,
                                  'sequence_name': name})
        if not validate_trace(trace_file, frame_count - 1):
            raise RuntimeError('trace validation failed for {}'.format(name))
        item = {'sequence': name, 'dataset_frames': frame_count,
                'trace_frames': frame_count - 1}
        state['completed'][name] = item
        state['failures'].pop(name, None)
        state['updated_unix_time'] = time.time()
        state['status'] = 'RUNNING'
        atomic_json(state_path, state)
        trace_summaries.append(item)
        elapsed = time.time() - started
        done = len(state['completed'])
        print('sequence {}/{} {} frames={} complete={}/244 elapsed_s={:.1f}'.format(
            index, len(names), name, frame_count - 1, done, elapsed), flush=True)

    if len(names) == 244 and len(state['completed']) == 244:
        state['status'] = 'TRACE-COMPLETE'
    else:
        state['status'] = 'SMOKE-COMPLETE' if len(names) < 244 else 'RUNNING'
    state['updated_unix_time'] = time.time()
    atomic_json(state_path, state)
    trace_manifest = {
        'schema_version': 1, 'experiment_id': EXPERIMENT_ID,
        'source_manifest': str(manifest_path),
        'source_manifest_sha256': sha256(manifest_path),
        'split': 'LasHeR testingset', 'sequence_list_sha256': sha256(seq_list),
        'checkpoint_sha256': EXPECTED_CHECKPOINT, 'config_sha256': EXPECTED_CONFIG,
        'precision': 'FP32', 'sequences': trace_summaries,
        'completed_count': len(state['completed']), 'status': state['status'],
    }
    atomic_json(trace_dir / 'manifest.json', trace_manifest)
    summary = {
        'experiment_id': EXPERIMENT_ID, 'status': state['status'],
        'sequence_count_requested': len(names), 'sequence_count_completed': len(state['completed']),
        'trace_frames_completed': sum(row['trace_frames'] for row in state['completed'].values()),
        'elapsed_seconds_this_invocation': time.time() - started,
        'checkpoint_sha256': EXPECTED_CHECKPOINT, 'config_sha256': EXPECTED_CONFIG,
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'sequences': [state['completed'][name] for name in names
                      if name in state['completed']],
        'trace_dir': str(trace_dir), 'run_state': str(state_path),
        'limitations': ['Current working-tree tracker replay; not directly comparable to archived M1 predictions.',
                        'Per-frame localization gradients are measured in the separate gradient stage.',
                        'Descriptive test-set analysis only; not for module or insertion-position selection.']}
    atomic_json(output_dir / 'trace_summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
