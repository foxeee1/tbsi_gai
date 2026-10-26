#!/usr/bin/env python3
"""Run protected M1 closed-loop FP32 tracking and save compact test traces."""
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
CHECKPOINT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG = ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pair(sequence, index):
    visible = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    infrared = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if visible is None or infrared is None:
        raise IOError('failed to read paired frame {} {}'.format(sequence.name, index))
    if visible.shape[:2] != infrared.shape[:2]:
        raise RuntimeError('RGB/TIR image shape mismatch in {}'.format(sequence.name))
    return np.concatenate((cv2.cvtColor(visible, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB)), axis=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('refusing non-empty output directory: {}'.format(output_dir))
    if not manifest_path.is_file() or not CHECKPOINT.is_file() or not CONFIG.is_file():
        raise FileNotFoundError('manifest, protected checkpoint, or config is missing')
    manifest = json.loads(manifest_path.read_text())
    sequence_names = manifest['selected_sequences']
    if len(sequence_names) < 30 or len(sequence_names) != len(set(sequence_names)):
        raise RuntimeError('manifest must contain at least 30 unique sequences')

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / 'trace'
    trace_dir.mkdir()
    sequence_list = trace_dir / 'trace_sequences.txt'
    sequence_list.write_text('\n'.join(sequence_names) + '\n')

    os.environ['TBSI_M1_TRACE_DIR'] = str(trace_dir)
    os.environ['TBSI_TRACE_SAVE_RAW'] = '0'
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_PERTURBATION'] = ''
    os.environ.pop('TBSI_CHECKPOINT_OVERRIDE', None)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this M1 test diagnostic')

    params = parameters(CONFIG_NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'lasher_test')
    dataset = LasHeRDataset(split='testingset', sequence_list_file=str(sequence_list)).get_sequence_list()
    sequence_map = {sequence.name: sequence for sequence in dataset}
    missing = sorted(set(sequence_names) - set(sequence_map))
    if missing:
        raise RuntimeError('manifest sequences absent from LasHeR testingset: {}'.format(missing))

    sequence_summaries = []
    tracked_frames = 0
    started = time.time()
    for sequence_index, name in enumerate(sequence_names, 1):
        sequence = sequence_map[name]
        frame_count = min(len(sequence.frames[0]), len(sequence.frames[1]),
                          len(sequence.ground_truth_rect))
        if frame_count < 2 or len(sequence.frames[0]) != len(sequence.frames[1]):
            raise RuntimeError('invalid paired sequence length: {}'.format(name))
        first = read_pair(sequence, 0)
        tracker.initialize(first, {'init_bbox': sequence.ground_truth_rect[0],
                                   'sequence_name': name})
        start_count = tracked_frames
        for frame_index in range(1, frame_count):
            image = read_pair(sequence, frame_index)
            gt = [float(v) for v in sequence.ground_truth_rect[frame_index]]
            tracker.track(image, {'gt_bbox': gt, 'frame_num': frame_index,
                                  'sequence_name': name})
            tracked_frames += 1
            if tracked_frames % 2000 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print('tracked={} fps={:.2f} elapsed_s={:.1f}'.format(
                    tracked_frames, tracked_frames / elapsed, elapsed), flush=True)
        sequence_summaries.append({'sequence': name,
                                   'dataset_frames': frame_count,
                                   'trace_frames': tracked_frames - start_count})
        print('sequence {}/{} {} frames={} elapsed_s={:.1f}'.format(
            sequence_index, len(sequence_names), name, tracked_frames - start_count,
            time.time() - started), flush=True)

    elapsed = time.time() - started
    trace_manifest = {
        'schema_version': 1,
        'experiment_id': 'M1-FUSION-FLOW-TESTSET-DIAG-20261009',
        'source_selection_manifest': str(manifest_path),
        'source_selection_manifest_sha256': sha256(manifest_path),
        'split': 'testingset',
        'sequence_list': str(sequence_list),
        'sequence_list_sha256': sha256(sequence_list),
        'checkpoint_sha256': sha256(CHECKPOINT),
        'config_sha256': sha256(CONFIG),
        'precision': 'FP32',
        'sequences': sequence_summaries,
    }
    (trace_dir / 'manifest.json').write_text(
        json.dumps(trace_manifest, indent=2, ensure_ascii=False) + '\n')
    payload = {
        'experiment_id': 'M1-FUSION-FLOW-TESTSET-DIAG-20261009',
        'parent': CONFIG_NAME,
        'split': 'LasHeR testingset',
        'manifest': str(manifest_path),
        'manifest_sha256': sha256(manifest_path),
        'checkpoint': str(CHECKPOINT),
        'checkpoint_sha256': sha256(CHECKPOINT),
        'config': str(CONFIG),
        'config_sha256': sha256(CONFIG),
        'sequences': sequence_summaries,
        'sequence_count': len(sequence_summaries),
        'tracked_frames': tracked_frames,
        'precision': 'FP32',
        'model_mode': 'eval; normal closed-loop tracker; no gradients or parameter updates',
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'elapsed_seconds': elapsed,
        'trace_manifest_sha256': sha256(sequence_list),
        'trace_dir': str(trace_dir),
        'limitations': [
            'Diagnostic subset selected from official attributes only; no performance metric used for selection.',
            'This test-set diagnostic is descriptive and must not choose a module or injection position.',
            'Checkpoint, config, tracker precision, and initialization protocol remain M1.'
        ]
    }
    (output_dir / 'trace_summary.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
