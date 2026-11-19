#!/usr/bin/env python3
"""Matched M1-weight evaluation on the frozen archived-low-AUC sequence set."""
import argparse
import csv
import hashlib
import json
import math
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
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track' / (NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
SCENE_CSV = ROOT / 'output/experiments/M1_FULL244_SCENE_MINING_20261009/per_sequence_scene.csv'
CONFIG_SHA = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pair(sequence, index):
    rgb = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    tir = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if rgb is None or tir is None or rgb.shape[:2] != tir.shape[:2]:
        raise RuntimeError('missing/misaligned image: {} frame {}'.format(sequence.name, index))
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)


def iou_xywh(a, b):
    if min(a[2], a[3], b[2], b[3]) <= 0:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(1e-12, a[2] * a[3] + b[2] * b[3] - intersection)


def metrics(predictions, ground_truth):
    overlap, center, normalized, _valid = calc_seq_err_robust(
        torch.tensor(predictions, dtype=torch.float64),
        torch.tensor(ground_truth, dtype=torch.float64), 'lasher')
    auc = float(torch.stack([(overlap > threshold).double().mean()
                             for threshold in torch.arange(0, 1.0001, 0.05)]).mean()) * 100
    precision = float((center <= 20).double().mean()) * 100
    norm_precision = float((normalized <= 0.2).double().mean()) * 100
    op50 = float((overlap > 0.5).double().mean()) * 100
    op75 = float((overlap > 0.75).double().mean()) * 100
    return dict(auc=auc, precision=precision, norm_precision=norm_precision,
                op50=op50, op75=op75)


def signal_capture(tracker):
    capture = {}

    def fusion_hook(_module, inputs, output):
        x = inputs[0].detach().float()
        rgb, tir = x.chunk(2, dim=1)
        capture['rgb_norm'] = float(rgb.norm().item())
        capture['tir_norm'] = float(tir.norm().item())
        capture['fused_norm'] = float(output.detach().float().norm().item())
        capture['branch_cosine'] = float(torch.nn.functional.cosine_similarity(
            rgb.flatten(1), tir.flatten(1)).item())

    def head_hook(_module, _inputs, output):
        capture['score'] = output[0].detach().float()[0, 0].cpu().numpy()

    hooks = [tracker.network.tbsi_fuse_search.register_forward_hook(fusion_hook),
             tracker.network.box_head.register_forward_hook(head_hook)]
    return capture, hooks


def information_row(capture, name, frame, state_before, gt, pred, search_factor, window):
    if min(gt[2], gt[3]) <= 0:
        return None
    side = int(math.ceil(math.sqrt(float(state_before[2]) * float(state_before[3])) *
                         float(search_factor)))
    cx = state_before[0] + state_before[2] / 2.0
    cy = state_before[1] + state_before[3] / 2.0
    x0, y0 = round(cx - side / 2.0), round(cy - side / 2.0)
    gcx, gcy = gt[0] + gt[2] / 2.0, gt[1] + gt[3] / 2.0
    gx, gy = (gcx - x0) / side * 16, (gcy - y0) / side * 16
    score = capture.get('score')
    if score is None:
        raise RuntimeError('missing CENTER capture')
    windowed = score * window
    peak_y, peak_x = np.unravel_index(int(windowed.argmax()), windowed.shape)
    inside = 0 <= gx < 16 and 0 <= gy < 16
    gt_row = min(15, max(0, int(gy)))
    gt_col = min(15, max(0, int(gx)))
    rank = int(np.count_nonzero(windowed > windowed[gt_row, gt_col]) + 1) if inside else 257
    probability = score.reshape(-1).astype(np.float64)
    probability = probability / max(1e-12, probability.sum())
    entropy = float(-(probability * np.log(np.maximum(probability, 1e-12))).sum() /
                    math.log(probability.size))
    row = {'sequence': name, 'frame': frame, 'gt_inside_crop': int(inside),
           'gt_rank_hann': rank, 'peak_distance_tokens': float(math.hypot(peak_x + 0.5 - gx,
                                                                          peak_y + 0.5 - gy)),
           'score_entropy': entropy, 'iou': iou_xywh(pred, gt)}
    row.update({key: capture[key] for key in ('rgb_norm', 'tir_norm', 'fused_norm',
                                             'branch_cosine')})
    gate = capture.get('gate')
    row['gate_prediction'] = gate['prediction'] if gate else 'NA'
    row['gate_direction'] = gate['direction'] if gate else 'NA'
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--signal', choices=('off', 'fixed_rgb', 'fixed_tir', 's1', 's2', 's3', 's1s2', 's1s2s3'), default='off')
    parser.add_argument('--artifact-dir', default=None)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--max-sequences', type=int, default=62,
                        help='preflight only when less than 62; final table requires 62')
    parser.add_argument('--sequence-list', default=None,
                        help='explicit frozen sequence list; overrides archived low-AUC selection')
    parser.add_argument('--info-stride', type=int, default=32)
    args = parser.parse_args()
    if args.signal not in ('off', 'fixed_rgb', 'fixed_tir') and (not args.artifact_dir or
                                  not (Path(args.artifact_dir) / 'signal_calibration.json').is_file()):
        raise RuntimeError('candidate signal artifact missing')
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output directory: ' + str(output))
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('protected M1 hash mismatch')
    if not torch.cuda.is_available() or args.info_stride < 1 or not 1 <= args.max_sequences <= 62:
        raise RuntimeError('invalid CUDA/stride/sequence request')

    with SCENE_CSV.open(newline='') as stream:
        scene_rows = {row['sequence']: row for row in csv.DictReader(stream)}
    if args.sequence_list:
        selected = [line.strip() for line in Path(args.sequence_list).read_text().splitlines()
                    if line.strip()]
        if len(selected) != len(set(selected)) or not selected:
            raise RuntimeError('invalid explicit sequence list')
        if any(name not in scene_rows for name in selected):
            raise RuntimeError('unknown sequence in explicit list')
    else:
        selected = sorted(name for name, row in scene_rows.items()
                          if float(row['m1_auc']) < 40)
        if len(selected) != 62:
            raise RuntimeError('archived low-AUC membership changed: ' + str(len(selected)))
    output.mkdir(parents=True)
    manifest = {'id': 'M1-SINGLE-SIGNAL-20261012-' + ('EXPLICIT20-' if args.sequence_list else 'LOWAUC-') + args.signal,
                'parent': NAME, 'signal': args.signal, 'selection': 'archived M1 sequence AUC<40',
                'all_selected_sequences': selected, 'run_sequences': selected[:args.max_sequences],
                'config_sha256': CONFIG_SHA, 'checkpoint_sha256': CHECKPOINT_SHA,
                'scene_csv_sha256': sha256(SCENE_CSV), 'info_stride': args.info_stride,
                'comparison_scope': 'matched current-code rerun; not archived M1 frame parity'}
    if args.sequence_list:
        manifest['selection'] = 'explicit frozen normal-scene control list'
    if args.signal not in ('off', 'fixed_rgb', 'fixed_tir'):
        manifest['signal_artifact_dir'] = str(Path(args.artifact_dir).resolve())
        manifest['signal_calibration_sha256'] = sha256(Path(args.artifact_dir) / 'signal_calibration.json')
        manifest['alpha'] = args.alpha
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'sequence_list.txt').write_text('\n'.join(selected[:args.max_sequences]) + '\n')
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_PERTURBATION'] = ''
    os.environ['TBSI_M1_TRACE_DIR'] = str(output / 'trace')
    os.environ['TBSI_TRACE_SAVE_RAW'] = '0'
    os.environ.pop('TBSI_CHECKPOINT_OVERRIDE', None)
    os.environ['TBSI_SINGLE_SIGNAL_MODE'] = args.signal
    if args.signal not in ('off', 'fixed_rgb', 'fixed_tir'):
        os.environ['TBSI_SINGLE_SIGNAL_ARTIFACT_DIR'] = str(Path(args.artifact_dir).resolve())
        os.environ['TBSI_SINGLE_SIGNAL_ALPHA'] = str(args.alpha)
    else:
        os.environ.pop('TBSI_SINGLE_SIGNAL_ARTIFACT_DIR', None)
        os.environ.pop('TBSI_SINGLE_SIGNAL_ALPHA', None)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters(NAME)
    params.checkpoint = str(CHECKPOINT)
    params.debug = 0
    params.save_all_boxes = False
    tracker = TBSITrack(params, 'lasher_test')
    capture, hooks = signal_capture(tracker)
    window = tracker.output_window[0, 0].float().cpu().numpy()
    dataset = LasHeRDataset(split='testingset',
                            sequence_list_file=str(output / 'sequence_list.txt')).get_sequence_list()
    if sorted(sequence.name for sequence in dataset) != sorted(selected[:args.max_sequences]):
        raise RuntimeError('dataset and frozen sequence list differ')
    rows, info_rows = [], []
    started = time.time()
    try:
        for index, sequence in enumerate(dataset, 1):
            sequence_started = time.time()
            if len(sequence.frames[0]) != len(sequence.frames[1]):
                raise RuntimeError('RGB/TIR frame mismatch: ' + sequence.name)
            gt = sequence.ground_truth_rect
            tracker.initialize(read_pair(sequence, 0), {
                'init_bbox': gt[0], 'sequence_name': sequence.name})
            predictions = [list(map(float, gt[0]))]
            for frame in range(1, len(gt)):
                capture.clear()
                before = list(map(float, tracker.state))
                tracker.track(read_pair(sequence, frame), {
                    'gt_bbox': list(map(float, gt[frame])),
                    'frame_num': frame, 'sequence_name': sequence.name})
                gate = getattr(tracker.network, 'single_signal_gate', None)
                if gate is not None:
                    capture['gate'] = gate.last_signal
                pred = list(map(float, tracker.state))
                predictions.append(pred)
                if frame == 1 or frame % args.info_stride == 0:
                    row = information_row(capture, sequence.name, frame, before,
                                          list(map(float, gt[frame])), pred,
                                          params.search_factor, window)
                    if row is not None:
                        info_rows.append(row)
            result = {'sequence': sequence.name, 'frames': len(gt),
                      'fps': (len(gt) - 1) / max(1e-9, time.time() - sequence_started)}
            result.update(metrics(predictions, gt))
            rows.append(result)
            np.savetxt(str(output / (sequence.name + '.txt')),
                       np.asarray(predictions), fmt='%.6f', delimiter='\t')
            print('{}/{} {} AUC={:.3f}'.format(index, len(dataset), sequence.name,
                                                result['auc']), flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    for name, data in (('per_sequence.csv', rows), ('information_flow.csv', info_rows)):
        if not data:
            raise RuntimeError('empty output: ' + name)
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {'id': manifest['id'], 'status': 'COMPLETE', 'n_sequences': len(rows),
               'n_frames': sum(row['frames'] for row in rows), 'n_info_samples': len(info_rows),
               'macro': {key: float(np.mean([row[key] for row in rows]))
                         for key in ('auc', 'precision', 'norm_precision', 'op50', 'op75')},
               'information_flow': {key: float(np.mean([row[key] for row in info_rows]))
                                    for key in ('gt_rank_hann', 'peak_distance_tokens',
                                                'score_entropy', 'rgb_norm', 'tir_norm',
                                                'fused_norm', 'branch_cosine', 'iou')},
               'elapsed_seconds': time.time() - started,
               'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
               'torch': torch.__version__, 'cuda': torch.version.cuda}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
