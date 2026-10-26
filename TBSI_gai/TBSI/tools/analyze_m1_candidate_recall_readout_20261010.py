#!/usr/bin/env python3
"""Fixed-state D1 candidate-recall and D2 CENTER regression-readout analysis."""
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

os.environ['TBSI_INFER_FP16'] = '0'
os.environ['TBSI_PERTURBATION'] = ''

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_m1_search_stage_20261009 import (
    CONFIG, CHECKPOINT, EXPECTED_CONFIG, EXPECTED_CHECKPOINT, iou,
    predicted_box, read_pair, select_frames, sha256, target_masks)
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box

DEFAULT_COHORT = ROOT / 'experiments/diagnostics/m1_candidate_recall_readout_v1'
DEFAULT_TRACE = ROOT / 'output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010'
DEFAULT_OUT = ROOT / 'output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/analysis'
K_VALUES = (1, 3, 5, 8, 16, 32, 256)


def decode_all_cells(out, state, image_shape, resize, search_size):
    size = out['size_map'].float()[0].permute(1, 2, 0).reshape(256, 2).cpu().numpy()
    offset = out['offset_map'].float()[0].permute(1, 2, 0).reshape(256, 2).cpu().numpy()
    index = np.arange(256)
    x, y = index % 16, index // 16
    cx = (x + offset[:, 0]) * (search_size / 16.0 / resize)
    cy = (y + offset[:, 1]) * (search_size / 16.0 / resize)
    width = size[:, 0] * (search_size / resize)
    height = size[:, 1] * (search_size / resize)
    state_cx = state[0] + .5 * state[2]
    state_cy = state[1] + .5 * state[3]
    half = .5 * search_size / resize
    boxes = []
    for i in range(256):
        box = [float(cx[i] + state_cx - half - .5 * width[i]),
               float(cy[i] + state_cy - half - .5 * height[i]),
               float(width[i]), float(height[i])]
        boxes.append(clip_box(box, image_shape[0], image_shape[1], margin=10))
    return np.asarray(boxes), offset, size


def calculate_event(network, tracker, seq, trace, index, label, name):
    item = trace[index]
    frame = int(item['frame_id'])
    state = [float(v) for v in item['tracker_state_before']]
    gt = np.asarray(seq.ground_truth_rect[frame], dtype=np.float64)
    image = read_pair(seq, frame)
    crop, resize, amask = sample_target(image, state, tracker.params.search_factor,
                                        output_sz=tracker.params.search_size)
    search = tracker.preprocessor.process(crop, amask, half=False)
    search_pair = [search.tensors[:, :3], search.tensors[:, 3:]]
    gt_crop = transform_image_to_crop(
        torch.tensor(gt, dtype=torch.float32), torch.tensor(state, dtype=torch.float32), resize,
        torch.tensor([tracker.params.search_size, tracker.params.search_size]), normalize=True).numpy()
    masks = target_masks(gt_crop)
    target, _background, gx, gy, gcx, gcy = masks
    with torch.inference_mode():
        out = network(template=[tracker.z_dict1.tensors[:, :3], tracker.z_dict1.tensors[:, 3:]],
                      search=search_pair, ce_template_mask=tracker.box_mask_z,
                      template_fg_masks=(tracker.template_fg_mask, tracker.template_fg_mask))
    online = (out['score_map'].float() * tracker.output_window.float())[0, 0].reshape(-1).cpu().numpy()
    order = np.argsort(-online, kind='stable')
    boxes, offsets, sizes = decode_all_cells(out, state, image.shape, resize, tracker.params.search_size)
    cell_errors = np.hypot(order % 16 + offsets[order, 0] - gcx,
                           order // 16 + offsets[order, 1] - gcy)
    all_center_errors = np.hypot(np.arange(256) % 16 + offsets[:, 0] - gcx,
                                 np.arange(256) // 16 + offsets[:, 1] - gcy)
    ious = np.asarray([iou(box, gt) for box in boxes], dtype=np.float64)
    size_l1_tokens = (np.abs(sizes[:, 0] * 16.0 - float(gt_crop[2]) * 16.0) +
                      np.abs(sizes[:, 1] * 16.0 - float(gt_crop[3]) * 16.0)) / 2.0
    size_l1_px = size_l1_tokens * tracker.params.search_size / 16.0 / resize
    offset_l1_tokens = (np.abs(offsets[:, 0] - (gcx - np.arange(256) % 16)) +
                        np.abs(offsets[:, 1] - (gcy - np.arange(256) // 16))) / 2.0
    gt_index = gy * 16 + gx
    support_indices = np.flatnonzero(target.reshape(-1))
    ranks = np.empty(256, dtype=np.int32)
    ranks[order] = np.arange(1, 257)
    support_rank = int(ranks[support_indices].min())
    gt_rank = int(ranks[gt_index])
    local = np.asarray([yy * 16 + xx for yy in range(max(0, gy - 1), min(16, gy + 2))
                        for xx in range(max(0, gx - 1), min(16, gx + 2))], dtype=np.int32)
    local_best_iou = int(local[np.argmax(ious[local])])
    local_best_center = int(local[np.argmin(all_center_errors[local])])

    # The vectorized M1 argmax decode must equal the closed-loop tracker result.
    parity = float(np.max(np.abs(boxes[int(order[0])] - np.asarray(item['pred_box']))))
    if parity > .01:
        raise RuntimeError('M1 top1 replay parity failure {} frame {}: {} px'.format(name, frame, parity))

    record = {
        'sequence': name, 'frame': frame, 'state_label': label,
        'trace_iou': float(iou(item['pred_box'], gt)),
        'gt_cell_rank': gt_rank, 'gt_support_rank': support_rank,
        'gt_cell_in_top1': int(gt_rank == 1), 'gt_support_in_top1': int(support_rank == 1),
        'top1_center_error_tokens': float(all_center_errors[order[0]]),
        'top1_size_l1_tokens': float(size_l1_tokens[order[0]]),
        'top1_size_l1_px': float(size_l1_px[order[0]]),
        'top1_iou': float(ious[order[0]]),
        'gt_cell_center_error_tokens': float(all_center_errors[gt_index]),
        'gt_cell_offset_l1_tokens': float(offset_l1_tokens[gt_index]),
        'gt_cell_size_l1_tokens': float(size_l1_tokens[gt_index]),
        'gt_cell_size_l1_px': float(size_l1_px[gt_index]),
        'gt_cell_iou': float(ious[gt_index]),
        'gt_cell_minus_top1_iou': float(ious[gt_index] - ious[order[0]]),
        'gt_cell_minus_top1_center_error_tokens': float(
            all_center_errors[gt_index] - all_center_errors[order[0]]),
        'gt_cell_minus_top1_size_l1_tokens': float(
            size_l1_tokens[gt_index] - size_l1_tokens[order[0]]),
        'gt_3x3_best_iou': float(ious[local_best_iou]),
        'gt_3x3_best_iou_center_error_tokens': float(all_center_errors[local_best_iou]),
        'gt_3x3_best_iou_offset_l1_tokens': float(offset_l1_tokens[local_best_iou]),
        'gt_3x3_best_iou_size_l1_tokens': float(size_l1_tokens[local_best_iou]),
        'gt_3x3_min_center_error_tokens': float(all_center_errors[local_best_center]),
        'gt_3x3_min_center_error_iou': float(ious[local_best_center]),
        'gt_3x3_min_center_error_size_l1_tokens': float(size_l1_tokens[local_best_center]),
        'gt_3x3_median_size_l1_tokens': float(np.median(size_l1_tokens[local])),
        'top1_index': int(order[0]), 'gt_cell_index': int(gt_index),
        'top8_oracle_iou': float(np.max(ious[order[:8]])),
        'top32_oracle_iou': float(np.max(ious[order[:32]])),
        'allmap_oracle_iou': float(np.max(ious)),
        'parity_max_px': parity,
    }
    for k in K_VALUES:
        cells = order[:k]
        best_iou_idx = int(cells[np.argmax(ious[cells])])
        best_center_idx = int(cells[np.argmin(all_center_errors[cells])])
        record['gt_cell_recall_at_{}'.format(k)] = int(gt_rank <= k)
        record['gt_support_recall_at_{}'.format(k)] = int(support_rank <= k)
        record['oracle_iou_at_{}'.format(k)] = float(ious[best_iou_idx])
        record['oracle_iou_center_error_at_{}'.format(k)] = float(all_center_errors[best_iou_idx])
        record['oracle_iou_offset_l1_at_{}'.format(k)] = float(offset_l1_tokens[best_iou_idx])
        record['oracle_iou_size_l1_tokens_at_{}'.format(k)] = float(size_l1_tokens[best_iou_idx])
        record['oracle_iou_size_l1_px_at_{}'.format(k)] = float(size_l1_px[best_iou_idx])
        record['oracle_min_center_error_at_{}'.format(k)] = float(all_center_errors[best_center_idx])
    return record


def mean_ci_by_sequence(rows, key, label, seed=6102026, draws=4000):
    grouped = {}
    for row in rows:
        if label is None or row['state_label'] == label:
            grouped.setdefault(row['sequence'], []).append(float(row[key]))
    per_sequence = np.asarray([np.mean(values) for values in grouped.values()], dtype=np.float64)
    if per_sequence.size == 0:
        return {'n_sequences': 0, 'mean_sequence': None, 'ci95': [None, None]}
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, per_sequence.size, size=(draws, per_sequence.size))
    boot = per_sequence[indices].mean(axis=1)
    return {'n_sequences': int(per_sequence.size), 'mean_sequence': float(per_sequence.mean()),
            'ci95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def summarize(rows):
    result = {}
    for label in ('all', 'matched_good', 'pre_onset', 'severe_onset'):
        subset = rows if label == 'all' else [r for r in rows if r['state_label'] == label]
        result[label] = {'events': len(subset), 'sequences': len({r['sequence'] for r in subset})}
        for k in K_VALUES:
            key = 'gt_cell_recall_at_{}'.format(k)
            support = 'gt_support_recall_at_{}'.format(k)
            oracle = 'oracle_iou_at_{}'.format(k)
            center = 'oracle_min_center_error_at_{}'.format(k)
            result[label]['K{}'.format(k)] = {
                'gt_cell_recall_pooled': float(np.mean([r[key] for r in subset])) if subset else None,
                'gt_cell_recall_sequence_macro_ci95': mean_ci_by_sequence(
                    rows, key, None if label == 'all' else label),
                'gt_support_recall_pooled': float(np.mean([r[support] for r in subset])) if subset else None,
                'oracle_iou_median': float(np.median([r[oracle] for r in subset])) if subset else None,
                'oracle_iou_center_error_tokens_median': float(np.median(
                    [r['oracle_iou_center_error_at_{}'.format(k)] for r in subset])) if subset else None,
                'oracle_iou_offset_l1_tokens_median': float(np.median(
                    [r['oracle_iou_offset_l1_at_{}'.format(k)] for r in subset])) if subset else None,
                'oracle_iou_size_l1_tokens_median': float(np.median(
                    [r['oracle_iou_size_l1_tokens_at_{}'.format(k)] for r in subset])) if subset else None,
                'oracle_min_center_error_tokens_median': float(np.median([r[center] for r in subset])) if subset else None,
            }
        for key in ('top1_iou', 'top1_center_error_tokens', 'top1_size_l1_tokens',
                    'top1_size_l1_px', 'gt_cell_iou', 'gt_cell_minus_top1_iou',
                    'gt_cell_minus_top1_center_error_tokens', 'gt_cell_minus_top1_size_l1_tokens',
                    'gt_cell_center_error_tokens',
                    'gt_cell_offset_l1_tokens', 'gt_cell_size_l1_tokens', 'gt_cell_size_l1_px',
                    'gt_3x3_best_iou', 'gt_3x3_best_iou_center_error_tokens',
                    'gt_3x3_best_iou_offset_l1_tokens', 'gt_3x3_best_iou_size_l1_tokens',
                    'gt_3x3_min_center_error_tokens', 'gt_3x3_median_size_l1_tokens'):
            result[label][key + '_median'] = float(np.median([r[key] for r in subset])) if subset else None
        result[label]['gt_cell_center_error_sequence_macro_ci95'] = mean_ci_by_sequence(
            rows, 'gt_cell_center_error_tokens', None if label == 'all' else label)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-dir', default=str(DEFAULT_COHORT))
    parser.add_argument('--trace-dir', default=str(DEFAULT_TRACE))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    cohort_dir = Path(args.cohort_dir).resolve()
    trace_dir = Path(args.trace_dir).resolve() / 'trace'
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output: ' + str(output))
    if sha256(CONFIG) != EXPECTED_CONFIG or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters('TBSI_E_MHSA_FULL_4090D_seed42')
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'lasher_train')
    network = tracker.network.eval()
    seq_file = cohort_dir / 'sequence_list.txt'
    names = seq_file.read_text().splitlines()
    sequences = LasHeRDataset('trainingset', str(seq_file)).get_sequence_list()
    seq_map = {seq.name: seq for seq in sequences}
    records = []
    start = time.time()
    for seq_index, name in enumerate(names, 1):
        seq = seq_map[name]
        trace_file = trace_dir / name / 'frames.jsonl'
        trace = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
        selected = select_frames(trace, 0, 'onset-3-iou-0.5')
        selected = [(idx, 'matched_good' if label == 'good_only' else label)
                    for idx, label in selected if label in ('matched_good', 'pre_onset', 'severe_onset', 'good_only')]
        tracker.initialize(read_pair(seq, 0), {'init_bbox': seq.ground_truth_rect[0], 'sequence_name': name})
        for idx, label in selected:
            records.append(calculate_event(network, tracker, seq, trace, idx, label, name))
        print('sequence {}/{} {} events={} total={}'.format(
            seq_index, len(names), name, len(selected), len(records)), flush=True)
    if not records:
        raise RuntimeError('no fixed-state events selected')
    output.mkdir(parents=True)
    with (output / 'per_event.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    by_sequence = {}
    metric_keys = [key for key in records[0] if key not in ('sequence', 'frame', 'state_label')]
    for row in records:
        key = (row['sequence'], row['state_label'])
        by_sequence.setdefault(key, []).append(row)
    sequence_rows = []
    for (sequence, label), group in by_sequence.items():
        item = {'sequence': sequence, 'state_label': label, 'events': len(group)}
        for key in metric_keys:
            item[key] = float(np.mean([r[key] for r in group]))
        sequence_rows.append(item)
    with (output / 'per_sequence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(sequence_rows[0]))
        writer.writeheader(); writer.writerows(sequence_rows)
    summary = {
        'id': 'M1-CANDIDATE-RECALL-READOUT-20261010', 'status': 'DIAGNOSTIC-COMPLETE',
        'parent': 'Protected M1 TBSI_E_MHSA_FULL_4090D_seed42 epoch15',
        'cohort_manifest_sha256': sha256(cohort_dir / 'manifest.json'),
        'sequence_list_sha256': sha256(seq_file), 'config_sha256': EXPECTED_CONFIG,
        'checkpoint_sha256': EXPECTED_CHECKPOINT, 'events': len(records),
        'sequences': len({r['sequence'] for r in records}), 'event_counts': {
            label: sum(r['state_label'] == label for r in records)
            for label in ('matched_good', 'pre_onset', 'severe_onset')},
        'max_top1_trace_parity_px': max(r['parity_max_px'] for r in records),
        'K_values': list(K_VALUES), 'summary': summarize(records),
        'environment': {'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                        'torch': torch.__version__, 'cuda': torch.version.cuda},
        'elapsed_seconds': time.time() - start,
        'protocol': 'M1-only fixed-state replay; actual Hann score ordering; decode every grid cell with the same CENTER size/offset maps; exact GT-cell and GT-support recall; top-K oracle metrics are offline upper bounds only.',
        'limitations': ['Train-side screening cohort; not test-244 or official test benchmark.',
                        'Current worktree replay is not frame-equivalent to archived M1 predictions.',
                        'Top-K oracle selection is diagnostic only and not an online method.',
                        'Scene quantiles are metadata proxies, not official attributes.'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
