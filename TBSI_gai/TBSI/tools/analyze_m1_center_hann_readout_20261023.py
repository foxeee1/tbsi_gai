#!/usr/bin/env python3
"""Compare raw and Hann-weighted CENTER decoding on paired saved maps."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_m1_center_geometry_crossover_20261022 import decode, iou_xywh, target_cell


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows):
    output = {}
    for model in ('parent', 'candidate'):
        model_rows = [row for row in rows if row['model'] == model]
        output[model] = {}
        for phase in ('pre_onset', 'severe_onset', 'all'):
            subset = [row for row in model_rows if phase == 'all' or row['phase'] == phase]
            paired = {}
            for event in {row['event_id'] for row in subset}:
                values = {row['window']: row for row in subset if row['event_id'] == event}
                if len(values) == 2:
                    paired[event] = values
            deltas = [v['hann']['iou'] - v['ones']['iou'] for v in paired.values()]
            rank_deltas = [v['hann']['gt_rank'] - v['ones']['gt_rank'] for v in paired.values()]
            by_sequence = {}
            for event, values in paired.items():
                sequence = event.split(':')[0]
                by_sequence.setdefault(sequence, []).append(values['hann']['iou'] - values['ones']['iou'])
            sequence_deltas = np.asarray([np.mean(values) for values in by_sequence.values()], dtype=np.float64)
            rng = np.random.RandomState(42)
            bootstrap = np.mean(rng.choice(sequence_deltas, (10000, len(sequence_deltas)), replace=True), axis=1)
            output[model][phase] = {
                'n': len(paired),
                'n_sequences': len(sequence_deltas),
                'ones_mean_iou': float(np.mean([v['ones']['iou'] for v in paired.values()])),
                'hann_mean_iou': float(np.mean([v['hann']['iou'] for v in paired.values()])),
                'mean_iou_delta_hann_minus_ones': float(np.mean(deltas)),
                'sequence_cluster_bootstrap_95ci_mean_iou_delta': [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])],
                'median_iou_delta_hann_minus_ones': float(np.median(deltas)),
                'iou_win_tie_loss': [int(sum(d > 1e-12 for d in deltas)),
                                     int(sum(abs(d) <= 1e-12 for d in deltas)),
                                     int(sum(d < -1e-12 for d in deltas))],
                'mean_gt_rank_ones': float(np.mean([v['ones']['gt_rank'] for v in paired.values()])),
                'mean_gt_rank_hann': float(np.mean([v['hann']['gt_rank'] for v in paired.values()])),
                'mean_gt_rank_delta_hann_minus_ones': float(np.mean(rank_deltas)),
                'peak_changed_count': sum(v['ones']['selected_cell'] != v['hann']['selected_cell']
                                          for v in paired.values()),
                'mean_center_error_ones_px': float(np.mean([v['ones']['center_error_px'] for v in paired.values()])),
                'mean_center_error_hann_px': float(np.mean([v['hann']['center_error_px'] for v in paired.values()])),
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--search-size', type=int, default=256)
    args = parser.parse_args()
    parent_path, candidate_path = Path(args.parent), Path(args.candidate)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    parent = np.load(parent_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    if parent['event_id'].tolist() != candidate['event_id'].tolist():
        raise RuntimeError('event order mismatch')
    for key in ('gt_box', 'crop_xywh', 'state_before', 'gt_mask', 'resize_factor', 'hann'):
        if not np.array_equal(parent[key], candidate[key]):
            raise RuntimeError('paired input mismatch: {}'.format(key))

    rows = []
    for model_name, data in (('parent', parent), ('candidate', candidate)):
        hann = data['hann'].astype(np.float64)
        ones = np.ones_like(hann)
        for i, event_id in enumerate(data['event_id'].tolist()):
            gt = data['gt_box'][i].astype(np.float64)
            crop = data['crop_xywh'][i].astype(np.float64)
            state = data['state_before'][i].astype(np.float64)
            mask = data['gt_mask'][i].astype(bool)
            size = data['size_map'][i].astype(np.float64)
            offset = data['offset_map'][i].astype(np.float64)
            score = data['score_map'][i].astype(np.float64)
            cell = target_cell(gt, crop, mask.shape[0])
            phase = event_id.split(':')[1]
            for window_name, window in (('ones', ones), ('hann', hann)):
                response = score * window
                y, x = np.unravel_index(int(np.argmax(response)), response.shape)
                gt_peak = float(response[mask].max())
                bg_peak = float(response[~mask].max())
                rank = 1 + int(np.sum(response[~mask] > gt_peak))
                box, selected = decode(score, size, offset, window, state,
                                       data['resize_factor'][i], args.search_size)
                center_error = float(np.hypot(
                    box[0] + box[2] * 0.5 - (gt[0] + gt[2] * 0.5),
                    box[1] + box[3] * 0.5 - (gt[1] + gt[3] * 0.5)))
                rows.append({
                    'event_id': event_id, 'phase': phase, 'model': model_name,
                    'window': window_name, 'selected_x': int(x), 'selected_y': int(y),
                    'selected_cell': '{}:{}'.format(y, x), 'gt_rank': rank,
                    'gt_hard_bg_margin': gt_peak - bg_peak,
                    'selected_cell_in_gt': bool(mask[y, x]),
                    'bbox_xywh': json.dumps(box.tolist()), 'iou': iou_xywh(box, gt),
                    'center_error_px': center_error,
                    'size_error_l1_px': float(abs(box[2] - gt[2]) + abs(box[3] - gt[3])),
                    'gt_center_cell': '{}:{}'.format(cell[0], cell[1]),
                })

    fields = list(rows[0])
    with (out / 'per_state.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'n_states': len(parent['event_id']),
        'summary': summarize(rows),
        'inputs': {
            'parent_head_maps': str(parent_path), 'parent_sha256': sha256(parent_path),
            'candidate_head_maps': str(candidate_path), 'candidate_sha256': sha256(candidate_path),
        },
        'decoder': 'CENTER score * window argmax; gather same-arm size/offset; exact saved crossover decoder',
        'scope': 'offline fixed-state calibration diagnostic; no model inference or tracking',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
