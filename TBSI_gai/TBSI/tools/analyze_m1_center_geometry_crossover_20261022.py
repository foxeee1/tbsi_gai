#!/usr/bin/env python3
"""Counterfactual CENTER score/geometry decoding on paired fixed-state maps."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def iou_xywh(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def decode(score, size, offset, hann, state, resize, search_size, forced_cell=None):
    response = score * hann
    if forced_cell is None:
        index = int(np.argmax(response))
        y, x = divmod(index, response.shape[1])
    else:
        y, x = forced_cell
    h, w = response.shape
    cx = (x + float(offset[0, y, x])) / w
    cy = (y + float(offset[1, y, x])) / h
    bw = float(size[0, y, x])
    bh = float(size[1, y, x])
    scale = float(search_size) / float(resize)
    cx *= scale
    cy *= scale
    bw *= scale
    bh *= scale
    state_cx = float(state[0] + 0.5 * state[2])
    state_cy = float(state[1] + 0.5 * state[3])
    side = float(search_size) / float(resize)
    cx += state_cx - 0.5 * side
    cy += state_cy - 0.5 * side
    return np.asarray([cx - 0.5 * bw, cy - 0.5 * bh, bw, bh], dtype=np.float64), (y, x)


def target_cell(gt, crop, grid_size):
    gx = (gt[0] + 0.5 * gt[2] - crop[0]) * grid_size / crop[2] - 0.5
    gy = (gt[1] + 0.5 * gt[3] - crop[1]) * grid_size / crop[3] - 0.5
    x = int(np.clip(round(gx), 0, grid_size - 1))
    y = int(np.clip(round(gy), 0, grid_size - 1))
    return (y, x)


def box_errors(box, gt):
    return {
        'center_error_px': float(np.hypot(
            (box[0] + 0.5 * box[2]) - (gt[0] + 0.5 * gt[2]),
            (box[1] + 0.5 * box[3]) - (gt[1] + 0.5 * gt[3]))),
        'width_abs_error_px': float(abs(box[2] - gt[2])),
        'height_abs_error_px': float(abs(box[3] - gt[3])),
        'width_relative_error': float(abs(box[2] - gt[2]) / max(float(gt[2]), 1e-6)),
        'height_relative_error': float(abs(box[3] - gt[3]) / max(float(gt[3]), 1e-6)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--search-size', type=int, default=256)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    parent = np.load(args.parent, allow_pickle=False)
    candidate = np.load(args.candidate, allow_pickle=False)
    p_ids = parent['event_id'].tolist()
    c_ids = candidate['event_id'].tolist()
    if p_ids != c_ids:
        raise RuntimeError('parent and candidate event order differs')
    for key in ('gt_box', 'crop_xywh', 'state_before', 'gt_mask'):
        if not np.array_equal(parent[key], candidate[key]):
            raise RuntimeError('fixed-state input mismatch: {}'.format(key))
    rows = []
    arms = {'parent': parent, 'candidate': candidate}
    for i, event_id in enumerate(p_ids):
        gt = parent['gt_box'][i].astype(np.float64)
        crop = parent['crop_xywh'][i].astype(np.float64)
        state = parent['state_before'][i].astype(np.float64)
        mask = parent['gt_mask'][i].astype(bool)
        hann = parent['hann'].astype(np.float64)
        cell = target_cell(gt, crop, mask.shape[0])
        gt_grid_x = (gt[0] + 0.5 * gt[2] - crop[0]) * mask.shape[1] / crop[2] - 0.5
        gt_grid_y = (gt[1] + 0.5 * gt[3] - crop[1]) * mask.shape[0] / crop[3] - 0.5
        for score_name, score_arm in arms.items():
            score = score_arm['score_map'][i].astype(np.float64)
            response = score * hann
            gt_peak = float(response[mask].max())
            bg_peak = float(response[~mask].max())
            gt_rank = 1 + int(np.sum(response[~mask] > gt_peak))
            top_y, top_x = np.unravel_index(int(np.argmax(response)), response.shape)
            raw_y, raw_x = np.unravel_index(int(np.argmax(score)), score.shape)
            raw_peak_in_gt = bool(mask[raw_y, raw_x])
            hann_peak_distance = float(np.hypot(top_x - gt_grid_x, top_y - gt_grid_y))
            raw_peak_distance = float(np.hypot(raw_x - gt_grid_x, raw_y - gt_grid_y))
            for geom_name, geom_arm in arms.items():
                size = geom_arm['size_map'][i].astype(np.float64)
                offset = geom_arm['offset_map'][i].astype(np.float64)
                box, peak_cell = decode(score, size, offset, hann, state,
                                        parent['resize_factor'][i], args.search_size)
                oracle_box, _ = decode(score, size, offset, hann, state,
                                       parent['resize_factor'][i], args.search_size,
                                       forced_cell=cell)
                rows.append({
                    'event_id': event_id,
                    'score_source': score_name,
                    'geometry_source': geom_name,
                    'decode_mode': 'predicted_peak',
                    'center_gt_hard_bg_margin': gt_peak - bg_peak,
                    'center_gt_rank': gt_rank,
                    'center_top1_in_gt': bool(mask[top_y, top_x]),
                    'center_top1_x': int(top_x), 'center_top1_y': int(top_y),
                    'raw_argmax_in_gt': raw_peak_in_gt,
                    'raw_argmax_x': int(raw_x), 'raw_argmax_y': int(raw_y),
                    'raw_argmax_distance_to_gt_center_cells': raw_peak_distance,
                    'hann_argmax_distance_to_gt_center_cells': hann_peak_distance,
                    'raw_hann_argmax_displacement_cells': float(np.hypot(top_x - raw_x, top_y - raw_y)),
                    'selected_x': int(peak_cell[1]), 'selected_y': int(peak_cell[0]),
                    'predicted_bbox_xywh': json.dumps(box.tolist()),
                    'iou': iou_xywh(box, gt),
                    **box_errors(box, gt),
                })
                rows.append({
                    'event_id': event_id,
                    'score_source': score_name,
                    'geometry_source': geom_name,
                    'decode_mode': 'gt_center_cell_oracle',
                    'center_gt_hard_bg_margin': gt_peak - bg_peak,
                    'center_gt_rank': gt_rank,
                    'center_top1_in_gt': bool(mask[top_y, top_x]),
                    'center_top1_x': int(top_x), 'center_top1_y': int(top_y),
                    'selected_x': int(cell[1]), 'selected_y': int(cell[0]),
                    'predicted_bbox_xywh': json.dumps(oracle_box.tolist()),
                    'iou': iou_xywh(oracle_box, gt),
                    **box_errors(oracle_box, gt),
                })
    csv_path = out / 'crossover_per_event.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for score_name in arms:
        for geom_name in arms:
            for mode in ('predicted_peak', 'gt_center_cell_oracle'):
                values = [row['iou'] for row in rows if row['score_source'] == score_name
                          and row['geometry_source'] == geom_name and row['decode_mode'] == mode]
                summary['{}__{}__{}'.format(score_name, geom_name, mode)] = {
                    'n': len(values), 'mean_iou': float(np.mean(values)),
                    'median_iou': float(np.median(values)),
                }
    summary['center_peak_diagnostics'] = {}
    for score_name in arms:
        selected = [row for row in rows if row['score_source'] == score_name
                    and row['geometry_source'] == score_name
                    and row['decode_mode'] == 'predicted_peak']
        for phase in ('pre_onset', 'severe_onset', 'all'):
            subset = [row for row in selected if phase == 'all' or
                      row['event_id'].split(':')[1] == phase]
            summary['center_peak_diagnostics']['{}__{}'.format(score_name, phase)] = {
                'n': len(subset),
                'raw_argmax_in_gt_count': sum(row['raw_argmax_in_gt'] for row in subset),
                'hann_argmax_in_gt_count': sum(row['center_top1_in_gt'] for row in subset),
                'mean_raw_argmax_distance_to_gt_center_cells': float(np.mean(
                    [row['raw_argmax_distance_to_gt_center_cells'] for row in subset])),
                'mean_hann_argmax_distance_to_gt_center_cells': float(np.mean(
                    [row['hann_argmax_distance_to_gt_center_cells'] for row in subset])),
                'mean_raw_hann_argmax_displacement_cells': float(np.mean(
                    [row['raw_hann_argmax_displacement_cells'] for row in subset])),
            }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'events': len(p_ids), 'rows': len(rows), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
