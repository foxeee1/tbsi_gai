#!/usr/bin/env python3
"""Pair the adaptive AMCM full-244 result with the protected M1 run."""
import csv
import json
import os
import pickle

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M1_EVAL = os.path.join(ROOT, 'output/test/result_plots/M1_S1_FULL244_CONFIRM_20261014/eval_data.pkl')
ADAPTIVE_EVAL = os.path.join(ROOT, 'output/test/result_plots/lasher_test/eval_data.pkl')
M1_RESULTS = os.path.join(ROOT, 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_1030')
ADAPTIVE_RESULTS = os.path.join(ROOT, 'output/test/tracking_results/tbsi_track/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021_3')
DATASET = os.path.join(ROOT, '../data/lasher/testingset')
OUT = os.path.join(ROOT, 'output/experiments/M1_AMCM_ADAPTIVE_FULL244_DIAG_20261021')


def load_pickle(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)


def sequence_scores(data, sequence_index, tracker_index):
    overlap_curve = np.asarray(
        data['ave_success_rate_plot_overlap'][sequence_index][tracker_index], dtype=np.float64)
    center_curve = np.asarray(
        data['ave_success_rate_plot_center'][sequence_index][tracker_index], dtype=np.float64)
    norm_curve = np.asarray(
        data['ave_success_rate_plot_center_norm'][sequence_index][tracker_index], dtype=np.float64)
    average_overlap = float(data['avg_overlap_all'][sequence_index][tracker_index])
    return {
        'auc': float(overlap_curve.mean() * 100.0),
        'op50': float(overlap_curve[10] * 100.0),
        'op75': float(overlap_curve[15] * 100.0),
        'precision': float(center_curve[20] * 100.0),
        'norm_precision': float(norm_curve[20] * 100.0),
        'average_overlap': average_overlap * 100.0,
    }


def bbox_iou(boxes, ground_truth):
    x1 = np.maximum(boxes[:, 0], ground_truth[:, 0])
    y1 = np.maximum(boxes[:, 1], ground_truth[:, 1])
    x2 = np.minimum(boxes[:, 0] + boxes[:, 2], ground_truth[:, 0] + ground_truth[:, 2])
    y2 = np.minimum(boxes[:, 1] + boxes[:, 3], ground_truth[:, 1] + ground_truth[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = boxes[:, 2] * boxes[:, 3] + ground_truth[:, 2] * ground_truth[:, 3] - intersection
    return intersection / np.maximum(union, 1e-12)


def load_boxes(path):
    boxes = np.loadtxt(path, dtype=np.float64)
    return np.atleast_2d(boxes)


def paired_bootstrap(values, seed=20261021, samples=50000):
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.randint(0, n, size=(count, n))
        means[start:start + count] = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 50.0, 97.5])]


def main():
    m1 = load_pickle(M1_EVAL)
    adaptive = load_pickle(ADAPTIVE_EVAL)
    m1_names = m1['sequences']
    adaptive_index = {name: i for i, name in enumerate(adaptive['sequences'])}
    if set(m1_names) != set(adaptive_index) or len(m1_names) != 244:
        raise RuntimeError('M1/adaptive sequence manifests do not match 244/244')
    if m1['trackers'][0]['run_id'] not in ('1030', 1030):
        raise RuntimeError('Expected protected M1 comparison run 1030')
    if adaptive['trackers'][0]['param'] != 'M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021':
        raise RuntimeError('Unexpected adaptive result cache')

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for i, name in enumerate(m1_names):
        j = adaptive_index[name]
        baseline = sequence_scores(m1, i, 0)
        candidate = sequence_scores(adaptive, j, 0)
        gt_path = os.path.join(DATASET, name, 'init.txt')
        gt = np.atleast_2d(np.loadtxt(gt_path, delimiter=',', dtype=np.float64))
        m1_boxes = load_boxes(os.path.join(M1_RESULTS, name + '.txt'))
        adaptive_boxes = load_boxes(os.path.join(ADAPTIVE_RESULTS, name + '.txt'))
        frame_count = min(len(gt), len(m1_boxes), len(adaptive_boxes))
        gt, m1_boxes, adaptive_boxes = gt[:frame_count], m1_boxes[:frame_count], adaptive_boxes[:frame_count]
        m1_iou = bbox_iou(m1_boxes, gt)
        adaptive_iou = bbox_iou(adaptive_boxes, gt)
        updates = adaptive_boxes[1:]
        rows.append({
            'sequence': name,
            'frames': frame_count,
            **{'m1_' + key: value for key, value in baseline.items()},
            **{'adaptive_' + key: value for key, value in candidate.items()},
            **{'delta_' + key: candidate[key] - baseline[key] for key in baseline},
            'm1_first_update_iou': float(m1_iou[1]) if frame_count > 1 else float('nan'),
            'adaptive_first_update_iou': float(adaptive_iou[1]) if frame_count > 1 else float('nan'),
            'm1_first5_update_iou': float(m1_iou[1:min(frame_count, 6)].mean()),
            'adaptive_first5_update_iou': float(adaptive_iou[1:min(frame_count, 6)].mean()),
            'adaptive_small_box_fraction': float(((updates[:, 2] <= 10.0) | (updates[:, 3] <= 10.0)).mean()) if len(updates) else 0.0,
            'adaptive_border_fraction': float(((updates[:, 0] <= 1.0) | (updates[:, 1] <= 1.0)).mean()) if len(updates) else 0.0,
            'adaptive_iou_below_005_fraction': float((adaptive_iou[1:] < 0.05).mean()) if frame_count > 1 else 0.0,
        })

    fields = list(rows[0])
    csv_path = os.path.join(OUT, 'per_sequence_m1_vs_adaptive.csv')
    with open(csv_path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = ['auc', 'precision', 'norm_precision', 'op50', 'op75', 'average_overlap']
    deltas = {key: np.asarray([row['delta_' + key] for row in rows]) for key in metrics}
    auc_delta = deltas['auc']
    catastrophic = [row for row in rows if row['delta_auc'] < -20.0]
    other = [row for row in rows if row['delta_auc'] >= -20.0]
    all_frames = sum(row['frames'] for row in rows)
    summary = {
        'parent': 'TBSI_E_MHSA_FULL_4090D_seed42 runid 1030, protected epoch15 checkpoint',
        'candidate': 'M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021 runid 3, epoch3 checkpoint',
        'sequence_count': len(rows),
        'frame_count': all_frames,
        'paired_metric_mean': {
            key: {
                'm1': float(np.mean([row['m1_' + key] for row in rows])),
                'adaptive': float(np.mean([row['adaptive_' + key] for row in rows])),
                'delta': float(deltas[key].mean()),
            } for key in metrics
        },
        'auc_delta': {
            'wins': int((auc_delta > 1e-8).sum()),
            'losses': int((auc_delta < -1e-8).sum()),
            'ties': int((np.abs(auc_delta) <= 1e-8).sum()),
            'median': float(np.median(auc_delta)),
            'paired_bootstrap_95_ci': paired_bootstrap(auc_delta),
            'loss_below_20_points': int((auc_delta < -20.0).sum()),
            'loss_below_50_points': int((auc_delta < -50.0).sum()),
            'gain_above_5_points': int((auc_delta > 5.0).sum()),
        },
        'catastrophic_cohort_delta_auc_lt_minus20': {
            'n': len(catastrophic),
            'm1_auc_mean': float(np.mean([row['m1_auc'] for row in catastrophic])),
            'adaptive_auc_mean': float(np.mean([row['adaptive_auc'] for row in catastrophic])),
            'first_update_m1_iou_mean': float(np.mean([row['m1_first_update_iou'] for row in catastrophic])),
            'first_update_adaptive_iou_mean': float(np.mean([row['adaptive_first_update_iou'] for row in catastrophic])),
            'first5_m1_iou_mean': float(np.mean([row['m1_first5_update_iou'] for row in catastrophic])),
            'first5_adaptive_iou_mean': float(np.mean([row['adaptive_first5_update_iou'] for row in catastrophic])),
            'small_box_frame_fraction_mean': float(np.mean([row['adaptive_small_box_fraction'] for row in catastrophic])),
            'border_frame_fraction_mean': float(np.mean([row['adaptive_border_fraction'] for row in catastrophic])),
            'near_zero_iou_frame_fraction_mean': float(np.mean([row['adaptive_iou_below_005_fraction'] for row in catastrophic])),
        },
        'remaining_cohort_delta_auc_ge_minus20': {
            'n': len(other),
            'm1_auc_mean': float(np.mean([row['m1_auc'] for row in other])),
            'adaptive_auc_mean': float(np.mean([row['adaptive_auc'] for row in other])),
            'first_update_m1_iou_mean': float(np.mean([row['m1_first_update_iou'] for row in other])),
            'first_update_adaptive_iou_mean': float(np.mean([row['adaptive_first_update_iou'] for row in other])),
            'first5_m1_iou_mean': float(np.mean([row['m1_first5_update_iou'] for row in other])),
            'first5_adaptive_iou_mean': float(np.mean([row['adaptive_first5_update_iou'] for row in other])),
            'small_box_frame_fraction_mean': float(np.mean([row['adaptive_small_box_fraction'] for row in other])),
            'border_frame_fraction_mean': float(np.mean([row['adaptive_border_fraction'] for row in other])),
            'near_zero_iou_frame_fraction_mean': float(np.mean([row['adaptive_iou_below_005_fraction'] for row in other])),
        },
        'early_collapse_count': int(sum(row['adaptive_first_update_iou'] < 0.05 and row['m1_first_update_iou'] > 0.5 for row in rows)),
        'largest_losses': sorted(rows, key=lambda row: row['delta_auc'])[:20],
        'largest_gains': sorted(rows, key=lambda row: row['delta_auc'], reverse=True)[:20],
        'inputs': {
            'm1_eval_cache': os.path.relpath(M1_EVAL, ROOT),
            'adaptive_eval_cache': os.path.relpath(ADAPTIVE_EVAL, ROOT),
            'm1_results': os.path.relpath(M1_RESULTS, ROOT),
            'adaptive_results': os.path.relpath(ADAPTIVE_RESULTS, ROOT),
            'dataset': os.path.relpath(DATASET, ROOT),
        },
        'limitations': [
            'M1 epoch15 and the candidate epoch3-after-M1 checkpoint are not a matched training-progress comparison.',
            'Full-sequence metrics establish closed-loop failure patterns; they do not identify the exact internal operator that first changes the frame-two prediction.',
            'The evaluator caches and prediction directories are matched by the official 244 sequence names; sequence bootstrap treats sequences as paired units.',
        ],
    }
    json_path = os.path.join(OUT, 'summary.json')
    with open(json_path, 'w') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({key: summary[key] for key in (
        'sequence_count', 'frame_count', 'paired_metric_mean', 'auc_delta',
        'catastrophic_cohort_delta_auc_lt_minus20', 'remaining_cohort_delta_auc_ge_minus20',
        'early_collapse_count', 'largest_losses', 'largest_gains')}, indent=2, ensure_ascii=False))
    print('CSV:', csv_path)
    print('JSON:', json_path)


if __name__ == '__main__':
    main()
