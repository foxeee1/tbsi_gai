#!/usr/bin/env python3
"""Summarize 30-sequence M1 flow diagnostics and archived-prediction drift."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return corr(ranks(x), ranks(y))


def iou_xywh(a, b):
    x, y, w, h = map(float, a)
    X, Y, W, H = map(float, b)
    iw = max(0.0, min(x + w, X + W) - max(x, X))
    ih = max(0.0, min(y + h, Y + H) - max(y, Y))
    inter = iw * ih
    union = max(0.0, w) * max(0.0, h) + max(0.0, W) * max(0.0, H) - inter
    return inter / union if union > 0 else 0.0


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--archive-dir', required=True)
    parser.add_argument('--full-per-sequence-csv', required=True)
    args = parser.parse_args()

    root, out = Path(args.root), Path(args.output_dir)
    trace_dir = root / 'closed_loop_trace' / 'trace'
    grad_dir = root / 'gradient_sampled_32'
    manifest = json.loads((root / 'manifest.json').read_text())
    full_rows = {row['sequence']: row for row in csv.DictReader(
        open(args.full_per_sequence_csv, newline=''))}
    grad_rows = list(csv.DictReader((grad_dir / 'per_frame.csv').open(newline='')))
    correlations = list(csv.DictReader((grad_dir / 'metric_correlations.csv').open(newline='')))
    gradient_summary = json.loads((grad_dir / 'summary.json').read_text())
    names = manifest.get('selected_sequences', manifest.get('sequences'))
    attributes_by_sequence = manifest.get(
        'selected_sequence_attributes', manifest.get('sequence_attributes', {}))
    expected_counts = gradient_summary.get('per_sequence_sample_counts', {})
    expected_samples = sum(expected_counts.values())
    if len(names) < 1 or len(grad_rows) != expected_samples:
        raise RuntimeError('expected {} sequences and {} sampled gradient rows; got {}'.format(
            len(names), expected_samples, len(grad_rows)))

    by_seq = {name: [row for row in grad_rows if row['sequence'] == name] for name in names}
    if set(expected_counts) != set(names) or any(
            len(by_seq[name]) != int(expected_counts[name]) for name in names):
        raise RuntimeError('gradient sample counts do not match valid trace frames')

    metrics = [
        'response_gt_cell', 'response_peak', 'mhsa_3_entropy', 'mhsa_6_entropy',
        'mhsa_9_entropy', 'mhsa_9_tir_cross_message_norm',
        'mhsa_9_rgb_cross_message_norm', 'mhsa_9_qkv_weight_grad_norm',
        'search_input_grad_rgb_norm', 'search_input_grad_tir_norm',
    ]
    outcome = 'tracking_iou'
    per_sequence_rows = []
    rho_by_metric = {metric: [] for metric in metrics}
    for name in names:
        rows = by_seq[name]
        values = {key: np.asarray([float(row[key]) for row in rows])
                  for key in rows[0] if key not in
                  ('sequence', 'frame', 'state_before', 'gt_valid', 'gt_box',
                   'parity_with_frozen_trace')}
        record = {
            'sequence': name,
            'attributes': ';'.join(attributes_by_sequence[name]),
            'gradient_frames': len(rows),
            'sampled_trace_iou_mean': float(values['tracking_iou'].mean()),
            'sampled_trace_iou_median': float(np.median(values['tracking_iou'])),
            'response_gt_cell_mean': float(values['response_gt_cell'].mean()),
            'mhsa_9_entropy_mean': float(values['mhsa_9_entropy'].mean()),
            'mhsa_9_tir_cross_message_mean': float(values['mhsa_9_tir_cross_message_norm'].mean()),
            'mhsa_9_qkv_grad_mean': float(values['mhsa_9_qkv_weight_grad_norm'].mean()),
            'archived_m1_auc_percent': float(full_rows[name]['m1_auc']),
            'full_current_replay_mean_iou': float(np.mean([
                iou_xywh(json.loads(line)['pred_box'], json.loads(line)['gt_box'])
                for line in (trace_dir / name / 'frames.jsonl').open()
                if json.loads(line).get('gt_box') is not None and
                float(json.loads(line)['gt_box'][2]) > 0 and
                float(json.loads(line)['gt_box'][3]) > 0])),
        }
        for metric in metrics:
            rho = spearman(values[metric], values[outcome])
            record[metric + '_rho_vs_iou'] = rho
            rho_by_metric[metric].append(rho)
        per_sequence_rows.append(record)

    rng = np.random.RandomState(20261009)
    cluster_rows = []
    for metric, rhos in rho_by_metric.items():
        valid = np.asarray([x for x in rhos if np.isfinite(x)], dtype=np.float64)
        draws = rng.randint(0, len(valid), size=(10000, len(valid)))
        medians = np.median(valid[draws], axis=1)
        cluster_rows.append({
            'metric': metric,
            'outcome': outcome,
            'sequences': int(len(valid)),
            'median_sequence_spearman': float(np.median(valid)),
            'mean_sequence_spearman': float(np.mean(valid)),
            'positive_sequence_fraction': float(np.mean(valid > 0)),
            'bootstrap95_median_low': float(np.quantile(medians, 0.025)),
            'bootstrap95_median_high': float(np.quantile(medians, 0.975)),
        })

    archive = Path(args.archive_dir)
    archive_rows = []
    all_coord_diffs, all_iou_diffs, current_ious, archived_ious = [], [], [], []
    for name in names:
        predictions = np.loadtxt(str(archive / (name + '.txt')), ndmin=2)
        per_coord, per_iou = [], []
        first_gt_2 = None
        trace_rows = [json.loads(line) for line in
                      (trace_dir / name / 'frames.jsonl').open()]
        for row in trace_rows:
            frame_id = int(row['frame_id'])
            if frame_id >= len(predictions):
                raise RuntimeError('archive frame alignment is incomplete: {}'.format(name))
            archived_box = predictions[frame_id]
            replay_box = np.asarray(row['pred_box'], dtype=np.float64)
            gt = np.asarray(row['gt_box'], dtype=np.float64)
            coordinate_delta = float(np.max(np.abs(replay_box - archived_box)))
            per_coord.append(coordinate_delta)
            if coordinate_delta > 2 and first_gt_2 is None:
                first_gt_2 = frame_id
            if gt[2] > 0 and gt[3] > 0:
                old_iou = iou_xywh(archived_box, gt)
                new_iou = iou_xywh(replay_box, gt)
                per_iou.append(abs(old_iou - new_iou))
                archived_ious.append(old_iou)
                current_ious.append(new_iou)
        all_coord_diffs.extend(per_coord)
        all_iou_diffs.extend(per_iou)
        archive_rows.append({
            'sequence': name,
            'frames_compared': len(per_coord),
            'mean_max_coordinate_delta_px': float(np.mean(per_coord)),
            'max_coordinate_delta_px': float(np.max(per_coord)),
            'first_frame_with_delta_over_2px': first_gt_2,
            'mean_absolute_iou_delta': float(np.mean(per_iou)) if per_iou else float('nan'),
            'fraction_frames_iou_delta_over_005': float(np.mean(np.asarray(per_iou) > 0.05)) if per_iou else float('nan'),
        })

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'per_sequence_summary.csv', per_sequence_rows)
    write_csv(out / 'sequence_cluster_correlations.csv', cluster_rows)
    write_csv(out / 'archive_prediction_comparison.csv', archive_rows)
    nonfinite = []
    for row_index, row in enumerate(grad_rows):
        for key, value in row.items():
            if key in ('sequence', 'frame', 'state_before', 'gt_valid', 'gt_box',
                       'parity_with_frozen_trace'):
                continue
            if not np.isfinite(float(value)):
                nonfinite.append((row_index, key))
    positive_gradient_fields = [key for key in grad_rows[0]
                                if key.endswith('_grad_norm')]
    positive_gradient_counts = {
        key: sum(float(row[key]) > 0 for row in grad_rows)
        for key in positive_gradient_fields
    }
    payload = {
        'experiment_id': manifest.get(
            'experiment_id', 'M1-FUSION-FLOW-TESTSET-DIAG-20261009'),
        'parent_checkpoint_sha256': '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef',
        'manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
        'sequence_count': len(names),
        'official_attribute_count_covered': len(manifest['attribute_sequence_counts']),
        'attribute_sequence_counts': manifest['attribute_sequence_counts'],
        'closed_loop_frames': sum(int(item['trace_frames']) for item in
                                  json.loads((root / 'closed_loop_trace/trace_summary.json').read_text())['sequences']),
        'gradient_sample_count': len(grad_rows),
        'sampled_frames_target_per_sequence': 32,
        'actual_sampled_frames_per_sequence_min': min(len(by_seq[name]) for name in names),
        'actual_sampled_frames_per_sequence_max': max(len(by_seq[name]) for name in names),
        'mean_current_replay_iou_on_valid_frames': float(np.mean(current_ious)),
        'mean_archived_m1_iou_on_matched_frames': float(np.mean(archived_ious)),
        'mean_absolute_iou_difference_vs_archived': float(np.mean(all_iou_diffs)),
        'mean_max_coordinate_delta_vs_archived_px': float(np.mean(all_coord_diffs)),
        'max_coordinate_delta_vs_archived_px': float(np.max(all_coord_diffs)),
        'frames_coordinate_delta_over_2px': int(np.sum(np.asarray(all_coord_diffs) > 2)),
        'compared_frames': len(all_coord_diffs),
        'positive_gradient_counts': positive_gradient_counts,
        'nonfinite_numeric_values': len(nonfinite),
        'cluster_correlations': cluster_rows,
        'comparability': 'NOT-DIRECTLY-COMPARABLE: current working-tree replay differs from protected full-244 archived predictions; exact M1 training-time source snapshot was not found.',
        'limitations': [
            'All manifest sequences are analyzed; membership is frozen before diagnostic outcomes are computed.',
            'Per-sequence correlations use 32 uniformly spaced frames and summarize sequences as clusters; uncertainty remains high.',
            'These test-set observations are descriptive only and must not be used to choose modules or insertion locations.',
            'Gradient is frozen-checkpoint, per-frame localization-loss sensitivity; no optimizer step or temporal backpropagation.',
        ],
        'outputs': ['per_sequence_summary.csv', 'sequence_cluster_correlations.csv',
                    'archive_prediction_comparison.csv'],
    }
    (out / 'analysis_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
