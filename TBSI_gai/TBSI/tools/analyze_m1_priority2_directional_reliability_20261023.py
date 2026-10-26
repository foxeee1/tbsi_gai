#!/usr/bin/env python3
"""Paired analysis for the M1 priority-2 directional reliability mechanism."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOW_PATH = ROOT / 'tools/run_m1_screening_fixed_state_flow_20261022.py'
SPEC = importlib.util.spec_from_file_location('m1_fixed_flow', str(FLOW_PATH))
flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow)
base = flow.base

ARM_DIRS = {
    'parent': 'M1_PRIORITY2_DIRECTIONAL_RELIABILITY_PARENT_CAPTURE_FIXED46_20261023',
    'independent': 'M1_PRIORITY2_DIRECTIONAL_RELIABILITY_INDEPENDENT_CAPTURE_FIXED46_20261023',
    'shared': 'M1_PRIORITY2_DIRECTIONAL_RELIABILITY_SHARED_CAPTURE_FIXED46_20261023',
    'swapped': 'M1_PRIORITY2_DIRECTIONAL_RELIABILITY_SWAPPED_CAPTURE_FIXED46_20261023',
}


def load_arm(name):
    path = ROOT / 'output/experiments' / ARM_DIRS[name] / 'per_event.csv'
    rows = list(csv.DictReader(path.open(newline='')))
    result = {}
    for row in rows:
        key = '{}:{}:{}'.format(row['sequence'], row['phase'], row['frame'])
        row['native_msha_stats'] = json.loads(row['native_msha_stats'])
        for key_name in ('fixed_replay_iou', 'trace_iou', 'center_gt_hard_bg_margin',
                         'center_gt_rank', 'center_top1_to_gt_center_cells'):
            row[key_name] = float(row[key_name])
        row['center_gt_rank'] = int(row['center_gt_rank'])
        row['center_top1_is_gt'] = row['center_top1_is_gt'].lower() == 'true'
        row['predicted_bbox'] = json.loads(row['predicted_bbox'])
        row['gt_box'] = json.loads(row['gt_box'])
        row['crop_xywh'] = json.loads(row['trace_crop_xywh'])
        result[key] = row
    return result


def auc_binary(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = scores[labels], scores[~labels]
    if not len(positives) or not len(negatives):
        return None
    return float(np.mean((positives[:, None] > negatives[None, :]) +
                         0.5 * (positives[:, None] == negatives[None, :])))


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter_w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    return float(inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-12))


def win_tie_loss(deltas, eps=1e-10):
    return [int(sum(d > eps for d in deltas)),
            int(sum(abs(d) <= eps for d in deltas)),
            int(sum(d < -eps for d in deltas))]


def main():
    arms = {name: load_arm(name) for name in ARM_DIRS}
    ids = list(arms['parent'])
    if any(set(arm) != set(ids) for arm in arms.values()):
        raise RuntimeError('arm event membership mismatch')
    for event in ids:
        parent = arms['parent'][event]
        for name in ('independent', 'shared', 'swapped'):
            candidate = arms[name][event]
            if candidate['trace_crop_xywh'] != parent['trace_crop_xywh']:
                raise RuntimeError('fixed crop mismatch for {}'.format(event))
            if candidate['trace_state_before'] != parent['trace_state_before']:
                raise RuntimeError('fixed state mismatch for {}'.format(event))
            if candidate['gt_box'] != parent['gt_box']:
                raise RuntimeError('GT mismatch for {}'.format(event))

    evidence_rows = []
    for event in ids:
        row = arms['independent'][event]
        mask = base.grid_mask(row['gt_box'], row['crop_xywh']).astype(bool)
        stats = row['native_msha_stats'][0]
        for modality in ('rgb', 'tir'):
            index = int(round(stats['{}_candidate_index_mean'.format(modality)]))
            peak_in_gt = bool(mask.reshape(-1)[index])
            evidence_rows.append({
                'event_id': event, 'sequence': row['sequence'], 'phase': row['phase'],
                'modality': modality,
                'evidence_margin': stats['{}_evidence_margin_mean'.format(modality)],
                'confidence': stats['{}_evidence_mean'.format(modality)],
                'candidate_index': index, 'candidate_peak_in_gt': peak_in_gt,
            })

    summary = {'n_states': len(ids), 'n_sequences': len({row['sequence'] for row in arms['parent'].values()}),
               'arms': {}, 'evidence_quality': {}, 'message_mapping': {}}
    parent = arms['parent']
    trace_error = [row['fixed_replay_iou'] - row['trace_iou'] for row in parent.values()]
    summary['parent_replay_vs_trace'] = {
        'max_abs_iou_delta': float(np.max(np.abs(trace_error))),
        'mean_iou_delta': float(np.mean(trace_error)),
    }
    for name, rows in arms.items():
        ious = np.asarray([row['fixed_replay_iou'] for row in rows.values()])
        ranks = np.asarray([row['center_gt_rank'] for row in rows.values()])
        margins = np.asarray([row['center_gt_hard_bg_margin'] for row in rows.values()])
        summary['arms'][name] = {
            'mean_iou': float(ious.mean()), 'median_iou': float(np.median(ious)),
            'mean_center_gt_rank': float(ranks.mean()),
            'mean_center_margin': float(margins.mean()),
            'top1_in_gt_count': int(sum(row['center_top1_is_gt'] for row in rows.values())),
        }
        if name == 'parent':
            continue
        paired_iou = np.asarray([rows[event]['fixed_replay_iou'] - parent[event]['fixed_replay_iou']
                                 for event in ids])
        paired_rank = np.asarray([rows[event]['center_gt_rank'] - parent[event]['center_gt_rank']
                                  for event in ids])
        paired_margin = np.asarray([rows[event]['center_gt_hard_bg_margin'] -
                                    parent[event]['center_gt_hard_bg_margin'] for event in ids])
        paired_box_l1 = np.asarray([
            np.abs(np.asarray(rows[event]['predicted_bbox']) -
                   np.asarray(parent[event]['predicted_bbox'])).sum() for event in ids])
        summary['arms'][name].update({
            'delta_mean_iou_vs_parent': float(paired_iou.mean()),
            'delta_median_iou_vs_parent': float(np.median(paired_iou)),
            'iou_win_tie_loss_vs_parent': win_tie_loss(paired_iou),
            'delta_mean_center_rank_vs_parent': float(paired_rank.mean()),
            'rank_improved_tied_worsened_vs_parent': win_tie_loss(-paired_rank),
            'delta_mean_center_margin_vs_parent': float(paired_margin.mean()),
            'mean_box_l1_shift_px_vs_parent': float(paired_box_l1.mean()),
        })

    for modality in ('rgb', 'tir'):
        rows = [row for row in evidence_rows if row['modality'] == modality]
        labels = [row['candidate_peak_in_gt'] for row in rows]
        margins = [row['evidence_margin'] for row in rows]
        confidence = [row['confidence'] for row in rows]
        summary['evidence_quality'][modality] = {
            'candidate_peak_in_gt': int(sum(labels)), 'n': len(labels),
            'mean_margin_when_peak_in_gt': float(np.mean([x for x, y in zip(margins, labels) if y])) if any(labels) else None,
            'mean_margin_when_peak_outside_gt': float(np.mean([x for x, y in zip(margins, labels) if not y])) if not all(labels) else None,
            'margin_auc_for_candidate_peak_in_gt': auc_binary(margins, labels),
            'mean_confidence_when_peak_in_gt': float(np.mean([x for x, y in zip(confidence, labels) if y])) if any(labels) else None,
            'mean_confidence_when_peak_outside_gt': float(np.mean([x for x, y in zip(confidence, labels) if not y])) if not all(labels) else None,
        }

    for arm_name in ('independent', 'shared', 'swapped'):
        rows = arms[arm_name]
        rgb_ratio_error, tir_ratio_error = [], []
        rgb_delta, tir_delta = [], []
        for event in ids:
            stat = rows[event]['native_msha_stats'][0]
            c_rgb = stat['rgb_evidence_mean']
            c_tir = stat['tir_evidence_mean']
            if arm_name == 'independent':
                expected_rgb = 1 + 0.25 * c_rgb
                expected_tir = 1 + 0.25 * c_tir
            elif arm_name == 'shared':
                expected_rgb = expected_tir = 1 + 0.25 * 0.5 * (c_rgb + c_tir)
            else:
                expected_rgb = 1 + 0.25 * c_tir
                expected_tir = 1 + 0.25 * c_rgb
            rgb_ratio_error.append(abs(stat['rgb_to_tir_message_norm_ratio'] - expected_rgb))
            tir_ratio_error.append(abs(stat['tir_to_rgb_message_norm_ratio'] - expected_tir))
            rgb_delta.append(stat['rgb_to_tir_message_norm_ratio'] - 1.0)
            tir_delta.append(stat['tir_to_rgb_message_norm_ratio'] - 1.0)
        summary['message_mapping'][arm_name] = {
            'rgb_to_tir_mean_norm_ratio': float(np.mean([rows[e]['native_msha_stats'][0]['rgb_to_tir_message_norm_ratio'] for e in ids])),
            'tir_to_rgb_mean_norm_ratio': float(np.mean([rows[e]['native_msha_stats'][0]['tir_to_rgb_message_norm_ratio'] for e in ids])),
            'rgb_to_tir_max_abs_formula_error': float(max(rgb_ratio_error)),
            'tir_to_rgb_max_abs_formula_error': float(max(tir_ratio_error)),
            'rgb_to_tir_changed_state_count': int(sum(abs(d) > 1e-8 for d in rgb_delta)),
            'tir_to_rgb_changed_state_count': int(sum(abs(d) > 1e-8 for d in tir_delta)),
        }

    out = ROOT / 'output/experiments/M1_PRIORITY2_DIRECTIONAL_RELIABILITY_ANALYSIS_20261023'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'evidence_per_state.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence_rows[0]))
        writer.writeheader()
        writer.writerows(evidence_rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
