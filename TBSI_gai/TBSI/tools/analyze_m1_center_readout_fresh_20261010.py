#!/usr/bin/env python3
"""Aggregate the preregistered fresh M1 CENTER readout diagnostic."""

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/experiments/M1_CENTER_READOUT_FRESH_DIAG_20261010'
STAGE = RUN / 'stage/per_arm.csv'
PROBE = RUN / 'probe/probe_dev.csv'
OUT = RUN / 'analysis'


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def med(values):
    return statistics.median(values) if values else None


def mean(values):
    return statistics.mean(values) if values else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def paired_arm(rows, label, arm):
    index = {(r['sequence'], r['frame'], r['label'], r['arm']): r for r in rows}
    base = [r for r in rows if r['arm'] == 'baseline' and r['label'] == label]
    pairs = [(b, index[(b['sequence'], b['frame'], label, arm)]) for b in base]
    diou = [float(c['iou']) - float(b['iou']) for b, c in pairs]
    return {
        'n': len(pairs),
        'peak_flips': sum((b['window_peak_x'], b['window_peak_y']) !=
                          (c['window_peak_x'], c['window_peak_y']) for b, c in pairs),
        'iou_improved_gt_0_1': sum(delta > .1 for delta in diou),
        'iou_worsened_gt_0_1': sum(delta < -.1 for delta in diou),
        'median_paired_iou_delta': med(diou),
    }


def main():
    rows, probes = read_csv(STAGE), read_csv(PROBE)
    baseline = [r for r in rows if r['arm'] == 'baseline']
    labels = ('matched_good', 'pre_onset', 'severe_onset', 'good_only', 'probe_reference')
    label_counts = Counter(r['label'] for r in baseline)

    probe_index = {(r['sequence'], r['frame'], r['label'], r['stage']): r for r in probes}
    event_rows = []
    severe_sequences = sorted(r['sequence'] for r in baseline if r['label'] == 'severe_onset')
    for base in baseline:
        if base['label'] not in ('pre_onset', 'severe_onset', 'matched_good'):
            continue
        item = {
            'sequence': base['sequence'], 'frame': base['frame'], 'label': base['label'],
            'crop_coverage': base['crop_coverage'], 'trace_iou': base['trace_iou'],
            'raw_peak_gt_rank': base['raw_gt_rank'], 'hann_peak_gt_rank': base['window_gt_rank'],
            'hann_changed_peak': base['hann_changed_peak'], 'actual_iou': base['iou'],
            'gt_cell_iou': base['oracle_gt_cell_iou'],
            'raw_decoded_center_tokens': base['raw_decoded_center_distance_tokens'],
            'hann_decoded_center_tokens': base['window_decoded_center_distance_tokens'],
            'gt_cell_decoded_center_tokens': base['gt_cell_decoded_center_distance_tokens'],
        }
        for stage in ('rgb', 'tir', 'concat', 'fused'):
            probe = probe_index[(base['sequence'], base['frame'], base['label'], stage)]
            item[stage + '_margin'] = probe['target_bg_margin']
            item[stage + '_target_top'] = probe['target_top']
        event_rows.append(item)

    stage_probe = {}
    for label in labels:
        stage_probe[label] = {}
        for stage in ('rgb', 'tir', 'concat', 'fused'):
            subset = [r for r in probes if r['label'] == label and r['stage'] == stage]
            if subset:
                margins = [float(r['target_bg_margin']) for r in subset]
                stage_probe[label][stage] = {
                    'frames': len(subset), 'sequences': len(set(r['sequence'] for r in subset)),
                    'median_target_background_margin': med(margins),
                    'median_peak_distance_tokens': med([float(r['peak_distance_tokens']) for r in subset]),
                    'target_region_top_fraction': mean([int(r['target_top']) for r in subset]),
                    'positive_margin_sequences': sum(
                        max(float(r['target_bg_margin']) for r in subset
                            if r['sequence'] == name) > 0
                        for name in set(r['sequence'] for r in subset)),
                }

    perturbation_arms = (
        'rgb_feature_target_atten_25', 'rgb_feature_target_atten_50',
        'rgb_feature_background_control_50', 'tir_feature_target_atten_50',
        'tir_feature_background_control_50', 'rgb_target_blur_50', 'rgb_target_blur_90',
        'rgb_background_control_90', 'tir_target_blur_90', 'tir_background_control_90',
    )
    perturbations = {
        label: {arm: paired_arm(rows, label, arm) for arm in perturbation_arms}
        for label in ('matched_good', 'pre_onset', 'severe_onset', 'good_only')
    }

    severe_probe = [r for r in event_rows if r['label'] == 'severe_onset']
    modality_fusion_patterns = {
        'at_least_one_single_modality_positive_but_fused_negative': 0,
        'both_single_modalities_negative_but_fused_positive': 0,
        'rgb_positive_tir_negative_fused_negative': 0,
        'tir_positive_rgb_negative_fused_negative': 0,
    }
    for row in severe_probe:
        rgb, tir, fused = (float(row[key + '_margin']) for key in ('rgb', 'tir', 'fused'))
        if (rgb > 0 or tir > 0) and fused <= 0:
            modality_fusion_patterns['at_least_one_single_modality_positive_but_fused_negative'] += 1
        if rgb <= 0 and tir <= 0 and fused > 0:
            modality_fusion_patterns['both_single_modalities_negative_but_fused_positive'] += 1
        if rgb > 0 and tir <= 0 and fused <= 0:
            modality_fusion_patterns['rgb_positive_tir_negative_fused_negative'] += 1
        if tir > 0 and rgb <= 0 and fused <= 0:
            modality_fusion_patterns['tir_positive_rgb_negative_fused_negative'] += 1

    severe = [r for r in baseline if r['label'] == 'severe_onset']
    actual_iou = [float(r['iou']) for r in severe]
    gt_iou = [float(r['oracle_gt_cell_iou']) for r in severe]
    center_delta = [float(r['gt_cell_decoded_center_distance_tokens']) -
                    float(r['window_decoded_center_distance_tokens']) for r in severe]
    summary = {
        'id': 'M1-CENTER-READOUT-FRESH-20261010',
        'status': 'DIAGNOSTIC-COMPLETE / EXPLORATORY-14-EVENTS',
        'parent': 'Protected M1 epoch15; fixed config/checkpoint hashes in manifest',
        'event_selection_rule': 'onset-3-iou-0.5; crop coverage >=0.999 for all 3 frames; one event per sequence',
        'counts': {'selected_sequences': 24, 'selected_frames': len(baseline),
                   'frames_by_label': dict(label_counts),
                   'unique_onset_sequences': len(severe_sequences)},
        'center_readout': {
            'severe_actual_iou_median': med(actual_iou),
            'severe_gt_cell_iou_median': med(gt_iou),
            'gt_cell_iou_improved_sequences': sum(b > a for a, b in zip(actual_iou, gt_iou)),
            'gt_cell_iou_ge_0_5': sum(v >= .5 for v in gt_iou),
            'gt_cell_decoded_center_residual_tokens_median': med(
                [float(r['gt_cell_decoded_center_distance_tokens']) for r in severe]),
            'gt_cell_vs_hann_decoded_center_delta_tokens_median': med(center_delta),
            'raw_gt_rank_le_3': sum(int(r['raw_gt_rank']) <= 3 for r in severe),
            'hann_gt_rank_le_3': sum(int(r['window_gt_rank']) <= 3 for r in severe),
            'hann_changed_peak': sum(int(r['hann_changed_peak']) for r in severe),
        },
        'heldout_probe_by_label': stage_probe,
        'severe_modality_fusion_sign_patterns': modality_fusion_patterns,
        'paired_perturbations': perturbations,
        'limitations': [
            '14 onset sequences is exploratory and below the 18-sequence stability gate.',
            'Current-worktree fixed-state replay is not frame-equivalent to the protected archive.',
            'Linear probes measure decodability, not original CENTER use; channel counts differ across stages.',
            'GT-cell and target-region perturbations use GT offline and are not deployable interventions.',
            'No human visibility or RGB-TIR registration audit was performed.',
        ],
        'input_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in (STAGE, PROBE)},
    }

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    fields = list(event_rows[0])
    with (OUT / 'per_event.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(event_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
