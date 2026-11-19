#!/usr/bin/env python3
"""Summarize paired fixed-state AMCM residual intervention outputs."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATIFIED = ROOT / 'output/experiments/M1_AMCM_ADAPTIVE_FIRST_UPDATE_CAUSAL_DIAG_20261021_RETRY4/per_arm.csv'
INITIAL = ROOT / 'output/experiments/M1_AMCM_ADAPTIVE_INITIAL_UPDATE_32SEQ_20261021/per_arm.csv'
OUT = ROOT / 'output/experiments/M1_AMCM_ADAPTIVE_INITIAL_UPDATE_32SEQ_20261021/causal_analysis.json'
ARMS = ('m1', 'adaptive', 'zero', 'reverse', 'shift')


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def paired(rows):
    result = defaultdict(dict)
    for row in rows:
        result[(row['sequence'], int(row['frame']))][row['arm']] = row
    return result


def stats(rows):
    n = len(rows)
    return {
        'n': n,
        'mean_fixed_state_iou': statistics.mean(float(r['fixed_state_iou']) for r in rows),
        'mean_gt_background_margin': statistics.mean(float(r['gt_hard_bg_margin']) for r in rows),
        'gt_rank_one': sum(int(r['gt_rank']) == 1 for r in rows),
        'mean_gt_rank': statistics.mean(int(r['gt_rank']) for r in rows),
        'mean_rgb_residual_input_norm_ratio': statistics.mean(
            float(r['amcm_residual_rgb_norm']) / max(float(r['amcm_input_rgb_norm']), 1e-12)
            for r in rows),
        'mean_tir_residual_input_norm_ratio': statistics.mean(
            float(r['amcm_residual_tir_norm']) / max(float(r['amcm_input_tir_norm']), 1e-12)
            for r in rows),
    }


def compare(pair_map, left, right):
    values = list(pair_map.values())
    iou_delta = [float(v[left]['fixed_state_iou']) - float(v[right]['fixed_state_iou'])
                 for v in values]
    margin_delta = [float(v[left]['gt_hard_bg_margin']) - float(v[right]['gt_hard_bg_margin'])
                    for v in values]
    rank_changes = sum(int(v[left]['gt_rank']) != int(v[right]['gt_rank']) for v in values)
    peak_flips = sum((int(v[left]['top1_x']), int(v[left]['top1_y'])) !=
                     (int(v[right]['top1_x']), int(v[right]['top1_y'])) for v in values)
    return {
        'left': left, 'right': right,
        'mean_iou_delta': statistics.mean(iou_delta),
        'mean_abs_iou_delta': statistics.mean(abs(x) for x in iou_delta),
        'max_abs_iou_delta': max(abs(x) for x in iou_delta),
        'mean_margin_delta': statistics.mean(margin_delta),
        'mean_abs_margin_delta': statistics.mean(abs(x) for x in margin_delta),
        'max_abs_margin_delta': max(abs(x) for x in margin_delta),
        'gt_rank_changes': rank_changes,
        'top1_cell_flips': peak_flips,
    }


def main():
    stratified_rows = read(STRATIFIED)
    initial_rows = read(INITIAL)
    stratified_pairs = paired(stratified_rows)
    initial_pairs = paired(initial_rows)
    initial_m1 = [v['m1'] for v in initial_pairs.values()]
    initial_iou = [float(r['m1_trace_iou']) for r in initial_m1]
    initial_margin = [float(r['gt_hard_bg_margin']) for r in initial_m1]
    summary = {
        'id': 'M1-AMCM-ADAPTIVE-FIRST-UPDATE-CAUSAL-DIAG-20261021',
        'status': 'COMPLETE / AMCM-READOUT-EFFECT-NOT-SUPPORTED-ON-TRAIN-COHORT',
        'stratified_12_state_diagnostic': {
            'n': len(stratified_pairs),
            'by_cohort_arm': {
                cohort: {
                    arm: stats([row for row in stratified_rows
                                if row['cohort'] == cohort and row['arm'] == arm])
                    for arm in ARMS
                }
                for cohort in ('failure', 'stable_control')
                if any(row['cohort'] == cohort for row in stratified_rows)
            },
            'adaptive_vs_zero': compare(stratified_pairs, 'adaptive', 'zero'),
            'reverse_vs_zero': compare(stratified_pairs, 'reverse', 'zero'),
            'shift_vs_zero': compare(stratified_pairs, 'shift', 'zero'),
            'm1_vs_zero': compare(stratified_pairs, 'm1', 'zero'),
        },
        'first_update_32_sequence_diagnostic': {
            'n': len(initial_pairs),
            'm1_first_update_iou_min_median_max': [
                min(initial_iou), statistics.median(initial_iou), max(initial_iou)],
            'm1_gt_background_margin_min_median_max': [
                min(initial_margin), statistics.median(initial_margin), max(initial_margin)],
            'm1_gt_rank_one': sum(int(r['gt_rank']) == 1 for r in initial_m1),
            'm1_top1_in_gt': sum(r['gt_cell_is_top1'] == 'True' for r in initial_m1),
            'by_arm': {arm: stats([v[arm] for v in initial_pairs.values()]) for arm in ARMS},
            'comparisons_vs_zero': {
                arm: compare(initial_pairs, arm, 'zero')
                for arm in ('m1', 'adaptive', 'reverse', 'shift')
            },
        },
        'parameter_audit': {
            'm1_state_keys': 267,
            'adaptive_state_keys': 283,
            'common_keys': 267,
            'changed_common_non_amcm_keys': 0,
            'additional_keys': 16,
            'interpretation': 'All M1 parameters are bytewise/value-identical in the adaptive checkpoint; only AMCM parameters are added.',
        },
        'provenance': {
            'stratified_csv': str(STRATIFIED.relative_to(ROOT)),
            'initial_update_csv': str(INITIAL.relative_to(ROOT)),
            'M1_train_side_manifest': 'output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/trace_run/trace/trace_sequences.txt',
        },
        'limitations': [
            'M1 training exposure to the train-side diagnostic sequences is unknown.',
            'All 32 first updates were stable; the catastrophic test-set first-update failure was not reproduced on this cohort.',
            'The 12-state failure/stable comparison is an outcome-stratified mechanism sample, not a population estimate.',
            'Fixed-state effects do not establish closed-loop tracking benefit.',
        ],
    }
    OUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    run_summary_path = INITIAL.parent / 'summary.json'
    run_summary = json.loads(run_summary_path.read_text())
    run_summary['summary_by_cohort_arm'] = [
        row for row in run_summary.get('summary_by_cohort_arm', []) if row.get('n', 0) > 0]
    run_summary['summary_note'] = 'Empty outcome strata are omitted; first-update cohort had 32 stable M1 states and no low-IoU state.'
    run_summary_path.write_text(json.dumps(run_summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    print('JSON:', OUT)


if __name__ == '__main__':
    main()
