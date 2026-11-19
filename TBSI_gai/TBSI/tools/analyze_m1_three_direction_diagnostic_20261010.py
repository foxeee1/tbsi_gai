#!/usr/bin/env python3
"""Read-only paired aggregation for three M1 fusion/readout hypotheses."""

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/analysis_final/per_arm.csv"
ORACLE = ROOT / "output/experiments/M1_STRIP_GT_CELL_COUNTERFACTUAL_20261010/oracle_final_hardneg_v2"
FRAME = ORACLE / "frame_results.csv"
PROBE = ORACLE / "probe/heldout_by_sequence.csv"
OUT = ROOT / "output/experiments/M1_THREE_DIRECTION_DIAGNOSTIC_20261010"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def med(values):
    return statistics.median(values) if values else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def intervention_summary(rows, label, arm):
    indexed = {(r["sequence"], r["frame"], r["label"], r["arm"]): r for r in rows}
    base = [r for r in rows if r["arm"] == "baseline" and r["label"] == label]
    paired = []
    for b in base:
        c = indexed[(b["sequence"], b["frame"], label, arm)]
        diou = float(c["iou"]) - float(b["iou"])
        flipped = (c["window_peak_x"], c["window_peak_y"]) != (b["window_peak_x"], b["window_peak_y"])
        paired.append((diou, flipped))
    deltas = [item[0] for item in paired]
    return {
        "n": len(paired),
        "peak_flips": sum(item[1] for item in paired),
        "iou_improved_gt_0_1": sum(d > 0.1 for d in deltas),
        "iou_worsened_gt_0_1": sum(d < -0.1 for d in deltas),
        "median_paired_iou_delta": med(deltas),
    }


def main():
    stage = read_csv(STAGE)
    frame = read_csv(FRAME)
    probe = read_csv(PROBE)

    event_labels = Counter(r["label"] for r in stage if r["arm"] == "baseline")
    hann_effects = Counter(r["m1_hann_effect_vs_raw"] for r in frame)
    prior_hann_changes = {
        label: sum(int(r["hann_changed_peak"]) for r in stage if r["arm"] == "baseline" and r["label"] == label)
        for label in ("matched_good", "pre_onset", "severe_onset", "good_only")
    }
    raw_error = [float(r["m1_raw_center_error_px"]) for r in frame]
    hann_error = [float(r["m1_hann_center_error_px"]) for r in frame]
    gt_error = [float(r["m1_gt_center_error_px"]) for r in frame]
    hann_iou = [float(r["m1_hann_iou"]) for r in frame]
    gt_iou = [float(r["m1_gt_iou"]) for r in frame]
    paired_delta_error = [float(r["m1_gt_minus_hann_center_error_px"]) for r in frame]
    paired_delta_iou = [float(r["m1_gt_minus_hann_iou"]) for r in frame]
    positive_margins = [float(r["target_hard_negative_margin"]) for r in probe if float(r["target_hard_negative_margin"]) > 0]

    interventions = {}
    arms = (
        "rgb_feature_target_atten_25", "rgb_feature_target_atten_50",
        "tir_feature_target_atten_25", "tir_feature_target_atten_50",
        "rgb_target_blur_50", "rgb_target_blur_90", "tir_target_blur_50", "tir_target_blur_90",
    )
    for label in ("severe_onset", "pre_onset", "good_only"):
        interventions[label] = {arm: intervention_summary(stage, label, arm) for arm in arms}

    result = {
        "status": "READ_ONLY_DIAGNOSTIC",
        "parent": "TBSI_E_MHSA_FULL_4090D_seed42 epoch15",
        "source_replay_note": "Current-worktree fixed-state replay; not frame-equivalent to protected archived predictions.",
        "cohorts": {
            "stage_probe_baseline_frames_by_label": dict(event_labels),
            "independent_gt_cell_events": len(frame),
            "independent_hard_negative_probe_sequences": len(probe),
        },
        "online_peak_supervision_mismatch": {
            "raw_to_hann_effect_counts_in_gt_cell_cohort": dict(hann_effects),
            "raw_to_hann_changed_peak_counts_in_prior_fixed_state_cohort": prior_hann_changes,
            "raw_center_error_median_px": med(raw_error),
            "hann_center_error_median_px": med(hann_error),
            "gt_cell_center_error_median_px": med(gt_error),
            "hann_iou_median": med(hann_iou),
            "gt_cell_iou_median": med(gt_iou),
            "gt_cell_center_error_improved": sum(e < 0 for e in paired_delta_error),
            "gt_cell_iou_improved": sum(i > 0 for i in paired_delta_iou),
            "gt_cell_both_center_and_iou_strictly_improved": sum(e < 0 and i > 0 for e, i in zip(paired_delta_error, paired_delta_iou)),
            "gt_cell_iou_ge_0_5": sum(v >= 0.5 for v in gt_iou),
            "gt_cell_center_error_lt_5px": sum(v < 5 for v in gt_error),
        },
        "center_readout_and_representation": {
            "hard_negative_positive_margin_sequences": len(positive_margins),
            "hard_negative_margin_median": med([float(r["target_hard_negative_margin"]) for r in probe]),
            "probe_peak_distance_tokens_median": med([float(r["probe_peak_distance_tokens"]) for r in probe]),
            "interventions": interventions,
        },
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (STAGE, FRAME, PROBE)},
        "scripts": {
            "source_stage_analysis": "tools/diagnose_m1_search_stage_20261009.py",
            "source_gt_cell_analysis": "tools/analyze_strip_gt_cell_counterfactual_20261010.py",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "summary.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
