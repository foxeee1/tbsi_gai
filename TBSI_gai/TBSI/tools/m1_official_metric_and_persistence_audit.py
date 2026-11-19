#!/usr/bin/env python3
"""Recompute official LasHeR metrics and persistent-failure events read-only.

The official metric path is imported from lib.test.analysis.extract_results:
calc_seq_err_robust, including first-frame replacement, length handling, and
invalid-frame semantics. No tracker is executed and no test sequence is
selected based on the resulting scores.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_boxes(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append([float(x) for x in line.replace(",", "\t").split()[:4]])
    return np.asarray(rows, dtype=np.float64)


def load_eval_data(path):
    import pickle
    with open(path, "rb") as handle:
        return pickle.load(handle)


def runs(mask):
    result = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i if value and i == len(mask) - 1 else i - 1
            result.append((start, end, end - start + 1))
            start = None
    return result


def persistence_events(overlap, short_len, persistent_len):
    short = overlap < 0.5
    persistent = []
    for start, end, length in runs(short):
        if length >= persistent_len:
            recovery = next((i for i in range(end + 1, len(overlap)) if overlap[i] >= 0.5), None)
            persistent.append({
                "start_frame_1based": start + 1,
                "end_frame_1based": end + 1,
                "length_at_least_persistent_len": length,
                "recovery_frame_1based": recovery + 1 if recovery is not None else None,
                "recovery_after_start_frames": recovery - start if recovery is not None else None,
            })
    short_events = [r for r in runs(short) if r[2] < persistent_len]
    return persistent, short_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--dataset", default="lasher_test")
    ap.add_argument("--report-eval-data", default=None)
    ap.add_argument("--persistent-len", type=int, default=30)
    ap.add_argument("--short-len", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from lib.test.analysis.extract_results import calc_seq_err_robust
    from lib.test.evaluation import get_dataset

    dataset = get_dataset(args.dataset)
    thresholds = np.arange(0.0, 1.0 + 0.05, 0.05, dtype=np.float64)
    trackers = {"M1": Path(args.pred_dir), "B0": Path(args.baseline_dir)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sequence = []
    missing = []
    curves = {key: [] for key in trackers}

    for seq in dataset:
        gt = torch.tensor(seq.ground_truth_rect, dtype=torch.float64)
        row = {"sequence": seq.name, "gt_frames": int(len(gt)), "dataset": seq.dataset}
        condition_rows = {}
        for label, directory in trackers.items():
            path = directory / (seq.name + ".txt")
            if not path.exists():
                missing.append({"tracker": label, "sequence": seq.name, "path": str(path)})
                continue
            raw = load_boxes(path)
            pred = torch.tensor(raw, dtype=torch.float64)
            raw_len = int(len(pred))
            err_overlap, err_center, err_center_norm, valid = calc_seq_err_robust(pred, gt, seq.dataset)
            overlap = err_overlap.numpy()
            valid_np = valid.numpy().astype(bool)
            seq_curve = np.array([(overlap > threshold).sum() / len(gt) for threshold in thresholds], dtype=np.float64)
            official_mean_overlap = float(overlap[valid_np].mean()) if valid_np.any() else None
            # The official success AUC is computed by averaging sequence curves,
            # then taking the mean over the 21 sampled thresholds.
            curve_auc = float(seq_curve.mean())
            p_persistent, p_short = persistence_events(overlap, args.short_len, args.persistent_len)
            condition_rows[label] = {
                "raw_prediction_frames": raw_len,
                "official_processed_frames": int(len(overlap)),
                "raw_length_relation": "equal" if raw_len == len(gt) else ("truncated" if raw_len > len(gt) else "padded"),
                "official_first_frame_replaced_with_gt": True,
                "official_valid_gt_frames": int(valid_np.sum()),
                "official_invalid_gt_frames": int((~valid_np).sum()),
                "official_success_denominator": int(len(gt)),
                "official_mean_overlap_valid_frames": official_mean_overlap,
                "official_sequence_curve_auc_sampled": curve_auc,
                "iou_lt_0.5_frames": int((overlap < 0.5).sum()),
                "iou_lt_0.2_frames": int((overlap < 0.2).sum()),
                "frame_weighted_failure_rate_lt_0.5": float((overlap < 0.5).mean()),
                "frame_weighted_failure_rate_lt_0.2": float((overlap < 0.2).mean()),
                "short_failure_segments_lt_persistent_len": len(p_short),
                "persistent_failure_segments": p_persistent,
                "persistent_failure_count": len(p_persistent),
                "longest_failure_segment": max((x[2] for x in runs(overlap < 0.5)), default=0),
            }
            row[label] = condition_rows[label]
            curves[label].append(seq_curve)
        if len(condition_rows) == len(trackers):
            per_sequence.append(row)

    aggregate = {}
    for label in trackers:
        rows = [x[label] for x in per_sequence]
        curve = np.asarray(curves[label][:len(per_sequence)], dtype=np.float64)
        aggregate[label] = {
            "n_sequences": len(rows),
            "official_auc_percent": float(curve.mean(axis=0).mean() * 100.0),
            "official_op50_percent": float(curve.mean(axis=0)[10] * 100.0),
            "official_op75_percent": float(curve.mean(axis=0)[15] * 100.0),
            "sequence_equal_mean_valid_overlap": float(np.mean([x["official_mean_overlap_valid_frames"] for x in rows])),
            "sequence_equal_failure_rate_lt_0.5": float(np.mean([x["frame_weighted_failure_rate_lt_0.5"] for x in rows])),
            "sequence_equal_failure_rate_lt_0.2": float(np.mean([x["frame_weighted_failure_rate_lt_0.2"] for x in rows])),
            "frame_weighted_failure_rate_lt_0.5": float(sum(x["iou_lt_0.5_frames"] for x in rows) / sum(x["official_processed_frames"] for x in rows)),
            "frame_weighted_failure_rate_lt_0.2": float(sum(x["iou_lt_0.2_frames"] for x in rows) / sum(x["official_processed_frames"] for x in rows)),
            "sequences_with_persistent_failure": int(sum(x["persistent_failure_count"] > 0 for x in rows)),
            "total_persistent_failure_segments": int(sum(x["persistent_failure_count"] for x in rows)),
            "longest_failure_segment": int(max(x["longest_failure_segment"] for x in rows)),
            "median_first_persistent_failure_frame": None,
        }
        starts = [event["start_frame_1based"] for x in rows for event in x["persistent_failure_segments"]]
        if starts:
            aggregate[label]["median_first_persistent_failure_frame"] = float(np.median(starts))

    existing_reference = None
    if args.report_eval_data:
        d = load_eval_data(args.report_eval_data)
        existing_reference = {
            "path": args.report_eval_data,
            "trackers": d.get("trackers"),
            "sequences": len(d.get("sequences", [])),
            "thresholds": d.get("threshold_set_overlap"),
        }

    result = {
        "contract": {
            "official_source": "lib/test/analysis/extract_results.py::calc_seq_err_robust and official extract_results aggregation",
            "dataset": args.dataset,
            "n_dataset_sequences": len(dataset),
            "n_common_sequences": len(per_sequence),
            "prediction_format": "x,y,w,h tab-delimited; official evaluation replaces pred[0] with GT[0]",
            "length_handling": "official calc truncates long predictions and pads short predictions with zeros for non-lasot datasets",
            "invalid_gt": "overlap=-1 on invalid GT frames; default success denominator remains full annotation length",
            "success_thresholds": thresholds.tolist(),
            "aggregation": "sequence equal-weight success curve, then mean across thresholds; frame-weighted rates reported separately",
            "persistent_failure_rule": "IoU<0.5 run with length >= persistent_len; short runs are below persistent_len",
            "persistent_len": args.persistent_len,
            "short_len": args.short_len,
            "read_only": True,
        },
        "aggregate": aggregate,
        "missing": missing,
        "existing_reference_eval_data": existing_reference,
        "per_sequence": per_sequence,
        "interpretation_guard": "This audit reconciles official metrics and describes persistence. It is not a module-selection or independent-test confirmation procedure.",
    }
    (output_dir / "official_metric_and_persistence.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["sequence", "gt_frames"]
    for label in trackers:
        fields += [label + ".raw_prediction_frames", label + ".official_valid_gt_frames", label + ".official_mean_overlap_valid_frames", label + ".official_sequence_curve_auc_sampled", label + ".iou_lt_0.5_frames", label + ".iou_lt_0.2_frames", label + ".frame_weighted_failure_rate_lt_0.5", label + ".frame_weighted_failure_rate_lt_0.2", label + ".persistent_failure_count", label + ".longest_failure_segment"]
    with (output_dir / "per_sequence_official.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_sequence:
            flat = {"sequence": row["sequence"], "gt_frames": row["gt_frames"]}
            for label in trackers:
                for field in fields:
                    prefix = label + "."
                    if field.startswith(prefix):
                        flat[field] = row[label][field[len(prefix):]]
            writer.writerow(flat)
    print(json.dumps({"aggregate": aggregate, "missing": missing, "reference": existing_reference}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
