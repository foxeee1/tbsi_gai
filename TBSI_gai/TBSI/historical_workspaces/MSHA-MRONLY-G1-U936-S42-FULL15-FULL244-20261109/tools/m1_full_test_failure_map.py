#!/usr/bin/env python3
"""Read-only full-test failure map for an existing LasHeR tracker run.

No tracker, model, feature hook, or test-set selection is executed here. The
script joins archived prediction boxes with test-set init.txt annotations.
Crop-out is reported as unavailable unless a future run archives per-frame
crop transforms.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_boxes(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([float(x) for x in line.replace(",", "\t").split()[:4]])
    return np.asarray(rows, dtype=np.float64)


def load_gt(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([float(x) for x in line.replace(",", "\t").split()[:4]])
    return np.asarray(rows, dtype=np.float64)


def iou_array(pred, gt):
    px1, py1 = pred[:, 0], pred[:, 1]
    px2, py2 = px1 + pred[:, 2], py1 + pred[:, 3]
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gx1 + gt[:, 2], gy1 + gt[:, 3]
    ix = np.maximum(0.0, np.minimum(px2, gx2) - np.maximum(px1, gx1))
    iy = np.maximum(0.0, np.minimum(py2, gy2) - np.maximum(py1, gy1))
    inter = ix * iy
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def contiguous_runs(mask):
    runs = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == len(mask) - 1):
            end = idx if value and idx == len(mask) - 1 else idx - 1
            runs.append((start, end, end - start + 1))
            start = None
    return runs


def metric_summary(values):
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def sequence_map(sequence, pred, gt, baseline=None):
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]
    eps = 1e-8
    ious = iou_array(pred, gt)
    pcx = pred[:, 0] + pred[:, 2] / 2.0
    pcy = pred[:, 1] + pred[:, 3] / 2.0
    gcx = gt[:, 0] + gt[:, 2] / 2.0
    gcy = gt[:, 1] + gt[:, 3] / 2.0
    center_px = np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)
    gt_diag = np.sqrt(gt[:, 2] ** 2 + gt[:, 3] ** 2)
    center_norm = center_px / np.maximum(gt_diag, eps)
    width_log = np.abs(np.log(np.maximum(pred[:, 2], eps) / np.maximum(gt[:, 2], eps)))
    height_log = np.abs(np.log(np.maximum(pred[:, 3], eps) / np.maximum(gt[:, 3], eps)))
    area_log = np.abs(np.log(np.maximum(pred[:, 2] * pred[:, 3], eps) / np.maximum(gt[:, 2] * gt[:, 3], eps)))
    aspect_log = np.abs(np.log(np.maximum(pred[:, 2] / np.maximum(pred[:, 3], eps), eps) /
                               np.maximum(gt[:, 2] / np.maximum(gt[:, 3], eps), eps)))

    failure = ious < 0.5
    severe = ious < 0.2
    runs = contiguous_runs(failure)
    severe_runs = contiguous_runs(severe)
    first_failure = next((i for i, x in enumerate(failure) if x), None)
    first_severe = next((i for i, x in enumerate(severe) if x), None)
    recovery_after_failure = None
    if first_failure is not None:
        recovery_after_failure = next((i for i in range(first_failure + 1, n) if ious[i] >= 0.5), None)

    out = {
        "sequence": sequence,
        "frames_pred": int(len(pred)),
        "frames_gt": int(len(gt)),
        "length_match": bool(len(pred) == len(gt)),
        "auc_mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "op50": float(np.mean(ious >= 0.5)),
        "op75": float(np.mean(ious >= 0.75)),
        "failure_rate_iou_lt_0.5": float(np.mean(failure)),
        "severe_rate_iou_lt_0.2": float(np.mean(severe)),
        "center_error_px": metric_summary(center_px),
        "center_error_normalized_gt_diag": metric_summary(center_norm),
        "width_log_error": metric_summary(width_log),
        "height_log_error": metric_summary(height_log),
        "area_log_error": metric_summary(area_log),
        "aspect_log_error": metric_summary(aspect_log),
        "failure_segments_iou_lt_0.5": len(runs),
        "max_failure_segment": max((r[2] for r in runs), default=0),
        "mean_failure_segment": float(np.mean([r[2] for r in runs])) if runs else 0.0,
        "severe_segments_iou_lt_0.2": len(severe_runs),
        "max_severe_segment": max((r[2] for r in severe_runs), default=0),
        "first_failure_frame_1based": first_failure + 1 if first_failure is not None else None,
        "first_severe_frame_1based": first_severe + 1 if first_severe is not None else None,
        "first_failure_fraction": first_failure / max(1, n - 1) if first_failure is not None else None,
        "recovery_frame_after_first_failure_1based": recovery_after_failure + 1 if recovery_after_failure is not None else None,
        "crop_out": "NOT_AVAILABLE_NO_PER_FRAME_CROP_TRANSFORM",
    }
    if baseline is not None:
        b = baseline[:n]
        biou = iou_array(b, gt)
        out["baseline_auc_mean_iou"] = float(np.mean(biou))
        out["m1_minus_baseline_auc_iou"] = float(np.mean(ious) - np.mean(biou))
        out["m1_minus_baseline_failure_rate"] = float(np.mean(failure) - np.mean(biou < 0.5))
        out["baseline_length_match"] = bool(len(b) == len(gt))
    return out


def flatten_row(row):
    result = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                result[key + "_" + subkey] = subvalue
        else:
            result[key] = value
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--sequence-list", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_root = Path(args.gt_root)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [x.strip() for x in Path(args.sequence_list).read_text().splitlines() if x.strip()]
    rows = []
    missing = []
    for name in names:
        pred_path = pred_dir / (name + ".txt")
        gt_path = gt_root / name / "init.txt"
        if not pred_path.exists() or not gt_path.exists():
            missing.append({"sequence": name, "prediction": pred_path.exists(), "gt": gt_path.exists()})
            continue
        pred = load_boxes(pred_path)
        gt = load_gt(gt_path)
        baseline = None
        if baseline_dir is not None:
            bp = baseline_dir / (name + ".txt")
            if bp.exists():
                baseline = load_boxes(bp)
        rows.append(sequence_map(name, pred, gt, baseline))

    flat = [flatten_row(x) for x in rows]
    if flat:
        fields = sorted({key for row in flat for key in row})
        with (output_dir / "per_sequence.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)

    def values(key):
        return [x[key] for x in rows if x.get(key) is not None]

    summary = {
        "contract": {
            "prediction_dir": str(pred_dir),
            "gt_root": str(gt_root),
            "sequence_list": str(args.sequence_list),
            "n_requested": len(names),
            "n_analyzed": len(rows),
            "n_missing": len(missing),
            "baseline_dir": str(baseline_dir) if baseline_dir else None,
            "read_only": True,
            "crop_out": "NOT_AVAILABLE_NO_PER_FRAME_CROP_TRANSFORM",
        },
        "aggregate": {
            "mean_sequence_auc_mean_iou": float(np.mean(values("auc_mean_iou"))),
            "median_sequence_auc_mean_iou": float(np.median(values("auc_mean_iou"))),
            "sequence_mean_failure_rate": float(np.mean(values("failure_rate_iou_lt_0.5"))),
            "sequence_mean_severe_rate": float(np.mean(values("severe_rate_iou_lt_0.2"))),
            "sequences_with_any_failure": int(sum(x["failure_segments_iou_lt_0.5"] > 0 for x in rows)),
            "sequences_with_severe_failure": int(sum(x["severe_segments_iou_lt_0.2"] > 0 for x in rows)),
            "total_failure_segments": int(sum(x["failure_segments_iou_lt_0.5"] for x in rows)),
            "total_severe_segments": int(sum(x["severe_segments_iou_lt_0.2"] for x in rows)),
            "max_failure_segment": int(max(values("max_failure_segment"))),
            "max_severe_segment": int(max(values("max_severe_segment"))),
            "median_first_failure_frame": float(np.median(values("first_failure_frame_1based"))) if values("first_failure_frame_1based") else None,
            "median_center_error_normalized_gt_diag": float(np.median([x["center_error_normalized_gt_diag"]["median"] for x in rows])),
            "median_area_log_error": float(np.median([x["area_log_error"]["median"] for x in rows])),
            "median_aspect_log_error": float(np.median([x["aspect_log_error"]["median"] for x in rows])),
        },
        "missing": missing,
        "failure_map": rows,
        "interpretation_guard": "This is a full-test descriptive risk map. It is not an independent confirmation set and must not be used to select a module or threshold.",
        "next_diagnostic_boundary": [
            "If crop-out is required, rerun only a fixed training-side event list with per-frame crop transforms archived.",
            "Use sequence-level and event-level results together; contiguous frames are not independent failures.",
            "Optional B0 comparison is a clue only when training/configuration contracts differ.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["contract"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
