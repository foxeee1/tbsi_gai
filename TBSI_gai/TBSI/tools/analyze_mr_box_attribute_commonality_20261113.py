#!/usr/bin/env python3
"""Offline box-vs-annotation and attribute diagnostics for MR/B0 seed pairs."""
import csv
import json
import math
import os
import statistics
from collections import defaultdict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(
    ROOT,
    "output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/analysis/"
    "mr_box_attribute_commonality_20261113",
)
GROUPS = os.path.join(
    ROOT,
    "output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/analysis/"
    "two_seed_failure_20261113/per_sequence_two_seed.csv",
)
GT_ROOT = os.path.join(ROOT, "../data/lasher/testingset")
RUNS = {
    42: {
        "b0": os.path.join(ROOT, "output/experiments/MSHA-B0-U936-S42-FULL15-FULL244-20261109/full244"),
        "mr": os.path.join(ROOT, "output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109/full244"),
    },
    43: {
        "b0": os.path.join(ROOT, "output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/full244"),
        "mr": os.path.join(ROOT, "output/experiments/MSHA-MRONLY-G1-U936-S43-FULL15-FULL244-RETRY3-20261112/full244"),
    },
}


def read_boxes(path):
    boxes = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().replace("\t", ",")
            if line:
                boxes.append(tuple(float(value) for value in line.split(",")))
    return boxes


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def frame_features(pred, gt):
    features = []
    for p, g in zip(pred, gt):
        gx, gy, gw, gh = g
        if gw <= 0 or gh <= 0:
            continue
        px, py, pw, ph = p
        pcx, pcy = px + pw / 2.0, py + ph / 2.0
        gcx, gcy = gx + gw / 2.0, gy + gh / 2.0
        center_px = math.hypot(pcx - gcx, pcy - gcy)
        center_norm = center_px / math.hypot(gw, gh)
        log_area = math.log(max(pw * ph, 1e-6) / (gw * gh))
        features.append(
            {
                "iou": iou(p, g),
                "center_norm": center_norm,
                "center_px": center_px,
                "abs_log_area": abs(log_area),
                "log_area": log_area,
                "dx": pcx - gcx,
                "dy": pcy - gcy,
            }
        )
    return features


def mean(values):
    return statistics.mean(values) if values else float("nan")


def summarize(features):
    n = len(features)
    return {
        "valid_frames": n,
        "iou_mean": mean([x["iou"] for x in features]),
        "center_norm_mean": mean([x["center_norm"] for x in features]),
        "center_px_mean": mean([x["center_px"] for x in features]),
        "abs_log_area_mean": mean([x["abs_log_area"] for x in features]),
        "log_area_mean": mean([x["log_area"] for x in features]),
        "dx_mean": mean([x["dx"] for x in features]),
        "dy_mean": mean([x["dy"] for x in features]),
        "iou_lt_050_rate": sum(x["iou"] < 0.50 for x in features) / n if n else float("nan"),
        "iou_lt_025_rate": sum(x["iou"] < 0.25 for x in features) / n if n else float("nan"),
        "center_gt_025_rate": sum(x["center_norm"] > 0.25 for x in features) / n if n else float("nan"),
        "center_gt_050_rate": sum(x["center_norm"] > 0.50 for x in features) / n if n else float("nan"),
        "center_gt_100_rate": sum(x["center_norm"] > 1.00 for x in features) / n if n else float("nan"),
        "area_gt_150_rate": sum(x["abs_log_area"] > math.log(1.5) for x in features) / n if n else float("nan"),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(GROUPS, encoding="utf-8") as handle:
        groups = list(csv.DictReader(handle))
    group_by_sequence = {row["sequence"]: row["cross_seed_group"] for row in groups}
    attrs = [key for key in groups[0] if key.startswith("attr_")]
    diagnostics = []
    for seed, arms in RUNS.items():
        for row in groups:
            sequence = row["sequence"]
            gt = read_boxes(os.path.join(GT_ROOT, sequence, "infrared.txt"))
            arm_summaries = {}
            for arm, directory in arms.items():
                pred = read_boxes(os.path.join(directory, sequence + ".txt"))
                arm_summaries[arm] = summarize(frame_features(pred, gt))
            record = {
                "sequence": sequence,
                "group": group_by_sequence[sequence],
                "seed": seed,
                "annotation_frames": len(gt),
                "invalid_annotation_frames": len(gt) - arm_summaries["b0"]["valid_frames"],
            }
            for attr in attrs:
                record[attr] = int(row[attr])
            for metric in arm_summaries["b0"]:
                record["b0_" + metric] = arm_summaries["b0"][metric]
                record["mr_" + metric] = arm_summaries["mr"][metric]
                record["delta_" + metric] = arm_summaries["mr"][metric] - arm_summaries["b0"][metric]
            diagnostics.append(record)

    fields = list(diagnostics[0])
    with open(os.path.join(OUT, "per_sequence_box_diagnostics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)

    summary = {"groups": {}, "attributes": {}, "sources": {"groups": GROUPS, "gt_root": GT_ROOT, "runs": RUNS}}
    metric_names = [
        "iou_mean", "center_norm_mean", "center_px_mean", "abs_log_area_mean", "log_area_mean",
        "dx_mean", "dy_mean", "iou_lt_050_rate", "iou_lt_025_rate", "center_gt_025_rate",
        "center_gt_050_rate", "center_gt_100_rate", "area_gt_150_rate",
    ]
    for group in ("consistent_mr_gain", "consistent_mr_regression", "seed_direction_flip"):
        summary["groups"][group] = {}
        for seed in RUNS:
            subset = [x for x in diagnostics if x["group"] == group and x["seed"] == seed]
            summary["groups"][group][str(seed)] = {"n_sequences": len(subset), "n_valid_frames": sum(x["b0_valid_frames"] for x in subset)}
            for metric in metric_names:
                summary["groups"][group][str(seed)][metric + "_b0"] = mean([x["b0_" + metric] for x in subset])
                summary["groups"][group][str(seed)][metric + "_mr"] = mean([x["mr_" + metric] for x in subset])
                summary["groups"][group][str(seed)][metric + "_delta"] = mean([x["delta_" + metric] for x in subset])
    for attr in attrs:
        summary["attributes"][attr[5:]] = {}
        for group in ("consistent_mr_gain", "consistent_mr_regression", "seed_direction_flip"):
            subset = [x for x in groups if x["cross_seed_group"] == group]
            summary["attributes"][attr[5:]][group] = {
                "count": sum(x[attr] == "1" for x in subset),
                "n": len(subset),
                "fraction": sum(x[attr] == "1" for x in subset) / len(subset),
            }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary["groups"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
