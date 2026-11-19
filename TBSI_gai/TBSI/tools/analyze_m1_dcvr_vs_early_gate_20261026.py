#!/usr/bin/env python3
"""Compare retained DCVR and Early-conditioned gate per-sequence analyses."""

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = {
    "auc": ("auc_delta", "delta_auc", "auc_parent", "parent_auc",
            "auc_candidate", "candidate_auc"),
    "ao": ("mean_iou_delta", "delta_ao", "mean_iou_parent", "parent_ao",
           "mean_iou_candidate", "candidate_ao"),
    "op50": ("op50_delta", "delta_op50", "op50_parent", "parent_op50",
             "op50_candidate", "candidate_op50"),
    "op75": ("op75_delta", "delta_op75", "op75_parent", "parent_op75",
             "op75_candidate", "candidate_op75"),
    "precision": ("precision_delta", "delta_precision", "precision_parent",
                  "parent_precision", "precision_candidate", "candidate_precision"),
    "norm_precision": ("norm_precision_delta", "delta_norm_precision",
                        "norm_precision_parent", "parent_norm_precision",
                        "norm_precision_candidate", "candidate_norm_precision"),
}
TOL = 1e-10


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = {row["sequence"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError("No sequence rows in {}".format(path))
    return rows


def number(row, key):
    return float(row[key])


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "positive": sum(value > TOL for value in values),
        "negative": sum(value < -TOL for value in values),
        "tie": sum(abs(value) <= TOL for value in values),
    }


def pearson(xs, ys):
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    std_x = statistics.pstdev(xs)
    std_y = statistics.pstdev(ys)
    if std_x == 0.0 or std_y == 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / len(xs)
    return covariance / (std_x * std_y)


def direction_group(x, y):
    if x > TOL and y > TOL:
        return "both_up"
    if x < -TOL and y < -TOL:
        return "both_down"
    if x > TOL and y < -TOL:
        return "early_up_dcvr_down"
    if x < -TOL and y > TOL:
        return "early_down_dcvr_up"
    return "tie_boundary"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-csv", required=True)
    parser.add_argument("--dcvr-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    early_rows = read_rows(args.early_csv)
    dcvr_rows = read_rows(args.dcvr_csv)
    if set(early_rows) != set(dcvr_rows):
        raise ValueError("Sequence sets differ between analysis inputs")
    sequences = sorted(early_rows)

    parent_checks = {}
    for metric, columns in METRICS.items():
        _, _, early_parent_col, dcvr_parent_col, _, _ = columns
        diffs = [abs(number(early_rows[seq], early_parent_col) -
                     number(dcvr_rows[seq], dcvr_parent_col)) for seq in sequences]
        parent_checks[metric] = {
            "max_abs_difference": max(diffs),
            "exact_matches": sum(diff <= TOL for diff in diffs),
        }
        if max(diffs) > TOL:
            raise ValueError("Parent rows differ for {}".format(metric))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "per_sequence_comparison.csv"
    fieldnames = ["sequence", "frames_early", "frames_dcvr"]
    for metric in METRICS:
        fieldnames.extend([
            "parent_{}".format(metric), "early_candidate_{}".format(metric),
            "early_delta_{}".format(metric), "dcvr_candidate_{}".format(metric),
            "dcvr_delta_{}".format(metric), "dcvr_minus_early_delta_{}".format(metric),
        ])
    fieldnames.extend(["auc_direction_group", "ao_direction_group"])

    output_rows = []
    for sequence in sequences:
        early = early_rows[sequence]
        dcvr = dcvr_rows[sequence]
        out = {
            "sequence": sequence,
            "frames_early": early.get("frames", "NA"),
            "frames_dcvr": dcvr.get("frames", dcvr.get("valid_frames", "NA")),
        }
        deltas = {}
        for metric, columns in METRICS.items():
            early_delta_col, dcvr_delta_col, early_parent_col, _, early_candidate_col, dcvr_candidate_col = columns
            early_delta = number(early, early_delta_col)
            dcvr_delta = number(dcvr, dcvr_delta_col)
            out["parent_{}".format(metric)] = number(early, early_parent_col)
            out["early_candidate_{}".format(metric)] = number(early, early_candidate_col)
            out["early_delta_{}".format(metric)] = early_delta
            out["dcvr_candidate_{}".format(metric)] = number(dcvr, dcvr_candidate_col)
            out["dcvr_delta_{}".format(metric)] = dcvr_delta
            out["dcvr_minus_early_delta_{}".format(metric)] = dcvr_delta - early_delta
            deltas[metric] = (early_delta, dcvr_delta)
        out["auc_direction_group"] = direction_group(*deltas["auc"])
        out["ao_direction_group"] = direction_group(*deltas["ao"])
        output_rows.append(out)

    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metrics_summary = {}
    for metric, columns in METRICS.items():
        early_delta_col, dcvr_delta_col = columns[:2]
        early_values = [number(early_rows[seq], early_delta_col) for seq in sequences]
        dcvr_values = [number(dcvr_rows[seq], dcvr_delta_col) for seq in sequences]
        metrics_summary[metric] = {
            "early_gate": summarize(early_values),
            "dcvr": summarize(dcvr_values),
            "delta_pearson": pearson(early_values, dcvr_values),
        }

    groups = {}
    for metric, group_key in (("auc", "auc_direction_group"),
                              ("ao", "ao_direction_group")):
        groups[metric] = {}
        for group in ("both_up", "both_down", "early_up_dcvr_down",
                      "early_down_dcvr_up", "tie_boundary"):
            selected = [row for row in output_rows if row[group_key] == group]
            groups[metric][group] = {
                "count": len(selected),
                "early_mean_delta": (statistics.mean(row["early_delta_{}".format(metric)]
                                                     for row in selected) if selected else None),
                "dcvr_mean_delta": (statistics.mean(row["dcvr_delta_{}".format(metric)]
                                                    for row in selected) if selected else None),
                "sequences": [row["sequence"] for row in selected],
            }

    summary = {
        "analysis": "descriptive comparison of retained prediction-directory CSVs",
        "early_input": str(Path(args.early_csv)),
        "dcvr_input": str(Path(args.dcvr_csv)),
        "sequence_count": len(sequences),
        "same_sequence_set": True,
        "common_parent_row_check": parent_checks,
        "protected_m1_anchor_status": "UNRESOLVED: current M1 output directory does not reproduce fixed protected metrics",
        "causal_attribution_valid": False,
        "metrics": metrics_summary,
        "direction_groups": groups,
        "limitations": [
            "The Early-gate prediction directory has known per-sequence provenance drift.",
            "The current M1 parent directory does not reproduce the protected M1 aggregate metrics.",
            "The shared parent rows establish arithmetic pairing only, not a verified common replay.",
            "The candidate runs do not isolate a single operator under a verified identical training and evaluation contract.",
        ],
        "output_csv": str(comparison_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({
        "sequence_count": len(sequences),
        "auc": metrics_summary["auc"],
        "ao": metrics_summary["ao"],
        "auc_groups": {key: value["count"] for key, value in groups["auc"].items()},
        "parent_rows_exact": {key: value["exact_matches"] for key, value in parent_checks.items()},
        "output_csv": str(comparison_path),
        "summary_json": str(summary_path),
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
