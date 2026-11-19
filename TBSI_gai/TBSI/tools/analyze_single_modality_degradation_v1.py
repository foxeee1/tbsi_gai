#!/usr/bin/env python
"""Offline analysis for the single-modality degradation four-cell diagnostic.

This script reads only the diagnostic CSV. It does not load a checkpoint, use
CUDA, or read the cancelled blind-review assets. Sequence-cluster bootstrap is
used for uncertainty; adjacent frames are never treated as independent
clusters.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np


CONDITIONS = ("00", "10", "01", "11")
SUITES = ("blur", "lowlight", "contrast")


def read_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"event_id", "seq", "suite", "condition", "offset", "iou", "status"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError("missing CSV columns: {}".format(sorted(missing)))
    if any(row["status"] != "OK" for row in rows):
        raise ValueError("input contains non-OK rows")
    for row in rows:
        row["offset"] = int(row["offset"])
        row["iou"] = float(row["iou"])
    return rows


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def median_or_nan(values):
    return float(np.median(values)) if values else float("nan")


def quantile_or_nan(values, q):
    return float(np.quantile(values, q)) if values else float("nan")


def ci_text(values):
    if not values:
        return {"low": None, "high": None}
    return {"low": quantile_or_nan(values, 0.025), "high": quantile_or_nan(values, 0.975)}


def paired_cells(rows, suite, offsets):
    cells = defaultdict(dict)
    for row in rows:
        if row["suite"] == suite and row["offset"] in offsets:
            cells[(row["event_id"], row["seq"], row["offset"])][row["condition"]] = row["iou"]
    incomplete = [key for key, values in cells.items() if set(values) != set(CONDITIONS)]
    if incomplete:
        raise ValueError("incomplete four-cell groups for {}: {}".format(suite, incomplete[:3]))
    return cells


def event_metrics(cells):
    """Return one row per event, averaging offsets before sequence summaries."""
    by_event = defaultdict(list)
    for (event_id, seq, offset), values in cells.items():
        i00, i10, i01, i11 = (values[c] for c in CONDITIONS)
        by_event[(event_id, seq)].append({
            "offset": offset,
            "iou_00": i00,
            "iou_10": i10,
            "iou_01": i01,
            "iou_11": i11,
            "loss_10": i00 - i10,
            "loss_01": i00 - i01,
            "loss_11": i00 - i11,
            "iou_interaction": i11 - i10 - i01 + i00,
            "error_interaction": -i11 + i10 + i01 - i00,
        })
    output = []
    for (event_id, seq), points in sorted(by_event.items()):
        offsets = sorted(points, key=lambda x: x["offset"])
        item = {"event_id": event_id, "seq": seq, "n_offsets": len(offsets)}
        for field in ("iou_00", "iou_10", "iou_01", "iou_11", "loss_10", "loss_01",
                      "loss_11", "iou_interaction", "error_interaction"):
            item[field] = mean_or_nan([point[field] for point in offsets])
        output.append(item)
    return output


def sequence_summary(event_rows):
    grouped = defaultdict(list)
    for row in event_rows:
        grouped[row["seq"]].append(row)
    output = []
    for seq, values in sorted(grouped.items()):
        item = {"seq": seq, "n_events": len(values)}
        fields = [key for key in values[0] if key not in ("event_id", "seq", "n_offsets")]
        for field in fields:
            item[field] = mean_or_nan([value[field] for value in values])
        output.append(item)
    return output


def cluster_bootstrap(seq_rows, field, seed=42, n_boot=5000):
    values = np.asarray([row[field] for row in seq_rows], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n_sequences": 0, "estimate": None, "ci95": {"low": None, "high": None}}
    rng = np.random.RandomState(seed)
    sampled = rng.randint(0, len(values), size=(n_boot, len(values)))
    boot = values[sampled].mean(axis=1)
    return {"n_sequences": int(len(values)), "estimate": float(values.mean()),
            "ci95": {"low": float(np.quantile(boot, 0.025)),
                     "high": float(np.quantile(boot, 0.975))},
            "bootstrap_seed": seed, "bootstrap_replicates": n_boot}


def summarize_field(event_rows, seq_rows, field):
    values = [row[field] for row in event_rows]
    order = np.argsort(np.asarray(values, dtype=float))[::-1]
    top3 = [values[index] for rank, index in enumerate(order) if rank >= min(3, len(values))]
    keep_two_thirds = [values[index] for rank, index in enumerate(order)
                       if rank >= int(math.ceil(len(values) / 3.0))]
    return {
        "event_mean": mean_or_nan(values),
        "event_median": median_or_nan(values),
        "event_ci95_iid_descriptive": ci_text(values),
        "event_mean_drop_top_3": mean_or_nan(top3),
        "event_mean_drop_top_third": mean_or_nan(keep_two_thirds),
        "sequence_cluster": cluster_bootstrap(seq_rows, field),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_path, output_dir, bootstrap_replicates):
    rows = read_rows(input_path)
    os.makedirs(output_dir, exist_ok=True)
    event_csv_rows = []
    sequence_csv_rows = []
    report = {
        "experiment": "single_modality_degradation_v1",
        "input_csv": os.path.abspath(input_path),
        "input_sha256": hashlib.sha256(open(input_path, "rb").read()).hexdigest(),
        "bootstrap": {"unit": "sequence", "seed": 42, "replicates": bootstrap_replicates},
        "suites": {},
    }
    for suite in SUITES:
        suite_rows = [row for row in rows if row["suite"] == suite]
        offsets = sorted(set(row["offset"] for row in suite_rows))
        if offsets != list(range(6)):
            raise ValueError("{} offsets are {} rather than 0..5".format(suite, offsets))
        suite_report = {"n_rows": len(suite_rows), "n_events": 0, "n_sequences": 0,
                        "offsets": offsets, "by_offset": {}, "post_only": {}}
        for offset in offsets:
            event_rows = event_metrics(paired_cells(suite_rows, suite, {offset}))
            seq_rows = sequence_summary(event_rows)
            suite_report["n_events"] = len(event_rows)
            suite_report["n_sequences"] = len(seq_rows)
            fields = ("iou_00", "iou_10", "iou_01", "iou_11", "loss_10", "loss_01",
                      "loss_11", "iou_interaction", "error_interaction")
            suite_report["by_offset"][str(offset)] = {
                field: summarize_field(event_rows, seq_rows, field) for field in fields
            }
            for item in event_rows:
                event_csv_rows.append(dict({"suite": suite, "offset": offset}, **item))
            for item in seq_rows:
                sequence_csv_rows.append(dict({"suite": suite, "offset": offset}, **item))
        post_events = event_metrics(paired_cells(suite_rows, suite, set(range(1, 6))))
        post_sequences = sequence_summary(post_events)
        fields = ("iou_00", "iou_10", "iou_01", "iou_11", "loss_10", "loss_01",
                  "loss_11", "iou_interaction", "error_interaction")
        suite_report["post_only"] = {field: summarize_field(post_events, post_sequences, field)
                                      for field in fields}
        report["suites"][suite] = suite_report
    write_csv(os.path.join(output_dir, "event_summary.csv"), event_csv_rows)
    write_csv(os.path.join(output_dir, "sequence_summary.csv"), sequence_csv_rows)
    with open(os.path.join(output_dir, "analysis.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"status": "COMPLETED", "output_dir": os.path.abspath(output_dir),
                      "events": 72, "sequences_per_suite": 24}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    analyze(args.input, args.output_dir, args.bootstrap_replicates)


if __name__ == "__main__":
    main()
