#!/usr/bin/env python3
"""Pairwise Full244 analysis for the lr_expected MR-only candidate."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.utils.load_text import load_text


RUN = ROOT / "output/experiments/MR-LR-EXPECTED-U936-S42-FULL15-FULL244-20261110"
ANALYSIS = RUN / "analysis"
CANDIDATE = RUN / "full244"
MR = ROOT / "output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109/full244"
B0 = ROOT / "output/experiments/MSHA-B0-U936-S42-FULL15-FULL244-20261109/full244"


def load_metrics(result_dir, sequence):
    path = result_dir / (sequence.name + ".txt")
    if not path.is_file():
        raise FileNotFoundError(path)
    pred = torch.tensor(load_text(str(path), delimiter=("\t", ","), dtype=np.float64))
    anno = torch.tensor(sequence.ground_truth_rect)
    visible = torch.tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
    overlap, center, center_norm, valid = calc_seq_err_robust(pred, anno, sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    success = (overlap.view(-1, 1) > thresholds.view(1, -1)).sum(0).double() / anno.shape[0]
    # Match the existing MR-only reports: sequence AUC is the equal-weight
    # mean over the 21 success thresholds (0.00 through 1.00).
    auc = float(success.mean().item() * 100.0)
    precision = float((center <= 20.0).sum().item() / anno.shape[0] * 100.0)
    norm_precision = float((center_norm <= 0.2).sum().item() / anno.shape[0] * 100.0)
    op50 = float((overlap > 0.50).sum().item() / anno.shape[0] * 100.0)
    op75 = float((overlap > 0.75).sum().item() / anno.shape[0] * 100.0)
    return {
        "sequence": sequence.name,
        "frames": int(anno.shape[0]),
        "auc": auc,
        "precision": precision,
        "norm_precision": norm_precision,
        "op50": op50,
        "op75": op75,
    }


def aggregate(rows):
    keys = ["auc", "precision", "norm_precision", "op50", "op75"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired(rows, left_key, right_key):
    metrics = ["auc", "precision", "norm_precision", "op50", "op75"]
    output = []
    for left, right in zip(rows[left_key], rows[right_key]):
        if left["sequence"] != right["sequence"]:
            raise RuntimeError("sequence order mismatch")
        row = {"sequence": left["sequence"]}
        for metric in metrics:
            row[left_key + "_" + metric] = left[metric]
            row[right_key + "_" + metric] = right[metric]
            row["delta_" + metric] = left[metric] - right[metric]
        output.append(row)
    return output


def stats(deltas):
    values = np.asarray(deltas, dtype=np.float64)
    eps = 1e-9
    top = np.sort(values)[-5:]
    positive_total = values[values > 0].sum()
    return {
        "mean_pp": float(np.mean(values)),
        "median_pp": float(np.median(values)),
        "wins": int(np.sum(values > eps)),
        "ties": int(np.sum(np.abs(values) <= eps)),
        "losses": int(np.sum(values < -eps)),
        "positive_fraction": float(np.mean(values > eps)),
        "top5_sum_pp": float(np.sum(top)),
        "top5_gross_positive_contribution": float(np.sum(top) / positive_total) if positive_total > 0 else None,
        "top5_signed_net_contribution": float(np.sum(top) / values.sum()) if abs(values.sum()) > eps else None,
        "min_pp": float(np.min(values)),
        "p05_pp": float(np.percentile(values, 5)),
        "p25_pp": float(np.percentile(values, 25)),
        "p75_pp": float(np.percentile(values, 75)),
        "p95_pp": float(np.percentile(values, 95)),
    }


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset("lasher_test")
    rows = {"candidate": [], "mr": [], "b0": []}
    for sequence in dataset:
        for key, result_dir in (("candidate", CANDIDATE), ("mr", MR), ("b0", B0)):
            rows[key].append(load_metrics(result_dir, sequence))

    metric_fields = ["sequence", "frames", "auc", "precision", "norm_precision", "op50", "op75"]
    write_csv(ANALYSIS / "per_sequence_metrics.csv", rows["candidate"], metric_fields)
    metrics = ["auc", "precision", "norm_precision", "op50", "op75"]
    paired_b0 = paired(rows, "candidate", "b0")
    paired_mr = paired(rows, "candidate", "mr")
    pair_fields = ["sequence"] + [x for metric in metrics for x in ("candidate_" + metric, "b0_" + metric, "delta_" + metric)]
    write_csv(ANALYSIS / "per_sequence_vs_b0.csv", paired_b0, pair_fields)
    pair_fields_mr = ["sequence"] + [x for metric in metrics for x in ("candidate_" + metric, "mr_" + metric, "delta_" + metric)]
    write_csv(ANALYSIS / "per_sequence_vs_mr.csv", paired_mr, pair_fields_mr)

    completed = json.loads((RUN / "full244/completed.json").read_text())
    summary = {
        "run_id": RUN.name,
        "sequence_count": len(dataset),
        "frames": int(sum(row["frames"] for row in rows["candidate"])),
        "metrics": aggregate(rows["candidate"]),
        "mr_reference_metrics": aggregate(rows["mr"]),
        "b0_reference_metrics": aggregate(rows["b0"]),
        "candidate_minus_b0": stats([row["delta_auc"] for row in paired_b0]),
        "candidate_minus_mr": stats([row["delta_auc"] for row in paired_mr]),
        "fps": completed["throughput_fps"],
        "p50_latency_ms": completed["per_frame_latency_ms_p50"],
        "p95_latency_ms": completed["per_frame_latency_ms_p95"],
        "full244_seconds": completed["seconds"],
        "checkpoint_sha256": json.loads((RUN / "train/checkpoint_u7020.json").read_text())["sha256"],
        "protocol": {"precision": "FP32", "tracker_threads": 0, "template_update": False,
                     "dataset": "LasHeR official test244", "seed": 42},
    }
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    top = sorted(paired_b0, key=lambda row: row["delta_auc"], reverse=True)[:10]
    bottom = sorted(paired_b0, key=lambda row: row["delta_auc"])[:10]
    report = ["# MR lr_expected Full244 Analysis", "", "## Protocol", "",
              "The candidate uses public SOT, G1/U936 training and LasHeR official test244 FP32/threads0/template-update-off evaluation. Relative to the trained MR parent, it changes the pair MLP to a low-rank scorer, score limit from 0.25 to 4, and conserved cross mass to expected-score block acceptance. No trained lr_joint control exists. Six common fusion weight matrices were initialized differently (historical initial hashes exactly reconstructed); this is a descriptive experiment comparison, not a strictly isolated causal ablation. See attribute_failure_audit_20261110/报告.md.", "",
              "## Overall", "", "| metric | candidate | MR-only | B0 | candidate-MR | candidate-B0 |", "|---|---:|---:|---:|---:|---:|"]
    for metric in metrics:
        report.append("| %s | %.4f | %.4f | %.4f | %+.4f | %+.4f |" % (
            metric, summary["metrics"][metric], summary["mr_reference_metrics"][metric], summary["b0_reference_metrics"][metric],
            summary["metrics"][metric] - summary["mr_reference_metrics"][metric], summary["metrics"][metric] - summary["b0_reference_metrics"][metric]))
    report += ["", "## AUC Per-sequence Deltas", "", "| comparison | mean pp | median pp | wins | ties | losses | positive fraction | top5/gross positive | min pp |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in (("candidate-B0", summary["candidate_minus_b0"]), ("candidate-MR", summary["candidate_minus_mr"])):
        report.append("| %s | %+.4f | %+.4f | %d | %d | %d | %.1f%% | %s | %+.4f |" % (name, value["mean_pp"], value["median_pp"], value["wins"], value["ties"], value["losses"], value["positive_fraction"] * 100, "NA" if value["top5_gross_positive_contribution"] is None else "%.2f%%" % (value["top5_gross_positive_contribution"] * 100), value["min_pp"]))
    report += ["", "## Candidate-B0 Largest Gains", "", "| sequence | AUC delta pp |", "|---|---:|"]
    report += ["| %s | %+.4f |" % (row["sequence"], row["delta_auc"]) for row in top]
    report += ["", "## Candidate-B0 Largest Losses", "", "| sequence | AUC delta pp |", "|---|---:|"]
    report += ["| %s | %+.4f |" % (row["sequence"], row["delta_auc"]) for row in bottom]
    report += ["", "## Throughput", "", "- Full244 wall time: %.3f s" % completed["seconds"], "- Tracker throughput: %.4f FPS" % completed["throughput_fps"], "- Per-frame latency P50/P95: %.4f / %.4f ms" % (completed["per_frame_latency_ms_p50"], completed["per_frame_latency_ms_p95"]), "", "The AUC comparison is sequence-equal and single-seed paired evidence. It does not establish seed robustness or a general mechanism claim."]
    (ANALYSIS / "REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
