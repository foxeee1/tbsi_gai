#!/usr/bin/env python3
"""Paired analysis for the receiver-conditioned message residual quick run."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text

RUN = ROOT / "output/experiments/MSHA-MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115"
LIST = ROOT / "experiments/early_gate_closed_loop_20261026/p1_sot_v3/screen96_max2000.txt"
METRICS = ("auc", "precision", "npr", "op50", "op75")


def metrics(result_dir, sequence):
    pred = torch.tensor(load_text(str(result_dir / (sequence.name + ".txt")),
                                  delimiter=("\t", ","), dtype=np.float64))
    anno = torch.tensor(sequence.ground_truth_rect)
    visible = (torch.tensor(sequence.target_visible, dtype=torch.uint8)
               if sequence.target_visible is not None else None)
    overlap, center, center_norm, _ = calc_seq_err_robust(pred, anno,
                                                           sequence.dataset, visible)
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    return {
        "sequence": sequence.name,
        "frames": int(anno.shape[0]),
        "auc": float(((overlap[:, None] > thresholds[None, :]).sum(0).double() /
                       anno.shape[0]).mean() * 100),
        "precision": float((center <= 20).sum() / anno.shape[0] * 100),
        "npr": float((center_norm <= 0.2).sum() / anno.shape[0] * 100),
        "op50": float((overlap > 0.5).sum() / anno.shape[0] * 100),
        "op75": float((overlap > 0.75).sum() / anno.shape[0] * 100),
    }


def mean(rows):
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def main():
    names = [line.strip() for line in LIST.read_text().splitlines() if line.strip()]
    dataset = {seq.name: seq for seq in get_dataset("lasher_test")}
    sequences = [dataset[name] for name in names]
    control = [metrics(RUN / "eval_control", seq) for seq in sequences]
    candidate = [metrics(RUN / "eval_candidate", seq) for seq in sequences]
    rows = []
    for cand, ctrl in zip(candidate, control):
        if cand["sequence"] != ctrl["sequence"]:
            raise RuntimeError("sequence ordering mismatch")
        row = {"sequence": cand["sequence"], "frames": cand["frames"]}
        for metric in METRICS:
            row["control_" + metric] = ctrl[metric]
            row["candidate_" + metric] = cand[metric]
            row["delta_" + metric] = cand[metric] - ctrl[metric]
        rows.append(row)

    def stats(metric):
        values = np.asarray([row["delta_" + metric] for row in rows], dtype=float)
        return {
            "mean_pp": float(values.mean()),
            "median_pp": float(np.median(values)),
            "wins": int((values > 1e-9).sum()),
            "ties": int((np.abs(values) <= 1e-9).sum()),
            "losses": int((values < -1e-9).sum()),
            "min_pp": float(values.min()),
            "max_pp": float(values.max()),
            "p25_pp": float(np.percentile(values, 25)),
            "p75_pp": float(np.percentile(values, 75)),
            "worst_decile_mean_pp": float(np.mean(np.sort(values)[:max(1, len(values)//10)])),
        }

    control_mean, candidate_mean = mean(control), mean(candidate)
    summary = {
        "run_id": RUN.name,
        "sequence_count": len(rows),
        "frames": int(sum(row["frames"] for row in rows)),
        "control": control_mean,
        "candidate": candidate_mean,
        "candidate_minus_control": {metric: candidate_mean[metric] - control_mean[metric]
                                     for metric in METRICS},
        "stats": {metric: stats(metric) for metric in METRICS},
        "eval_protocol": {
            "precision": "FP32", "tracker_threads": 0, "template_update": False,
            "seed": 42, "cohort": "fixed screen96_max2000 exploratory cohort",
        },
        "scope": "468 successful updates from public pretrained single-stage; one seed; quick diagnostic",
    }
    analysis = RUN / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / "per_sequence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = ["# Receiver-conditioned message residual quick paired analysis", "",
              "| metric | control | candidate | delta (pp) |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        report.append("| %s | %.4f | %.4f | %+.4f |" %
                      (metric, control_mean[metric], candidate_mean[metric],
                       summary["candidate_minus_control"][metric]))
    auc = summary["stats"]["auc"]
    report += ["", "## AUC sequence comparison", "",
               "| mean pp | median pp | wins | ties | losses | min pp | max pp | worst decile mean |",
               "|---:|---:|---:|---:|---:|---:|---:|---:|",
               "| %+.4f | %+.4f | %d | %d | %d | %+.4f | %+.4f | %+.4f |" %
               (auc["mean_pp"], auc["median_pp"], auc["wins"], auc["ties"], auc["losses"],
                auc["min_pp"], auc["max_pp"], auc["worst_decile_mean_pp"]), "",
               "This is a single-seed, fixed 96-sequence exploratory quick result; it is not a full-data or multi-seed claim."]
    (analysis / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
