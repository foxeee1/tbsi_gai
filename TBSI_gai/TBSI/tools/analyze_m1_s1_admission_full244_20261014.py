#!/usr/bin/env python3
"""Summarize official per-sequence S1 admission results by frozen cohorts."""

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014"
LOW_LIST = ROOT / "output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/s1_full62/sequence_list.txt"
NORMAL_LIST = ROOT / "output/experiments/M1_SINGLE_SIGNAL_NORMAL20_20261012/s1_full20/sequence_list.txt"
METRICS = ("AUC", "PR", "NPR", "OP50", "OP75")
ARMS = ("S1_CONTROL", "S1_SELECTIVE_RISK", "S1_MOTION_INNOVATION")


def read_names(path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main():
    source = EXP / "evaluation/per_sequence_official.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = {row["sequence"] for row in rows}
    low = read_names(LOW_LIST)
    normal = read_names(NORMAL_LIST)
    if len(rows) != 244 or len(names) != 244 or len(low) != 62 or len(normal) != 20:
        raise ValueError("Unexpected full-test or cohort membership size")
    if low & normal or not (low | normal) <= names:
        raise ValueError("Cohort lists overlap or contain unknown sequences")

    cohorts = {
        "low_auc62": low,
        "normal20_sentinel": normal,
        "remaining162": names - low - normal,
        "all244": names,
    }
    output = {
        "source": str(source),
        "n_sequences": len(rows),
        "cohorts": {},
    }
    for cohort, members in cohorts.items():
        selected = [row for row in rows if row["sequence"] in members]
        entry = {"n_sequences": len(selected), "arms": {}}
        for arm in ARMS:
            arm_result = {}
            for metric in METRICS:
                values = [float(row[f"{arm}_{metric}"]) for row in selected]
                delta_m1 = [float(row[f"{arm}_{metric}"]) - float(row[f"M1_{metric}"]) for row in selected]
                delta_control = [float(row[f"{arm}_{metric}"]) - float(row[f"S1_CONTROL_{metric}"]) for row in selected]
                arm_result[metric] = {
                    "mean": mean(values),
                    "delta_vs_m1_mean": mean(delta_m1),
                    "delta_vs_control_mean": mean(delta_control),
                }
            auc_delta = [float(row[f"{arm}_AUC"]) - float(row["S1_CONTROL_AUC"]) for row in selected]
            arm_result["auc_wins_losses_ties_vs_control"] = [
                sum(value > 1e-8 for value in auc_delta),
                sum(value < -1e-8 for value in auc_delta),
                sum(abs(value) <= 1e-8 for value in auc_delta),
            ]
            entry["arms"][arm] = arm_result
        output["cohorts"][cohort] = entry

    target = EXP / "evaluation/cohort_summary.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(target)


if __name__ == "__main__":
    main()
