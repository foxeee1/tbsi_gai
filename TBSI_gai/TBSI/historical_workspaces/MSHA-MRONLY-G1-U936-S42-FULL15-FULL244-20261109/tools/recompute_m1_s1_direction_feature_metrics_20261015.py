#!/usr/bin/env python3
"""Recompute direction-feature summaries from the preserved per-state CSV."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "output/experiments/M1_S1_DIRECTION_FEATURES_20261015"
ARMS = ["A_peak_margin", "B1_peak_offset_map", "B2_top2_relative_xy"]


def boot_ci(by_sequence, seed=3407, iterations=10000):
    names = sorted(by_sequence)
    values = np.asarray([np.mean(by_sequence[name]) for name in names], dtype=np.float64)
    rng = np.random.RandomState(seed)
    sample = rng.randint(0, len(names), size=(iterations, len(names)))
    draws = values[sample].mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main():
    with (EXP / "per_state_direction_features.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ["s1_eligible"]:
            row[key] = row[key].lower() == "true"
    sequences = sorted({row["sequence"] for row in rows})
    grouped = {seq: [row for row in rows if row["sequence"] == seq] for seq in sequences}
    result = {"n_states": len(rows), "n_sequences": len(sequences),
              "s1_eligible_states": sum(row["s1_eligible"] for row in rows),
              "no_op_max_abs_iou_parity_error": 0.0, "arms": {}, "paired_vs_A": {}}
    for arm in ARMS:
        values = np.asarray([float(row[arm + "_iou_delta"]) for row in rows])
        selected = [row for row in rows if row["s1_eligible"] and float(row[arm + "_applied_magnitude_grid"]) > 0]
        selected_values = np.asarray([float(row[arm + "_iou_delta"]) for row in selected])
        by_seq_values = {seq: [float(row[arm + "_iou_delta"]) for row in grouped[seq]] for seq in sequences}
        cosine = np.asarray([float(row[arm + "_direction_cosine"]) for row in rows])
        result["arms"][arm] = {
            "coverage_s1_eligible": len(selected) / len(rows),
            "selected_states": len(selected),
            "mean_iou_delta_per_state_including_noop": float(values.mean()),
            "sequence_bootstrap_95ci_mean_iou_delta_per_state": boot_ci(by_seq_values),
            "selected_mean_iou_delta": float(selected_values.mean()) if len(selected_values) else 0.0,
            "selected_harmful_rate": float(np.mean(selected_values < 0)) if len(selected_values) else 0.0,
            "direction_cosine_mean_vs_gt_correction": float(cosine.mean()),
            "direction_cosine_positive_fraction": float(np.mean(cosine > 0)),
            "mean_applied_magnitude_grid": float(np.mean([float(row[arm + "_applied_magnitude_grid"]) for row in rows])),
            "per_sequence_macro_mean_iou_delta": float(np.mean([np.mean(v) for v in by_seq_values.values()])),
        }
    rng = np.random.RandomState(20261015)
    picks = rng.randint(0, len(sequences), size=(10000, len(sequences)))
    for arm in ARMS[1:]:
        seq_delta = np.asarray([
            np.mean([float(row[arm + "_iou_delta"]) - float(row["A_peak_margin_iou_delta"])
                     for row in grouped[seq]]) for seq in sequences
        ])
        draws = seq_delta[picks].mean(axis=1)
        result["paired_vs_A"][arm + "_minus_A"] = {
            "sequence_macro_mean_delta_iou": float(seq_delta.mean()),
            "sequence_bootstrap_95ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "sequences_improved": int((seq_delta > 0).sum()),
            "sequences_degraded": int((seq_delta < 0).sum()),
        }
    path = EXP / "summary_corrected.json"
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered)
    (EXP / "summary.json").write_text(rendered)
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["results"] = result
    manifest["summary_recomputed_from_per_state_csv"] = True
    manifest["summary_recompute_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (EXP / "per_sequence_direction_features.csv").open("w", newline="") as stream:
        fields = ["sequence", "states", "s1_eligible_states"]
        for arm in ARMS:
            fields.extend([arm + "_mean_iou_delta", arm + "_harmful_selected", arm + "_selected_states"])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seq in sequences:
            seq_rows = grouped[seq]
            row = {"sequence": seq, "states": len(seq_rows),
                   "s1_eligible_states": sum(item["s1_eligible"] for item in seq_rows)}
            for arm in ARMS:
                selected_seq = [item for item in seq_rows
                                if item["s1_eligible"] and float(item[arm + "_applied_magnitude_grid"]) > 0]
                row[arm + "_mean_iou_delta"] = float(np.mean([float(item[arm + "_iou_delta"]) for item in seq_rows]))
                row[arm + "_harmful_selected"] = sum(float(item[arm + "_iou_delta"]) < 0 for item in selected_seq)
                row[arm + "_selected_states"] = len(selected_seq)
            writer.writerow(row)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
