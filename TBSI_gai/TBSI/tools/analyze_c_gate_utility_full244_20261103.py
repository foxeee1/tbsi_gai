#!/usr/bin/env python3
"""Posthoc aggregation for the saved all-sequence fixed-state C gate probe."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output/experiments/EG_C_GATE_UTILITY_FIXEDSTATE_FULL244_RETRY1_20261103"
RAW = RUN / "per_case.jsonl"


def bootstrap_mean(values, seed=20261103, draws=20000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        idx = rng.randint(0, len(values), size=(count, len(values)))
        samples[start:start + count] = values[idx].mean(axis=1)
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    if len(rows) != 732 or manifest["sequence_count"] != 244 or manifest["case_count"] != 732:
        raise RuntimeError("raw coverage mismatch: expected 732 fixed-state comparisons on 244 sequences")
    if not all(row.get("state_crop_parity") and row.get("native_aa_exact") for row in rows):
        raise RuntimeError("state/crop or native A/A parity failure exists")
    drift = set(manifest["m1_prediction_known_provenance_drift_sequences"])
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"]["sequence"], []).append(row)
    if len(grouped) != 244 or any(len(items) != 3 for items in grouped.values()):
        raise RuntimeError("expected exactly 3 frames for each of 244 unique sequences")

    sequence_rows = []
    for name in manifest["sequences"]:
        items = grouped[name]
        diou = np.asarray([x["delta_iou_native_minus_identity"] for x in items], dtype=np.float64)
        dcenter = np.asarray([x["delta_center_error_identity_minus_native"] for x in items], dtype=np.float64)
        native = np.asarray([x["native_a"]["iou"] for x in items], dtype=np.float64)
        identity = np.asarray([x["identity"]["iou"] for x in items], dtype=np.float64)
        sequence_rows.append({
            "sequence": name,
            "m1_prediction_known_provenance_drift": name in drift,
            "frames": ";".join(str(x["case"]["frame"]) for x in items),
            "anchor_fractions": ";".join("%.6f" % x["case"]["anchor_fraction"] for x in items),
            "native_iou_mean": float(native.mean()),
            "identity_iou_mean": float(identity.mean()),
            "delta_iou_mean": float(diou.mean()),
            "delta_iou_median": float(np.median(diou)),
            "delta_center_error_identity_minus_native_mean": float(dcenter.mean()),
        })
    seq_delta = np.asarray([x["delta_iou_mean"] for x in sequence_rows], dtype=np.float64)
    clean_rows = [x for x in sequence_rows if not x["m1_prediction_known_provenance_drift"]]
    clean_delta = np.asarray([x["delta_iou_mean"] for x in clean_rows], dtype=np.float64)
    frame_delta = np.asarray([x["delta_iou_native_minus_identity"] for x in rows], dtype=np.float64)
    frame_center = np.asarray([x["delta_center_error_identity_minus_native"] for x in rows], dtype=np.float64)

    layer_values = {}
    for row in rows:
        for layer in row["native_a"]["gate_log_gain"]:
            layer_values.setdefault(layer["layer_id"], []).append(layer)
    gain_summary = {}
    for layer_id, values in sorted(layer_values.items()):
        means = np.asarray([x["mean_by_source"] for x in values], dtype=np.float64)
        stds = np.asarray([x["std_by_source"] for x in values], dtype=np.float64)
        gain_summary[str(layer_id)] = {
            "mean_log_gain_by_source": means.mean(axis=0).tolist(),
            "mean_within_token_std_by_source": stds.mean(axis=0).tolist(),
            "mean_absolute_log_gain": float(np.abs(means).mean()),
        }

    summary = {
        "status": "FULL244_FIXEDSTATE_GATE_UTILITY_POSTHOC_COMPLETE",
        "sequence_count": len(sequence_rows),
        "frame_comparisons": len(rows),
        "frames_per_sequence": 3,
        "m1_known_provenance_drift_sequences": len(drift),
        "m1_known_provenance_clean_sequences": len(clean_rows),
        "all_native_aa_exact": all(x["native_aa_exact"] for x in rows),
        "all_state_crop_parity": all(x["state_crop_parity"] for x in rows),
        "frame_delta_iou": {
            "mean": float(frame_delta.mean()), "median": float(np.median(frame_delta)),
            "mean_absolute": float(np.abs(frame_delta).mean()),
            "native_better": int((frame_delta > 0).sum()),
            "identity_better": int((frame_delta < 0).sum()),
            "ties": int((frame_delta == 0).sum()),
        },
        "equal_sequence_delta_iou": {
            "all244_mean": float(seq_delta.mean()),
            "all244_median": float(np.median(seq_delta)),
            "all244_95pct_sequence_bootstrap_ci": bootstrap_mean(seq_delta),
            "all244_native_better": int((seq_delta > 0).sum()),
            "all244_identity_better": int((seq_delta < 0).sum()),
            "all244_ties": int((seq_delta == 0).sum()),
            "clean237_mean": float(clean_delta.mean()),
            "clean237_median": float(np.median(clean_delta)),
            "clean237_95pct_sequence_bootstrap_ci": bootstrap_mean(clean_delta, seed=20261104),
            "clean237_native_better": int((clean_delta > 0).sum()),
            "clean237_identity_better": int((clean_delta < 0).sum()),
            "clean237_ties": int((clean_delta == 0).sum()),
        },
        "delta_center_error_identity_minus_native_pixels": {
            "mean": float(frame_center.mean()), "median": float(np.median(frame_center)),
            "native_better": int((frame_center > 0).sum()),
            "identity_better": int((frame_center < 0).sum()),
            "ties": int((frame_center == 0).sum()),
        },
        "native_gate_log_gain": gain_summary,
        "largest_native_iou_gains": sorted(sequence_rows, key=lambda x: x["delta_iou_mean"], reverse=True)[:10],
        "largest_native_iou_losses": sorted(sequence_rows, key=lambda x: x["delta_iou_mean"])[:10],
        "anchor_definition": manifest["state_anchor"],
        "interpretation_boundary": manifest["interpretation_boundary"],
        "limitation": "Immediate fixed-state C gate utility only on protected M1 trajectory anchors; no C free-running state claim, no retrained pairwise gate claim, no closed-loop benefit or seed uncertainty.",
        "raw_jsonl_sha256": sha256_file(RAW),
        "elapsed_seconds_from_run_manifest": float(time.time() - manifest["start_time"]),
    }
    (RUN / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (RUN / "per_sequence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sequence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sequence_rows)
    complete = {
        "status": summary["status"], "sequence_count": len(sequence_rows),
        "frame_comparisons": len(rows), "all_native_aa_exact": summary["all_native_aa_exact"],
        "all_state_crop_parity": summary["all_state_crop_parity"],
        "summary_sha256": sha256_file(RUN / "summary.json"),
        "per_sequence_sha256": sha256_file(RUN / "per_sequence.csv"),
        "raw_jsonl_sha256": sha256_file(RAW),
    }
    (RUN / "completed.json").write_text(json.dumps(complete, indent=2) + "\n")
    (RUN / "progress.json").write_text(json.dumps({
        "completed_cases": len(rows), "total_cases": len(rows),
        "completed_sequences": len(sequence_rows), "elapsed_seconds": summary["elapsed_seconds_from_run_manifest"],
        "posthoc_reconciled": True,
    }, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("largest_native_iou_gains", "largest_native_iou_losses")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
