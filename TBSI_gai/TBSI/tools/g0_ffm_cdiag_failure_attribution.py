#!/usr/bin/env python3
"""Read-only attribution audit for the archived Cdiag seed runs.

This script intentionally does not run a model or select a new subset. It
joins the existing sequence metrics, hard-event records, tracking boxes,
checkpoints, and training diagnostics.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ADAPTER_PREFIX = "backbone.post_tbsi_residual."


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def box_iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center(box):
    return float(box[0] + box[2] / 2.0), float(box[1] + box[3] / 2.0)


def load_boxes(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rows.append([float(x) for x in line.replace(",", "\t").split()])
    return rows


def event_error(event):
    pred = event["pred_box"]
    gt = event["gt_box"]
    px, py = center(pred)
    gx, gy = center(gt)
    area_ratio = (pred[2] * pred[3]) / max(1e-12, gt[2] * gt[3])
    aspect_ratio = (pred[2] / max(1e-12, pred[3])) / (gt[2] / max(1e-12, gt[3]))
    # Fixed descriptive buckets, chosen before reading candidate deltas.
    center_bucket = "center_le_1_token" if event["center_distance_tokens"] <= 1.0 else "center_gt_1_token"
    scale_bucket = "area_ratio_0.67_1.50" if 2.0 / 3.0 <= area_ratio <= 1.5 else "area_ratio_outside"
    if center_bucket == "center_le_1_token" and scale_bucket == "area_ratio_0.67_1.50":
        error_type = "neither_dominant_by_fixed_bucket"
    elif center_bucket == "center_gt_1_token" and scale_bucket == "area_ratio_outside":
        error_type = "center_and_scale"
    elif center_bucket == "center_gt_1_token":
        error_type = "center_dominant_by_fixed_bucket"
    else:
        error_type = "scale_dominant_by_fixed_bucket"
    return {
        "center_distance_tokens": float(event["center_distance_tokens"]),
        "pred_area_over_gt_area": float(area_ratio),
        "pred_aspect_over_gt_aspect": float(aspect_ratio),
        "center_bucket": center_bucket,
        "scale_bucket": scale_bucket,
        "error_type": error_type,
        "m1_iou": float(event["m1_iou"]),
        "target_in_crop": bool(event.get("target_in_crop", True)),
    }


def read_sequence_metrics(path):
    d = read_json(path)
    result = {}
    if "per_sequence_deltas" in d:
        for row in d["per_sequence_deltas"]["Cdiag"]:
            result[row["sequence"]] = row
    else:
        # The seed3407 run predates the multi-condition analysis schema.
        for row in d["per_sequence"]:
            result[row["sequence"]] = {
                "sequence": row["sequence"],
                "delta_auc": row["delta_auc"],
                "delta_iou": row["delta_iou"],
            }
    return result


def checkpoint_summary(path, init_path):
    ckpt = torch.load(path, map_location="cpu")
    init = torch.load(init_path, map_location="cpu")
    state = ckpt["net"]
    init_state = init["net"]
    adapter = {}
    for key, value in state.items():
        if key.startswith(ADAPTER_PREFIX):
            arr = value.detach().float()
            adapter[key] = {
                "shape": list(value.shape),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "l2": float(torch.linalg.vector_norm(arr)),
                "abs_max": float(arr.abs().max()),
            }
    non_adapter_diffs = []
    missing = []
    for key, value in init_state.items():
        if key.startswith(ADAPTER_PREFIX):
            continue
        if key not in state:
            missing.append(key)
            continue
        diff = (state[key].detach().float() - value.detach().float()).abs()
        if diff.numel() and float(diff.max()) != 0.0:
            non_adapter_diffs.append({"key": key, "max_abs_diff": float(diff.max()), "l2_diff": float(torch.linalg.vector_norm(diff))})
    return {
        "checkpoint": str(path),
        "epoch": ckpt.get("epoch"),
        "adapter": adapter,
        "non_adapter_state_diff_count": len(non_adapter_diffs),
        "non_adapter_state_diffs": non_adapter_diffs[:20],
        "non_adapter_missing_count": len(missing),
        "optimizer_param_groups": len(ckpt.get("optimizer", {}).get("param_groups", [])),
        "settings_repr": repr(ckpt.get("settings", {})),
    }


def read_csv_summary(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))
    if not rows:
        return {"rows": 0}
    numeric = {}
    for key in rows[0]:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (TypeError, ValueError):
                pass
        if vals:
            numeric[key] = {"first": vals[0], "last": vals[-1], "mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    return {"rows": len(rows), "numeric": numeric}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--events", required=True)
    ap.add_argument("--seed42-analysis", required=True)
    ap.add_argument("--seed3407-analysis", required=True)
    ap.add_argument("--seed42-windows", required=True)
    ap.add_argument("--seed3407-windows", required=True)
    ap.add_argument("--c0-dir", required=True)
    ap.add_argument("--candidate42-dir", required=True)
    ap.add_argument("--candidate3407-dir", required=True)
    ap.add_argument("--checkpoint42", required=True)
    ap.add_argument("--checkpoint3407", required=True)
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--seed42-gradient", required=True)
    ap.add_argument("--seed3407-gradient", required=True)
    ap.add_argument("--seed42-signals", required=True)
    ap.add_argument("--seed3407-signals", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    events = []
    for line in Path(args.events).read_text().splitlines():
        event = json.loads(line)
        if "rescue_failure" in event.get("groups", []):
            event = dict(event)
            event["error_decomposition"] = event_error(event)
            events.append(event)

    sequence_metrics = {
        "seed42": read_sequence_metrics(args.seed42_analysis),
        "seed3407": read_sequence_metrics(args.seed3407_analysis),
    }
    evaluated_sequences = set(sequence_metrics["seed42"]) & set(sequence_metrics["seed3407"])
    events = [event for event in events if event["sequence"] in evaluated_sequences]
    sequence_rows = []
    names = sorted(set(sequence_metrics["seed42"]) | set(sequence_metrics["seed3407"]))
    for name in names:
        a = sequence_metrics["seed42"].get(name, {})
        b = sequence_metrics["seed3407"].get(name, {})
        d42 = float(a.get("delta_auc", float("nan")))
        d3407 = float(b.get("delta_auc", float("nan")))
        sequence_rows.append({
            "sequence": name,
            "seed42_delta_auc": d42,
            "seed3407_delta_auc": d3407,
            "direction_agree": bool((d42 > 0 and d3407 > 0) or (d42 < 0 and d3407 < 0)),
            "seed42_delta_iou": float(a.get("delta_iou", float("nan"))),
            "seed3407_delta_iou": float(b.get("delta_iou", float("nan"))),
        })

    box_rows = []
    dirs = {"C0": Path(args.c0_dir), "seed42": Path(args.candidate42_dir), "seed3407": Path(args.candidate3407_dir)}
    for event in events:
        seq = event["sequence"]
        row = {"event_id": event["event_id"], "sequence": seq, "frame_1based": event["frame"], **event["error_decomposition"]}
        c0_path = dirs["C0"] / (seq + ".txt")
        base = load_boxes(c0_path)
        event_index = max(0, int(event["frame"]) - 1)
        row["available_tracking_frames"] = len(base)
        row["first_divergence_frame_1based"] = {}
        row["first_large_event_iou_drop_unavailable"] = True
        for label in ["seed42", "seed3407"]:
            cand = load_boxes(dirs[label] / (seq + ".txt"))
            n = min(len(base), len(cand))
            divergences = []
            for i in range(n):
                delta = max(abs(base[i][j] - cand[i][j]) for j in range(4))
                if delta > 1.0:
                    divergences.append(i + 1)
            row["first_divergence_frame_1based"][label] = divergences[0] if divergences else None
            row[label + "_event_frame_box_iou"] = box_iou(cand[event_index], event["gt_box"]) if event_index < len(cand) else None
            row[label + "_event_frame_vs_c0_max_box_delta"] = max(abs(base[event_index][j] - cand[event_index][j]) for j in range(4)) if event_index < n else None
        box_rows.append(row)

    categories = {}
    for row in box_rows:
        key = row["error_type"]
        categories.setdefault(key, []).append(row)
    category_summary = {}
    for key, rows in categories.items():
        category_summary[key] = {
            "events": len(rows),
            "sequences": len({x["sequence"] for x in rows}),
            "mean_m1_iou": float(np.mean([x["m1_iou"] for x in rows])),
            "mean_center_distance_tokens": float(np.mean([x["center_distance_tokens"] for x in rows])),
            "mean_pred_area_over_gt_area": float(np.mean([x["pred_area_over_gt_area"] for x in rows])),
        }

    result = {
        "parent": "G0-FFM-CDIAG-SEED3407-STOP-20261005",
        "question": "Distinguish feature/readout signal, train-seed sensitivity, and closed-loop failure mode using only archived assets.",
        "read_only": True,
        "selection_contract": "All rescue_failure events from the already frozen G0-Hard events.jsonl; no reselection by candidate results.",
        "sequence_pair_table": sequence_rows,
        "sequence_direction_agreement": {
            "n_sequences": len(sequence_rows),
            "same_direction": int(sum(x["direction_agree"] for x in sequence_rows)),
            "opposite_or_tie": int(sum(not x["direction_agree"] for x in sequence_rows)),
        },
        "rescue_event_error_decomposition": {
            "n_events": len(box_rows),
            "n_sequences": len({x["sequence"] for x in box_rows}),
            "fixed_bucket_definition": "center <= 1 token versus > 1 token; area ratio in [2/3, 1.5] versus outside",
            "categories": category_summary,
            "rows": box_rows,
        },
        "checkpoint_and_training_diagnostics": {
            "seed42": checkpoint_summary(args.checkpoint42, args.init_checkpoint),
            "seed3407": checkpoint_summary(args.checkpoint3407, args.init_checkpoint),
            "gradient_flow": {"seed42": read_csv_summary(args.seed42_gradient), "seed3407": read_csv_summary(args.seed3407_gradient)},
            "internal_signals": {"seed42": read_csv_summary(args.seed42_signals), "seed3407": read_csv_summary(args.seed3407_signals)},
        },
        "unavailable_without_new_instrumentation": [
            "Per-frame target-in-crop transition: archived tracking files lack the tracker crop transform for each frame.",
            "First severe per-frame IoU decline: archived result files contain boxes, but no per-frame GT-aligned IoU table; only event/window aggregates are available.",
            "Exact dataloader RNG trajectory: final configs/logs record seed and worker settings, but not a serialized worker/sample-order trace.",
            "Residual-to-main feature norm at inference: no per-frame residual activation dump was archived for Cdiag.",
        ],
        "interpretation_guard": "Event error buckets are descriptive and post-run; they do not authorize selecting a new subgroup or claiming a causal mechanism.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = json_safe(result)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "n_sequences": len(sequence_rows),
        "n_events": len(box_rows),
        "same_direction": result["sequence_direction_agreement"]["same_direction"],
        "categories": category_summary,
        "checkpoint_state_diff": {k: result["checkpoint_and_training_diagnostics"][k]["non_adapter_state_diff_count"] for k in ["seed42", "seed3407"]},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
