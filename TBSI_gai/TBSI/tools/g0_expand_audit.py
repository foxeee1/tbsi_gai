#!/usr/bin/env python3
"""Read-only G0-Expand audit with sequence bootstrap and XY controls."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


CONDITIONS = ("L_zero", "L_E", "L_Lcopy", "L_E_spatial", "L_E_event", "L_XY", "L_XY_E")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def align(value):
    value = torch.tensor(value, dtype=torch.float32)
    if tuple(value.shape) != (256, 768):
        raise ValueError("expected [256, 768], got {}".format(tuple(value.shape)))
    return value


def box_iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def event_record(row, feature_dir):
    feature = np.load(Path(feature_dir) / (row["event_id"] + ".npz"))
    rgb_late, tir_late = align(feature["late_input_rgb"]), align(feature["late_input_tir"])
    rgb_early, tir_early = align(feature["early_rgb"]), align(feature["early_tir"])
    late = torch.cat([rgb_late, tir_late], dim=1)
    early = torch.cat([rgb_early, tir_early], dim=1)
    target = torch.tensor(row["target_mask"], dtype=torch.float32)
    y, x = torch.meshgrid(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 16))
    xy = torch.stack([x, y], dim=-1).reshape(256, 2)
    gt = row.get("projected_gt_xywh")
    if gt is None:
        gt = row.get("gt_box_search")
    center_x = (float(gt[0]) + float(gt[2]) / 2.0) if gt else float("nan")
    center_y = (float(gt[1]) + float(gt[3]) / 2.0) if gt else float("nan")
    center_offset = float(np.hypot(center_x - 128.0, center_y - 128.0) / 128.0) if gt else float("nan")
    pred_iou = box_iou_xywh(row["pred_box"], row["gt_box_original"])
    return {"row": row, "late": late, "early": early, "xy": xy, "target": target,
            "center_offset": center_offset, "m1_iou": pred_iou}


def make_input(item, condition, donor=None, spatial_perm=None):
    late, early, xy = item["late"], item["early"], item["xy"]
    if condition == "L_zero":
        return torch.cat([late, torch.zeros_like(early)], dim=1)
    if condition == "L_E":
        return torch.cat([late, early], dim=1)
    if condition == "L_Lcopy":
        return torch.cat([late, late], dim=1)
    if condition == "L_E_spatial":
        return torch.cat([late, early[spatial_perm]], dim=1)
    if condition == "L_E_event":
        return torch.cat([late, donor], dim=1)
    if condition == "L_XY":
        return torch.cat([late, xy], dim=1)
    if condition == "L_XY_E":
        return torch.cat([late, xy, early], dim=1)
    raise ValueError(condition)


class Probe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, value):
        return self.net(value).squeeze(-1)


def average_precision(scores, labels):
    order = np.argsort(-scores)
    labels = np.asarray(labels)[order]
    count = float(labels.sum())
    if count == 0:
        return float("nan")
    return float((np.cumsum(labels) / np.arange(1, len(labels) + 1) * labels).sum() / count)


def event_metrics(scores, target):
    scores = np.asarray(scores)
    target = np.asarray(target) > 0.5
    order = np.argsort(-scores)
    positive = np.where(target)[0]
    ranks = [int(np.where(order == index)[0][0] + 1) for index in positive]
    return {"ap": average_precision(scores, target),
            "best_positive_rank": min(ranks) if ranks else None,
            "positive_count": int(target.sum())}


def fit(items, condition, seed, steps, donors, spatial_perm):
    inputs, targets = [], []
    for index, item in enumerate(items):
        donor = donors[(index + 1) % len(donors)]["early"] if donors else None
        inputs.append(make_input(item, condition, donor, spatial_perm))
        targets.append(item["target"])
    x = torch.cat(inputs)
    y = torch.cat(targets)
    positive = torch.where(y > 0.5)[0]
    negative = torch.where(y <= 0.5)[0]
    generator = torch.Generator().manual_seed(seed)
    selected_negative = negative[torch.randperm(len(negative), generator=generator)[: min(len(negative), 3 * len(positive))]]
    selected = torch.cat([positive, selected_negative])
    mean = x[selected].mean(0)
    std = x[selected].std(0).clamp_min(1e-6)
    model = Probe(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    torch.manual_seed(seed)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model((x[selected] - mean) / std), y[selected])
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def bootstrap(values, seed, count=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "q025": float(np.quantile(draws, 0.025)),
            "q975": float(np.quantile(draws, 0.975)), "n_sequences": int(len(values))}


def summarize_strata(rows, delta_key, thresholds):
    output = []
    for name, lower, upper in thresholds:
        selected = [row for row in rows if lower <= row[delta_key] < upper]
        values = [row["delta"] for row in selected]
        output.append({"stratum": name, "lower": lower, "upper": upper,
                       "sequences": len(set(row["sequence"] for row in selected)),
                       "events": len(selected),
                       "mean_delta": float(np.mean(values)) if values else None})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--old-labels", required=True)
    parser.add_argument("--old-features", required=True)
    parser.add_argument("--fresh-events", required=True)
    parser.add_argument("--fresh-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=640)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing non-empty output {}".format(output))
    output.mkdir(parents=True)
    manifest = json.loads(Path(args.manifest).read_text())
    old_rows = [row for row in read_jsonl(Path(args.old_labels) / "token_labels.jsonl") if row.get("rescue_eligible")]
    fresh_rows = [row for row in read_jsonl(args.fresh_events) if row.get("target_in_crop") and row.get("target_token_count", 0) > 0]
    train = [event_record(row, args.old_features) for row in old_rows]
    valid = [event_record(row, args.fresh_features) for row in fresh_rows]
    if len(train) < 20 or len(valid) < 20:
        raise RuntimeError("insufficient train/valid events")
    all_results = []
    for seed in manifest["probe_seeds"]:
        spatial_perm = torch.roll(torch.arange(256), shifts=37)
        for condition in CONDITIONS:
            model, mean, std = fit(train, condition, seed, args.steps, train, spatial_perm)
            with torch.no_grad():
                for index, item in enumerate(valid):
                    donor = valid[(index + 1) % len(valid)]["early"]
                    x = make_input(item, condition, donor, spatial_perm)
                    metrics = event_metrics(torch.sigmoid(model((x - mean) / std)).numpy(), item["target"].numpy())
                    metrics.update({"seed": seed, "condition": condition, "event_id": item["row"]["event_id"],
                                   "sequence": item["row"]["sequence"], "center_offset": item["center_offset"],
                                   "m1_iou": item["m1_iou"]})
                    all_results.append(metrics)
            print(seed, condition, len(valid), flush=True)
    (output / "per_event.jsonl").write_text("\n".join(json.dumps(row) for row in all_results) + "\n")
    summary = {"conditions": list(CONDITIONS), "seeds": manifest["probe_seeds"],
               "train_events": len(train), "valid_events": len(valid), "steps": args.steps,
               "sequence_bootstrap": {}}
    for seed in manifest["probe_seeds"]:
        rows = [row for row in all_results if row["seed"] == seed]
        by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
        base = {row["event_id"]: row["ap"] for row in by_condition["L_zero"]}
        for condition in CONDITIONS:
            sequence_values = {}
            for row in by_condition[condition]:
                sequence_values.setdefault(row["sequence"], []).append(row["ap"] - base[row["event_id"]])
            summary["sequence_bootstrap"].setdefault(str(seed), {})[condition] = bootstrap(
                [np.mean(values) for values in sequence_values.values()], seed + len(condition))
        for condition in ("L_E", "L_Lcopy", "L_E_spatial", "L_E_event", "L_XY", "L_XY_E"):
            diffs = {row["event_id"]: row["ap"] - base[row["event_id"]] for row in by_condition[condition]}
            summary.setdefault(str(seed), {})[condition] = {
                "mean_delta_vs_L_zero": float(np.mean(list(diffs.values()))),
                "positive_event_fraction": float(np.mean(np.asarray(list(diffs.values())) > 0)),
                "mean_best_positive_rank": float(np.mean([row["best_positive_rank"] for row in by_condition[condition]])),
            }
        # Paired event deltas answer the control questions without treating events as independent bootstrap units.
        for left, right in (("L_E", "L_E_spatial"), ("L_E", "L_E_event"), ("L_XY_E", "L_XY"), ("L_E", "L_XY_E")):
            left_map = {row["event_id"]: row["ap"] for row in by_condition[left]}
            right_map = {row["event_id"]: row["ap"] for row in by_condition[right]}
            delta_rows = [{"sequence": row["sequence"], "event_id": row["event_id"], "delta": left_map[row["event_id"]] - right_map[row["event_id"]],
                           "center_offset": row["center_offset"], "m1_iou": row["m1_iou"]} for row in by_condition[left]]
            key = "{}__minus__{}".format(left, right)
            per_sequence = {}
            for row in delta_rows:
                per_sequence.setdefault(row["sequence"], []).append(row["delta"])
            summary["sequence_bootstrap"][str(seed)][key] = bootstrap(
                [np.mean(values) for values in per_sequence.values()], seed + 100 + len(key))
            summary.setdefault(str(seed), {})[key] = {
                "mean_event_delta": float(np.mean([row["delta"] for row in delta_rows])),
                "positive_event_fraction": float(np.mean(np.asarray([row["delta"] for row in delta_rows]) > 0)),
                "center_offset_strata": summarize_strata(delta_rows, "center_offset", [("near", 0.0, 0.35), ("mid", 0.35, 0.70), ("far", 0.70, 10.0)]),
                "m1_iou_strata": summarize_strata(delta_rows, "m1_iou", [("low", 0.0, 0.40), ("mid", 0.40, 0.70), ("high", 0.70, 1.01)]),
            }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
