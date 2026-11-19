import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "output/experiments/M1_FULL244_SCENE_MINING_20261009/per_sequence_scene.csv"
FLOW = ROOT / "output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/analysis/per_sequence_summary.csv"
OUT = ROOT / "output/experiments/M1_FUSION_FAILURE_SUBGROUPS_20261010"
BOOTSTRAPS = 2000
SEED = 20261010

METRICS = [
    "archived_m1_auc_percent",
    "full_current_replay_mean_iou",
    "response_gt_cell_rho_vs_iou",
    "response_peak_rho_vs_iou",
    "mhsa_9_entropy_rho_vs_iou",
    "mhsa_9_tir_cross_message_norm_rho_vs_iou",
    "mhsa_9_rgb_cross_message_norm_rho_vs_iou",
    "mhsa_9_qkv_weight_grad_norm_rho_vs_iou",
    "search_input_grad_rgb_norm_rho_vs_iou",
    "search_input_grad_tir_norm_rho_vs_iou",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def values(rows, key):
    out = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value == value and abs(value) != float("inf"):
            out.append(value)
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def bootstrap_median_delta(group, complement, rng):
    if len(group) < 2 or len(complement) < 2:
        return None, None
    a = np.asarray(group, dtype=np.float64)
    b = np.asarray(complement, dtype=np.float64)
    ai = rng.integers(0, len(a), size=(BOOTSTRAPS, len(a)))
    bi = rng.integers(0, len(b), size=(BOOTSTRAPS, len(b)))
    deltas = np.median(a[ai], axis=1) - np.median(b[bi], axis=1)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def main():
    scene_rows = read_csv(SCENE)
    flow_rows = read_csv(FLOW)
    scene = {r["sequence"]: r for r in scene_rows}
    flow = {r["sequence"]: r for r in flow_rows}
    common = sorted(set(scene) & set(flow))
    if len(common) != 244 or len(scene) != 244 or len(flow) != 244:
        raise SystemExit("Expected an exact 244-sequence join; refusing partial analysis")

    rows = [{**scene[name], **flow[name]} for name in common]
    area_order = sorted(rows, key=lambda r: (float(r["median_gt_area_fraction"]), r["sequence"]))
    motion_order = sorted(rows, key=lambda r: (float(r["p90_gt_motion_per_diag"]), r["sequence"]))
    area_rank = {r["sequence"]: i for i, r in enumerate(area_order)}
    motion_rank = {r["sequence"]: i for i, r in enumerate(motion_order)}
    n = len(rows)
    area_cut = n // 4
    motion_cut = n - area_cut

    groups = {
        "all": lambda r: True,
        "archived_auc_lt40": lambda r: float(r["m1_auc"]) < 40,
        "archived_auc_ge40": lambda r: float(r["m1_auc"]) >= 40,
        "smallest_area_quartile": lambda r: area_rank[r["sequence"]] < area_cut,
        "largest_area_quartile": lambda r: area_rank[r["sequence"]] >= n - area_cut,
        "highest_motion_quartile": lambda r: motion_rank[r["sequence"]] >= motion_cut,
        "small_and_high_motion": lambda r: area_rank[r["sequence"]] < area_cut and motion_rank[r["sequence"]] >= motion_cut,
        "low_auc_and_LR": lambda r: float(r["m1_auc"]) < 40 and "LR" in r["attributes"].split(";"),
        "low_auc_and_SA": lambda r: float(r["m1_auc"]) < 40 and "SA" in r["attributes"].split(";"),
        "low_auc_and_TC": lambda r: float(r["m1_auc"]) < 40 and "TC" in r["attributes"].split(";"),
    }
    for attr in ("LR", "SA", "TC", "LI", "BC", "OV", "AIV", "ARV", "HO", "HI", "MB", "SV", "FM", "PO", "TO", "CM", "DEF", "FL", "NO"):
        groups["attr_" + attr] = lambda r, attr=attr: attr in r["attributes"].split(";")

    rng = np.random.default_rng(SEED)
    out_rows = []
    counts = {}
    for group_name, predicate in groups.items():
        selected = [r for r in rows if predicate(r)]
        complement = [r for r in rows if not predicate(r)]
        counts[group_name] = len(selected)
        for metric in METRICS:
            xs = values(selected, metric)
            ys = values(complement, metric)
            lo, hi = bootstrap_median_delta(xs, ys, rng)
            out_rows.append({
                "group": group_name,
                "n": len(xs),
                "metric": metric,
                "group_median": median(xs),
                "group_mean": sum(xs) / len(xs) if xs else None,
                "complement_n": len(ys),
                "complement_median": median(ys),
                "median_delta_vs_complement": median(xs) - median(ys) if xs and ys else None,
                "bootstrap95_median_delta_low": lo,
                "bootstrap95_median_delta_high": hi,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "subgroup_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    summary = {
        "id": "M1-FUSION-FAILURE-SUBGROUPS-20261010",
        "n_sequences": len(common),
        "exact_join": True,
        "scene_source": str(SCENE.relative_to(ROOT)),
        "flow_source": str(FLOW.relative_to(ROOT)),
        "group_counts": counts,
        "metrics": METRICS,
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": SEED,
        "interpretation_limits": [
            "Official test-set attributes and archived AUC are descriptive only; they must not select a module or insertion point.",
            "Fusion-flow sequence metrics come from a current-working-tree replay that is not frame-equivalent to archived M1 predictions.",
            "Attributes overlap, groups are not independent, and no multiple-comparison correction or causal inference is intended.",
            "Per-sequence correlations are estimates from 31-32 uniformly sampled frames and have within-sequence temporal dependence.",
            "Gradient norm correlations are loss sensitivity, not contribution, reliability, or optimization update evidence.",
        ],
        "outputs": ["subgroup_metrics.csv", "summary.json"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for group in ("archived_auc_lt40", "archived_auc_ge40", "smallest_area_quartile", "highest_motion_quartile", "low_auc_and_LR", "low_auc_and_SA", "low_auc_and_TC", "attr_OV"):
        print("\n" + group, "n=", counts[group])
        for metric in ("response_gt_cell_rho_vs_iou", "response_peak_rho_vs_iou", "mhsa_9_entropy_rho_vs_iou", "mhsa_9_tir_cross_message_norm_rho_vs_iou", "mhsa_9_rgb_cross_message_norm_rho_vs_iou", "mhsa_9_qkv_weight_grad_norm_rho_vs_iou"):
            row = next(x for x in out_rows if x["group"] == group and x["metric"] == metric)
            print(metric, "median=", round(row["group_median"], 3), "vs_rest=", round(row["median_delta_vs_complement"], 3), "CI=", (round(row["bootstrap95_median_delta_low"], 3), round(row["bootstrap95_median_delta_high"], 3)))
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
