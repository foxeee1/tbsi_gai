import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PRED_DIR = ROOT / "output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42"
SCENE_CSV = ROOT / "output/experiments/M1_FULL244_SCENE_MINING_20261009/per_sequence_scene.csv"
OUT = ROOT / "output/experiments/M1_ARCHIVED_PR_SR_SCENES_20261010"
BOOTSTRAPS = 5000
SEED = 20261011


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    from lib.test.analysis.extract_results import calc_seq_err_robust
    from lib.test.utils.load_text import load_text
    from lib.test.evaluation import get_dataset

    scene_rows = read_csv(SCENE_CSV)
    scene = {r["sequence"]: r for r in scene_rows}
    dataset = get_dataset("lasher_test")
    if len(dataset) != 244 or len(scene) != 244:
        raise SystemExit("Expected the protected 244-sequence LasHeR test manifest")

    per_sequence = []
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    for seq in dataset:
        path = PRED_DIR / (seq.name + ".txt")
        if not path.is_file():
            raise SystemExit("Missing protected M1 prediction: " + str(path))
        pred = torch.as_tensor(load_text(str(path), delimiter=("\t", ","), dtype=np.float64), dtype=torch.float64)
        gt = torch.as_tensor(seq.ground_truth_rect, dtype=torch.float64)
        visible = torch.as_tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
        overlap, center, center_norm, valid = calc_seq_err_robust(pred, gt, seq.dataset, visible)
        denominator = len(gt)
        row = {
            "sequence": seq.name,
            "attributes": scene[seq.name]["attributes"],
            "archived_auc_scene_file": float(scene[seq.name]["m1_auc"]),
            "valid_gt_frames": int(valid.sum().item()),
            "frames": denominator,
            "success_auc": float((overlap[:, None] > thresholds[None, :]).float().mean(dim=0).mean().item() * 100.0),
            "precision_20px": float((center <= 20).float().sum().item() / denominator * 100.0),
            "norm_precision_0p2": float((center_norm <= 0.2).float().sum().item() / denominator * 100.0),
            "op50": float((overlap > 0.50).float().sum().item() / denominator * 100.0),
            "op75": float((overlap > 0.75).float().sum().item() / denominator * 100.0),
            "mean_overlap_valid": float(overlap[valid].mean().item()),
            "center_error_median_valid": float(center[valid].median().item()),
            "frac_center_gt20_valid": float((center[valid] > 20).float().mean().item() * 100.0),
            "frac_overlap_lt05_valid": float((overlap[valid] < 0.5).float().mean().item() * 100.0),
        }
        per_sequence.append(row)

    area_sorted = sorted(per_sequence, key=lambda r: (float(scene[r["sequence"]]["median_gt_area_fraction"]), r["sequence"]))
    motion_sorted = sorted(per_sequence, key=lambda r: (float(scene[r["sequence"]]["p90_gt_motion_per_diag"]), r["sequence"]))
    area_rank = {r["sequence"]: i for i, r in enumerate(area_sorted)}
    motion_rank = {r["sequence"]: i for i, r in enumerate(motion_sorted)}
    cut = len(per_sequence) // 4
    groups = {
        "all": lambda r: True,
        "archived_auc_lt40": lambda r: r["archived_auc_scene_file"] < 40,
        "archived_auc_ge40": lambda r: r["archived_auc_scene_file"] >= 40,
        "smallest_area_quartile": lambda r: area_rank[r["sequence"]] < cut,
        "largest_area_quartile": lambda r: area_rank[r["sequence"]] >= len(per_sequence) - cut,
        "highest_motion_quartile": lambda r: motion_rank[r["sequence"]] >= len(per_sequence) - cut,
        "small_and_high_motion": lambda r: area_rank[r["sequence"]] < cut and motion_rank[r["sequence"]] >= len(per_sequence) - cut,
    }
    for attr in ("LR", "SA", "TC", "LI", "BC", "OV", "AIV", "ARV", "HO", "HI", "MB", "SV", "FM", "PO", "TO", "CM", "DEF", "FL", "NO"):
        groups["attr_" + attr] = lambda r, attr=attr: attr in r["attributes"].split(";")
    groups.update({
        "low_auc_and_LR": lambda r: r["archived_auc_scene_file"] < 40 and "LR" in r["attributes"].split(";"),
        "low_auc_and_SA": lambda r: r["archived_auc_scene_file"] < 40 and "SA" in r["attributes"].split(";"),
        "low_auc_and_TC": lambda r: r["archived_auc_scene_file"] < 40 and "TC" in r["attributes"].split(";"),
    })
    metric_names = ("success_auc", "precision_20px", "norm_precision_0p2", "op50", "op75", "mean_overlap_valid", "center_error_median_valid", "frac_center_gt20_valid", "frac_overlap_lt05_valid")
    rng = np.random.default_rng(SEED)
    group_rows = []
    for name, predicate in groups.items():
        selected = [r for r in per_sequence if predicate(r)]
        rest = [r for r in per_sequence if not predicate(r)]
        for metric in metric_names:
            a = np.asarray([r[metric] for r in selected], dtype=np.float64)
            b = np.asarray([r[metric] for r in rest], dtype=np.float64)
            if len(b):
                ai = rng.integers(0, len(a), size=(BOOTSTRAPS, len(a)))
                bi = rng.integers(0, len(b), size=(BOOTSTRAPS, len(b)))
                delta = np.mean(a[ai], axis=1) - np.mean(b[bi], axis=1)
                rest_mean = float(np.mean(b))
                mean_delta = float(np.mean(a) - rest_mean)
                ci_low, ci_high = float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))
            else:
                rest_mean = None
                mean_delta = None
                ci_low, ci_high = None, None
            group_rows.append({
                "group": name,
                "n_sequences": len(selected),
                "metric": metric,
                "group_mean": float(np.mean(a)),
                "rest_n_sequences": len(rest),
                "rest_mean": rest_mean,
                "mean_delta_vs_rest": mean_delta,
                "bootstrap95_mean_delta_low": ci_low,
                "bootstrap95_mean_delta_high": ci_high,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_sequence_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sequence[0]))
        writer.writeheader()
        writer.writerows(per_sequence)
    with (OUT / "scene_group_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    summary = {
        "id": "M1-ARCHIVED-PR-SR-SCENES-20261010",
        "n_sequences": len(per_sequence),
        "prediction_dir": str(PRED_DIR.relative_to(ROOT)),
        "dataset": "LasHeR testingset, official 244-sequence evaluation API",
        "official_semantics": {
            "success_auc": "mean of sequence overlap-success curve over thresholds 0.00..1.00 step 0.05",
            "precision_20px": "center error <=20 px divided by full annotation length",
            "norm_precision_0p2": "normalized center error <=0.2 divided by full annotation length",
            "op50_op75": "overlap strictly greater than threshold divided by full annotation length",
            "first_frame": "official calc_seq_err_robust replaces first predicted box with first GT box",
        },
        "aggregate": {m: float(np.mean([r[m] for r in per_sequence])) for m in metric_names},
        "group_counts": {name: sum(1 for r in per_sequence if pred(r)) for name, pred in groups.items()},
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": SEED,
        "limits": [
            "This uses the protected archived M1 predictions and is descriptive; it does not select modules or establish causal mechanisms.",
            "Attributes overlap; slices are correlated and no multiplicity correction is applied.",
            "Sequence means are the unit of analysis; no frame-level inference is made.",
        ],
        "outputs": ["per_sequence_metrics.csv", "scene_group_metrics.csv"],
    }
    reference = {"success_auc": 56.2077, "precision_20px": 70.21, "norm_precision_0p2": 66.20, "op50": 68.33, "op75": 47.13}
    for key, expected in reference.items():
        if abs(summary["aggregate"][key] - expected) > 0.03:
            raise SystemExit("Protected M1 metric parity failed for {}: {} vs {}".format(key, summary["aggregate"][key], expected))
    summary["protected_reference_parity"] = {k: {"computed": summary["aggregate"][k], "reference": v, "passed": True} for k, v in reference.items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": summary["aggregate"], "group_counts": summary["group_counts"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
