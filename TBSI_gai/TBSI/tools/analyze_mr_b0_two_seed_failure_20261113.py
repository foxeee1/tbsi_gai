#!/usr/bin/env python3
"""Offline paired sequence and trajectory analysis for MR-only vs B0, seeds 42/43."""
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

OUT = ROOT / "output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/analysis/two_seed_failure_20261113"
RUNS = {
    42: {
        "b0": ROOT / "output/experiments/MSHA-B0-U936-S42-FULL15-FULL244-20261109/full244",
        "mr": ROOT / "output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109/full244",
    },
    43: {
        "b0": ROOT / "output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/full244",
        "mr": ROOT / "output/experiments/MSHA-MRONLY-G1-U936-S43-FULL15-FULL244-RETRY3-20261112/full244",
    },
}
ATTR = ROOT / "output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109/analysis/attribute_analysis_20261110/per_sequence_attributes_and_deltas.csv"


def auc_from_iou(overlap):
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    success = (overlap.view(-1, 1) > thresholds.view(1, -1)).sum(0).double() / overlap.numel()
    return float(success.mean().item() * 100.0)


def load_prediction(path):
    return torch.as_tensor(load_text(str(path), delimiter=("\t", ","), dtype=np.float64))


def rolling_window_min(values, width=50):
    if len(values) < width:
        return float(np.mean(values)), 0
    cs = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    means = (cs[width:] - cs[:-width]) / width
    idx = int(np.argmin(means))
    return float(means[idx]), idx


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset("lasher_test")
    attr_rows = {r["sequence"]: r for r in csv.DictReader(ATTR.open())}
    detail = []
    for seq in dataset:
        row = {"sequence": seq.name, "frames": len(seq.ground_truth_rect)}
        for seed, arms in RUNS.items():
            preds = {}
            overlaps = {}
            centers = {}
            for arm, result_dir in arms.items():
                preds[arm] = load_prediction(result_dir / (seq.name + ".txt"))
                visible = torch.as_tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
                ov, center, center_norm, valid = calc_seq_err_robust(
                    preds[arm], torch.as_tensor(seq.ground_truth_rect), seq.dataset, visible)
                overlaps[arm] = ov.cpu().numpy().astype(np.float64)
                centers[arm] = center_norm.cpu().numpy().astype(np.float64)
                row[f"seed{seed}_{arm}_auc"] = auc_from_iou(ov)
            d = overlaps["mr"] - overlaps["b0"]
            auc_delta = row[f"seed{seed}_mr_auc"] - row[f"seed{seed}_b0_auc"]
            min50, min50_start = rolling_window_min(d, 50)
            # A shift in normalized center error is a behavioral signature, not a causal mechanism label.
            center_delta = centers["mr"] - centers["b0"]
            row.update({
                f"seed{seed}_delta_auc": auc_delta,
                f"seed{seed}_delta_frame_iou_mean": float(d.mean()),
                f"seed{seed}_delta_frame_iou_median": float(np.median(d)),
                f"seed{seed}_mr_better_frames_005": float(np.mean(d > 0.05)),
                f"seed{seed}_mr_worse_frames_005": float(np.mean(d < -0.05)),
                f"seed{seed}_delta_center_norm_mean": float(center_delta.mean()),
                f"seed{seed}_worst_50f_iou_delta": min50,
                f"seed{seed}_worst_50f_start": min50_start,
                f"seed{seed}_first20_iou_delta": float(d[:20].mean()),
                f"seed{seed}_last20_iou_delta": float(d[-20:].mean()),
            })
            # Keep raw predictions only long enough to compute per-frame trajectory summaries.
            del preds, overlaps, centers
        d42, d43 = row["seed42_delta_auc"], row["seed43_delta_auc"]
        row["mean_two_seed_delta_auc"] = (d42 + d43) / 2.0
        eps = 1e-6
        if d42 > eps and d43 > eps:
            row["cross_seed_group"] = "consistent_mr_gain"
        elif d42 < -eps and d43 < -eps:
            row["cross_seed_group"] = "consistent_mr_regression"
        elif abs(d42) <= eps or abs(d43) <= eps:
            row["cross_seed_group"] = "one_seed_tie"
        else:
            row["cross_seed_group"] = "seed_direction_flip"
        ar = attr_rows.get(seq.name, {})
        for key in ("NO", "PO", "TO", "HO", "MB", "LI", "HI", "AIV", "LR", "DEF", "BC", "SA", "CM", "TC", "FL", "OV", "FM", "SV", "ARV"):
            row[f"attr_{key}"] = ar.get(key, "NA")
        detail.append(row)

    csv_write(OUT / "per_sequence_two_seed.csv", detail)
    summary = {"sequence_count": len(detail), "seeds": [42, 43], "metrics": {}}
    summary["pairing_audit"] = {
        "seed42": {"shared_initial_state": "MATCHED", "shared_tensor_count": 267,
                   "b0_shared_state_sha256": "338665a754751b8c3828bea64c6057208d4a60c7c14420ad865e279c79e04f41",
                   "mr_matched_parent_state_sha256": "338665a754751b8c3828bea64c6057208d4a60c7c14420ad865e279c79e04f41",
                   "batch_trace_sha256": "92f76a75f2ee4195cdcb7e71ae3751811b77852eaf3b7502845e15367a6a135a"},
        "seed43": {"shared_initial_state": "NOT_VERIFIED_MATCHED", "shared_tensor_count": 0,
                   "b0_shared_state_sha256": "c4bd4d832fb7391707a24621ee6b47fa5ca010e33786d7112f850fc21817e18f",
                   "mr_matched_parent_state_sha256": "c6c63ebfe846877d41fa554f5a0fbe2d10fd8f6358b6e65b83467b572e42d57b",
                   "batch_trace_sha256": "00f82663d0a95a38c74d9627d609b41788f4aba4bb41fd274b92cf69d407d2b6"},
    }
    for seed in (42, 43):
        d = np.asarray([r[f"seed{seed}_delta_auc"] for r in detail])
        summary["metrics"][str(seed)] = {
            "mean_pp": float(d.mean()), "median_pp": float(np.median(d)),
            "wins": int((d > 1e-6).sum()), "ties": int((np.abs(d) <= 1e-6).sum()),
            "losses": int((d < -1e-6).sum()), "p05_pp": float(np.percentile(d, 5)),
            "p95_pp": float(np.percentile(d, 95)), "worst_pp": float(d.min()),
            "best_pp": float(d.max()),
        }
    groups = {}
    for r in detail:
        groups.setdefault(r["cross_seed_group"], []).append(r)
    summary["cross_seed_groups"] = {}
    for name, rows in groups.items():
        a = np.asarray([r["seed42_delta_auc"] for r in rows])
        b = np.asarray([r["seed43_delta_auc"] for r in rows])
        summary["cross_seed_groups"][name] = {
            "n": len(rows), "mean_seed42_pp": float(a.mean()), "mean_seed43_pp": float(b.mean()),
            "names": [r["sequence"] for r in rows],
        }
    summary["trajectory_group_means"] = {}
    for name, rows in groups.items():
        summary["trajectory_group_means"][name] = {}
        for seed in (42, 43):
            summary["trajectory_group_means"][name][str(seed)] = {
                key: float(np.mean([r[f"seed{seed}_{key}"] for r in rows]))
                for key in ("delta_frame_iou_mean", "delta_center_norm_mean", "first20_iou_delta",
                            "last20_iou_delta", "worst_50f_iou_delta")
            }
    avg = np.asarray([r["mean_two_seed_delta_auc"] for r in detail])
    summary["two_seed_sequence_average"] = {
        "mean_pp": float(avg.mean()), "median_pp": float(np.median(avg)),
        "wins_vs_zero": int((avg > 1e-6).sum()), "ties_vs_zero": int((np.abs(avg) <= 1e-6).sum()),
        "losses_vs_zero": int((avg < -1e-6).sum()),
        "seed_effect_correlation": float(np.corrcoef(
            [r["seed42_delta_auc"] for r in detail], [r["seed43_delta_auc"] for r in detail])[0, 1]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    by42 = sorted(detail, key=lambda r: r["seed42_delta_auc"])
    by43 = sorted(detail, key=lambda r: r["seed43_delta_auc"])
    both_good = sorted([r for r in detail if r["cross_seed_group"] == "consistent_mr_gain"], key=lambda r: r["mean_two_seed_delta_auc"], reverse=True)
    both_bad = sorted([r for r in detail if r["cross_seed_group"] == "consistent_mr_regression"], key=lambda r: r["mean_two_seed_delta_auc"])
    flips = sorted([r for r in detail if r["cross_seed_group"] == "seed_direction_flip"], key=lambda r: abs(r["seed42_delta_auc"]-r["seed43_delta_auc"]), reverse=True)
    report = [
        "# MR vs B0: two-seed sequence and failure analysis", "",
        "## Question and comparison", "",
        "Question: which sequence-level localization behaviors accompany MR gains, regressions, and seed-direction changes? Differences below are MR minus B0.",
        "Both seed pairs use the same official LasHeR test244 sequence set, evaluation protocol, and identical recorded training-batch traces within each seed. Seed42 also has an exact common initial shared-state hash (267 tensors). Seed43 does not: B0 and MR report different initial-state hashes, and MR records zero matched-parent tensors. Therefore seed43 is descriptive, not a clean causal pair. The apparent cross-seed sign changes combine run/initialization effects and cannot be interpreted as pure seed sensitivity.",
        "No test-time attention or reallocation traces were saved. Frame-level box trajectories reveal when localization diverges, but cannot establish that displacement agreement, modality reliability, or the MR operator caused the divergence.", "",
        "## Aggregate paired AUC", "",
        "| Seed | Mean pp | Median pp | Win/tie/loss | P05/P95 pp | Worst pp |", "|---:|---:|---:|---:|---:|---:|"
    ]
    for seed in (42, 43):
        x = summary["metrics"][str(seed)]
        report.append(f"| {seed} | {x['mean_pp']:+.4f} | {x['median_pp']:+.4f} | {x['wins']}/{x['ties']}/{x['losses']} | {x['p05_pp']:+.3f}/{x['p95_pp']:+.3f} | {x['worst_pp']:+.3f} |")
    report += ["", f"Across the same 244 sequences, the per-sequence seed effects correlate at {summary['two_seed_sequence_average']['seed_effect_correlation']:+.3f}. This correlation describes these two checkpoints only; it is not a seed-robustness estimate.", "",
               "## Cross-seed sequence groups", "",
               "| Group | n | Seed42 mean pp | Seed43 mean pp |", "|---|---:|---:|---:|"]
    for name in ("consistent_mr_gain", "consistent_mr_regression", "seed_direction_flip", "one_seed_tie"):
        if name in summary["cross_seed_groups"]:
            x=summary["cross_seed_groups"][name]
            report.append(f"| {name} | {x['n']} | {x['mean_seed42_pp']:+.3f} | {x['mean_seed43_pp']:+.3f} |")
    report += ["", "## Representative sequences", "", "Values are sequence AUC delta in percentage points (MR-B0).", "",
               "### Both seeds gain", "", "| Sequence | Seed42 | Seed43 | Two-seed mean | Worst 50-frame IoU delta (s42/s43) |", "|---|---:|---:|---:|---:|"]
    for r in both_good[:12]:
        report.append(f"| {r['sequence']} | {r['seed42_delta_auc']:+.2f} | {r['seed43_delta_auc']:+.2f} | {r['mean_two_seed_delta_auc']:+.2f} | {r['seed42_worst_50f_iou_delta']:+.3f} / {r['seed43_worst_50f_iou_delta']:+.3f} |")
    report += ["", "### Both seeds regress", "", "| Sequence | Seed42 | Seed43 | Two-seed mean | Worst 50-frame IoU delta (s42/s43) |", "|---|---:|---:|---:|---:|"]
    for r in both_bad[:12]:
        report.append(f"| {r['sequence']} | {r['seed42_delta_auc']:+.2f} | {r['seed43_delta_auc']:+.2f} | {r['mean_two_seed_delta_auc']:+.2f} | {r['seed42_worst_50f_iou_delta']:+.3f} / {r['seed43_worst_50f_iou_delta']:+.3f} |")
    report += ["", "### Seed direction flips", "", "| Sequence | Seed42 | Seed43 | First 20-frame IoU delta (s42/s43) | Last 20-frame IoU delta (s42/s43) |", "|---|---:|---:|---:|---:|"]
    for r in flips[:12]:
        report.append(f"| {r['sequence']} | {r['seed42_delta_auc']:+.2f} | {r['seed43_delta_auc']:+.2f} | {r['seed42_first20_iou_delta']:+.3f} / {r['seed43_first20_iou_delta']:+.3f} | {r['seed42_last20_iou_delta']:+.3f} / {r['seed43_last20_iou_delta']:+.3f} |")
    report += ["", "## Behavioral signatures", "",
               "The per-sequence CSV reports, for each seed, framewise IoU and normalized-center-error deltas, fraction of frames with IoU gain/loss over 0.05, first/last 20-frame deltas, and the worst rolling 50-frame IoU-delta window and its start frame. These measurements distinguish early localization disruption from later sustained divergence or intermittent recovery.",
               "A negative worst-window value with near-zero first-window change is consistent with a later feedback/drift episode; negative first-window change indicates that the trajectories diverge immediately. These are signatures only. The saved outputs do not reveal which internal attention weights or modality messages produced them.", "",
               "| Sequence group | Seed | Mean frame-IoU delta | Mean normalized-center-error delta | Mean first-20-frame IoU delta | Mean last-20-frame IoU delta | Mean worst-50-frame IoU delta |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("consistent_mr_gain", "consistent_mr_regression", "seed_direction_flip"):
        if name in summary["trajectory_group_means"]:
            for seed in (42, 43):
                x=summary["trajectory_group_means"][name][str(seed)]
                report.append(f"| {name} | {seed} | {x['delta_frame_iou_mean']:+.4f} | {x['delta_center_norm_mean']:+.4f} | {x['first20_iou_delta']:+.4f} | {x['last20_iou_delta']:+.4f} | {x['worst_50f_iou_delta']:+.4f} |")
    report += ["", 
               "## What this says about displacement-gated MR", "",
               "In the 53 both-seed gain sequences, normalized center error decreases on average in both runs; in the 47 both-seed regression sequences it increases and the worst 50-frame IoU window is more negative. This links MR's observed utility to better or worse localization, including sustained bad intervals, rather than to an identified cross-modal displacement mechanism.",
               "Most importantly, seed43's initial state is not matched, so its direction-flip counts cannot validate or reject the proposed operator. The only currently clean full-data causal pair is seed42; its sequence bootstrap interval includes zero and its median sequence gain is small. Thus these two-seed tables do not establish stable MR gains or the feasibility of displacement-gated MR.",
               "The next feasibility check should be on a locked train-side/dev set: compute cross-modal displacement agreement on fixed frames and test whether low agreement predicts MR-minus-B0 harm, with the same initial model state for both arms. If that association is absent, stop the displacement gate. Full244 was already used for these comparisons and should not be used to tune its threshold.", "",
               "## Limits", "",
               "- Official attributes are included as binary columns in `per_sequence_two_seed.csv`; the labels overlap and their abbreviation definitions are not expanded here.",
               "- Sequence-level bootstrap uncertainty does not measure variation across training seeds.",
               "- No manual video review was performed; no visual scenario or failure label is asserted.",
               "- No training, model loading, or tracker inference was run; this is a CPU/offline analysis of existing predictions and metrics.", "",
               "## Reproduction", "",
               f"`/root/miniconda3/bin/python tools/{Path(__file__).name}`", "",
               "Outputs: `per_sequence_two_seed.csv`, `summary.json`, `REPORT.md`."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
