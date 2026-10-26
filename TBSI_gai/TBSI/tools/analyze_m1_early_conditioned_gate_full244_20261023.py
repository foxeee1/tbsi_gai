#!/usr/bin/env python3
"""Paired sequence and frame analysis for the full-244 early MSHA gate run."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust


def load_boxes(path, n):
    arr = np.loadtxt(str(path), delimiter=None, ndmin=2, dtype=np.float64)
    if arr.shape[1] > 4:
        arr = arr[:, :4]
    if len(arr) != n:
        raise ValueError("{} has {} rows; expected {}".format(path, len(arr), n))
    return arr


def frame_iou(pred, gt):
    tl = np.maximum(pred[:, :2], gt[:, :2])
    br = np.minimum(pred[:, :2] + pred[:, 2:] - 1.0, gt[:, :2] + gt[:, 2:] - 1.0)
    sz = np.maximum(br - tl + 1.0, 0.0)
    inter = sz[:, 0] * sz[:, 1]
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def sequence_metrics(pred, gt, dataset_name):
    pred = pred.copy()
    pred[0] = gt[0]
    gt_t = torch.as_tensor(gt, dtype=torch.float64)
    pred_t = torch.as_tensor(pred, dtype=torch.float64)
    iou_t, center_t, norm_t, valid_t = calc_seq_err_robust(pred_t, gt_t, dataset_name)
    iou = iou_t.numpy()
    center = center_t.numpy()
    norm = norm_t.numpy()
    valid = valid_t.numpy().astype(bool)
    thresholds = np.arange(0, 1.0001, 0.05)
    success = np.array([(iou > t).sum() / len(gt) for t in thresholds])
    return {
        "auc": float(success.mean() * 100),
        "op50": float((iou > 0.50).mean() * 100),
        "op75": float((iou > 0.75).mean() * 100),
        "precision": float((center <= 20).sum() / len(gt) * 100),
        "norm_precision": float((norm <= 0.20).sum() / len(gt) * 100),
        "mean_iou": float(iou[valid].mean()),
        "iou": iou,
    }


def bootstrap_ci(values, seed=42023, reps=10000):
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="TBSI_E_MHSA_FULL_4090D_seed42")
    ap.add_argument("--candidate", default="M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023_RETRY1")
    ap.add_argument("--out", default="output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023/analysis")
    args = ap.parse_args()
    p_parent = ROOT / "output/test/tracking_results/tbsi_track" / args.parent
    p_cand = ROOT / "output/test/tracking_results/tbsi_track" / args.candidate
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset("lasher_test")
    rows, all_frame_deltas = [], []
    sequence_deltas = []
    totals = {k: {"parent": [], "candidate": []} for k in
              ("auc", "op50", "op75", "precision", "norm_precision", "mean_iou")}
    for seq in dataset:
        n = len(seq.ground_truth_rect)
        f0, f1 = p_parent / (seq.name + ".txt"), p_cand / (seq.name + ".txt")
        if not f0.exists() or not f1.exists():
            raise FileNotFoundError("missing sequence output: {} / {}".format(f0, f1))
        gt = np.asarray(seq.ground_truth_rect, dtype=np.float64)
        m0 = sequence_metrics(load_boxes(f0, n), gt, seq.dataset)
        m1 = sequence_metrics(load_boxes(f1, n), gt, seq.dataset)
        d = m1["iou"] - m0["iou"]
        row = {"sequence": seq.name, "frames": n}
        for key in totals:
            row[key + "_parent"] = totals[key]["parent"][-1] if totals[key]["parent"] else None
            totals[key]["parent"].append(m0[key])
            totals[key]["candidate"].append(m1[key])
            row[key + "_parent"] = m0[key]
            row[key + "_candidate"] = m1[key]
            row[key + "_delta"] = m1[key] - m0[key]
        row.update({
            "frames_parent_iou_lt_0.1": int((m0["iou"] < 0.1).sum()),
            "frames_candidate_iou_lt_0.1": int((m1["iou"] < 0.1).sum()),
            "parent_failure_recovered_iou_ge_0.5": int(((m0["iou"] < 0.1) & (m1["iou"] >= 0.5)).sum()),
            "parent_good_regressed_iou_lt_0.1": int(((m0["iou"] >= 0.5) & (m1["iou"] < 0.1)).sum()),
            "frame_delta_first_quarter": float(d[:max(1, n // 4)].mean()),
            "frame_delta_last_quarter": float(d[(3*n)//4:].mean()),
        })
        rows.append(row)
        sequence_deltas.append(row["mean_iou_delta"])
        all_frame_deltas.extend(d.tolist())

    keys = list(totals)
    summary = {"parent": args.parent, "candidate": args.candidate, "dataset": "LasHeR test full-244",
               "sequences": len(rows), "frames": int(sum(r["frames"] for r in rows)),
               "metrics": {}, "sequence_mean_iou_delta_bootstrap95": bootstrap_ci(sequence_deltas),
               "frame_mean_iou_delta": float(np.mean(all_frame_deltas)),
               "sequence_mean_iou_delta": float(np.mean(sequence_deltas)),
               "sequence_median_iou_delta": float(np.median(sequence_deltas)),
               "wins_ties_losses_mean_iou": [int(np.sum(np.array(sequence_deltas) > 1e-12)),
                                               int(np.sum(np.abs(sequence_deltas) <= 1e-12)),
                                               int(np.sum(np.array(sequence_deltas) < -1e-12))],
               "low_auc_sequences": sorted(rows, key=lambda r: r["auc_candidate"])[:20],
               "largest_gains": sorted(rows, key=lambda r: r["mean_iou_delta"], reverse=True)[:20],
               "largest_regressions": sorted(rows, key=lambda r: r["mean_iou_delta"])[:20],
               "frame_recovery": {
                   "parent_iou_lt_0.1_to_candidate_ge_0.5": int(sum(r["parent_failure_recovered_iou_ge_0.5"] for r in rows)),
                   "parent_iou_ge_0.5_to_candidate_lt_0.1": int(sum(r["parent_good_regressed_iou_lt_0.1"] for r in rows)),
                   "net_frames_iou_lt_0.1": int(sum(r["frames_candidate_iou_lt_0.1"] - r["frames_parent_iou_lt_0.1"] for r in rows)),
               }}
    for key, values in totals.items():
        delta = np.asarray(values["candidate"]) - np.asarray(values["parent"])
        summary["metrics"][key] = {"parent_macro": float(np.mean(values["parent"])),
                                    "candidate_macro": float(np.mean(values["candidate"])),
                                    "delta": float(np.mean(delta)), "bootstrap95": bootstrap_ci(delta)}
    auc_strata = []
    for lo, hi in ((0, 40), (40, 60), (60, 80), (80, 101)):
        group = [r for r in rows if lo <= r["auc_parent"] < hi]
        deltas = np.asarray([r["auc_delta"] for r in group], dtype=np.float64)
        auc_strata.append({"parent_auc_range": "{}-{}".format(lo, hi), "sequences": len(group),
                           "mean_auc_delta": float(deltas.mean()) if len(group) else None,
                           "wins": int((deltas > 0).sum()), "losses": int((deltas < 0).sum())})
    auc_deltas = np.asarray([r["auc_delta"] for r in rows])
    summary["descriptive_parent_auc_strata"] = auc_strata
    temporal_strata = []
    for lo, hi in ((0, 40), (40, 60), (60, 80), (80, 101)):
        group = [r for r in rows if lo <= r["auc_parent"] < hi]
        temporal_strata.append({"parent_auc_range": "{}-{}".format(lo, hi), "sequences": len(group),
                                "first_quarter_iou_delta_macro": float(np.mean([r["frame_delta_first_quarter"] for r in group])),
                                "last_quarter_iou_delta_macro": float(np.mean([r["frame_delta_last_quarter"] for r in group]))})
    summary["temporal_delta_by_parent_auc_strata"] = temporal_strata
    summary["auc_delta_top5_gain_sum"] = float(np.sort(auc_deltas)[-5:].sum())
    summary["auc_delta_all_sequence_sum"] = float(auc_deltas.sum())
    csv_path = out / "per_sequence.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = ["# Early-conditioned MSHA gate full-244 sequence analysis", "",
          "Parent: `{}`; candidate: `{}`; LasHeR test: {} sequences, {} frames.".format(args.parent,args.candidate,summary['sequences'],summary['frames']),
          "", "## Paired macro metrics", "", "| Metric | Parent | Candidate | Delta | Sequence bootstrap 95% CI |", "|---|---:|---:|---:|---:|"]
    for key, v in summary["metrics"].items():
        md.append("| {} | {:.4f} | {:.4f} | {:+.4f} | [{:+.4f}, {:+.4f}] |".format(key,v['parent_macro'],v['candidate_macro'],v['delta'],*v['bootstrap95']))
    md += ["", "## Distribution", "", "- Per-sequence mean-IoU wins/ties/losses: {}.".format(summary['wins_ties_losses_mean_iou']),
           "- Sequence-mean IoU delta: {:.6f}; median: {:.6f}; bootstrap 95% CI [{:.6f}, {:.6f}].".format(summary['sequence_mean_iou_delta'],summary['sequence_median_iou_delta'],*summary['sequence_mean_iou_delta_bootstrap95']),
           "- Frame-level mean IoU delta: {:.6f}.".format(summary['frame_mean_iou_delta']),
           "- Top five sequence AUC gains sum to {:.3f} points across sequences, while the net sum across all sequences is {:.3f}; large gains and losses offset each other.".format(summary['auc_delta_top5_gain_sum'],summary['auc_delta_all_sequence_sum']),
           "- Descriptive strata by parent AUC (selection on parent score; interpret cautiously):", ""]
    for s in summary["descriptive_parent_auc_strata"]:
        md.append("  - {}: n={}, mean AUC delta {:+.3f}, wins/losses {}/{}".format(s["parent_auc_range"],s["sequences"],s["mean_auc_delta"],s["wins"],s["losses"]))
    md += ["- First/last quarter frame-IoU delta by the same strata (descriptive):"]
    for s in summary["temporal_delta_by_parent_auc_strata"]:
        md.append("  - {}: first quarter {:+.4f}, last quarter {:+.4f}".format(s["parent_auc_range"],s["first_quarter_iou_delta_macro"],s["last_quarter_iou_delta_macro"]))
    md += [
           "- Frames moving from parent IoU <0.1 to candidate IoU >=0.5: {}; reverse transitions: {}; net change in IoU <0.1 frames: {}.".format(summary['frame_recovery']['parent_iou_lt_0.1_to_candidate_ge_0.5'],summary['frame_recovery']['parent_iou_ge_0.5_to_candidate_lt_0.1'],summary['frame_recovery']['net_frames_iou_lt_0.1']),
           "", "## Largest gains", "", "| Sequence | Frames | Mean-IoU delta | AUC delta | OP50 delta | Low-IoU recovery / regression |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary['largest_gains'][:12]:
        md.append("| {} | {} | {:+.4f} | {:+.3f} | {:+.3f} | {} / {} |".format(r['sequence'],r['frames'],r['mean_iou_delta'],r['auc_delta'],r['op50_delta'],r['parent_failure_recovered_iou_ge_0.5'],r['parent_good_regressed_iou_lt_0.1']))
    md += ["", "## Largest regressions", "", "| Sequence | Frames | Mean-IoU delta | AUC delta | OP50 delta | Low-IoU recovery / regression |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary['largest_regressions'][:12]:
        md.append("| {} | {} | {:+.4f} | {:+.3f} | {:+.3f} | {} / {} |".format(r['sequence'],r['frames'],r['mean_iou_delta'],r['auc_delta'],r['op50_delta'],r['parent_failure_recovered_iou_ge_0.5'],r['parent_good_regressed_iou_lt_0.1']))
    md += ["", "## Attribution limits", "", "These outputs measure where the closed-loop tracking difference occurs. They do not identify the internal gate activations or prove that gate changes caused a specific sequence outcome. Frame transitions and sequence concentration are descriptive; inspect early gate logs and sequence videos before assigning appearance, modality, or update-policy causes.", ""]
    (out / "report.md").write_text("\n".join(md))
    print(json.dumps({"out": str(out), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
