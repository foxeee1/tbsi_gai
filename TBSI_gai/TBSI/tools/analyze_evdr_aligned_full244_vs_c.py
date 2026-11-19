#!/usr/bin/env python3
"""Re-score two existing LasHeR prediction directories; never runs a tracker."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.test.analysis.plot_results import check_and_load_precomputed_results
from lib.test.evaluation import get_dataset, trackerlist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--dataset", default="lasher_test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-name", required=True)
    args = parser.parse_args()

    trackers = []
    for name in (args.parent, args.candidate):
        trackers.extend(trackerlist("tbsi_track", name, args.dataset, run_ids=None,
                                    display_name=name))
    dataset = get_dataset(args.dataset)
    data = check_and_load_precomputed_results(trackers, dataset, args.cache_name,
                                             force_evaluation=True)

    seqs = data["sequences"]
    valid = data["valid_sequence"]
    auc_curves = data["ave_success_rate_plot_overlap"]
    prec_curves = data["ave_success_rate_plot_center"]
    norm_curves = data["ave_success_rate_plot_center_norm"]
    ao = data["avg_overlap_all"]
    thresholds = data["threshold_set_overlap"]
    op50_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.5))
    op75_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.75))

    rows = []
    for i, seq in enumerate(seqs):
        if not valid[i]:
            continue
        parent_auc = sum(auc_curves[i][0]) / len(auc_curves[i][0]) * 100.0
        cand_auc = sum(auc_curves[i][1]) / len(auc_curves[i][1]) * 100.0
        parent_pr = prec_curves[i][0][20] * 100.0
        cand_pr = prec_curves[i][1][20] * 100.0
        parent_npr = norm_curves[i][0][20] * 100.0
        cand_npr = norm_curves[i][1][20] * 100.0
        rows.append({
            "sequence": seq,
            "parent_AUC": parent_auc,
            "candidate_AUC": cand_auc,
            "delta_AUC_pp": cand_auc - parent_auc,
            "parent_PR": parent_pr,
            "candidate_PR": cand_pr,
            "delta_PR_pp": cand_pr - parent_pr,
            "parent_NPR": parent_npr,
            "candidate_NPR": cand_npr,
            "delta_NPR_pp": cand_npr - parent_npr,
            "parent_OP50": auc_curves[i][0][op50_idx] * 100.0,
            "candidate_OP50": auc_curves[i][1][op50_idx] * 100.0,
            "delta_OP50_pp": (auc_curves[i][1][op50_idx] - auc_curves[i][0][op50_idx]) * 100.0,
            "parent_OP75": auc_curves[i][0][op75_idx] * 100.0,
            "candidate_OP75": auc_curves[i][1][op75_idx] * 100.0,
            "delta_OP75_pp": (auc_curves[i][1][op75_idx] - auc_curves[i][0][op75_idx]) * 100.0,
            "parent_AO": ao[i][0] * 100.0,
            "candidate_AO": ao[i][1] * 100.0,
            "delta_AO_pp": (ao[i][1] - ao[i][0]) * 100.0,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    deltas = [r["delta_AUC_pp"] for r in rows]
    print("valid_sequences", n)
    print("AUC_mean_delta_pp", sum(deltas) / n)
    print("AUC_win_tie_loss", sum(x > 1e-9 for x in deltas),
          sum(abs(x) <= 1e-9 for x in deltas), sum(x < -1e-9 for x in deltas))
    for label, key in (("AUC gains", "delta_AUC_pp"), ("AUC losses", "delta_AUC_pp")):
        ordered = sorted(rows, key=lambda r: r[key], reverse=(label == "AUC gains"))
        if label == "AUC losses":
            ordered = ordered[:10]
        else:
            ordered = ordered[:10]
        print(label)
        for r in ordered:
            print(r["sequence"], "AUC", round(r["delta_AUC_pp"], 3),
                  "AO", round(r["delta_AO_pp"], 3),
                  "PR", round(r["delta_PR_pp"], 3))
    for metric in ("delta_AUC_pp", "delta_PR_pp", "delta_NPR_pp", "delta_OP50_pp", "delta_OP75_pp", "delta_AO_pp"):
        vals = [r[metric] for r in rows]
        print(metric, "positive", sum(x > 1e-9 for x in vals),
              "negative", sum(x < -1e-9 for x in vals),
              "mean", sum(vals) / n)


if __name__ == "__main__":
    main()
