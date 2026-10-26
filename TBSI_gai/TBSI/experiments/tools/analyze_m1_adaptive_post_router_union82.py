import csv
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset, trackerlist


MANIFEST = "experiments/manifests/m1_normal20_lowauc62_union_20261013.txt"
NORMAL_MANIFEST = "experiments/manifests/m1_normal20_control_20261012.txt"
LOWAUC_MANIFEST = "output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/matched_m1_full62/sequence_list.txt"
M1_PARAM, M1_RUN = "TBSI_E_MHSA_FULL_4090D_seed42", 9904
P0_PARAM, P0_RUN = "M1_ADAPTIVE_POST_ROUTER_P0_QUICK_seed42", 9905
OUTPUT_DIR = "output/experiments/M1_ADAPTIVE_POST_ROUTER_P0_UNION82_20261013/paired_analysis"
N_BOOT = 10000
BOOT_SEED = 20261013


def read_names(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def metric_summary(curves, center_curves, center_norm_curves, indices):
    success = curves[indices, :, :]
    precision = center_curves[indices, :, :]
    norm_precision = center_norm_curves[indices, :, :]
    per_sequence = {
        "AUC": success.mean(axis=2) * 100.0,
        "PR": precision[:, :, 20] * 100.0,
        "NPR": norm_precision[:, :, 20] * 100.0,
        "OP50": success[:, :, 10] * 100.0,
        "OP75": success[:, :, 15] * 100.0,
    }
    return {key: val.mean(axis=0) for key, val in per_sequence.items()}, per_sequence


def bootstrap_delta(values, rng):
    n = values.shape[0]
    samples = rng.integers(0, n, size=(N_BOOT, n))
    means = values[samples].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    os.environ["TBSI_RPP_TEST_LIST"] = MANIFEST
    all_names = read_names(MANIFEST)
    normal_names = read_names(NORMAL_MANIFEST)
    lowauc_names = read_names(LOWAUC_MANIFEST)
    assert len(all_names) == 82 and len(normal_names) == 20 and len(lowauc_names) == 62
    assert len(set(all_names)) == 82 and set(normal_names).isdisjoint(lowauc_names)
    assert all_names == normal_names + lowauc_names

    dataset = get_dataset("rpp_lasher_test")
    assert [seq.name for seq in dataset] == all_names
    trackers = trackerlist("tbsi_track", M1_PARAM, "rpp_lasher_test", M1_RUN, "M1")
    trackers += trackerlist("tbsi_track", P0_PARAM, "rpp_lasher_test", P0_RUN, "P0")
    data = extract_results(trackers, dataset, "m1_adaptive_post_router_union82_9904_9905", plot_bin_gap=0.05)

    curves = np.asarray(data["ave_success_rate_plot_overlap"], dtype=np.float64)
    centers = np.asarray(data["ave_success_rate_plot_center"], dtype=np.float64)
    centers_norm = np.asarray(data["ave_success_rate_plot_center_norm"], dtype=np.float64)
    valid = np.asarray(data["valid_sequence"], dtype=bool)
    assert valid.all() and curves.shape[:2] == (82, 2)

    groups = {"normal20": np.arange(0, 20), "lowauc62": np.arange(20, 82), "union82": np.arange(82)}
    rng = np.random.default_rng(BOOT_SEED)
    summaries = {}
    for group, indices in groups.items():
        summary, per_seq = metric_summary(curves, centers, centers_norm, indices)
        deltas = {metric: per_seq[metric][:, 1] - per_seq[metric][:, 0] for metric in per_seq}
        auc_delta = deltas["AUC"]
        wins = int((auc_delta > 1e-12).sum())
        losses = int((auc_delta < -1e-12).sum())
        ties = int((np.abs(auc_delta) <= 1e-12).sum())
        summaries[group] = {
            "n_sequences": int(len(indices)),
            "metrics": {metric: {"M1": float(summary[metric][0]),
                                  "P0": float(summary[metric][1]),
                                  "delta": float(deltas[metric].mean()),
                                  "paired_sequence_bootstrap_95ci": bootstrap_delta(deltas[metric], rng)}
                        for metric in deltas},
            "auc_win_loss_tie": [wins, losses, ties],
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "per_sequence_auc.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "cohort", "M1_AUC", "P0_AUC", "delta_P0_minus_M1"])
        for i, name in enumerate(all_names):
            cohort = "normal20" if i < 20 else "lowauc62"
            m1_auc = float(curves[i, 0].mean() * 100.0)
            p0_auc = float(curves[i, 1].mean() * 100.0)
            writer.writerow([name, cohort, f"{m1_auc:.8f}", f"{p0_auc:.8f}", f"{p0_auc - m1_auc:.8f}"])
    with open(os.path.join(OUTPUT_DIR, "stratified_metrics.json"), "w") as f:
        json.dump({"groups": summaries, "bootstrap_replicates": N_BOOT, "bootstrap_seed": BOOT_SEED,
                   "manifest": MANIFEST}, f, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
