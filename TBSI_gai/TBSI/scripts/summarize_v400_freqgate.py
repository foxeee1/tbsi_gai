import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset, trackerlist


DATASET_NAME = "mini_lasher_test"
OUT_DIR = Path("output/diagnostics/freqgate_v400")
REPORT_NAME = "mini_lasher_test_freqgate_v400"
CATEGORY_FILE = Path("output/diagnostics/signal_decouple/per_sequence_proxy_attr_metrics.json")
PREVIOUS_SUMMARY_FILE = Path("output/diagnostics/signal_decouple_v165/attr_metrics.json")
V172_SUMMARY_FILE = Path("output/diagnostics/template_search_competition_v172/attr_metrics_v172_compare.json")
V320_SUMMARY_FILE = Path("output/diagnostics/competitive_bridge_v320/attr_metrics_v320_compare.json")
V321_SUMMARY_FILE = Path("output/diagnostics/competitive_residual_bridge_v321/attr_metrics_v321_compare.json")

TRACKER_PARAMS = [
    "v4.0.0-freqgate-A",
    "v4.0.0-freqgate-B",
    "v4.0.0-freqgate-C",
]

REFERENCE_PARAMS = {
    "baseline": "v1.0.0-完美基线-{seed}",
    "v160": "v1.6.0-signal-decouple-v1-{seed}",
    "v165": "v1.6.5-signal-decouple-learnscale-all-smax05-init01-{seed}",
    "v172": "v1.7.2-template-search-competition-{seed}",
    "v320": "v3.2.0-competitive-bridge-{seed}",
    "v321": "v3.2.1-competitive-residual-bridge-{seed}",
}

METRIC_KEYS = ["AUC", "OP50", "OP75", "Precision", "NormPrecision"]


def metric_from_curves(eval_data, seq_ids, trk_id):
    overlap = torch.tensor(eval_data["ave_success_rate_plot_overlap"], dtype=torch.float64)
    center = torch.tensor(eval_data["ave_success_rate_plot_center"], dtype=torch.float64)
    center_norm = torch.tensor(eval_data["ave_success_rate_plot_center_norm"], dtype=torch.float64)
    overlap_thresholds = torch.tensor(eval_data["threshold_set_overlap"], dtype=torch.float64)

    success_curve = overlap[seq_ids, trk_id, :].mean(0) * 100.0
    precision_curve = center[seq_ids, trk_id, :].mean(0) * 100.0
    norm_precision_curve = center_norm[seq_ids, trk_id, :].mean(0) * 100.0
    op50_id = int(torch.argmin(torch.abs(overlap_thresholds - 0.50)).item())
    op75_id = int(torch.argmin(torch.abs(overlap_thresholds - 0.75)).item())

    return {
        "AUC": float(success_curve.mean().item()),
        "OP50": float(success_curve[op50_id].item()),
        "OP75": float(success_curve[op75_id].item()),
        "Precision": float(precision_curve[20].item()),
        "NormPrecision": float(norm_precision_curve[20].item()),
    }


def sub_metrics(a, b):
    return {k: a[k] - b[k] for k in METRIC_KEYS}


def mean_metrics(items):
    return {k: sum(x[k] for x in items) / len(items) for k in METRIC_KEYS}


def category_delta_mean(deltas):
    categories = next(iter(deltas.values()))["by_category"].keys()
    return {
        cat: mean_metrics([seed_delta["by_category"][cat] for seed_delta in deltas.values()])
        for cat in categories
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = json.load(PREVIOUS_SUMMARY_FILE.open())
    v172_summary = json.load(V172_SUMMARY_FILE.open())
    v320_summary = json.load(V320_SUMMARY_FILE.open())
    v321_summary = json.load(V321_SUMMARY_FILE.open())
    categories = json.load(CATEGORY_FILE.open())["categories"]

    overall = dict(previous["overall"])
    by_category = dict(previous["by_category"])
    overall.update(v172_summary["v172_overall"])
    by_category.update(v172_summary["v172_by_category"])
    overall.update(v320_summary["overall"])
    by_category.update(v320_summary["by_category"])
    overall.update(v321_summary["overall"])
    by_category.update(v321_summary["by_category"])

    dataset = get_dataset(DATASET_NAME)
    trackers = []
    for param in TRACKER_PARAMS:
        trackers.extend(trackerlist("tbsi_track", param, DATASET_NAME, None, param))

    eval_data = extract_results(
        trackers,
        dataset,
        REPORT_NAME,
        skip_missing_seq=False,
        plot_bin_gap=0.05,
    )

    seq_names = eval_data["sequences"]
    valid_ids = [i for i, valid in enumerate(eval_data["valid_sequence"]) if valid]
    category_ids = {}
    for i, seq_name in enumerate(seq_names):
        category_ids.setdefault(categories[seq_name], []).append(i)

    for trk_id, tracker in enumerate(eval_data["trackers"]):
        param = tracker["param"]
        overall[param] = metric_from_curves(eval_data, valid_ids, trk_id)
        by_category[param] = {
            cat: metric_from_curves(eval_data, ids, trk_id)
            for cat, ids in category_ids.items()
        }

    deltas = {}
    for seed in "ABC":
        current = f"v4.0.0-freqgate-{seed}"
        for tag, pattern in REFERENCE_PARAMS.items():
            ref = pattern.format(seed=seed)
            key = f"{seed}_v400_minus_{tag}"
            deltas[key] = {
                "overall": sub_metrics(overall[current], overall[ref]),
                "by_category": {
                    cat: sub_metrics(by_category[current][cat], by_category[ref][cat])
                    for cat in category_ids
                },
            }

    mean = {
        "v400": mean_metrics([overall[f"v4.0.0-freqgate-{s}"] for s in "ABC"]),
        "baseline": mean_metrics([overall[REFERENCE_PARAMS["baseline"].format(seed=s)] for s in "ABC"]),
        "v160": mean_metrics([overall[REFERENCE_PARAMS["v160"].format(seed=s)] for s in "ABC"]),
        "v165": mean_metrics([overall[REFERENCE_PARAMS["v165"].format(seed=s)] for s in "ABC"]),
        "v172": mean_metrics([overall[REFERENCE_PARAMS["v172"].format(seed=s)] for s in "ABC"]),
        "v320": mean_metrics([overall[REFERENCE_PARAMS["v320"].format(seed=s)] for s in "ABC"]),
        "v321": mean_metrics([overall[REFERENCE_PARAMS["v321"].format(seed=s)] for s in "ABC"]),
    }
    mean_deltas_by_category = {}
    for tag in REFERENCE_PARAMS:
        seed_deltas = {s: deltas[f"{s}_v400_minus_{tag}"] for s in "ABC"}
        mean[f"v400_minus_{tag}"] = mean_metrics([d["overall"] for d in seed_deltas.values()])
        mean_deltas_by_category[f"v400_minus_{tag}"] = category_delta_mean(seed_deltas)

    summary = {
        "overall": overall,
        "by_category": by_category,
        "deltas": deltas,
        "mean": mean,
        "mean_deltas_by_category": mean_deltas_by_category,
        "categories": categories,
        "report_name": REPORT_NAME,
        "hypothesis": (
            "Spatial-frequency gated attention modulation should use RGB high-frequency "
            "and TIR low-frequency dominance as token-wise bridge-attention evidence, "
            "improving TBSI cross-modal interaction without prompt tuning, learned "
            "reliability statistics, or output residual gating."
        ),
    }
    out_path = OUT_DIR / "attr_metrics_v400_compare.json"
    json.dump(summary, out_path.open("w"), indent=2, ensure_ascii=False)
    print(json.dumps({"wrote": str(out_path), "mean": mean}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
