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
OUT_DIR = Path("output/diagnostics/signal_decouple_v165")
REPORT_NAME = "mini_lasher_test_signal_decouple_v165"
CATEGORY_FILE = Path("output/diagnostics/signal_decouple/per_sequence_proxy_attr_metrics.json")
PREVIOUS_SUMMARY_FILE = Path("output/diagnostics/signal_decouple_v162/attr_and_bypass_metrics.json")

TRACKER_PARAMS = [
    "v1.6.5-signal-decouple-learnscale-all-smax05-init01-A",
    "v1.6.5-signal-decouple-learnscale-all-smax05-init01-B",
    "v1.6.5-signal-decouple-learnscale-all-smax05-init01-C",
]


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
    keys = ["AUC", "OP50", "OP75", "Precision", "NormPrecision"]
    return {k: a[k] - b[k] for k in keys}


def mean_metrics(items):
    keys = ["AUC", "OP50", "OP75", "Precision", "NormPrecision"]
    return {k: sum(x[k] for x in items) / len(items) for k in keys}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    categories = json.load(CATEGORY_FILE.open())["categories"]
    previous = json.load(PREVIOUS_SUMMARY_FILE.open())
    overall = dict(previous["overall"])
    by_category = dict(previous["by_category"])

    dataset = get_dataset(DATASET_NAME)
    trackers = []
    for param in TRACKER_PARAMS:
        trackers.extend(
            trackerlist("tbsi_track", param, DATASET_NAME, None, param)
        )
    eval_data = extract_results(trackers, dataset, REPORT_NAME, skip_missing_seq=False, plot_bin_gap=0.05)

    seq_names = eval_data["sequences"]
    valid_ids = [i for i, v in enumerate(eval_data["valid_sequence"]) if v]
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
        v165 = f"v1.6.5-signal-decouple-learnscale-all-smax05-init01-{seed}"
        for tag, ref in [
            ("minus_baseline", f"v1.0.0-完美基线-{seed}"),
            ("minus_v160", f"v1.6.0-signal-decouple-v1-{seed}"),
            ("minus_v161", f"v1.6.1-signal-decouple-zeroinit-{seed}"),
            ("minus_v162", f"v1.6.2-signal-decouple-residual-l12-{seed}"),
        ]:
            deltas[f"{seed}_v165_{tag}"] = {
                "overall": sub_metrics(overall[v165], overall[ref]),
                "by_category": {
                    cat: sub_metrics(by_category[v165][cat], by_category[ref][cat])
                    for cat in category_ids
                },
            }

    mean = {
        "v165": mean_metrics([overall[f"v1.6.5-signal-decouple-learnscale-all-smax05-init01-{s}"] for s in "ABC"]),
        "baseline": mean_metrics([overall[f"v1.0.0-完美基线-{s}"] for s in "ABC"]),
        "v160": mean_metrics([overall[f"v1.6.0-signal-decouple-v1-{s}"] for s in "ABC"]),
        "v161": mean_metrics([overall[f"v1.6.1-signal-decouple-zeroinit-{s}"] for s in "ABC"]),
        "v162": mean_metrics([overall[f"v1.6.2-signal-decouple-residual-l12-{s}"] for s in "ABC"]),
    }
    for tag in ["minus_baseline", "minus_v160", "minus_v161", "minus_v162"]:
        mean[f"v165_{tag}"] = mean_metrics([deltas[f"{s}_v165_{tag}"]["overall"] for s in "ABC"])

    summary = {
        "overall": overall,
        "by_category": by_category,
        "deltas": deltas,
        "mean": mean,
        "categories": categories,
        "note": "A/B checkpoints were deleted after testing to recover disk space; raw tracking results are preserved.",
    }
    out_path = OUT_DIR / "attr_metrics.json"
    json.dump(summary, out_path.open("w"), indent=2, ensure_ascii=False)
    print(json.dumps({"wrote": str(out_path), "mean": mean}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
