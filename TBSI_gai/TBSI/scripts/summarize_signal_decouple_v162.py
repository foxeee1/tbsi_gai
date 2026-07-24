import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset, trackerlist


DATASET_NAME = "mini_lasher_test"
OUT_DIR = Path("output/diagnostics/signal_decouple_v162")
REPORT_NAME = "mini_lasher_test_signal_decouple_v162_full"

TRACKER_PARAMS = [
    "v1.6.2-signal-decouple-residual-l12-A",
    "v1.6.2-signal-decouple-residual-l12-B",
    "v1.6.2-signal-decouple-residual-l12-C",
    "v1.6.2-signal-decouple-residual-l12-A-bypass",
    "v1.6.2-signal-decouple-residual-l12-B-bypass",
    "v1.6.2-signal-decouple-residual-l12-C-bypass",
]

CATEGORY_FILE = Path("output/diagnostics/signal_decouple/per_sequence_proxy_attr_metrics.json")
PREVIOUS_SUMMARY_FILE = Path("output/diagnostics/signal_decouple_v161/attr_and_bypass_metrics.json")


def load_categories():
    with CATEGORY_FILE.open("r") as f:
        data = json.load(f)
    return data["categories"]


def build_trackers():
    trackers = []
    for param in TRACKER_PARAMS:
        trackers.extend(
            trackerlist(
                name="tbsi_track",
                parameter_name=param,
                dataset_name=DATASET_NAME,
                run_ids=None,
                display_name=param,
            )
        )
    return trackers


def metric_from_curves(eval_data, seq_ids, trk_id):
    overlap = torch.tensor(eval_data["ave_success_rate_plot_overlap"], dtype=torch.float64)
    center = torch.tensor(eval_data["ave_success_rate_plot_center"], dtype=torch.float64)
    center_norm = torch.tensor(eval_data["ave_success_rate_plot_center_norm"], dtype=torch.float64)
    avg_overlap = torch.tensor(eval_data["avg_overlap_all"], dtype=torch.float64)
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
        "MeanIoU": float(avg_overlap[seq_ids, trk_id].mean().item() * 100.0),
    }


def sub_metrics(a, b):
    return {k: a[k] - b[k] for k in ["AUC", "OP50", "OP75", "Precision", "NormPrecision"]}


def mean_metrics(items):
    keys = ["AUC", "OP50", "OP75", "Precision", "NormPrecision"]
    return {k: sum(x[k] for x in items) / len(items) for k in keys}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PREVIOUS_SUMMARY_FILE.open("r") as f:
        previous_summary = json.load(f)

    dataset = get_dataset(DATASET_NAME)
    trackers = build_trackers()
    eval_data = extract_results(
        trackers,
        dataset,
        REPORT_NAME,
        skip_missing_seq=False,
        plot_bin_gap=0.05,
    )

    categories = load_categories()
    seq_names = eval_data["sequences"]
    valid_ids = [i for i, v in enumerate(eval_data["valid_sequence"]) if v]
    category_ids = {}
    for i, seq_name in enumerate(seq_names):
        category_ids.setdefault(categories[seq_name], []).append(i)

    overall = dict(previous_summary["overall"])
    by_category = dict(previous_summary["by_category"])
    per_sequence = {}
    for trk_id, tracker in enumerate(eval_data["trackers"]):
        param = tracker["param"]
        overall[param] = {
            k: v
            for k, v in metric_from_curves(eval_data, valid_ids, trk_id).items()
            if k != "MeanIoU"
        }
        by_category[param] = {
            cat: {
                k: v
                for k, v in metric_from_curves(eval_data, ids, trk_id).items()
                if k != "MeanIoU"
            }
            for cat, ids in category_ids.items()
        }
        per_sequence[param] = {
            seq_names[i]: metric_from_curves(eval_data, [i], trk_id)
            for i in valid_ids
        }

    deltas = {}
    for seed in ["A", "B", "C"]:
        v162 = f"v1.6.2-signal-decouple-residual-l12-{seed}"
        bypass = f"{v162}-bypass"
        base = f"v1.0.0-完美基线-{seed}"
        v160 = f"v1.6.0-signal-decouple-v1-{seed}"
        v161 = f"v1.6.1-signal-decouple-zeroinit-{seed}"
        for tag, ref in [
            ("minus_bypass", bypass),
            ("minus_baseline", base),
            ("minus_v160", v160),
            ("minus_v161", v161),
        ]:
            key = f"{seed}_v162_{tag}"
            deltas[key] = {
                "overall": sub_metrics(overall[v162], overall[ref]),
                "by_category": {
                    cat: sub_metrics(by_category[v162][cat], by_category[ref][cat])
                    for cat in category_ids
                },
            }

    def seed_keys(tag):
        return [deltas[f"{seed}_v162_{tag}"]["overall"] for seed in ["A", "B", "C"]]

    mean = {
        "v162": mean_metrics([overall[f"v1.6.2-signal-decouple-residual-l12-{s}"] for s in ["A", "B", "C"]]),
        "baseline": mean_metrics([overall[f"v1.0.0-完美基线-{s}"] for s in ["A", "B", "C"]]),
        "v160": mean_metrics([overall[f"v1.6.0-signal-decouple-v1-{s}"] for s in ["A", "B", "C"]]),
        "v161": mean_metrics([overall[f"v1.6.1-signal-decouple-zeroinit-{s}"] for s in ["A", "B", "C"]]),
        "v162_bypass": mean_metrics([overall[f"v1.6.2-signal-decouple-residual-l12-{s}-bypass"] for s in ["A", "B", "C"]]),
        "v162_minus_bypass": mean_metrics(seed_keys("minus_bypass")),
        "v162_minus_baseline": mean_metrics(seed_keys("minus_baseline")),
        "v162_minus_v160": mean_metrics(seed_keys("minus_v160")),
        "v162_minus_v161": mean_metrics(seed_keys("minus_v161")),
    }

    summary = {
        "overall": overall,
        "by_category": by_category,
        "per_sequence": per_sequence,
        "deltas": deltas,
        "mean": mean,
        "categories": categories,
        "report_name": REPORT_NAME,
    }
    out_path = OUT_DIR / "attr_and_bypass_metrics.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"wrote": str(out_path), "mean": mean}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
