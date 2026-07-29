import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset, trackerlist


DATASET_NAME = "attribute_sentinel_lasher"
OUT_DIR = Path("output/diagnostics/stage1_fcc_v410_v420")
REPORT_NAME = "stage1_fcc_v410_v420"
MANIFEST_FILE = Path("output/diagnostics/attribute_sentinel/attribute_sentinel_manifest.json")

TARGET_ATTRS = ["NO", "HO", "LI", "BC", "LR", "CM", "FM", "TC", "PO", "SV"]
METRIC_KEYS = ["AUC", "OP50", "OP75", "Precision", "NormPrecision"]

TRACKER_GROUPS = {
    "v160_full": ["v1.6.0-signal-decouple-v1-全量"],
    "v165_full": ["v1.6.5-signal-decouple-learnscale-all-smax05-init01-全量"],
    "v410_fcc": [
        "v4.1.0-fcc-A",
        "v4.1.0-fcc-B",
        "v4.1.0-fcc-C",
    ],
    "v420_fcc_residual": [
        "v4.2.0-fcc-residual-A",
        "v4.2.0-fcc-residual-B",
        "v4.2.0-fcc-residual-C",
    ],
}


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


def mean_metrics(items):
    return {k: sum(x[k] for x in items) / len(items) for k in METRIC_KEYS}


def sub_metrics(a, b):
    return {k: a[k] - b[k] for k in METRIC_KEYS}


def coverage_from_delta(attr_deltas):
    return {
        "attr_count": len(attr_deltas),
        "auc_positive": sum(1 for v in attr_deltas.values() if v["AUC"] > 0),
        "auc_non_negative": sum(1 for v in attr_deltas.values() if v["AUC"] >= 0),
        "auc_drop_lt_minus_0_5": sum(1 for v in attr_deltas.values() if v["AUC"] < -0.5),
        "auc_drop_lt_minus_1_0": sum(1 for v in attr_deltas.values() if v["AUC"] < -1.0),
    }


def write_per_experiment(param, payload, manifest_path):
    exp_log_dir = Path("output") / "experiments" / param / "logs" / "stage1_attribute_sentinel"
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    json.dump(payload, (exp_log_dir / "attribute_sentinel_metrics.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    shutil.copyfile(OUT_DIR / "stage1_fcc_v410_v420_summary.json",
                    exp_log_dir / "stage1_fcc_v410_v420_summary.json")
    shutil.copyfile(manifest_path, exp_log_dir / "attribute_sentinel_manifest.json")
    seq_list = Path("experiments") / "tbsi_track" / "attribute_sentinel_sequences.txt"
    if seq_list.exists():
        shutil.copyfile(seq_list, exp_log_dir / "attribute_sentinel_sequences.txt")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.load(MANIFEST_FILE.open(encoding="utf-8"))
    seq_attrs = manifest["sequence_attributes"]
    dataset = get_dataset(DATASET_NAME)

    trackers = []
    for params in TRACKER_GROUPS.values():
        for param in params:
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
    attr_ids = {
        attr: [i for i, seq in enumerate(seq_names) if attr in seq_attrs.get(seq, [])]
        for attr in TARGET_ATTRS
    }

    tracker_metrics = {}
    tracker_attr_metrics = {}
    for trk_id, tracker in enumerate(eval_data["trackers"]):
        param = tracker["param"]
        tracker_metrics[param] = metric_from_curves(eval_data, valid_ids, trk_id)
        tracker_attr_metrics[param] = {
            attr: metric_from_curves(eval_data, ids, trk_id)
            for attr, ids in attr_ids.items() if ids
        }

    group_metrics = {}
    group_attr_metrics = {}
    for group, params in TRACKER_GROUPS.items():
        present = [p for p in params if p in tracker_metrics]
        group_metrics[group] = mean_metrics([tracker_metrics[p] for p in present])
        group_attr_metrics[group] = {}
        for attr in TARGET_ATTRS:
            attr_present = [p for p in present if attr in tracker_attr_metrics[p]]
            if attr_present:
                group_attr_metrics[group][attr] = mean_metrics(
                    [tracker_attr_metrics[p][attr] for p in attr_present]
                )

    comparisons = {
        "v410_minus_v160_full": ("v410_fcc", "v160_full"),
        "v410_minus_v165_full": ("v410_fcc", "v165_full"),
        "v420_minus_v160_full": ("v420_fcc_residual", "v160_full"),
        "v420_minus_v165_full": ("v420_fcc_residual", "v165_full"),
        "v420_minus_v410": ("v420_fcc_residual", "v410_fcc"),
    }
    deltas = {}
    coverage = {}
    for name, (cur, ref) in comparisons.items():
        attr_delta = {
            attr: sub_metrics(group_attr_metrics[cur][attr], group_attr_metrics[ref][attr])
            for attr in TARGET_ATTRS
            if attr in group_attr_metrics[cur] and attr in group_attr_metrics[ref]
        }
        deltas[name] = {
            "overall": sub_metrics(group_metrics[cur], group_metrics[ref]),
            "attributes": attr_delta,
        }
        coverage[name] = coverage_from_delta(attr_delta)

    summary = {
        "dataset": DATASET_NAME,
        "report_name": REPORT_NAME,
        "sequence_count": len(seq_names),
        "valid_sequence_count": len(valid_ids),
        "target_attrs": TARGET_ATTRS,
        "attr_sequence_counts": {attr: len(ids) for attr, ids in attr_ids.items()},
        "tracker_metrics": tracker_metrics,
        "tracker_attr_metrics": tracker_attr_metrics,
        "group_metrics": group_metrics,
        "group_attr_metrics": group_attr_metrics,
        "deltas": deltas,
        "coverage": coverage,
        "comparability_note": (
            "v4.1.0/v4.2.0 are MiniLasHeR-trained ABC checkpoints evaluated on the "
            "same attribute-sentinel sequence list. v1.6.0/v1.6.5 full checkpoints are "
            "included only as historical full-trained comparators; they are useful for "
            "failure-risk screening but are not an apples-to-apples mini-training baseline."
        ),
    }
    out_path = OUT_DIR / "stage1_fcc_v410_v420_summary.json"
    json.dump(summary, out_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    for group in ["v410_fcc", "v420_fcc_residual"]:
        for param in TRACKER_GROUPS[group]:
            per_exp = {
                "dataset": DATASET_NAME,
                "report_name": REPORT_NAME,
                "config": param,
                "group": group,
                "metrics": tracker_metrics.get(param),
                "attributes": tracker_attr_metrics.get(param),
                "group_metrics": group_metrics,
                "deltas": deltas,
                "coverage": coverage,
                "comparability_note": summary["comparability_note"],
            }
            write_per_experiment(param, per_exp, MANIFEST_FILE)

    print(json.dumps({
        "wrote": str(out_path),
        "group_metrics": group_metrics,
        "coverage": coverage,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
