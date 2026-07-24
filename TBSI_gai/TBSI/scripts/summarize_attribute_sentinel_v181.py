import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset, trackerlist


DATASET_NAME = "attribute_sentinel_lasher"
OUT_DIR = Path("output/diagnostics/attribute_sentinel")
REPORT_NAME = "attribute_sentinel_v181"
MANIFEST_FILE = OUT_DIR / "attribute_sentinel_manifest.json"

TRACKER_GROUPS = {
    "v160_full": ["v1.6.0-signal-decouple-v1-全量"],
    "v165_full": ["v1.6.5-signal-decouple-learnscale-all-smax05-init01-全量"],
    "v181_mini": [
        "v1.8.1-reliability-guided-late-layers-A",
        "v1.8.1-reliability-guided-late-layers-B",
        "v1.8.1-reliability-guided-late-layers-C",
    ],
}
TARGET_ATTRS = ["NO", "HO", "LI", "BC", "LR", "CM", "FM", "TC", "PO", "SV"]
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


def mean_metrics(items):
    return {k: sum(x[k] for x in items) / len(items) for k in METRIC_KEYS}


def sub_metrics(a, b):
    return {k: a[k] - b[k] for k in METRIC_KEYS}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.load(MANIFEST_FILE.open())
    seq_attrs = manifest["sequence_attributes"]
    dataset = get_dataset(DATASET_NAME)

    trackers = []
    tracker_to_group = {}
    for group, params in TRACKER_GROUPS.items():
        for param in params:
            trackers.extend(trackerlist("tbsi_track", param, DATASET_NAME, None, param))
            tracker_to_group[param] = group

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
                    [tracker_attr_metrics[p][attr] for p in attr_present])

    deltas = {}
    for ref in ["v160_full", "v165_full"]:
        deltas[f"v181_mini_minus_{ref}"] = {
            "overall": sub_metrics(group_metrics["v181_mini"], group_metrics[ref]),
            "attributes": {
                attr: sub_metrics(group_attr_metrics["v181_mini"][attr],
                                  group_attr_metrics[ref][attr])
                for attr in TARGET_ATTRS
                if attr in group_attr_metrics["v181_mini"] and attr in group_attr_metrics[ref]
            },
        }

    coverage = {}
    for key, delta in deltas.items():
        attrs = delta["attributes"]
        coverage[key] = {
            "attr_count": len(attrs),
            "auc_non_negative": sum(1 for v in attrs.values() if v["AUC"] >= 0),
            "auc_drop_lt_minus_0_5": sum(1 for v in attrs.values() if v["AUC"] < -0.5),
            "auc_drop_lt_minus_1_0": sum(1 for v in attrs.values() if v["AUC"] < -1.0),
        }

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
            "v181_mini uses MiniLasHeR-trained checkpoints; v160_full/v165_full "
            "use full-trained checkpoints and existing full tracking results. "
            "This Stage-1 sentinel analysis is a failure-risk screen, not a "
            "claim-carrying full comparison."
        ),
    }
    out_path = OUT_DIR / "attribute_sentinel_v181_summary.json"
    json.dump(summary, out_path.open("w"), indent=2, ensure_ascii=False)
    print(json.dumps({
        "wrote": str(out_path),
        "group_metrics": group_metrics,
        "coverage": coverage,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
