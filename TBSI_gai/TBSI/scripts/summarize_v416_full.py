import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset, trackerlist


DATASET_NAME = "lasher_test"
OUT_DIR = Path("output/diagnostics/fcc_v416_full")
REPORT_NAME = "lasher_test_fcc_v416_full"
ATTR_ZIP = ROOT.parent / "data" / "lasher" / "Attributes&Order.zip"
TEST_LIST = ROOT.parent / "data" / "lasher" / "testingsetList.txt"
SKIP = {"advancedredcup", "cameraman_1202", "mirroratleft"}

TRACKER_PARAMS = [
    "v1.0.0-完美基线-全量",
    "v1.6.0-signal-decouple-v1-全量",
    "v1.6.5-signal-decouple-learnscale-all-smax05-init01-全量",
    "v4.1.0-fcc-全量",
    "v4.1.6-fcc-middle-全量",
]

DISPLAY_KEYS = [
    ("AUC", "AUC"),
    ("OP50", "OP50"),
    ("OP75", "OP75"),
    ("Precision", "Precision"),
    ("NormPrecision", "Norm Precision"),
]
METRIC_KEYS = [k for k, _ in DISPLAY_KEYS]


def parse_full_attributes():
    test_sequences = [s.strip() for s in TEST_LIST.read_text(encoding="utf-8").splitlines() if s.strip()]
    test_sequences = [s for s in test_sequences if s not in SKIP]
    with zipfile.ZipFile(ATTR_ZIP) as zf:
        order = zf.read("Attributes_order.txt").decode().strip().replace(" ", "")
        attr_names = [x for x in order.split(",") if x]
        seq_attrs = {}
        for seq in test_sequences:
            name = f"AttriSeqsTxt/{seq}.txt"
            if name not in zf.namelist():
                continue
            values = [int(x) for x in zf.read(name).decode().strip().split(",")]
            seq_attrs[seq] = [attr for attr, value in zip(attr_names, values) if value == 1]
    return attr_names, seq_attrs


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attr_names, seq_attrs = parse_full_attributes()

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
    attr_ids = defaultdict(list)
    for i, seq_name in enumerate(seq_names):
        for attr in seq_attrs.get(seq_name, []):
            attr_ids[attr].append(i)

    overall = {}
    attr_metrics = {}
    for trk_id, tracker in enumerate(eval_data["trackers"]):
        param = tracker["param"]
        overall[param] = metric_from_curves(eval_data, valid_ids, trk_id)
        attr_metrics[param] = {
            attr: metric_from_curves(eval_data, ids, trk_id)
            for attr, ids in attr_ids.items() if ids
        }

    baseline = "v1.0.0-完美基线-全量"
    current = "v4.1.6-fcc-middle-全量"
    rows = []
    for attr in attr_names:
        ids = attr_ids.get(attr, [])
        if not ids:
            continue
        base = attr_metrics[baseline][attr]
        cur = attr_metrics[current][attr]
        row = {"attr": attr, "seq_count": len(ids)}
        for key, display in DISPLAY_KEYS:
            row[f"baseline_{key}"] = base[key]
            row[f"v416_{key}"] = cur[key]
            row[f"delta_{key}"] = cur[key] - base[key]
        rows.append(row)

    summary = {
        "trackers": TRACKER_PARAMS,
        "overall": overall,
        "attributes": {attr: len(ids) for attr, ids in attr_ids.items() if ids},
        "rows": rows,
        "deltas": {
            "v416_minus_baseline": sub_metrics(overall[current], overall[baseline]),
            "v416_minus_v160": sub_metrics(overall[current], overall["v1.6.0-signal-decouple-v1-全量"]),
            "v416_minus_v165": sub_metrics(
                overall[current], overall["v1.6.5-signal-decouple-learnscale-all-smax05-init01-全量"]),
            "v416_minus_v410": sub_metrics(overall[current], overall["v4.1.0-fcc-全量"]),
        },
        "note": (
            "Baseline and prior full result files are reused from their standard tracking_results paths. "
            "v4.1.6 uses middle-layer threshold FCC: layers=[1], margin=0.05, learnable_scale=True."
        ),
    }
    out_path = OUT_DIR / "full_attr_compare_v416.json"
    json.dump(summary, out_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({"wrote": str(out_path), "deltas": summary["deltas"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
