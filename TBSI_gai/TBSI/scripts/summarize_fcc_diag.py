import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATEGORY_FILE = ROOT / "output/diagnostics/signal_decouple/per_sequence_proxy_attr_metrics.json"
METRIC_KEYS = ("frequency_disagreement", "penalty", "bias", "rgb_low_ratio", "tir_low_ratio")


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/summarize_fcc_diag.py <diag_dir> <out_json>")
    diag_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    categories = json.load(CATEGORY_FILE.open())["categories"]

    by_seq = defaultdict(lambda: defaultdict(list))
    for path in sorted(diag_dir.glob("fcc_stats_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seq = row.get("sequence", "unknown")
            for key in METRIC_KEYS:
                if key in row:
                    by_seq[seq][f"{key}_mean"].append(row[key]["mean"])
                    by_seq[seq][f"{key}_std"].append(row[key]["std"])
                    by_seq[seq][f"{key}_max"].append(row[key]["max"])

    sequence_summary = {}
    by_category_raw = defaultdict(lambda: defaultdict(list))
    for seq, values in by_seq.items():
        sequence_summary[seq] = {k: mean(v) for k, v in values.items()}
        cat = categories.get(seq, "unknown")
        for k, v in sequence_summary[seq].items():
            by_category_raw[cat][k].append(v)

    by_category = {
        cat: {k: mean(v) for k, v in values.items()}
        for cat, values in by_category_raw.items()
    }
    summary = {
        "diag_dir": str(diag_dir),
        "num_sequences": len(sequence_summary),
        "sequence_summary": sequence_summary,
        "by_category": by_category,
        "interpretation": (
            "Higher frequency_disagreement_mean and penalty_mean indicate more "
            "above-average RGB/TIR spatial-frequency conflict under FCC."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, out_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({"wrote": str(out_path), "num_sequences": len(sequence_summary)}, indent=2))


if __name__ == "__main__":
    main()
