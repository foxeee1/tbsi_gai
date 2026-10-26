#!/usr/bin/env python3
"""Compare full-test sequence failure and low-AUC maps for LOC[4,7,10] vs M1."""
import argparse
import csv
import json
import subprocess
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return {row["sequence"]: row for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--m1-dir", required=True)
    ap.add_argument("--candidate-dir", required=True)
    ap.add_argument("--sequence-list", default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sequence_list = args.sequence_list or str(out / "sequence_list.txt")
    if args.sequence_list is None:
        names = sorted(p.name for p in Path(args.gt_root).iterdir() if (p / "init.txt").is_file())
        Path(sequence_list).write_text("\n".join(names) + "\n", encoding="utf-8")
    mapper = root / "tools/m1_full_test_failure_map.py"
    maps = {}
    for label, pred_dir in (("m1", args.m1_dir), ("loc4710", args.candidate_dir)):
        map_dir = out / label
        subprocess.run([
            "python", str(mapper), "--pred-dir", pred_dir, "--gt-root", args.gt_root,
            "--sequence-list", sequence_list, "--baseline-dir", args.m1_dir,
            "--output-dir", str(map_dir)
        ], check=True)
        maps[label] = read_csv(map_dir / "per_sequence.csv")

    rows = []
    for seq in sorted(maps["m1"]):
        m = maps["m1"][seq]
        c = maps["loc4710"].get(seq)
        if c is None:
            continue
        m_auc = float(m["auc_mean_iou"])
        c_auc = float(c["auc_mean_iou"])
        rows.append({
            "sequence": seq,
            "m1_auc": m_auc,
            "loc4710_auc": c_auc,
            "delta_auc": c_auc - m_auc,
            "m1_failure_rate": float(m["failure_rate_iou_lt_0.5"]),
            "loc4710_failure_rate": float(c["failure_rate_iou_lt_0.5"]),
            "delta_failure_rate": float(c["failure_rate_iou_lt_0.5"]) - float(m["failure_rate_iou_lt_0.5"]),
            "m1_severe_rate": float(m["severe_rate_iou_lt_0.2"]),
            "loc4710_severe_rate": float(c["severe_rate_iou_lt_0.2"]),
            "delta_severe_rate": float(c["severe_rate_iou_lt_0.2"]) - float(m["severe_rate_iou_lt_0.2"]),
            "m1_any_failure": int(float(m["failure_segments_iou_lt_0.5"]) > 0),
            "loc4710_any_failure": int(float(c["failure_segments_iou_lt_0.5"]) > 0),
            "m1_severe_failure": int(float(m["severe_segments_iou_lt_0.2"]) > 0),
            "loc4710_severe_failure": int(float(c["severe_segments_iou_lt_0.2"]) > 0),
        })
    fields = list(rows[0])
    with (out / "per_sequence_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    def low(rows_, key):
        return {r["sequence"] for r in rows_ if r[key] < 0.40}
    m_low = low(rows, "m1_auc")
    c_low = low(rows, "loc4710_auc")
    failure_m = {r["sequence"] for r in rows if r["m1_any_failure"]}
    failure_c = {r["sequence"] for r in rows if r["loc4710_any_failure"]}
    severe_m = {r["sequence"] for r in rows if r["m1_severe_failure"]}
    severe_c = {r["sequence"] for r in rows if r["loc4710_severe_failure"]}
    ranked = sorted(rows, key=lambda r: r["delta_auc"])
    summary = {
        "n_sequences": len(rows),
        "low_auc_threshold": 0.40,
        "m1_low_auc_n": len(m_low), "loc4710_low_auc_n": len(c_low),
        "low_auc_intersection_n": len(m_low & c_low),
        "low_auc_new_in_loc4710": sorted(c_low - m_low),
        "low_auc_lost_from_loc4710": sorted(m_low - c_low),
        "failure_any_m1_n": len(failure_m), "failure_any_loc4710_n": len(failure_c),
        "failure_intersection_n": len(failure_m & failure_c),
        "failure_new_in_loc4710": sorted(failure_c - failure_m),
        "failure_lost_from_loc4710": sorted(failure_m - failure_c),
        "severe_m1_n": len(severe_m), "severe_loc4710_n": len(severe_c),
        "severe_intersection_n": len(severe_m & severe_c),
        "largest_auc_regressions": ranked[:20],
        "largest_auc_gains": list(reversed(ranked[-20:])),
        "evidence": {"m1_map": "m1/summary.json", "loc4710_map": "loc4710/summary.json", "comparison": "per_sequence_comparison.csv"},
        "interpretation": "Descriptive full-test sequence comparison; one seed and contiguous frames are not independent.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
