#!/usr/bin/env python3
"""Audit frozen risk-gated state commits for the official 244-sequence run."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/M1_S1_SELECTIVE_STATE_COMMIT_FULL244_20261015/full244"
TRACE = OUT / "trace"
LOW = ROOT / "output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/s1_full62/sequence_list.txt"
NORMAL = ROOT / "output/experiments/M1_SINGLE_SIGNAL_NORMAL20_20261012/s1_full20/sequence_list.txt"


def members(path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main():
    low, normal = members(LOW), members(NORMAL)
    files = sorted(TRACE.glob("*/frames.jsonl"))
    if len(files) != 244:
        raise RuntimeError("expected 244 trace files, got {}".format(len(files)))
    rows = []
    cohort_totals = defaultdict(lambda: [0, 0, 0, 0.0])
    total_frames = total_commit = invalid = 0
    for path in files:
        sequence = path.parent.name
        frames = committed = 0
        risk_sum = 0.0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            risk = float(row["state_commit_risk_probability"])
            threshold = float(row["state_commit_risk_threshold"])
            expected = risk <= threshold
            actual = bool(row["state_commit_corrected"])
            invalid += int(actual != expected)
            frames += 1
            committed += int(actual)
            risk_sum += risk
        if not frames:
            raise RuntimeError("empty trace for {}".format(sequence))
        ratio = committed / float(frames)
        rows.append({"sequence": sequence, "frames": frames, "corrected_state_commits": committed,
                     "m1_fallbacks": frames - committed, "commit_coverage": ratio,
                     "mean_risk_probability": risk_sum / frames})
        cohorts = ["all244"]
        if sequence in low:
            cohorts.append("low_auc62")
        elif sequence in normal:
            cohorts.append("normal20_sentinel")
        else:
            cohorts.append("remaining162")
        for cohort in cohorts:
            value = cohort_totals[cohort]
            value[0] += frames
            value[1] += committed
            value[2] += 1
            value[3] += risk_sum
        total_frames += frames
        total_commit += committed
    if invalid:
        raise RuntimeError("{} state decisions violated the frozen risk threshold".format(invalid))
    rows.sort(key=lambda x: x["sequence"])
    with (OUT / "state_commit_by_sequence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sequences": len(rows), "frames": total_frames, "corrected_state_commits": total_commit,
        "m1_fallbacks": total_frames - total_commit, "overall_commit_coverage": total_commit / float(total_frames),
        "threshold_violations": invalid, "cohorts": {},
    }
    for name, value in cohort_totals.items():
        summary["cohorts"][name] = {
            "sequences": value[2], "frames": value[0], "corrected_state_commits": value[1],
            "m1_fallbacks": value[0] - value[1], "commit_coverage": value[1] / float(value[0]),
            "mean_risk_probability": value[3] / value[0],
        }
    (OUT / "state_commit_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("stages", {})["state_commit_audit"] = {
        "status": "COMPLETE", "sequences": len(rows), "frames": total_frames,
        "threshold_violations": invalid, "summary": str(OUT / "state_commit_audit.json"),
        "per_sequence": str(OUT / "state_commit_by_sequence.csv"),
    }
    manifest["analysis_code"] = str(Path(__file__).resolve())
    manifest["analysis_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["state_commit_summary"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
