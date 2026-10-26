#!/usr/bin/env python3
"""Merge human labels into a new, audited copy of the frozen queue."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TRUTH = {"1", "true", "yes", "y", "是"}
FALSE = {"0", "false", "no", "n", "否"}
LABELS = {"P", "S", "N", "UNCERTAIN"}
BOX_RE = re.compile(r"^\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]\s*$")


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value):
    value = (value or "").strip().lower()
    if value in TRUTH:
        return True
    if value in FALSE:
        return False
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    for path in (args.out, args.report):
        if path.exists():
            raise SystemExit(f"Refuse to overwrite existing output: {path}")
    candidates, queue, review = map(read_rows, (args.candidates, args.queue, args.review))
    errors = []
    source = {row["event_id"]: row for row in candidates}
    frozen = {row["event_id"]: row for row in queue}
    labels = {row.get("event_id", ""): row for row in review}
    if len(source) != len(candidates) or len(frozen) != len(queue) or len(labels) != len(review):
        errors.append("duplicate or missing event_id in candidates, queue, or review")
    if set(labels) != set(frozen):
        errors.append(f"review IDs do not exactly match queue: missing={sorted(set(frozen)-set(labels))[:5]}, extra={sorted(set(labels)-set(frozen))[:5]}")
    immutable = ("seq", "frame", "screen_pass", "rgb_box", "tir_box")
    merged = []
    for base in queue:
        event_id = base["event_id"]
        row = dict(base)
        label = labels.get(event_id, {})
        for key, value in label.items():
            if key in {"event_type", "same_physical_object", "visible_in_both", "alignment_ok", "reviewer", "comment", "distractor_box_rgb", "distractor_box_tir", "non_target_confirmed", "physical_object_id"}:
                row[key] = value
        for key in immutable:
            if row.get(key) != frozen[event_id].get(key):
                errors.append(f"{event_id}: immutable field {key} changed")
        event_type = (row.get("event_type") or "UNCERTAIN").strip().upper()
        if event_type not in LABELS:
            errors.append(f"{event_id}: invalid event_type {event_type}")
        if not (row.get("reviewer") or "").strip():
            errors.append(f"{event_id}: missing reviewer")
        if event_type == "P":
            for key in ("same_physical_object", "visible_in_both", "alignment_ok", "non_target_confirmed"):
                if truth(row.get(key)) is not True:
                    errors.append(f"{event_id}: P needs {key}=true")
            if not (row.get("physical_object_id") or "").strip():
                errors.append(f"{event_id}: P missing physical_object_id")
            for key in ("distractor_box_rgb", "distractor_box_tir"):
                if not BOX_RE.match((row.get(key) or "").strip()):
                    errors.append(f"{event_id}: P needs valid {key}=[x,y,w,h]")
        if args.require_complete and event_type == "UNCERTAIN":
            errors.append(f"{event_id}: incomplete review")
        merged.append(row)
    labels_count = Counter((row.get("event_type") or "UNCERTAIN").strip().upper() for row in merged)
    verified_p = [row for row in merged if (row.get("event_type") or "").strip().upper() == "P"]
    p_keys = {(row.get("seq"), row.get("frame"), row.get("physical_object_id")) for row in verified_p}
    report = {
        "queue_rows": len(queue), "review_rows": len(review), "labels": dict(labels_count),
        "verified_p_rows": len(verified_p), "verified_p_distinct_frame_object_keys": len(p_keys),
        "verified_p_distinct_frames": len({(row.get("seq"), row.get("frame")) for row in verified_p}),
        "verified_p_distinct_sequences": len({row.get("seq") for row in verified_p}),
        "errors": errors,
        "status": "READY_FOR_GATE_REVIEW" if not errors and len(p_keys) >= 20 else "AWAITING-BLIND-REVIEW" if not errors else "REVIEW-INVALID",
        "note": "P count is administrative and requires independent review/adjudication before causal intervention claims.",
    }
    if errors:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fields = list(queue[0].keys()) if queue else []
    for row in merged:
        for key in row:
            if key not in fields:
                fields.append(key)
    with args.out.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
