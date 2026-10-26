#!/usr/bin/env python3
"""Audit blind labels and source-row integrity without intervention outputs."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


YES = {"1", "true", "yes", "y", "是"}
NO = {"0", "false", "no", "n", "否"}


def truth(value):
    value = (value or "").strip().lower()
    if value in YES:
        return True
    if value in NO:
        return False
    return None


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(candidates, queue):
    errors = []
    candidate_ids = [row.get("event_id", "") for row in candidates]
    queue_ids = [row.get("event_id", "") for row in queue]
    if len(set(candidate_ids)) != len(candidates):
        errors.append("duplicate candidate event_id")
    if len(set(queue_ids)) != len(queue):
        errors.append("duplicate queue event_id")

    source = {row["event_id"]: row for row in candidates if row.get("event_id")}
    queued = {row["event_id"]: row for row in queue if row.get("event_id")}
    missing = set(queued) - set(source)
    if missing:
        errors.append(f"queue IDs absent from source: {sorted(missing)[:5]}")

    passed = {row["event_id"] for row in candidates if row.get("screen_pass") == "1"}
    if not passed <= set(queued):
        errors.append("screened-in candidates missing from queue")

    immutable = ("seq", "frame", "screen_pass", "rgb_box", "tir_box")
    for row in queue:
        event_id = row.get("event_id", "")
        if event_id not in source:
            continue
        original = source[event_id]
        for key in immutable:
            if row.get(key) != original.get(key):
                errors.append(f"{event_id}: immutable field {key} changed")
        label = (row.get("event_type") or "uncertain").strip().upper()
        if label not in {"P", "S", "N", "UNCERTAIN"}:
            errors.append(f"{event_id}: invalid event_type {label}")
        if label == "UNCERTAIN":
            continue
        if not (row.get("reviewer") or "").strip():
            errors.append(f"{event_id}: decided label without reviewer")
        if label == "P":
            for field in ("same_physical_object", "visible_in_both", "alignment_ok"):
                if truth(row.get(field)) is not True:
                    errors.append(f"{event_id}: P needs {field}=true")
            for field in ("distractor_box_rgb", "distractor_box_tir"):
                if not (row.get(field) or "").strip():
                    errors.append(f"{event_id}: P missing reviewed {field}")

    labels = collections.Counter(
        (row.get("event_type") or "uncertain").strip().upper() for row in queue
    )
    verified = [
        row for row in queue
        if (row.get("event_type") or "").strip().upper() == "P"
        and all(truth(row.get(field)) is True for field in
                ("same_physical_object", "visible_in_both", "alignment_ok"))
        and (row.get("reviewer") or "").strip()
    ]
    reviewed = [
        row for row in queue
        if (row.get("event_type") or "uncertain").strip().upper() != "UNCERTAIN"
    ]
    rejected_verified = [row for row in verified if row.get("screen_pass") == "0"]
    rejected_reviewed = [row for row in reviewed if row.get("screen_pass") == "0"]
    frame_keys = {(row["seq"], row["frame"]) for row in verified}
    return {
        "original_rows": len(candidates),
        "queue_rows": len(queue),
        "screen_pass_rows": len(passed),
        "reviewed_rows": len(reviewed),
        "labels": dict(labels),
        "verified_p_rows": len(verified),
        "verified_p_distinct_frames": len(frame_keys),
        "verified_p_distinct_sequences": len({row["seq"] for row in verified}),
        "rejected_sample_reviewed": len(rejected_reviewed),
        "rejected_sample_verified_p": len(rejected_verified),
        "screened_in_verified_p": len(verified) - len(rejected_verified),
        "errors": errors,
        "warning": "S/N labels in a paired-candidate queue are not independently sampled controls; deduplicate repeated physical distractors after object_id annotation.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(read_csv(args.candidates), read_csv(args.queue))
    report["sha256"] = {
        "candidates": digest(args.candidates),
        "queue": digest(args.queue),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
