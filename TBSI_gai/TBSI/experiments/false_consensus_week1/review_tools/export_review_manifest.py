#!/usr/bin/env python3
"""Export blind-review metadata without scores or screening metadata."""
import argparse
import csv
from pathlib import Path


REVIEW_FIELDS = (
    "event_id", "seq", "frame", "rgb_box", "tir_box", "gt_box",
    "distractor_box_rgb", "distractor_box_tir", "event_type",
    "same_physical_object", "visible_in_both", "alignment_ok", "reviewer", "comment",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("input/output must differ: do not overwrite the frozen queue")

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event_id", "seq", "frame", "rgb_box", "tir_box", "gt_box"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("missing required queue columns")
        rows = list(reader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REVIEW_FIELDS})
    print(f"exported {len(rows)} rows; hidden baseline_box, response scores, screening reason and priority")


if __name__ == "__main__":
    main()
