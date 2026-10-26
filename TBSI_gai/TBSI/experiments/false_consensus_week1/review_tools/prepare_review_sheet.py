#!/usr/bin/env python3
"""Prepare a versioned human-review copy without changing the manifest."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.manifest.resolve() == args.out.resolve() or args.out.exists():
        raise SystemExit("Refuse to overwrite input or existing output")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not {"event_id", "event_type", "reviewer"} <= set(fields):
        raise ValueError("not a reviewer manifest")
    fields += [field for field in ("non_target_confirmed", "physical_object_id") if field not in fields]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} review rows in {args.out}; image review is still required")


if __name__ == "__main__":
    main()
