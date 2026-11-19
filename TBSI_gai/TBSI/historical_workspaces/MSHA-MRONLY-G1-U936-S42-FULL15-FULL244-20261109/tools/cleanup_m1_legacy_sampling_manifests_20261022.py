#!/usr/bin/env python3
"""Remove superseded experiment-level sequence manifests; preserve current protocol."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments"
PROTOCOL = ROOT / "experiments/protocols/m1_screening_v1_20261022"
MEMBER_NAMES = {
    "sample_manifest.json",
    "fixed_event_manifest.json",
    "test_manifest.json",
    "trace_sequences.txt",
}
OBSOLETE_SELECTOR = ROOT / "tools/select_m1_precision_screen_sequences_20261020.py"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect():
    paths = []
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith("M1_SCREENING_") for part in path.parts):
            continue
        name = path.name.lower()
        if name in MEMBER_NAMES or "sequence_list" in name:
            paths.append(path)
    if OBSOLETE_SELECTOR.is_file():
        paths.append(OBSOLETE_SELECTOR)
    pyc = ROOT / "tools/__pycache__/select_m1_precision_screen_sequences_20261020.cpython-38.pyc"
    if pyc.is_file():
        paths.append(pyc)
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    paths = collect()
    entries = [{
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in paths]
    result = {
        "id": "M1-LEGACY-SAMPLING-MANIFEST-CLEANUP-20261022",
        "mode": "delete" if args.delete else "dry-run",
        "excluded": "All M1_SCREENING_* outputs and m1_screening_v1_20261022 protocol assets",
        "count": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    if not args.delete:
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
        return
    manifest = PROTOCOL / "legacy_sampling_cleanup_20261022.json"
    if manifest.exists():
        raise SystemExit("refusing to overwrite existing cleanup manifest: {}".format(manifest))
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for path in paths:
        path.unlink()
    result["mode"] = "deleted"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
