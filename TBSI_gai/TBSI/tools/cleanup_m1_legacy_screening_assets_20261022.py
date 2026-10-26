#!/usr/bin/env python3
"""Remove superseded M1 cohort splits and tracking text outputs with a hash manifest."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiments/protocols/m1_screening_v1_20261022"
TRACKS = REPO / "output/test/tracking_results/tbsi_track"
KEEP_TRACK_DIRS = {
    "TBSI_E_MHSA_FULL_4090D_seed42",
    "v1.0.1-单卡4090D-24G-等效全局128加速",
}
OLD_COHORT_FILES = {
    "m1_action_space_comparison_20261013",
    "m1_candidate_recall_readout_v1",
    "m1_center_readout_fresh_v1",
    "m1_fusion_signal_admission_20261012",
    "m1_fusion_signal_expansion_20261012",
    "m1_identity_calibration_v1",
    "m1_identity_calibration_v2",
    "m1_message_position_20261012",
    "m1_need_aware_action_sweep_20261012",
    "m1_prefusion_mcca_admission_20261012",
    "m1_prefusion_mcca_admission_v2_20261012",
    "m1_two_direction_cohort_v1",
}
OLD_BUILDERS = {
    "analyze_m1_two_direction_cohort_20261010.py",
    "build_m1_candidate_recall_readout_cohort_20261010.py",
    "build_m1_identity_calibration_cohort_20261010.py",
    "build_m1_message_position_cohort_20261012.py",
    "build_m1_need_aware_action_sweep_cohort_20261012.py",
    "build_m1_two_direction_cohort_20261010.py",
    "revise_m1_identity_calibration_cohort_20261010.py",
}
PROTECTED = {
    REPO / "experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml":
        "fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e",
    REPO / "output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar":
        "1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_entry(path, kind):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {
        "path": str(path.relative_to(REPO)),
        "kind": kind,
        "bytes": sum(p.stat().st_size for p in files),
        "files": len(files),
        "file_hashes": [
            {"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in files
        ],
    }


def collect_targets():
    targets = []
    diag_root = REPO / "experiments/diagnostics"
    for name in sorted(OLD_COHORT_FILES):
        folder = diag_root / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and (
                path.name == "manifest.json"
                or "sequence" in path.name and path.suffix == ".txt"
                or path.name in {"calibration_sequences.txt", "validation_sequences.txt", "final_holdout_sequences.txt"}
            ):
                targets.append((path, "superseded_cohort_manifest"))

    tools = REPO / "tools"
    for name in sorted(OLD_BUILDERS):
        for path in (tools / name, tools / "__pycache__" / (name.replace(".py", ".cpython-38.pyc"))):
            if path.is_file():
                targets.append((path, "superseded_cohort_builder"))

    if TRACKS.is_dir():
        for path in sorted(TRACKS.iterdir()):
            if path.is_dir() and path.name not in KEEP_TRACK_DIRS:
                targets.append((path, "historical_tracking_result"))
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="write manifest, then delete listed targets")
    args = parser.parse_args()

    for path, expected in PROTECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise SystemExit("protected M1 hash mismatch: {} ({})".format(path, actual))

    targets = collect_targets()
    entries = [tree_entry(path, kind) for path, kind in targets]
    summary = {
        "id": "M1-LEGACY-SCREENING-ASSET-CLEANUP-20261022",
        "mode": "delete" if args.delete else "dry-run",
        "retained_tracking_result_directories": sorted(KEEP_TRACK_DIRS),
        "removed_entries": len(entries),
        "removed_files": sum(item["files"] for item in entries),
        "removed_bytes": sum(item["bytes"] for item in entries),
        "protected_m1_hashes_verified_before": {
            str(path.relative_to(REPO)): sha256(path) for path in PROTECTED
        },
        "entries": entries,
    }
    if not args.delete:
        print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, indent=2, ensure_ascii=False))
        return

    manifest = PROTOCOL / "legacy_asset_cleanup_20261022.json"
    if manifest.exists():
        raise SystemExit("cleanup manifest already exists; refusing to overwrite: {}".format(manifest))
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for path, _kind in targets:
        if path.is_dir():
            shutil.rmtree(str(path))
        elif path.exists():
            path.unlink()

    for path, expected in PROTECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise SystemExit("protected M1 hash mismatch after cleanup: {} ({})".format(path, actual))
    summary["protected_m1_hashes_verified_after"] = {
        str(path.relative_to(REPO)): sha256(path) for path in PROTECTED
    }
    summary["mode"] = "deleted_and_verified"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
