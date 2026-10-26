#!/usr/bin/env python3
"""Train and evaluate the seed-42 Cdiag confirmation against fixed M1.

The candidate starts from the protected M1 checkpoint and TRAIN.CPLAIN_ONLY
freezes every M1 parameter except POST_TBSI_RESIDUAL.  The script writes a
manifest before starting, refuses to overwrite output, and runs train -> test
-> official metric audit serially.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "M1_FUSION_CDIAG_FULL_15EP_seed42"
M1 = "TBSI_E_MHSA_FULL_4090D_seed42"
M1_CKPT = ROOT / "output/experiments" / M1 / "checkpoints/TBSITrack_ep0015.pth.tar"
M1_RESULTS = ROOT / "output/test/tracking_results/tbsi_track" / M1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, env, log_path):
    started = time.time()
    with open(log_path, "w") as stream:
        subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return time.time() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()
    config_name = args.config
    out = Path(args.output_root).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing non-empty output root: {}".format(out))
    out.mkdir(parents=True, exist_ok=True)
    cfg = ROOT / "experiments/tbsi_track" / (config_name + ".yaml")
    if not cfg.exists() or not M1_CKPT.exists():
        raise RuntimeError("missing config or protected M1 checkpoint")
    if not M1_RESULTS.is_dir():
        raise RuntimeError("missing fixed M1 full-244 results: {}".format(M1_RESULTS))
    manifest = {
        "experiment_id": "M1-FUSION-CDIAG-FULL-15EP-20261013",
        "parent": M1,
        "candidate_config": str(cfg),
        "candidate_config_sha256": sha256(cfg),
        "parent_checkpoint": str(M1_CKPT),
        "parent_checkpoint_sha256": sha256(M1_CKPT),
        "seed": 42,
        "dataset_train": "LasHeR_train full configured SAMPLE_PER_EPOCH=59904",
        "dataset_test": "LasHeR full 244 lasher_test sequences",
        "candidate_contract": "M1 checkpoint initialization; TRAIN.CPLAIN_ONLY freezes M1 and trains only post_tbsi_residual diag",
        "resource_deviation": "NUM_WORKER=1 (main + one worker) to satisfy current global process limit; M1 historical config uses worker=4",
        "commands": [],
        "stages": {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1", "TBSI_DEBUG_FINITE": "1"})
    train_root = out / "training"
    train_cmd = [sys.executable, "lib/train/run_training.py", "--script", "tbsi_track",
                 "--config", config_name, "--save_dir", str(train_root), "--seed", "42",
                 "--use_lmdb", "0"]
    manifest["commands"].append(train_cmd)
    ckpt = train_root / "experiments" / config_name / "checkpoints" / "TBSITrack_ep0015.pth.tar"
    if not args.skip_train:
        manifest["stages"]["train"] = {"status": "RUNNING", "log": str(out / "train.stdout.log")}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["stages"]["train"]["seconds"] = run(train_cmd, env, out / "train.stdout.log")
        manifest["stages"]["train"]["status"] = "COMPLETE"
    if not ckpt.exists():
        raise RuntimeError("candidate checkpoint missing: {}".format(ckpt))

    test_cfg = config_name
    test_runid = "9102"
    result_dir = ROOT / "output/test/tracking_results/tbsi_track" / (config_name + "_" + test_runid)
    if result_dir.exists():
        raise RuntimeError("refusing existing candidate result dir: {}".format(result_dir))
    test_cmd = [sys.executable, "tracking/test.py", "tbsi_track", test_cfg,
                "--runid", test_runid, "--dataset_name", "lasher_test",
                "--threads", "0", "--num_gpus", "1"]
    manifest["commands"].append(test_cmd)
    env["TBSI_CHECKPOINT_OVERRIDE"] = str(ckpt)
    if not args.skip_test:
        manifest["stages"]["test"] = {"status": "RUNNING", "log": str(out / "test.stdout.log")}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["stages"]["test"]["seconds"] = run(test_cmd, env, out / "test.stdout.log")
        manifest["stages"]["test"]["status"] = "COMPLETE"
    if not result_dir.is_dir():
        raise RuntimeError("candidate tracking result dir missing: {}".format(result_dir))

    audit_dir = out / "official_audit"
    audit_cmd = [sys.executable, "tools/m1_official_metric_and_persistence_audit.py",
                 "--repo", str(ROOT), "--pred-dir", str(result_dir),
                 "--baseline-dir", str(M1_RESULTS), "--dataset", "lasher_test",
                 "--output-dir", str(audit_dir)]
    manifest["commands"].append(audit_cmd)
    manifest["stages"]["official_audit"] = {"status": "RUNNING", "log": str(out / "official_audit.stdout.log")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["stages"]["official_audit"]["seconds"] = run(audit_cmd, env, out / "official_audit.stdout.log")
    manifest["stages"]["official_audit"]["status"] = "COMPLETE"
    manifest["candidate_checkpoint"] = str(ckpt)
    manifest["candidate_checkpoint_sha256"] = sha256(ckpt)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
