#!/usr/bin/env python3
"""Full15 + Full244 run for the final MR expected-score candidate.

The train/evaluate implementation is the already matched MR-only runner. This
wrapper only supplies a new run identity and an independent asset status file;
it never resumes a tracker checkpoint or optimizer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "MR-LR-EXPECTED-U936-S42-FULL15-FULL244-20261110"
RUN = ROOT / "output/experiments" / RUN_ID
CONFIG = ROOT / "experiments/tbsi_track" / (RUN_ID + ".yaml")
MANIFEST = ROOT / "experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json"
PUBLIC = ROOT / "pretrained_models/TBSITrack_SOT_Pretrained.pth.tar"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def resource_guard():
    query = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True).strip()
    if query:
        raise RuntimeError("another CUDA process is active: " + query)
    for line in subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines():
        if any(token in line for token in ("tracking/train.py", "tracking/test.py",
                                            "lib/train/run_training.py",
                                            "tracking/analysis_results.py")):
            raise RuntimeError("another train/test process is active: " + line.strip())


def record_status(status, conclusion, **extra):
    path = RUN / "run_status.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    prior.update({"run_id": RUN_ID, "status": status, "conclusion": conclusion,
                  "time_unix": time.time(), "config": str(CONFIG.resolve()),
                  "config_sha256": sha(CONFIG), "public_pretrained_sha256": sha(PUBLIC),
                  "manifest_sha256": sha(MANIFEST), **extra})
    write(path, prior)


def recorder(output, cfg, seed, arm, status, conclusion, checkpoint="NA", gpu_hours="NA"):
    record_status(status, conclusion, checkpoint=checkpoint, gpu_hours=gpu_hours,
                  seed=seed, arm=arm, runner_config=str(Path(cfg).resolve()))


def snapshot_sources():
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    with (RUN / "source_code_snapshot.tar.gz").open("wb") as stream:
        subprocess.run(["tar", "--exclude=__pycache__", "--exclude=*.pyc", "-czf", "-",
                        "lib", "tracking", "tools/run_dual_message_p1.py",
                        "tools/record_dual_message_p1.py", "tools/run_mr_lr_expected_full_20261110.py",
                        "experiments/tbsi_track", "experiments/early_gate_closed_loop_20261026/p1_sot_v3"],
                       cwd=str(ROOT), stdout=stream, check=True)
    (RUN / "source_code_snapshot.sha256").write_text(
        sha(RUN / "source_code_snapshot.tar.gz") + "  source_code_snapshot.tar.gz\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if not CONFIG.is_file() or not PUBLIC.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("candidate config, public checkpoint or manifest is missing")
    if sha(PUBLIC) != "1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b":
        raise RuntimeError("public SOT checkpoint hash mismatch")
    if RUN.exists() and any(RUN.iterdir()):
        raise RuntimeError("refusing to reuse non-empty run directory: " + str(RUN))
    resource_guard()
    RUN.mkdir(parents=True)
    snapshot_sources()
    write(RUN / "run_contract.json", {
        "run_id": RUN_ID, "status": "PREPARING", "parent": "MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109",
        "arm": "MR_LR_EXPECTED", "seed": 42,
        "method_delta": "only MODEL.BACKBONE.MASS_REDIST_MODE lr_joint -> lr_expected; all other MR-only runtime fields held",
        "config": str(CONFIG.resolve()), "config_sha256": sha(CONFIG),
        "public_pretrained": str(PUBLIC.resolve()), "public_pretrained_sha256": sha(PUBLIC),
        "train_manifest": str(MANIFEST.resolve()), "train_manifest_sha256": sha(MANIFEST),
        "train_list": str(MANIFEST.parent / "train_sequences.txt"),
        "train_list_sha256": sha(MANIFEST.parent / "train_sequences.txt"),
        "test_list": str(MANIFEST.parent / "test_sequences.txt"),
        "test_list_sha256": sha(MANIFEST.parent / "test_sequences.txt"),
        "training": {"epochs": 15, "successful_updates": 7020, "batch": 16, "grad_accum": 8,
                      "lr": 1e-4, "weight_decay": 1e-4, "scheduler": "Mstep gamma=.1 milestone=[2], step every 468 updates",
                      "workers": 4, "prefetch_factor": 2, "persistent_workers": False,
                      "AMP": True, "optimizer": "fresh AdamW", "INIT_CHECKPOINT": ""},
        "evaluation": {"dataset": "LasHeR official testing set", "sequences": 244,
                        "precision": "FP32", "tracker_threads": 0, "template_update": False},
        "baseline_metrics": {"auc": 57.6510066068563, "precision": 71.41411613307486,
                              "norm_precision": 67.87794180346087},
        "source_snapshot_sha256": sha(RUN / "source_code_snapshot.tar.gz")})
    record_status("PREPARING / CONTRACT-LOCKED", "No training or utility result yet.")
    sys.path.insert(0, str(ROOT / "tools"))
    from run_dual_message_p1 import SimpleNamespace, evaluate, train
    if args.mode in ("train", "all"):
        train_args = SimpleNamespace(config=str(CONFIG), output=str(RUN / "train"), seed=42,
                                     arm="MR_LR_EXPECTED", manifest=str(MANIFEST), updates=7020,
                                     milestones="936,7020", diagnostic_only=False,
                                     shared_init_snapshot=None, export_shared_init=None,
                                     expected_shared_state_sha256=None)
        record_status("RUNNING / FULL15-TRAIN", "Training launched from public SOT; no utility result yet.")
        train(train_args, recorder=recorder)
        record_status("FULL15-TRAINING-COMPLETE / FULL244-PENDING",
                      "7020 successful updates; checkpoint ready; Full244 pending.",
                      checkpoint=str(RUN / "train/checkpoint_u7020.pth.tar"),
                      checkpoint_sha256=sha(RUN / "train/checkpoint_u7020.pth.tar"))
    if args.mode in ("evaluate", "all"):
        resource_guard()
        checkpoint = RUN / "train/checkpoint_u7020.pth.tar"
        if not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        eval_args = SimpleNamespace(config=str(CONFIG), output=str(RUN / "full244"), seed=42,
                                    checkpoint=str(checkpoint), list=None)
        record_status("RUNNING / FULL244-EVALUATION", "FP32 threads0 Full244 evaluation in progress.")
        evaluate(eval_args)
        record_status("FULL15-FULL244-COMPLETE / ANALYSIS-PENDING",
                      "Training and Full244 prediction generation complete; paired analysis pending.",
                      checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))


if __name__ == "__main__":
    main()
