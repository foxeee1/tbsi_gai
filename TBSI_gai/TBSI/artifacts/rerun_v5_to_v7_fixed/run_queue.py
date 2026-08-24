#!/usr/bin/env python3
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/root/autodl-tmp/TBSI_gai/TBSI")
RUN_ROOT = ROOT / "artifacts/rerun_v5_to_v7_fixed"
SUMMARY = RUN_ROOT / "summary.tsv"
BASELINE_AUC = 55.46

CONFIGS = [
    "v6.1.1-rtm-aligned-local-fixed",
    "v7.0.1-egir-only-fixed",
    "v7.1.2-rsm-v2-imbalance-fixed",
]


def run(cmd, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def test_epoch(config_name):
    text = (ROOT / f"experiments/tbsi_track/{config_name}.yaml").read_text(encoding="utf-8")
    in_test = False
    for line in text.splitlines():
        if line.startswith("TEST:"):
            in_test = True
            continue
        if in_test and line and not line.startswith(" ") and not line.startswith("#"):
            in_test = False
        if in_test and line.strip().startswith("EPOCH:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"TEST.EPOCH not found for {config_name}")


def extract_auc(eval_log):
    text = eval_log.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"\|\s*([0-9]+\.[0-9]+)\s*\|\s*[0-9]+\.[0-9]+\s*\|\s*[0-9]+\.[0-9]+\s*\|\s*[0-9]+\.[0-9]+\s*\|\s*[0-9]+\.[0-9]+\s*\|",
        text,
    )
    if not matches:
        raise RuntimeError(f"AUC not found in {eval_log}")
    return float(matches[-1])


def append_summary(config, auc, status):
    if not SUMMARY.exists():
        SUMMARY.write_text("config\tbaseline_auc\tauc\tdelta\tstatus\n", encoding="utf-8")
    current = SUMMARY.read_text(encoding="utf-8")
    if f"{config}\t" in current:
        return
    auc_text = "NA" if auc is None else f"{auc:.2f}"
    delta_text = "NA" if auc is None else f"{auc - BASELINE_AUC:+.2f}"
    with SUMMARY.open("a", encoding="utf-8") as f:
        f.write(f"{config}\t{BASELINE_AUC:.2f}\t{auc_text}\t{delta_text}\t{status}\n")


def check_space():
    usage = shutil.disk_usage(ROOT)
    if usage.free < 2.5 * 1024**3:
        raise RuntimeError(f"Available disk below 2.5GB: {usage.free / 1024**3:.2f}GB")


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        if SUMMARY.exists() and f"{config}\t" in SUMMARY.read_text(encoding="utf-8"):
            continue
        check_space()
        run_dir = RUN_ROOT / config
        run_dir.mkdir(parents=True, exist_ok=True)
        epoch = test_epoch(config)
        ckpt = ROOT / f"output/experiments/{config}/checkpoints/TBSITrack_ep{epoch:04d}.pth.tar"
        if not ckpt.exists():
            run(
                [
                    "python",
                    "-u",
                    "lib/train/run_training.py",
                    "--script",
                    "tbsi_track",
                    "--config",
                    config,
                    "--save_dir",
                    "./output",
                    "--use_lmdb",
                    "0",
                    "--use_wandb",
                    "0",
                    "--seed",
                    "42",
                ],
                run_dir / "train.log",
            )
        if not ckpt.exists():
            raise RuntimeError(f"Expected checkpoint missing after training: {ckpt}")

        check_space()
        env = dict(**__import__("os").environ)
        env["TBSI_STRICT_CHECKPOINT"] = "1"
        env["TBSI_TEST_CPU_THREADS"] = "2"
        run(
            [
                "python",
                "-u",
                "tracking/test.py",
                "tbsi_track",
                config,
                "--dataset_name",
                "lasher_test",
                "--threads",
                "4",
                "--num_gpus",
                "1",
            ],
            run_dir / "test_lasher.log",
            env=env,
        )
        run(
            [
                "python",
                "tracking/analysis_results.py",
                "--tracker_param",
                config,
                "--dataset_name",
                "lasher_test",
            ],
            run_dir / "eval_lasher.log",
        )
        auc = extract_auc(run_dir / "eval_lasher.log")
        append_summary(config, auc, "done")


if __name__ == "__main__":
    main()
