#!/usr/bin/env python3
"""Incrementally register the B0 update-936 scheduler-only run."""
import argparse
import csv
from pathlib import Path

ROOT = Path("/root/autodl-tmp/TBSI_gai/TBSI")
CSV_PATH = ROOT / "experiments/experiment_asset_registry.csv"
RUN_ID = "B0-U936-LR-ONLY-S42-RETRY2-20261107"
C_ID = "EG-C-MSHA-MASS-REDIST-STEP-MATCHED-S42-20261106"
FIELDS = [
    "ID", "Date", "Parent", "CommitOrSnapshot", "Config", "Seed", "Dataset",
    "EvalScope", "AUC", "PR", "NPR", "OP50", "OP75", "FPS", "GPU",
    "HardwareConfig", "GPUHours", "Checkpoint", "AssumptionOrHypothesis",
    "Conclusion", "Status", "Evidence", "Notes",
]


def upsert(rows, asset_id, values):
    for row in rows:
        if row["ID"] == asset_id:
            row.update(values)
            return
    row = {key: "NA" for key in FIELDS}
    row.update(values)
    row["ID"] = asset_id
    rows.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="RUNNING / FULL15-THEN-FULL244")
    parser.add_argument("--auc", default="NA")
    parser.add_argument("--pr", default="NA")
    parser.add_argument("--npr", default="NA")
    parser.add_argument("--op50", default="NA")
    parser.add_argument("--op75", default="NA")
    parser.add_argument("--fps", default="NA")
    parser.add_argument("--gpuhours", default="NA")
    parser.add_argument("--checkpoint", default="NA (training in progress)")
    parser.add_argument("--conclusion", default="Training started; full-244 result pending.")
    args = parser.parse_args()

    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    c_row = next((row for row in rows if row["ID"] == C_ID), None)
    if c_row is None:
        raise SystemExit("C+Mass-Redist StepLR asset row is missing")
    c_row["Checkpoint"] = (
        "DELETED BY USER 2026-11-07; prior path retained in report; SHA256 "
        "8d6f91f9db79e0fc3a37b83da9d85c98a3e2b0456ea86a998bf9ef66a81fdfe8"
    )
    c_row["Status"] += " / CHECKPOINT-CLEANED-USER-REQUESTED"
    c_row["Notes"] += (
        " Epoch15 checkpoint deleted by user request on 2026-11-07 to release space; "
        "full244 box/timing outputs and logs retained; original checkpoint SHA256 "
        "8d6f91f9db79e0fc3a37b83da9d85c98a3e2b0456ea86a998bf9ef66a81fdfe8."
    )

    base = {
        "Date": "20261107",
        "Parent": "Matched B0 / StepLR seed42 EG-B0-V101-STEP-MATCHED-S42-20261106",
        "CommitOrSnapshot": str(ROOT / "output/experiments/B0_U936_LR_ONLY_S42_RETRY2_20261107/source_snapshot.sha256") +
                            " SHA256 b3fd589406fc175ff0743be210138e1840ed253b6aeff864dbe57fd8bfe1b751; "
                            "training entrypoint SHA256 7a892caffffdf80cd7f3907641f9adc268ab9828d07e52a51ed35f2b8bf8ddb2",
        "Config": str(ROOT / "experiments/tbsi_track/B0_U936_LR_ONLY_S42_RETRY2_20261107.yaml") +
                  " SHA256 230e718bb7f8725c69611e12c7d3e6e39d20f0e80d46df45bc71a08507997440",
        "Seed": "42",
        "Dataset": "LasHeR train977; p1_sot_v3 manifest SHA256 5258569430eab4acef1d50656309ee51c634c7a15881a99b40b876d4452a0469",
        "EvalScope": "15 epochs / planned 7020 updates; LasHeR full244; FP32; tracker threads0; template update off",
        "AUC": args.auc, "PR": args.pr, "NPR": args.npr, "OP50": args.op50,
        "OP75": args.op75, "FPS": args.fps,
        "GPU": "1x NVIDIA GeForce RTX 4090 D 24 GiB",
        "HardwareConfig": "Python3.8.20/PyTorch1.9.0+cu111/CUDA11.1; AMP; B16 accum8; worker4 prefetch2; persistent_workers=False; MultiStepLR gamma0.1 at epoch2 (after u936); serial task",
        "GPUHours": args.gpuhours,
        "Checkpoint": args.checkpoint,
        "AssumptionOrHypothesis": "Estimate B0 sensitivity to LR decay timing while holding the matched StepLR B0 configuration and all non-scheduler factors fixed.",
        "Conclusion": args.conclusion,
        "Status": args.status,
        "Evidence": str(ROOT / "output/experiments/B0_U936_LR_ONLY_S42_RETRY2_20261107") +
                    "; report experiments/reports/b0_u936_lr_only_seed42_20261107.md",
        "Notes": "Only config difference is TRAIN.SCHEDULER: step epoch10 -> Mstep milestone epoch2, gamma0.1. Public SOT init SHA256 1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b; fresh AdamW; exact source workspace from matched StepLR pair. Input-batch hash comparison against matched B0 required. Two pre-training setup failures are separately retained as output/experiments/B0_U936_LR_ONLY_S42_20261107 and ...RETRY1_20261107; zero optimizer updates.",
    }
    upsert(rows, RUN_ID, base)

    failed_rows = [
        {
            "ID": "B0-U936-LR-ONLY-S42-PREFLIGHT-FAIL-20261107",
            "Date": "20261107",
            "Parent": "Matched B0 / StepLR seed42",
            "CommitOrSnapshot": "Local preflight script; no training source executed",
            "Config": str(ROOT / "experiments/tbsi_track/B0_U936_LR_ONLY_S42_20261107.yaml"),
            "Seed": "42",
            "Dataset": "LasHeR manifest not loaded",
            "EvalScope": "Preflight only; 0 samples; 0 updates",
            "AUC": "NA", "PR": "NA", "NPR": "NA", "OP50": "NA", "OP75": "NA", "FPS": "NA",
            "GPU": "NA", "HardwareConfig": "CPU-only config validation", "GPUHours": "0",
            "Checkpoint": "None",
            "AssumptionOrHypothesis": "Verify scheduler-only config contract.",
            "Conclusion": "Preflight validator incorrectly expected runtime milestone in YAML; no model execution.",
            "Status": "FAILED-PREFLIGHT / ZERO-TRAINING / RETAINED",
            "Evidence": str(ROOT / "output/experiments/B0_U936_LR_ONLY_S42_20261107"),
            "Notes": "Retry uses new identity B0-U936-LR-ONLY-S42-RETRY2-20261107.",
        },
        {
            "ID": "B0-U936-LR-ONLY-S42-ENTRYPOINT-FAIL-20261107",
            "Date": "20261107",
            "Parent": "Matched B0 / StepLR seed42",
            "CommitOrSnapshot": "Local runner wrapper; no training source executed",
            "Config": str(ROOT / "experiments/tbsi_track/B0_U936_LR_ONLY_S42_RETRY1_20261107.yaml"),
            "Seed": "42",
            "Dataset": "LasHeR manifest not loaded",
            "EvalScope": "Entrypoint import only; 0 samples; 0 updates",
            "AUC": "NA", "PR": "NA", "NPR": "NA", "OP50": "NA", "OP75": "NA", "FPS": "NA",
            "GPU": "NA", "HardwareConfig": "Python3.8.20/PyTorch1.9.0+cu111; failure before GPU use", "GPUHours": "0",
            "Checkpoint": "None",
            "AssumptionOrHypothesis": "Execute the standard runner with runtime scheduler milestone injection.",
            "Conclusion": "Wrapper omitted lib/train from sys.path; standard entrypoint import failed before training.",
            "Status": "FAILED-ENTRYPOINT / ZERO-TRAINING / RETAINED",
            "Evidence": str(ROOT / "output/experiments/B0_U936_LR_ONLY_S42_RETRY1_20261107"),
            "Notes": "Retry adds workspace/lib/train to sys.path and uses new identity B0-U936-LR-ONLY-S42-RETRY2-20261107.",
        },
    ]
    for failed in failed_rows:
        upsert(rows, failed["ID"], failed)

    temp = CSV_PATH.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(CSV_PATH)
    print("updated", RUN_ID, "and cleanup/failure provenance rows")


if __name__ == "__main__":
    main()
