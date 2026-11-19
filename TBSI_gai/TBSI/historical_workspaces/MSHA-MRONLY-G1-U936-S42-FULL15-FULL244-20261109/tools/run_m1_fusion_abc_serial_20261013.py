#!/usr/bin/env python3
"""Serial A/B/C fusion comparison on the fixed M1 failure/control cohort.

 A: protected M1 C0 (original fusion)
 B: G0_FFM_CDIAG (pointwise cross-modal residual)
 C: G0_FFM_CSPATIAL (spatial cross-modal residual)

The script deliberately evaluates every arm against the same explicit sequence
list. It never trains, overwrites an existing result directory, or changes the
protected M1 config/checkpoint. Training of B/C must already have produced the
explicit epoch-1 checkpoints referenced by their configs.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("TBSI_PYTHON", sys.executable)
M1_CONFIG = "TBSI_E_MHSA_FULL_4090D_seed42"
CONFIG_SHA = "fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e"
CHECKPOINT_SHA = "1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef"
ARMS = {
    # The tracker framework parses run_id as an epoch unless a checkpoint
    # override is present; numeric IDs keep both code paths valid.
    "A_C0": {"config": M1_CONFIG, "runid": 8801},
    "B_CDIAG": {"config": "G0_FFM_CDIAG_seed42", "runid": 8802},
    "C_CSPATIAL": {"config": "G0_FFM_CSPATIAL_seed42", "runid": 8803},
}
LOW_LIST = ROOT / "output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/matched_m1_full62/sequence_list.txt"
NORMAL_LIST = ROOT / "experiments/manifests/m1_normal20_control_20261012.txt"
UNION_LIST = ROOT / "experiments/manifests/m1_normal20_lowauc62_union_20261013.txt"
SCENE_CSV = ROOT / "output/experiments/M1_FULL244_SCENE_MINING_20261009/per_sequence_scene.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_names(path):
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def sequence_metrics(result_dir, sequence):
    pred = np.loadtxt(Path(result_dir) / (sequence + ".txt"), delimiter="\t", ndmin=2).astype(float)
    gt_path = ROOT / "data/lasher/testingset" / sequence / "init.txt"
    gt = np.loadtxt(gt_path, delimiter=",", ndmin=2).astype(float)
    n = min(len(pred), len(gt))
    overlaps = np.asarray([iou(pred[i], gt[i]) for i in range(n)], dtype=float)
    thresholds = np.linspace(0.0, 1.0, 101)
    auc = float(np.mean([(overlaps >= t).mean() for t in thresholds]) * 100.0)
    # LasHeR reports center and normalized precision using the same standard
    # xywh boxes; these values are supplementary to the official evaluator.
    pc = np.column_stack((pred[:n, 0] + pred[:n, 2] / 2, pred[:n, 1] + pred[:n, 3] / 2))
    gc = np.column_stack((gt[:n, 0] + gt[:n, 2] / 2, gt[:n, 1] + gt[:n, 3] / 2))
    center = np.linalg.norm(pc - gc, axis=1)
    diag = np.sqrt(gt[:n, 2] * gt[:n, 3])
    return {"sequence": sequence, "frames": int(n), "auc": auc,
            "precision": float((center <= 20).mean() * 100),
            "norm_precision": float((center / np.maximum(diag, 1e-9) <= .2).mean() * 100),
            "op50": float((overlaps >= .5).mean() * 100),
            "op75": float((overlaps >= .75).mean() * 100),
            "mean_iou": float(overlaps.mean()), "fail_iou_lt_05": int((overlaps < .5).sum())}


def summarize(rows, categories):
    out = {"sequences": len(rows)}
    for key in ("auc", "precision", "norm_precision", "op50", "op75", "mean_iou"):
        out["mean_" + key] = float(np.mean([r[key] for r in rows]))
    out["wins_vs_A"] = None
    out["category_counts"] = {k: sum(v == k for v in categories.values()) for k in set(categories.values())}
    return out


def run_arm(arm, sequence_list, output_root, dry_run, training_root):
    spec = ARMS[arm]
    cfg = ROOT / "experiments/tbsi_track" / (spec["config"] + ".yaml")
    if not cfg.is_file():
        raise RuntimeError("missing config: {}".format(cfg))
    result_dir = ROOT / "output/test/tracking_results/tbsi_track" / (spec["config"] + "_" + str(spec["runid"]))
    if result_dir.exists():
        raise RuntimeError("refusing to overwrite result directory: {}".format(result_dir))
    env = os.environ.copy()
    env.update({"TBSI_RPP_TEST_LIST": str(sequence_list), "TBSI_INFER_FP16": "0",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    alternate = training_root / "experiments" / spec["config"] / "checkpoints" / ("TBSITrack_ep%04d.pth.tar" % (15 if arm == "A_C0" else 1))
    if arm == "A_C0":
        env["TBSI_CHECKPOINT_OVERRIDE"] = str(ROOT / "output/experiments" / M1_CONFIG / "checkpoints/TBSITrack_ep0015.pth.tar")
    elif alternate.is_file():
        env["TBSI_CHECKPOINT_OVERRIDE"] = str(alternate)
    cmd = [PYTHON, "tracking/test.py", "tbsi_track", spec["config"],
           "--runid", str(spec["runid"]), "--dataset_name", "rpp_lasher_test",
           "--threads", "0", "--num_gpus", "1"]
    log = output_root / (arm + ".log")
    if dry_run:
        return {"command": cmd, "result_dir": str(result_dir), "status": "DRY_RUN"}
    started = time.time()
    with log.open("w") as stream:
        subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    if not result_dir.is_dir():
        raise RuntimeError("tracker completed without result directory: {}".format(result_dir))
    rows = [sequence_metrics(result_dir, s) for s in read_names(sequence_list)]
    return {"command": cmd, "result_dir": str(result_dir), "status": "COMPLETE",
            "seconds": time.time() - started, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sequence-list", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--training-root", default=None,
                    help="separate save_dir used for retrained B/C checkpoints")
    ap.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing non-empty campaign directory: {}".format(out))
    out.mkdir(parents=True, exist_ok=True)
    training_root = Path(args.training_root).resolve() if args.training_root else (out / "training")
    m1_cfg = ROOT / "experiments/tbsi_track" / (M1_CONFIG + ".yaml")
    m1_ckpt = ROOT / "output/experiments" / M1_CONFIG / "checkpoints/TBSITrack_ep0015.pth.tar"
    if sha256(m1_cfg) != CONFIG_SHA or sha256(m1_ckpt) != CHECKPOINT_SHA:
        raise RuntimeError("protected M1 config/checkpoint hash mismatch")
    low = read_names(LOW_LIST)
    normal = read_names(NORMAL_LIST)
    selected = read_names(args.sequence_list) if args.sequence_list else read_names(UNION_LIST)
    if len(selected) != len(set(selected)) or len(selected) != 82:
        raise RuntimeError("expected 82 unique union sequences, got {}".format(len(selected)))
    if not set(low).issubset(selected) or not set(normal).issubset(selected):
        raise RuntimeError("union manifest does not cover low-AUC and normal sets")
    categories = {s: ("LOW_AUC_FAILURE" if s in set(low) else "NORMAL_SENTINEL") for s in selected}
    missing = []
    for arm in args.arms:
        spec = ARMS[arm]
        cfg_path = ROOT / "experiments/tbsi_track" / (spec["config"] + ".yaml")
        # The A arm uses the protected epoch-15 checkpoint; B/C use the epoch
        # recorded in their independent configs.
        epoch = 15 if arm == "A_C0" else 1
        checkpoint = ROOT / "output/experiments" / spec["config"] / "checkpoints" / ("TBSITrack_ep%04d.pth.tar" % epoch)
        alternate = training_root / "experiments" / spec["config"] / "checkpoints" / ("TBSITrack_ep%04d.pth.tar" % epoch)
        if not checkpoint.is_file() and not alternate.is_file():
            missing.append(str(checkpoint) + " (or " + str(alternate) + ")")
    if missing and not args.dry_run:
        raise RuntimeError("missing explicit arm checkpoint(s); no arm was started:\n" + "\n".join(missing))
    seq_file = out / "sequence_list.txt"
    seq_file.write_text("\n".join(selected) + "\n")
    (out / "categories.json").write_text(json.dumps(categories, indent=2, sort_keys=True) + "\n")
    contract = {"id": "M1-FUSION-ABC-SERIAL-20261013", "parent": M1_CONFIG,
                "cohort_count": len(selected), "low_auc_count": len(low), "normal_count": len(normal),
                "low_list_sha256": sha256(LOW_LIST), "normal_list_sha256": sha256(NORMAL_LIST),
                "union_list_sha256": sha256(UNION_LIST), "config_sha256": CONFIG_SHA,
                "checkpoint_sha256": CHECKPOINT_SHA, "arms": args.arms, "serial": True}
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    results = {}
    for arm in args.arms:
        print("START", arm, flush=True)
        results[arm] = run_arm(arm, seq_file, out, args.dry_run, training_root)
        if results[arm].get("rows"):
            results[arm]["summary"] = summarize(results[arm]["rows"], categories)
            for cat in ("LOW_AUC_FAILURE", "NORMAL_SENTINEL"):
                cat_rows = [r for r in results[arm]["rows"] if categories[r["sequence"]] == cat]
                results[arm]["summary"][cat] = summarize(cat_rows, categories)
    # Paired deltas and a conservative automatic decision.
    if all(results[a].get("rows") for a in results):
        base = {r["sequence"]: r for r in results["A_C0"]["rows"]}
        for arm in ("B_CDIAG", "C_CSPATIAL"):
            for r in results[arm]["rows"]:
                r["delta_auc_vs_A"] = r["auc"] - base[r["sequence"]]["auc"]
        decisions = {}
        for arm in ("B_CDIAG", "C_CSPATIAL"):
            rows = results[arm]["rows"]
            delta = [r["delta_auc_vs_A"] for r in rows]
            low_delta = [r["delta_auc_vs_A"] for r in rows if categories[r["sequence"]] == "LOW_AUC_FAILURE"]
            normal_delta = [r["delta_auc_vs_A"] for r in rows if categories[r["sequence"]] == "NORMAL_SENTINEL"]
            decisions[arm] = {"mean_delta_auc": float(np.mean(delta)),
                              "low_failure_delta_auc": float(np.mean(low_delta)),
                              "normal_delta_auc": float(np.mean(normal_delta)),
                              "wins": int(sum(x > 0 for x in delta)), "losses": int(sum(x < 0 for x in delta)),
                              "decision": ("PROMOTE_TO_FULL_CONFIRMATION" if np.mean(low_delta) > .5 and
                                           np.mean(normal_delta) >= -.25 and sum(x > 0 for x in delta) > sum(x < 0 for x in delta)
                                           else "NO_PROMOTION")}
        results["decision"] = decisions
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"output": str(out), "arms": list(results), "decision": results.get("decision", "PENDING")}, indent=2))


if __name__ == "__main__":
    main()
