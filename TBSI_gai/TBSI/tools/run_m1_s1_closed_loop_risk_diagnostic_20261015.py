#!/usr/bin/env python3
"""Paired train-side closed-loop rollout for S1-state vs M1-state risk signals."""

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAME = "TBSI_E_MHSA_FULL_4090D_seed42"
CHECKPOINT = ROOT / "output/experiments" / NAME / "checkpoints/TBSITrack_ep0015.pth.tar"
CONFIG = ROOT / "experiments/tbsi_track" / (NAME + ".yaml")
PREDICTOR = ROOT / "experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json"
POLICY = ROOT / "experiments/artifacts/m1_readout_independent_20261014.json"
SOURCE_LIST = ROOT / "output/experiments/predictor_sequence_list.txt"
OUT = ROOT / "output/experiments/M1_S1_CLOSED_LOOP_RISK_DIAGNOSTIC_20261015"
EXPECTED = {
    "config": "fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e",
    "checkpoint": "1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef",
    "predictor": "b97bf2fe4d576a447f6cabe62a7523a2276f6837ffcc8761714b181119d79c6c",
    "policy": "ff4cbef73341eae20c726911f916b25cd80b6a61458bba0a1e5196004840c098",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def read_pair(paths):
    visible = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    infrared = cv2.imread(paths[1], cv2.IMREAD_COLOR)
    if visible is None or infrared is None:
        raise RuntimeError("failed to decode paired frame: {} / {}".format(*paths))
    if visible.shape[:2] != infrared.shape[:2]:
        raise RuntimeError("RGB/TIR frame shape mismatch")
    return np.concatenate([visible, infrared], axis=2)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def corr(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def auc_binary(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    ranks = rankdata(scores)
    npos = int(labels.sum())
    return float((ranks[labels].sum() - npos * (npos + 1) / 2.0) /
                 (npos * (len(labels) - npos)))


def summarize(rows, horizon, score_field):
    usable = [r for r in rows if r["future_h{}".format(horizon)] is not None]
    score = np.asarray([r[score_field] for r in usable], dtype=np.float64)
    delta = np.asarray([r["future_h{}".format(horizon)] for r in usable], dtype=np.float64)
    harm = delta > 0.0
    cutoff = np.quantile(score, 0.75) if len(score) else 0.0
    top = score >= cutoff
    by_sequence = {}
    for row in usable:
        by_sequence.setdefault(row["sequence"], []).append(row)
    heldout = []
    for name, group in sorted(by_sequence.items()):
        s = [x[score_field] for x in group]
        d = [x["future_h{}".format(horizon)] for x in group]
        heldout.append({"sequence": name, "frames": len(group), "spearman": corr(rankdata(s), rankdata(d)),
                        "harm_rate": float(np.mean(np.asarray(d) > 0.0)),
                        "top_quartile_harm_rate": float(np.mean(np.asarray(d)[np.asarray(s) >= np.quantile(s, .75)] > 0.0))})
    return {
        "horizon": horizon, "score": score_field, "frames": len(usable),
        "positive_harm_fraction": float(harm.mean()) if len(harm) else None,
        "spearman_pooled": corr(rankdata(score), rankdata(delta)),
        "auroc_harm": auc_binary(harm, score),
        "top_quartile_harm_rate": float(harm[top].mean()) if top.any() else None,
        "top_quartile_enrichment": (float(harm[top].mean() / harm.mean())
                                      if top.any() and harm.mean() > 0 else None),
        "sequence_heldout": heldout,
        "sequence_mean_spearman": float(np.nanmean([x["spearman"] for x in heldout
                                                     if x["spearman"] is not None]))
                                    if any(x["spearman"] is not None for x in heldout) else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=12)
    args = parser.parse_args()
    if args.sequences < 4:
        raise RuntimeError("at least four whole sequences are required")
    for key, path in (("config", CONFIG), ("checkpoint", CHECKPOINT),
                      ("predictor", PREDICTOR), ("policy", POLICY)):
        if sha(path) != EXPECTED[key]:
            raise RuntimeError("{} hash mismatch: {}".format(key, path))
    if OUT.exists():
        raise RuntimeError("refusing to overwrite output: {}".format(OUT))
    names = [line.strip() for line in SOURCE_LIST.read_text().splitlines() if line.strip()]
    if len(names) != 72 or len(set(names)) != 72:
        raise RuntimeError("expected the frozen 72-sequence train-side source manifest")
    names = sorted(names, key=lambda n: hashlib.sha256(n.encode("utf-8")).hexdigest())[:args.sequences]
    OUT.mkdir(parents=True)
    manifest_path = OUT / "sequence_manifest.txt"
    manifest_path.write_text("\n".join(names) + "\n")
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["TBSI_INFER_FP16"] = "0"
    os.environ["TBSI_CHECKPOINT_OVERRIDE"] = str(CHECKPOINT)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TBSI_READOUT_STATE_MODE"] = "coupled"
    os.environ.pop("TBSI_STATE_RISK_POLICY", None)
    os.environ.pop("TBSI_READOUT_AUDIT_DIR", None)

    from lib.test.evaluation.lasherdataset import LasHeRDataset
    from lib.test.parameter.tbsi_track import parameters
    from lib.test.tracker.tbsi_track import TBSITrack
    dataset = LasHeRDataset("trainingset", str(manifest_path)).get_sequence_list()
    if [s.name for s in dataset] != names:
        raise RuntimeError("resolved training sequence order differs from frozen manifest")

    os.environ.pop("TBSI_BOUNDED_READOUT_PREDICTOR", None)
    os.environ.pop("TBSI_READOUT_POLICY", None)
    m1_params = parameters(NAME)
    m1_params.checkpoint = str(CHECKPOINT)
    m1_params.debug = 0
    m1 = TBSITrack(m1_params, "m1_s1_closed_loop_m1_state")
    os.environ["TBSI_BOUNDED_READOUT_PREDICTOR"] = str(PREDICTOR)
    os.environ["TBSI_READOUT_POLICY"] = str(POLICY)
    s1_params = parameters(NAME)
    s1_params.checkpoint = str(CHECKPOINT)
    s1_params.debug = 0
    s1 = TBSITrack(s1_params, "m1_s1_closed_loop_s1_state")
    m1.network.eval()
    s1.network.eval()

    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
    manifest = {
        "id": "M1-S1-CLOSED-LOOP-RISK-DIAGNOSTIC-20261015", "parent": "M1-S1-SELECTIVE-STATE-COMMIT-FULL244-20261015",
        "hypothesis": "Observable divergence between S1-state and M1-state predicts future localization harm from S1 state feedback.",
        "one_changed_variable": "closed-loop internal state path; same frozen M1 model, frames, initialization, FP32 and S1 readout policy",
        "dataset": "LasHeR trainingset; first N sequence names after SHA256 lexical ordering from the frozen 72-sequence train-side manifest",
        "dataset_path": dataset[0].frames[0][0].split("/visible/")[0] if dataset else "NA",
        "sequence_manifest": str(manifest_path), "sequence_manifest_sha256": sha(manifest_path),
        "sequences": names, "sequence_count": len(names), "sample_unit": "whole sequence",
        "label": "mean future IoU(M1-state)-IoU(S1-state), next 10/30 tracked frames; positive is S1-state harm",
        "scores": ["normalized center displacement", "log-size displacement", "joint L2 displacement"],
        "training": "none; no threshold or model fit; exploratory train-side mechanism diagnostic",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
        "config": str(CONFIG), "config_sha256": sha(CONFIG), "predictor": str(PREDICTOR), "predictor_sha256": sha(PREDICTOR),
        "policy": str(POLICY), "policy_sha256": sha(POLICY), "runner_sha256": sha(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT.parents[1])).decode().strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "gpu": gpu, "precision": "FP32",
                        "tracker_processes": 1, "workers": 0},
        "frames": 0, "completed_sequences": [], "status": "RUNNING",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    all_rows = []
    started = time.time()
    row_path = OUT / "paired_rollout.jsonl"
    for si, seq in enumerate(dataset, 1):
        if len(seq.frames[0]) != len(seq.frames[1]) or len(seq.frames[0]) != len(seq.ground_truth_rect):
            raise RuntimeError("frame/annotation count mismatch in {}".format(seq.name))
        frame0 = read_pair([seq.frames[0][0], seq.frames[1][0]])
        init = seq.init_info()
        m1.initialize(frame0, init)
        s1.initialize(frame0, init)
        local = []
        for frame_idx in range(1, len(seq.frames[0])):
            image = read_pair([seq.frames[0][frame_idx], seq.frames[1][frame_idx]])
            state_m = [float(x) for x in m1.state]
            state_s = [float(x) for x in s1.state]
            scale = max(math.sqrt(max(state_m[2] * state_m[3], 1.0)), 1.0)
            dx = ((state_s[0] + state_s[2] / 2.0) - (state_m[0] + state_m[2] / 2.0)) / scale
            dy = ((state_s[1] + state_s[3] / 2.0) - (state_m[1] + state_m[3] / 2.0)) / scale
            lw = math.log(max(state_s[2], 1e-6) / max(state_m[2], 1e-6))
            lh = math.log(max(state_s[3], 1e-6) / max(state_m[3], 1e-6))
            out_m = m1.track(image, {"sequence_name": seq.name, "frame_num": frame_idx})["target_bbox"]
            out_s = s1.track(image, {"sequence_name": seq.name, "frame_num": frame_idx})["target_bbox"]
            gt = np.asarray(seq.ground_truth_rect[frame_idx], dtype=np.float64).tolist()
            local.append({"sequence": seq.name, "frame": frame_idx, "score_center": math.sqrt(dx * dx + dy * dy),
                          "score_size": math.sqrt(lw * lw + lh * lh),
                          "score_joint": math.sqrt(dx * dx + dy * dy + lw * lw + lh * lh),
                          "state_gap_x": dx, "state_gap_y": dy, "state_gap_logw": lw, "state_gap_logh": lh,
                          "iou_m1_state": iou(out_m, gt), "iou_s1_state": iou(out_s, gt),
                          "instant_harm": iou(out_m, gt) - iou(out_s, gt),
                          "future_h10": None, "future_h30": None})
        for idx, row in enumerate(local):
            for horizon in (10, 30):
                end = idx + horizon + 1
                if end <= len(local):
                    row["future_h{}".format(horizon)] = float(np.mean([
                        x["iou_m1_state"] - x["iou_s1_state"] for x in local[idx + 1:end]]))
        all_rows.extend(local)
        with row_path.open("a") as stream:
            for row in local:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        manifest["frames"] = len(all_rows)
        manifest["completed_sequences"].append(seq.name)
        manifest["elapsed_seconds"] = time.time() - started
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print("ROLLOUT {}/{} {} frames={} elapsed={:.1f}s".format(
            si, len(dataset), seq.name, len(local), time.time() - started), flush=True)

    columns = ["sequence", "frame", "score_center", "score_size", "score_joint", "state_gap_x", "state_gap_y",
               "state_gap_logw", "state_gap_logh", "iou_m1_state", "iou_s1_state", "instant_harm", "future_h10", "future_h30"]
    with (OUT / "paired_rollout.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    manifest["frames"] = len(all_rows)
    manifest["elapsed_seconds"] = time.time() - started
    manifest["status"] = "ROLLOUT-COMPLETE"
    manifest["outputs"] = ["paired_rollout.csv", "paired_rollout.jsonl", "metrics.json"]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    metrics = {"status": "complete", "sequences": len(names), "frames": len(all_rows), "horizons": {}}
    for horizon in (10, 30):
        metrics["horizons"][str(horizon)] = {
            score: summarize(all_rows, horizon, score)
            for score in ("score_center", "score_size", "score_joint")}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
