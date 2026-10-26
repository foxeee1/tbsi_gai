#!/usr/bin/env python3
"""Compare frozen M1 fused and single-modality box candidates at fixed states."""

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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAME = "TBSI_E_MHSA_FULL_4090D_seed42"
CONFIG = ROOT / "experiments/tbsi_track" / (NAME + ".yaml")
CHECKPOINT = ROOT / "output/experiments" / NAME / "checkpoints/TBSITrack_ep0015.pth.tar"
FEATURES = ROOT / "output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz"
CAPTURE = ROOT / "output/experiments/M1_BOUNDED_READOUT_PREDICTOR_20261014_retry1/capture_rows.csv"
SEQUENCE_LIST = ROOT / "output/experiments/predictor_sequence_list.txt"
TRACE_DIRS = [
    ROOT / "output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run/trace",
    ROOT / "output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank24/trace",
    ROOT / "output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank48/trace",
]
OUT_DEFAULT = ROOT / "output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015"
EXPECTED_CONFIG = "fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e"
EXPECTED_CHECKPOINT = "1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def decode(network, tracker, feature, state, image_shape, resize):
    score, _, size_map, offset_map = network.box_head(feature)
    response = tracker.output_window * score
    box = network.box_head.cal_bbox(response, size_map, offset_map).view(-1, 4).mean(0)
    box = box * (tracker.params.search_size / resize)
    cx = float(state[0]) + 0.5 * float(state[2])
    cy = float(state[1]) + 0.5 * float(state[3])
    half = 0.5 * float(tracker.params.search_size) / resize
    xywh = [float(box[0] + cx - half - 0.5 * box[2]),
            float(box[1] + cy - half - 0.5 * box[3]),
            float(box[2]), float(box[3])]
    from lib.utils.box_ops import clip_box
    return clip_box(xywh, image_shape[0], image_shape[1], margin=10)


def clustered_interval(rows, field, seed=3407, n_boot=10000):
    groups = {}
    for row in rows:
        groups.setdefault(row["sequence"], []).append(float(row[field]))
    names = sorted(groups)
    means = np.asarray([np.mean(groups[name]) for name in names], dtype=np.float64)
    rng = np.random.RandomState(seed)
    sample = rng.randint(0, len(means), size=(n_boot, len(means)))
    values = means[sample].mean(axis=1)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def summarize(rows):
    arms = ("m1_fused", "rgb_only", "tir_only", "oracle_best_of_three")
    result = {"states": len(rows), "sequences": len(set(r["sequence"] for r in rows)), "arms": {}}
    for arm in arms:
        values = np.asarray([float(row[arm + "_iou"]) for row in rows], dtype=np.float64)
        delta_key = arm + "_delta_vs_m1_fused"
        deltas = ([0.0] * len(rows) if arm == "m1_fused" else
                  [float(row[delta_key]) for row in rows])
        result["arms"][arm] = {
            "mean_iou_micro": float(values.mean()),
            "median_iou": float(np.median(values)),
            "mean_delta_vs_m1_fused": float(np.mean(deltas)),
            "median_delta_vs_m1_fused": float(np.median(deltas)),
            "positive_delta_fraction": float(np.mean(np.asarray(deltas) > 1e-12)),
            "negative_delta_fraction": float(np.mean(np.asarray(deltas) < -1e-12)),
            "zero_delta_fraction": float(np.mean(np.abs(deltas) <= 1e-12)),
            "sequence_cluster_bootstrap_95ci_delta": ([0.0, 0.0] if arm == "m1_fused" else
                                                        clustered_interval(rows, delta_key)),
        }
    rgb_wins = sum(float(r["rgb_only_iou"]) > float(r["m1_fused_iou"]) + 1e-12 for r in rows)
    tir_wins = sum(float(r["tir_only_iou"]) > float(r["m1_fused_iou"]) + 1e-12 for r in rows)
    result["candidate_availability"] = {
        "rgb_only_beats_fused_states": rgb_wins,
        "tir_only_beats_fused_states": tir_wins,
        "either_single_modality_beats_fused_states": sum(
            max(float(r["rgb_only_iou"]), float(r["tir_only_iou"])) > float(r["m1_fused_iou"]) + 1e-12
            for r in rows),
        "oracle_union_strictly_improves_fused_states": sum(
            float(r["oracle_best_of_three_iou"]) > float(r["m1_fused_iou"]) + 1e-12 for r in rows),
        "oracle_union_nonworse_fraction": float(np.mean([
            float(r["oracle_best_of_three_iou"]) >= float(r["m1_fused_iou"]) - 1e-12 for r in rows])),
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing nonempty output directory: {}".format(out))
    if sha256(CONFIG) != EXPECTED_CONFIG or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("protected M1 config/checkpoint hash mismatch")
    for path in (FEATURES, CAPTURE, SEQUENCE_LIST):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    out.mkdir(parents=True, exist_ok=True)

    os.environ["TBSI_INFER_FP16"] = "0"
    os.environ["TBSI_CHECKPOINT_OVERRIDE"] = str(CHECKPOINT)
    os.environ.pop("TBSI_BOUNDED_READOUT_PREDICTOR", None)
    os.environ.pop("TBSI_READOUT_POLICY", None)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from lib.test.evaluation.lasherdataset import LasHeRDataset
    from lib.test.parameter.tbsi_track import parameters
    from lib.test.tracker.tbsi_track import TBSITrack
    from lib.train.data.processing_utils import sample_target

    arrays = np.load(str(FEATURES), allow_pickle=False)
    sequences = arrays["sequence"].astype(str)
    frames = arrays["frame"].astype(int)
    keys = list(zip(sequences, frames))
    if len(keys) != 144 or len(set(sequences)) != 72:
        raise RuntimeError("unexpected frozen feature cohort")
    captures = {(r["sequence"], int(r["frame"])): r for r in read_csv(CAPTURE)}
    if set(keys) != set(captures):
        raise RuntimeError("feature and capture key sets differ")
    for name in ("rgb", "tir", "fused"):
        if arrays[name].shape != (144, 16, 16, 768):
            raise RuntimeError("unexpected {} feature shape {}".format(name, arrays[name].shape))

    params = parameters(NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, "m1_modality_candidate_availability")
    network = tracker.network.eval()
    dataset = LasHeRDataset("trainingset", str(SEQUENCE_LIST)).get_sequence_list()
    seqmap = {seq.name: seq for seq in dataset}
    if not set(sequences).issubset(set(seqmap)):
        raise RuntimeError("frozen state sequences missing from LasHeR train split")
    state_lookup = {}
    for trace_root in TRACE_DIRS:
        for trace_path in trace_root.glob("*/frames.jsonl"):
            for line in trace_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    key = (trace_path.parent.name, int(row["frame_id"]))
                    if key in set(keys):
                        if key in state_lookup:
                            raise RuntimeError("duplicate trace state {}:{}".format(*key))
                        state_lookup[key] = row
    if set(state_lookup) != set(keys):
        raise RuntimeError("fixed state traces incomplete: {}/{}".format(len(state_lookup), len(keys)))

    manifest = {
        "id": "M1-S1-MODALITY-CANDIDATE-AUDIT-20261015",
        "status": "RUNNING",
        "parent": "M1-S1-TRUE-ACTION-UTILITY-AUDIT-20261015",
        "question": "At identical fixed crop/state, do RGB-only or TIR-only candidates contain better target localization than the M1 fused candidate?",
        "hypothesis": "If the union of frozen single-modality candidates has a meaningful oracle IoU gain over M1 fused, candidate-level routing is worth testing; otherwise stop this route.",
        "intervention": "Only candidate feature source changes: cached M1 fused, shared M1 fusion projection with TIR zeroed, or shared M1 fusion projection with RGB zeroed. Same frozen M1 CENTER head, crop, state, GT, and decoder.",
        "dataset": "LasHeR train-side frozen 72-sequence / 144-state cohort; same states as prior action-direction audits",
        "features": str(FEATURES), "features_sha256": sha256(FEATURES),
        "capture": str(CAPTURE), "capture_sha256": sha256(CAPTURE),
        "sequence_list": str(SEQUENCE_LIST), "sequence_list_sha256": sha256(SEQUENCE_LIST),
        "trace_dirs": [str(p) for p in TRACE_DIRS],
        "config": str(CONFIG), "config_sha256": sha256(CONFIG),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "runner_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT.parents[1])).decode().strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "precision": "FP32", "processes": 1, "workers": 0},
        "metrics": "Per-state decoded IoU; paired delta to M1 fused; per-sequence means; sequence-cluster bootstrap CI. Best-of-three is an oracle upper bound, not a deployable method.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    rows = []
    fusion_parity = []
    started = time.time()
    with torch.inference_mode():
        for index, key in enumerate(keys):
            sequence, frame = key
            trace = state_lookup[key]
            state = np.asarray(trace["tracker_state_before"], dtype=np.float64)
            gt = np.asarray(trace["gt_box"], dtype=np.float64)
            if not trace.get("gt_valid", False):
                raise RuntimeError("invalid GT for {}:{}".format(sequence, frame))
            seq = seqmap[sequence]
            rgb_img = cv2.imread(seq.frames[0][frame], cv2.IMREAD_COLOR)
            tir_img = cv2.imread(seq.frames[1][frame], cv2.IMREAD_COLOR)
            if rgb_img is None or tir_img is None or rgb_img.shape != tir_img.shape:
                raise RuntimeError("invalid paired frame {}:{}".format(sequence, frame))
            image = np.concatenate((cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(tir_img, cv2.COLOR_BGR2RGB)), axis=2)
            _, resize, _ = sample_target(image, state, params.search_factor,
                                          output_sz=params.search_size)

            rgb = torch.as_tensor(arrays["rgb"][index].transpose(2, 0, 1)[None],
                                  dtype=torch.float32, device="cuda")
            tir = torch.as_tensor(arrays["tir"][index].transpose(2, 0, 1)[None],
                                  dtype=torch.float32, device="cuda")
            fused_saved = torch.as_tensor(arrays["fused"][index].transpose(2, 0, 1)[None],
                                           dtype=torch.float32, device="cuda")
            zero = torch.zeros_like(rgb)
            rgb_only = network.tbsi_fuse_search(torch.cat((rgb, zero), dim=1))
            tir_only = network.tbsi_fuse_search(torch.cat((zero, tir), dim=1))
            reconstructed = network.tbsi_fuse_search(torch.cat((rgb, tir), dim=1))
            fusion_parity.append(float((reconstructed - fused_saved).abs().max().item()))

            boxes = {
                "m1_fused": decode(network, tracker, fused_saved, state, image.shape, resize),
                "rgb_only": decode(network, tracker, rgb_only, state, image.shape, resize),
                "tir_only": decode(network, tracker, tir_only, state, image.shape, resize),
            }
            values = {name: iou(box, gt) for name, box in boxes.items()}
            oracle_iou = max(values.values())
            base_iou = values["m1_fused"]
            reference = float(captures[key]["base_iou"])
            if abs(base_iou - reference) > 1e-3:
                raise RuntimeError("M1 fused baseline parity failed {}:{} diff={}".format(
                    sequence, frame, abs(base_iou - reference)))
            row = {"sequence": sequence, "frame": frame,
                   "m1_fused_iou": base_iou,
                   "rgb_only_iou": values["rgb_only"],
                   "tir_only_iou": values["tir_only"],
                   "oracle_best_of_three_iou": oracle_iou,
                   "rgb_only_delta_vs_m1_fused": values["rgb_only"] - base_iou,
                   "tir_only_delta_vs_m1_fused": values["tir_only"] - base_iou,
                   "oracle_best_of_three_delta_vs_m1_fused": oracle_iou - base_iou,
                   "best_single_modality": ("rgb" if values["rgb_only"] > values["tir_only"] else
                                             "tir" if values["tir_only"] > values["rgb_only"] else "tie"),
                   "m1_fused_box": [float(x) for x in boxes["m1_fused"]],
                   "rgb_only_box": [float(x) for x in boxes["rgb_only"]],
                   "tir_only_box": [float(x) for x in boxes["tir_only"]],
                   "gt_box": [float(x) for x in gt]}
            for arm in ("m1_fused", "rgb_only", "tir_only"):
                for j, name in enumerate(("x", "y", "w", "h")):
                    row[arm + "_box_" + name] = float(boxes[arm][j])
            rows.append(row)
            if (index + 1) % 24 == 0:
                print("PROGRESS {}/{} elapsed={:.1f}s".format(index + 1, len(keys), time.time() - started),
                      flush=True)

    max_parity = max(fusion_parity)
    if max_parity > 0.02:
        raise RuntimeError("cached feature recomposition mismatch: {}".format(max_parity))
    columns = [k for k, v in rows[0].items() if not isinstance(v, list)]
    with (out / "per_state_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{k: v for k, v in row.items() if k in columns} for row in rows])

    metrics = summarize(rows)
    metrics["max_fusion_feature_recomposition_abs"] = max_parity
    metrics["max_m1_fused_iou_parity_abs"] = max(
        abs(float(r["m1_fused_iou"]) - float(captures[(r["sequence"], int(r["frame"]))]["base_iou"]))
        for r in rows)
    metrics["elapsed_seconds"] = time.time() - started
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row["sequence"], []).append(row)
    seq_rows = []
    for sequence, group in sorted(by_sequence.items()):
        item = {"sequence": sequence, "states": len(group)}
        for arm in ("m1_fused", "rgb_only", "tir_only", "oracle_best_of_three"):
            item[arm + "_mean_iou"] = float(np.mean([r[arm + "_iou"] for r in group]))
        for arm in ("rgb_only", "tir_only", "oracle_best_of_three"):
            item[arm + "_mean_delta_vs_m1_fused"] = float(np.mean([
                r[arm + "_delta_vs_m1_fused"] for r in group]))
        item["rgb_only_beats_fused_states"] = sum(r["rgb_only_iou"] > r["m1_fused_iou"] + 1e-12 for r in group)
        item["tir_only_beats_fused_states"] = sum(r["tir_only_iou"] > r["m1_fused_iou"] + 1e-12 for r in group)
        seq_rows.append(item)
    with (out / "per_sequence_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seq_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seq_rows)

    manifest.update({"status": "DIAGNOSTIC-COMPLETE", "state_count": len(rows),
                     "sequence_count": len(by_sequence), "elapsed_seconds": metrics["elapsed_seconds"],
                     "outputs": ["metrics.json", "per_state_candidates.csv", "per_sequence_candidates.csv"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
