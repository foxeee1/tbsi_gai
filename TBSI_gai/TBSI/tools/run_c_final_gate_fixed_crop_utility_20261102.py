#!/usr/bin/env python3
"""Paired fixed-crop test of the trained C gate versus unconditional g=1."""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["TBSI_INFER_FP16"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBSI_DISABLE_TRACKER_CACHE"] = "1"

import cv2
import numpy as np
import torch

from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack


DEFAULT_CHECKPOINT = ROOT / "output/experiments/EG_C_EARLY_LR_DROP_U936_FULL15_FULL244_S42_20261101/C/train/checkpoint_u7020.pth.tar"
DEFAULT_CONFIG = "EG_C_EARLY_LR_DROP_U936_SOT_S42_20261101"
DEFAULT_CASES = ROOT / "experiments/early_gate_closed_loop_20261026/c_final_gate_utility_20261102/cases.json"
DEFAULT_OUTPUT = ROOT / "output/experiments/EG_C_FINAL_GATE_FIXED_CROP_UTILITY_20261102"
DEFAULT_SAVED_PREDICTIONS = Path("/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/EG_C_EARLY_LR_DROP_U936_FULL15_FULL244_S42_20261101/C/full244")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(value):
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(str((arr.dtype, arr.shape)).encode() + arr.tobytes()).hexdigest()


def hash_state(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        return hashlib.sha256(str((tensor.dtype, tuple(tensor.shape))).encode() + tensor.numpy().tobytes()).hexdigest()
    return hashlib.sha256(repr(value).encode()).hexdigest()


def clone_state(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    return copy.deepcopy(value)


def read_frame(sequence, frame_index):
    rgb = cv2.imread(sequence.frames[0][frame_index], cv2.IMREAD_COLOR)
    thermal = cv2.imread(sequence.frames[1][frame_index], cv2.IMREAD_COLOR)
    if rgb is None or thermal is None:
        raise RuntimeError("frame decode failed: %s[%d]" % (sequence.name, frame_index))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
    return np.concatenate((rgb, thermal), axis=2)


def box_iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / max(aw * ah + bw * bh - inter, 1e-12)


def center_error(a, b):
    ac = np.asarray([a[0] + a[2] / 2.0, a[1] + a[3] / 2.0])
    bc = np.asarray([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0])
    return float(np.linalg.norm(ac - bc))


def gate_summary(fusions):
    summaries = []
    for layer_id, fusion in fusions:
        values = fusion.last_gate_log_gain
        if values is None:
            raise RuntimeError("gate utility capture missing at layer %d" % layer_id)
        values = values.detach().float().cpu()
        summaries.append({
            "layer_id": int(layer_id),
            "log_gain_mean_by_source": values.mean(dim=(0, 1)).tolist(),
            "log_gain_std_by_source": values.std(dim=(0, 1), unbiased=False).tolist(),
            "log_gain_min_by_source": values.amin(dim=(0, 1)).tolist(),
            "log_gain_max_by_source": values.amax(dim=(0, 1)).tolist(),
            "fraction_suppressed_by_source": (values < 0).float().mean(dim=(0, 1)).tolist(),
            "fraction_boosted_by_source": (values > 0).float().mean(dim=(0, 1)).tolist(),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--saved-predictions", default=str(DEFAULT_SAVED_PREDICTIONS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    cases_path = Path(args.cases).resolve()
    output = Path(args.output).resolve()
    saved_predictions = Path(args.saved_predictions).resolve()
    if not str(ROOT.resolve()).startswith("/root/autodl-tmp/"):
        raise RuntimeError("project is outside the authorized data disk")
    if "/autodl-fs/" in str(Path(__file__).resolve()) or "/autodl-fs/" in str(output):
        raise RuntimeError("execution/output on /autodl-fs is forbidden")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing nonempty output directory: " + str(output))
    if not checkpoint.is_file() or not cases_path.is_file():
        raise RuntimeError("checkpoint or case manifest is missing")
    cases_doc = json.loads(cases_path.read_text())
    cases = cases_doc["cases"]
    if len(cases) != 8 or any(int(case["frame"]) != 50 for case in cases):
        raise RuntimeError("unexpected/unlocked case manifest")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["TBSI_CHECKPOINT_OVERRIDE"] = str(checkpoint)

    dataset = {seq.name: seq for seq in LasHeRDataset(split="testingset").get_sequence_list()}
    params = parameters(args.config)
    params.debug = 0
    params.save_all_boxes = False
    if Path(params.checkpoint).resolve() != checkpoint:
        raise RuntimeError("tracker checkpoint differs from explicit checkpoint")
    tracker = TBSITrack(params, "lasher_test")
    if getattr(params.cfg.MODEL, "TEMPORAL_TOKENS", False):
        raise RuntimeError("saved-box state replay requires temporal tokens disabled")
    fusions = []
    for layer_id, layer in enumerate(getattr(tracker.network.backbone, "tbsi_layers", [])):
        fusion = getattr(layer, "v2_template_fusion", None)
        if fusion is not None and getattr(fusion, "early_conditioned_cross_gate", False):
            fusion.capture_gate_utility = True
            fusions.append((layer_id, fusion))
    if not fusions:
        raise RuntimeError("native Early gate fusion layers were not found")

    import lib.test.tracker.tbsi_track as tracker_module
    original_sample_target = tracker_module.sample_target
    crop_hashes = []

    def capture_sample_target(*sample_args, **sample_kwargs):
        result = original_sample_target(*sample_args, **sample_kwargs)
        crop_hashes.append(hash_array(result[0]))
        return result

    tracker_module.sample_target = capture_sample_target
    start = time.time()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": cases_doc["id"], "selection": cases_doc["selection"], "cases": cases,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "config_name": args.config,
        "config_path": str(ROOT / "experiments/tbsi_track" / (args.config + ".yaml")),
        "config_sha256": sha256_file(ROOT / "experiments/tbsi_track" / (args.config + ".yaml")),
        "cases_path": str(cases_path), "cases_sha256": sha256_file(cases_path),
        "source_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "fusion": sha256_file(ROOT / "lib/models/layers/tbsi_layer.py"),
            "search_context": sha256_file(ROOT / "lib/models/layers/early_gate_search_context.py"),
            "tracker": sha256_file(ROOT / "lib/test/tracker/tbsi_track.py"),
            "backbone": sha256_file(ROOT / "lib/models/tbsi_track/vit_tbsi_care.py"),
        },
        "precision": "FP32", "tracker_threads": 0, "template_update": False,
        "interventions": ["native trained Early gate", "gate logits forced to zero => g=1"],
        "g1_override": "zero the native evidence logits and any enabled search-context residual logits before the shared tanh/exp action",
        "state_anchor": "saved C full244 prediction at target frame minus one; temporal tokens and template updates are disabled",
        "gt_usage": "post-hoc scoring only; never passed to gate/model",
        "training": False, "full_test": False, "command": sys.argv,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_count": torch.cuda.device_count(), "start_time": start,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    rows = []
    try:
        for case_idx, case in enumerate(cases, 1):
            seq = dataset.get(case["sequence"])
            if seq is None:
                raise RuntimeError("sequence not found: " + case["sequence"])
            target = int(case["frame"])
            if target >= len(seq.frames[0]) or target >= len(seq.ground_truth_rect):
                raise RuntimeError("target frame outside sequence: " + seq.name)
            result_file = saved_predictions / (seq.name + ".txt")
            if not result_file.is_file():
                raise RuntimeError("saved C prediction missing: " + str(result_file))
            saved = np.loadtxt(result_file, ndmin=2)
            tracker.initialize(read_frame(seq, 0), {"init_bbox": seq.ground_truth_rect[0]})

            if target <= 0 or target >= len(saved):
                raise RuntimeError("saved result does not contain state for %s@%d" % (seq.name, target))
            # C has no temporal-token or online-template state. Its only
            # inter-frame input is the previous predicted box. Restore that
            # state directly, avoiding a free-running prefix mismatch.
            tracker.state = saved[target - 1].astype(np.float64).tolist()
            tracker.frame_id = target - 1
            tracker.prev_search = None
            tracker.prev_temporal_tokens = None

            anchor = {
                "state": clone_state(tracker.state), "frame_id": tracker.frame_id,
                "prev_search": clone_state(tracker.prev_search),
                "prev_temporal_tokens": clone_state(tracker.prev_temporal_tokens),
            }
            anchor_hash = hash_state(anchor["state"])
            frame = read_frame(seq, target)
            frame_hash = hash_array(frame)
            info = {"sequence_name": seq.name, "frame_num": target}
            branches = []
            native_prediction = None
            for mode in ("native", "g1"):
                tracker.state = clone_state(anchor["state"])
                tracker.frame_id = anchor["frame_id"]
                tracker.prev_search = clone_state(anchor["prev_search"])
                tracker.prev_temporal_tokens = clone_state(anchor["prev_temporal_tokens"])
                crop_hashes.clear()
                originals = []
                if mode == "g1":
                    for _, fusion in fusions:
                        originals.append((fusion.early_gate, fusion.early_gate.forward))
                        fusion.early_gate.forward = lambda evidence: evidence.new_zeros(
                            (evidence.shape[0], evidence.shape[1], 2))
                        if getattr(fusion, "early_gate_search_context_mode", "off") != "off":
                            context_branch = fusion.search_context_gain
                            originals.append((context_branch, context_branch.forward))
                            context_branch.forward = lambda evidence, early_context, condition_context=None: evidence.new_zeros(
                                (evidence.shape[0], evidence.shape[1], 2))
                try:
                    pred = np.asarray(tracker.track(frame, info)["target_bbox"], dtype=np.float64).reshape(4)
                finally:
                    for gate, forward in originals:
                        gate.forward = forward
                if len(crop_hashes) != 1:
                    raise RuntimeError("expected one captured target crop, got %d" % len(crop_hashes))
                if mode == "native":
                    native_prediction = pred
                if mode == "native" and target < len(saved) and not np.array_equal(pred.astype(np.int64), saved[target].astype(np.int64)):
                    raise RuntimeError("native target parity failed at %s frame %d" % (seq.name, target))
                branches.append({
                    "mode": mode, "input_state_sha256": anchor_hash,
                    "frame_sha256": frame_hash, "search_crop_sha256": crop_hashes[0],
                    "gate_log_gain": gate_summary(fusions),
                    "prediction": pred.tolist(),
                    "gt": np.asarray(seq.ground_truth_rect[target], dtype=np.float64).tolist(),
                    "iou": box_iou(pred, seq.ground_truth_rect[target]),
                    "center_error_pixels": center_error(pred, seq.ground_truth_rect[target]),
                })
            if branches[0]["search_crop_sha256"] != branches[1]["search_crop_sha256"]:
                raise RuntimeError("paired crop mismatch: " + seq.name)
            if branches[0]["input_state_sha256"] != branches[1]["input_state_sha256"]:
                raise RuntimeError("paired state mismatch: " + seq.name)
            rows.append({"case": case, "anchor_state_sha256": anchor_hash, "branches": branches})
            (output / "partial_results.json").write_text(json.dumps(
                {"completed_cases": len(rows), "rows": rows}, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"case": case["sequence"], "frame": target,
                              "native_iou": branches[0]["iou"], "g1_iou": branches[1]["iou"],
                              "crop_match": True, "state_match": True}), flush=True)
    finally:
        tracker_module.sample_target = original_sample_target

    completed = {
        "status": "PAIRED-FIXED-CROP-DIAGNOSTIC-COMPLETE",
        "cases": rows, "elapsed_seconds": time.time() - start,
        "interpretation_limit": "Eight predeclared tail sequences, one fixed frame each; diagnostic only, no population or seed-level claim.",
    }
    (output / "completed.json").write_text(json.dumps(completed, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": completed["status"], "output": str(output),
                      "cases": len(rows), "elapsed_seconds": completed["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
