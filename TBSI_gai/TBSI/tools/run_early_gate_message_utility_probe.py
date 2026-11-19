#!/usr/bin/env python3
"""Fixed-state source/pair message utility audit for Early gate.

The script replays a fixed prefix with the protected Early gate checkpoint,
then forks from the identical tracker state at a predeclared target frame.
Interventions mask only the cross-modal attention entries in all active TBSI
fusion layers.  No weights are trained or modified.
"""

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
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TBSI_INFER_FP16"] = "0"
os.environ["TBSI_DISABLE_TRACKER_CACHE"] = "1"

import cv2
import numpy as np
import torch

from lib.models.layers.message_utility_probe import (
    CrossAttentionUtilityProbe, ProbeMode, module_contract,
)


CHECKPOINT = Path(os.environ.get(
    "EG_PROBE_CHECKPOINT",
    str(ROOT / "output/主实验/实验/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023/checkpoints/TBSITrack_ep0015.pth.tar")))
CONFIG_NAME = "M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023"
CONFIG = Path(os.environ.get(
    "EG_PROBE_CONFIG",
    str(ROOT / ("experiments/主实验/配置/%s.yaml" % CONFIG_NAME))))
SAVED_RESULTS = Path(os.environ.get(
    "EG_PROBE_SAVED_RESULTS",
    str(ROOT / ("output/主实验/测试结果/tracking_results/tbsi_track/%s" % CONFIG_NAME))))
CASES = [
    {"sequence": "10runone", "frame": 51, "role": "failure-onset"},
    {"sequence": "umbrellawillopen", "frame": 272, "role": "positive-control"},
    {"sequence": "darkcarturn", "frame": 60, "role": "normal-control"},
    {"sequence": "boyruninsnow", "frame": 100, "role": "normal-control"},
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        return hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode() +
                              value.numpy().tobytes()).hexdigest()
    if isinstance(value, (list, tuple)):
        return hashlib.sha256("".join(tensor_sha(item) for item in value).encode()).hexdigest()
    if isinstance(value, dict):
        return hashlib.sha256("".join(str(key) + tensor_sha(value[key])
                                      for key in sorted(value)).encode()).hexdigest()
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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_rgbt(sequence, frame_index):
    rgb = cv2.imread(sequence.frames[0][frame_index], cv2.IMREAD_COLOR)
    tir = cv2.imread(sequence.frames[1][frame_index], cv2.IMREAD_COLOR)
    if rgb is None or tir is None:
        raise RuntimeError("decode failed: %s frame %d" % (sequence.name, frame_index))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)
    return np.concatenate((rgb, tir), axis=2)


def bbox_iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ix = max(0., min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0., min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-12)


def center_error(a, b):
    acx, acy = float(a[0]) + float(a[2]) / 2., float(a[1]) + float(a[3]) / 2.
    bcx, bcy = float(b[0]) + float(b[2]) / 2., float(b[1]) + float(b[3]) / 2.
    return float(np.hypot(acx - bcx, acy - bcy))


def peak_pairs(records, direction, k=2):
    """Select receiver/source pairs from captured baseline attention mass."""
    values = {}
    for record in records:
        for row in record.get("attention_rows", []):
            if row["direction"] != direction:
                continue
            key = (int(row["receiver"]), int(row["source"]))
            values[key] = values.get(key, 0.) + float(row["value"])
    return [key for key, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)[:k]]


def source_tokens(records, direction, k=2):
    values = {}
    for record in records:
        for row in record.get("attention_rows", []):
            if row["direction"] != direction:
                continue
            source = int(row["source"])
            values[source] = values.get(source, 0.) + float(row["value"])
    return [source for source, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)[:k]]


def make_modes(baseline_records):
    modes = [ProbeMode("off"), ProbeMode("cross_all")]
    for direction in ("rt", "tr"):
        modes.append(ProbeMode("source_all", direction=direction))
        for source in source_tokens(baseline_records, direction, k=2):
            modes.append(ProbeMode("source_token", direction=direction, source=source))
        for receiver, source in peak_pairs(baseline_records, direction, k=2):
            modes.append(ProbeMode("pair", direction=direction,
                                   receiver=receiver, source=source))
    return modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT /
                        "experiments/early_gate_closed_loop_20261026/message_utility_probe_20261028"))
    parser.add_argument("--case", default="all", choices=["all"] + [c["sequence"] for c in CASES])
    parser.add_argument("--fixed-state-from-saved", action="store_true",
                        help="skip prefix replay and use saved target-1 box as a fixed exploratory state")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing nonempty output: " + str(output))
    if not CHECKPOINT.exists() or not CONFIG.exists():
        raise RuntimeError("protected checkpoint/config missing")
    output.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    start = time.time()

    from lib.test.evaluation import get_dataset, Tracker
    from lib.test.utils import TrackerParams
    from lib.config.tbsi_track.config import cfg, update_config_from_file
    os.environ["TBSI_CHECKPOINT_OVERRIDE"] = str(CHECKPOINT)
    dataset = {sequence.name: sequence for sequence in get_dataset("lasher_test")}
    wrapper = Tracker("tbsi_track", CONFIG_NAME, "lasher_test")
    # The protected M1 YAML intentionally lives outside the normal mutable
    # tracker-config directory.  Load it explicitly instead of copying or
    # changing that asset.
    update_config_from_file(str(CONFIG))
    params = TrackerParams()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = str(CHECKPOINT)
    params.save_all_boxes = False
    if Path(params.checkpoint).resolve() != CHECKPOINT.resolve():
        raise RuntimeError("unexpected checkpoint: %s" % params.checkpoint)
    params.debug = 0
    tracker = wrapper.create_tracker(params)
    fusion_modules = []
    for layer in getattr(tracker.network.backbone, "tbsi_layers", []):
        fusion = getattr(layer, "v2_template_fusion", None)
        if fusion is not None and hasattr(fusion, "attn_drop"):
            fusion_modules.append(fusion.attn_drop)
    if not fusion_modules:
        raise RuntimeError("no TBSI fusion modules with attn_drop found")
    probe = CrossAttentionUtilityProbe(fusion_modules, n_rgb=64)
    contract = module_contract(fusion_modules)
    selected_cases = CASES if args.case == "all" else [c for c in CASES if c["sequence"] == args.case]
    manifest = {
        "id": "EG-EARLY-GATE-MESSAGE-UTILITY-PROBE-20261028",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
        "config": str(CONFIG), "config_sha256": sha(CONFIG),
        "cases": selected_cases, "precision": "FP32", "threads": 0,
        "template_update": False, "fusion_contract": contract,
        "intervention": "post-softmax cross attention entry removal; no renormalization",
        "command": sys.argv, "start_time": start,
    }
    write_json(output / "manifest.json", manifest)

    results = []
    try:
        for case in selected_cases:
            sequence = dataset[case["sequence"]]
            target = int(case["frame"])
            saved_path = SAVED_RESULTS / (sequence.name + ".txt")
            saved = np.loadtxt(saved_path, ndmin=2)
            tracker.initialize(read_rgbt(sequence, 0), sequence.init_info())
            prefix_contract = "REPLAYED-SERIALIZED-BOX-PARITY"
            if args.fixed_state_from_saved:
                if target <= 0 or target >= len(saved):
                    raise RuntimeError("saved result has no target-1 state for %s@%d" %
                                       (sequence.name, target))
                tracker.state = saved[target - 1].astype(np.float64).tolist()
                tracker.frame_id = target - 1
                prefix_contract = "UNAVAILABLE / EXPLORATORY-SAVED-TARGET-MINUS-ONE-STATE"
            else:
                for frame_num in range(1, target):
                    info = sequence.frame_info(frame_num)
                    info.update(sequence_name=sequence.name, frame_num=frame_num,
                                gt_bbox=sequence.ground_truth_rect[frame_num])
                    pred = np.asarray(tracker.track(read_rgbt(sequence, frame_num), info)["target_bbox"], dtype=np.float64).reshape(4)
                    # Historical M1 result files store truncated integer boxes
                    # while tracker state is kept in float32.
                    if frame_num >= len(saved) or not np.array_equal(pred.astype(np.int64),
                                                                      saved[frame_num].astype(np.int64)):
                        expected = saved[frame_num].reshape(4).tolist() if frame_num < len(saved) else None
                        raise RuntimeError("prefix parity failed at %s frame %d: predicted=%s saved=%s" %
                                           (sequence.name, frame_num, pred.tolist(), expected))

            anchor = {"state": clone_state(tracker.state), "frame_id": tracker.frame_id,
                      "prev_search": clone_state(tracker.prev_search),
                      "prev_temporal_tokens": clone_state(tracker.prev_temporal_tokens)}
            anchor_hash = tensor_sha(anchor["state"])

            # Baseline target forward captures every cross attention entry.  This
            # determines the top tokens/pairs before any intervention is read.
            baseline_capture = {"output": None}
            original_forward = tracker.network.forward

            def capture_forward(*forward_args, **kwargs):
                result = original_forward(*forward_args, **kwargs)
                baseline_capture["output"] = result
                return result

            tracker.network.forward = capture_forward
            probe.set_mode(ProbeMode("off"))
            probe.capture = True
            info = sequence.frame_info(target)
            info.update(sequence_name=sequence.name, frame_num=target,
                        gt_bbox=sequence.ground_truth_rect[target])
            tracker.track(read_rgbt(sequence, target), info)
            # The runtime hook captured the actual post-softmax entries.  Use
            # those entries to choose source and pair interventions before
            # reading any intervention result.
            baseline_records = list(probe.records)
            if not baseline_records:
                raise RuntimeError("baseline attention hook did not capture records")
            modes = make_modes(baseline_records)
            baseline_output = baseline_capture["output"]
            baseline_probe_records = list(probe.records)
            # Restore the exact prefix anchor before every branch and keep the
            # same image and frame metadata.
            case_result = {"case": case, "prefix_contract": prefix_contract,
                           "anchor_state_sha256": anchor_hash,
                           "baseline_hook_records": baseline_probe_records,
                           "baseline_attention_records": baseline_records,
                           "branches": []}
            gt = np.asarray(sequence.ground_truth_rect[target], dtype=np.float64)
            for mode in modes:
                tracker.state = clone_state(anchor["state"])
                tracker.frame_id = anchor["frame_id"]
                tracker.prev_search = clone_state(anchor["prev_search"])
                tracker.prev_temporal_tokens = clone_state(anchor["prev_temporal_tokens"])
                probe.set_mode(mode)
                probe.capture = True
                probe.records = []
                out = tracker.track(read_rgbt(sequence, target), info)
                pred = np.asarray(out["target_bbox"], dtype=np.float64).reshape(4)
                branch = {
                    "mode": mode.__dict__, "input_state_sha256": tensor_sha(anchor["state"]),
                    "prediction": pred.tolist(), "gt": gt.tolist(),
                    "iou": bbox_iou(pred, gt), "center_error_pixels": center_error(pred, gt),
                    "hook_records": list(probe.records),
                }
                case_result["branches"].append(branch)
            tracker.network.forward = original_forward
            results.append(case_result)
            write_json(output / "partial_results.json", {"completed_cases": len(results), "cases": results})
            print(json.dumps({"case": sequence.name, "modes": len(modes), "paired": True}), flush=True)
    except Exception as error:
        write_json(output / "failed.json", {
            "status": "EXECUTION-FAILED / NO MECHANISM CONCLUSION",
            "error_type": type(error).__name__, "error": str(error),
            "completed_cases": len(results), "elapsed_seconds": time.time() - start,
        })
        raise
    finally:
        probe.close()

    completed = {"status": "P2-COMPLETE / FIXED-STATE-SOURCE-PAIR-UTILITY",
                 "cases": results, "elapsed_seconds": time.time() - start}
    write_json(output / "completed.json", completed)
    print(json.dumps({"status": completed["status"], "cases": len(results),
                      "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
