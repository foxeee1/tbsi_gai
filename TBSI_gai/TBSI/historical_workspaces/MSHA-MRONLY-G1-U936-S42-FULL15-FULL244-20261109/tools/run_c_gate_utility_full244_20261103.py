#!/usr/bin/env python3
"""Full244 fixed-state probe of the trained C gate against identity gain."""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/root/autodl-tmp/EG_C_SNAPSHOT_20261102/TBSI_gai/TBSI").resolve()
sys.path.insert(0, str(CODE_ROOT))

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


CHECKPOINT = PROJECT_ROOT / "output/experiments/EG_C_EARLY_LR_DROP_U936_FULL15_FULL244_S42_20261101/C/train/checkpoint_u7020.pth.tar"
CONFIG_NAME = "EG_C_EARLY_LR_DROP_U936_SOT_S42_20261101"
CONFIG_PATH = CODE_ROOT / "experiments/tbsi_track/EG_C_EARLY_LR_DROP_U936_SOT_S42_20261101.yaml"
SOURCE_MANIFEST = PROJECT_ROOT / "output/experiments/EG_C_EARLY_LR_DROP_U936_FULL15_FULL244_S42_20261101/C/source_at_launch/manifest.json"
PER_SEQUENCE = PROJECT_ROOT / "output/experiments/EG_C_EARLY_LR_DROP_U936_FULL15_FULL244_S42_20261101/analysis/m1_comparison_20261101/per_sequence_244.csv"
SAVED_PREDICTIONS = PROJECT_ROOT / "output/主实验/测试结果/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42"
OUTPUT = PROJECT_ROOT / "output/experiments/EG_C_GATE_UTILITY_FIXEDSTATE_FULL244_RETRY1_20261103"


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
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(str((arr.dtype, arr.shape)).encode() + arr.tobytes()).hexdigest()


def read_frame(sequence, frame_index):
    rgb = cv2.imread(sequence.frames[0][frame_index], cv2.IMREAD_COLOR)
    thermal = cv2.imread(sequence.frames[1][frame_index], cv2.IMREAD_COLOR)
    if rgb is None or thermal is None:
        raise RuntimeError("frame decode failed: %s[%d]" % (sequence.name, frame_index))
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)), axis=2)


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
    rows = []
    for layer_id, fusion in fusions:
        values = fusion.last_gate_log_gain
        if values is None:
            raise RuntimeError("gate utility capture missing at layer %d" % layer_id)
        values = values.detach().float().cpu()
        rows.append({
            "layer_id": int(layer_id),
            "mean_by_source": values.mean(dim=(0, 1)).tolist(),
            "std_by_source": values.std(dim=(0, 1), unbiased=False).tolist(),
            "min_by_source": values.amin(dim=(0, 1)).tolist(),
            "max_by_source": values.amax(dim=(0, 1)).tolist(),
        })
    return rows


def main():
    if not str(PROJECT_ROOT).startswith("/root/autodl-tmp/TBSI_gai/TBSI"):
        raise RuntimeError("runner must live in the authorized project tree")
    if "/autodl-fs/" in str(Path(__file__).resolve()) or "/autodl-fs/" in str(OUTPUT):
        raise RuntimeError("execution/output on /autodl-fs is forbidden")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("refusing nonempty output directory: " + str(OUTPUT))
    if not CHECKPOINT.is_file() or not PER_SEQUENCE.is_file() or not SOURCE_MANIFEST.is_file():
        raise RuntimeError("required checkpoint or registered full244 inputs are missing")
    expected_checkpoint = "8742d0cd456084e6bd2cb39e8524d1fa5e5b33c9cf7f1eba72ce3bb4cb29148e"
    if sha256_file(CHECKPOINT) != expected_checkpoint:
        raise RuntimeError("explicit C checkpoint hash differs from registered candidate")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    expected_sources = {
        "fusion": "8dc8697f984202ca511dc98a5d956ed01d4da32faad02970801a89c97447f286",
        "search_context": "14ab5adadf2103db4812af16729017a756e2e1786c060ac4904c0d9decafbf1c",
        "tracker": "4c850f04e7f70a72c5e5b62d1db3bb5c8aa7ebbff459837fdb6dae8e16e10aa0",
        "backbone": "b1ee54cb84d5b6891e1afbd3f8629725adad4ada29d0a318ec68ded249d64ce4",
    }
    source_paths = {
        "fusion": CODE_ROOT / "lib/models/layers/tbsi_layer.py",
        "search_context": CODE_ROOT / "lib/models/layers/early_gate_search_context.py",
        "tracker": CODE_ROOT / "lib/test/tracker/tbsi_track.py",
        "backbone": CODE_ROOT / "lib/models/tbsi_track/vit_tbsi_care.py",
    }
    observed_sources = {key: sha256_file(path) for key, path in source_paths.items()}
    if observed_sources != expected_sources:
        raise RuntimeError("C source snapshot hashes do not match the registered run")
    config_hash = "159ff0e31ea1e55dab5b3ac229e7472b050ea98ab257b3da581feb1c77a44f8d"
    if sha256_file(CONFIG_PATH) != config_hash:
        raise RuntimeError("C config hash does not match the recorded fixed-crop probe")

    names = []
    with PER_SEQUENCE.open(newline="") as stream:
        sequence_records = list(csv.DictReader(stream))
    names = [row["sequence"] for row in sequence_records]
    drift_names = [row["sequence"] for row in sequence_records
                   if row["known_m1_prediction_drift"].lower() == "true"]
    if len(names) != 244 or len(set(names)) != 244:
        raise RuntimeError("expected exactly 244 unique registered test sequences")

    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    os.environ["TBSI_CHECKPOINT_OVERRIDE"] = str(CHECKPOINT)

    dataset = {seq.name: seq for seq in LasHeRDataset(split="testingset").get_sequence_list()}
    if set(names) != set(dataset):
        raise RuntimeError("registered 244 names do not equal test dataset sequence names")
    params = parameters(CONFIG_NAME)
    params.debug = 0
    params.save_all_boxes = False
    if Path(params.checkpoint).resolve() != CHECKPOINT.resolve():
        raise RuntimeError("tracker checkpoint differs from the explicit registered checkpoint")
    tracker = TBSITrack(params, "lasher_test")
    if getattr(params.cfg.MODEL, "TEMPORAL_TOKENS", False):
        raise RuntimeError("saved-box fixed-state replay requires temporal tokens disabled")
    fusions = []
    for layer_id, layer in enumerate(getattr(tracker.network.backbone, "tbsi_layers", [])):
        fusion = getattr(layer, "v2_template_fusion", None)
        if fusion is not None and getattr(fusion, "early_conditioned_cross_gate", False):
            fusion.capture_gate_utility = True
            fusions.append((layer_id, fusion))
    if len(fusions) != 3:
        raise RuntimeError("expected three C Early-gate fusion layers")

    import lib.test.tracker.tbsi_track as tracker_module
    original_sample_target = tracker_module.sample_target
    crop_hashes = []

    def capture_sample_target(*args, **kwargs):
        result = original_sample_target(*args, **kwargs)
        crop_hashes.append(hash_array(result[0]))
        return result

    tracker_module.sample_target = capture_sample_target
    cases = []
    for name in names:
        seq = dataset[name]
        n = min(len(seq.frames[0]), len(seq.frames[1]), len(seq.ground_truth_rect))
        if n < 2:
            raise RuntimeError("sequence too short for fixed-state probe: " + name)
        targets = sorted(set(max(1, min(n - 1, int(round((n - 1) * q)))) for q in (0.25, 0.50, 0.75)))
        cases.extend({"sequence": name, "frame": int(target), "anchor_fraction": float(target / (n - 1))}
                     for target in targets)

    start = time.time()
    OUTPUT.mkdir(parents=True)
    manifest = {
        "id": "EG-C-GATE-UTILITY-FIXEDSTATE-FULL244-20261103",
        "scope": "244 sequences; 3 predetermined anchor fractions per sequence (25/50/75%); fixed protected-M1 saved state; same-crop C native A/A vs g=1; no training",
        "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "sequence_count": len(names), "case_count": len(cases), "sequences": names,
        "m1_prediction_known_provenance_drift_sequences": drift_names,
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "config": str(CONFIG_PATH), "config_sha256": sha256_file(CONFIG_PATH),
        "source_snapshot": str(CODE_ROOT), "source_hashes": observed_sources,
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "full244_metrics_source": str(PER_SEQUENCE), "full244_metrics_sha256": sha256_file(PER_SEQUENCE),
        "state_anchor": "protected M1 saved prediction at t-1; C model starts from same first-frame template; temporal tokens/template updates disabled",
        "arms": ["native C gate A/A replay 1", "native C gate A/A replay 2", "g=1 with evidence and search-context logits zeroed"],
        "interpretation_boundary": "The C prediction files were removed during prior artifact cleanup. These are M1-trajectory anchor states, not C free-running states. Seven M1 anchor sequence files carry known provenance-drift flags and remain included with flags.",
        "gt_usage": "post-hoc scoring only; not passed to model or gate",
        "precision": "FP32", "tracker_threads": 0, "torch_threads": 1,
        "training": False, "full_tracking_evaluation": False,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "runner_sha256": sha256_file(Path(__file__).resolve()), "start_time": start,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    saved_by_sequence = {}
    rows = []
    try:
        for case_index, case in enumerate(cases, 1):
            seq = dataset[case["sequence"]]
            target = case["frame"]
            if seq.name not in saved_by_sequence:
                pred_file = SAVED_PREDICTIONS / (seq.name + ".txt")
                if not pred_file.is_file():
                    raise RuntimeError("saved C prediction missing: " + str(pred_file))
                saved_by_sequence[seq.name] = np.loadtxt(pred_file, ndmin=2)
                tracker.initialize(read_frame(seq, 0), {"init_bbox": seq.ground_truth_rect[0]})
            saved = saved_by_sequence[seq.name]
            if target >= len(saved):
                raise RuntimeError("saved prediction lacks %s@%d" % (seq.name, target))
            anchor_state = saved[target - 1].astype(np.float64).tolist()
            anchor_hash = hash_state(anchor_state)
            frame = read_frame(seq, target)
            frame_hash = hash_array(frame)
            info = {"sequence_name": seq.name, "frame_num": target}
            branches = []
            for mode in ("native_a", "native_b", "g1"):
                tracker.state = list(anchor_state)
                tracker.frame_id = target - 1
                tracker.prev_search = None
                tracker.prev_temporal_tokens = None
                crop_hashes.clear()
                originals = []
                if mode == "g1":
                    for _, fusion in fusions:
                        originals.append((fusion.early_gate, fusion.early_gate.forward))
                        fusion.early_gate.forward = lambda evidence: evidence.new_zeros((evidence.shape[0], evidence.shape[1], 2))
                        if getattr(fusion, "early_gate_search_context_mode", "off") != "off":
                            branch = fusion.search_context_gain
                            originals.append((branch, branch.forward))
                            branch.forward = lambda evidence, early_context, condition_context=None: evidence.new_zeros((evidence.shape[0], evidence.shape[1], 2))
                try:
                    prediction = np.asarray(tracker.track(frame, info)["target_bbox"], dtype=np.float64).reshape(4)
                finally:
                    for module, forward in originals:
                        module.forward = forward
                if len(crop_hashes) != 1:
                    raise RuntimeError("expected exactly one search crop for %s@%d" % (seq.name, target))
                gt = np.asarray(seq.ground_truth_rect[target], dtype=np.float64)
                branches.append({
                    "mode": mode, "state_sha256": anchor_hash, "frame_sha256": frame_hash,
                    "crop_sha256": crop_hashes[0], "prediction": prediction.tolist(),
                    "iou": box_iou(prediction, gt), "center_error_pixels": center_error(prediction, gt),
                    "gate_log_gain": gate_summary(fusions),
                })
            if len({branch["crop_sha256"] for branch in branches}) != 1:
                raise RuntimeError("paired search crop mismatch for %s@%d" % (seq.name, target))
            if len({branch["state_sha256"] for branch in branches}) != 1:
                raise RuntimeError("paired tracker state mismatch for %s@%d" % (seq.name, target))
            if not np.array_equal(np.asarray(branches[0]["prediction"]), np.asarray(branches[1]["prediction"])):
                raise RuntimeError("native A/A exact replay failed for %s@%d" % (seq.name, target))
            rows.append({
                "case": case, "native_a": branches[0], "native_b": branches[1], "identity": branches[2],
                "delta_iou_native_minus_identity": branches[0]["iou"] - branches[2]["iou"],
                "delta_center_error_identity_minus_native": branches[2]["center_error_pixels"] - branches[0]["center_error_pixels"],
                "state_crop_parity": True,
                "native_aa_exact": True,
            })
            if case_index % 60 == 0:
                (OUTPUT / "progress.json").write_text(json.dumps({
                    "completed_cases": case_index, "total_cases": len(cases),
                    "completed_sequences": len({row["case"]["sequence"] for row in rows}),
                    "elapsed_seconds": time.time() - start,
                }, indent=2) + "\n")
    finally:
        tracker_module.sample_target = original_sample_target

    with (OUTPUT / "per_case.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row["case"]["sequence"], []).append(row)
    sequence_rows = []
    for name in names:
        seq_rows = by_sequence[name]
        sequence_rows.append({
            "sequence": name, "frames": [r["case"]["frame"] for r in seq_rows],
            "mean_delta_iou": float(np.mean([r["delta_iou_native_minus_identity"] for r in seq_rows])),
            "mean_delta_center_error_identity_minus_native": float(np.mean([r["delta_center_error_identity_minus_native"] for r in seq_rows])),
            "native_iou_mean": float(np.mean([r["native_a"]["iou"] for r in seq_rows])),
            "identity_iou_mean": float(np.mean([r["identity"]["iou"] for r in seq_rows])),
        })
    deltas = np.asarray([r["delta_iou_native_minus_identity"] for r in rows], dtype=np.float64)
    seq_deltas = np.asarray([r["mean_delta_iou"] for r in sequence_rows], dtype=np.float64)
    summary = {
        "status": "FULL244_FIXEDSTATE_GATE_UTILITY_COMPLETE",
        "sequence_count": len(sequence_rows), "case_count": len(rows),
        "all_sequences_present": len(sequence_rows) == 244,
        "all_state_crop_parity": all(r["state_crop_parity"] for r in rows),
        "all_native_aa_exact": all(r["native_aa_exact"] for r in rows),
        "native_saved_box_parity_count": len(rows),
        "frame_level_delta_iou_mean": float(deltas.mean()),
        "frame_level_delta_iou_median": float(np.median(deltas)),
        "frame_level_delta_iou_abs_mean": float(np.abs(deltas).mean()),
        "frame_native_better": int((deltas > 0).sum()),
        "frame_identity_better": int((deltas < 0).sum()),
        "frame_ties": int((deltas == 0).sum()),
        "sequence_equal_weight_delta_iou_mean": float(seq_deltas.mean()),
        "sequence_delta_iou_median": float(np.median(seq_deltas)),
        "sequence_native_better": int((seq_deltas > 0).sum()),
        "sequence_identity_better": int((seq_deltas < 0).sum()),
        "sequence_ties": int((seq_deltas == 0).sum()),
        "sequences_delta_gt_0_01": int((seq_deltas > 0.01).sum()),
        "sequences_delta_lt_minus_0_01": int((seq_deltas < -0.01).sum()),
        "elapsed_seconds": time.time() - start,
        "interpretation_limit": "Three predetermined frames per each of 244 sequences, anchored to protected M1 predictions. This estimates local immediate C-gate utility on M1-trajectory states only; it is not C free-running utility, retrained method effect, full tracking metric, closed-loop propagation, or seed uncertainty. Seven anchor files have known provenance-drift flags.",
        "per_sequence": sequence_rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUTPUT / "completed.json").write_text(json.dumps({
        "status": summary["status"], "sequence_count": len(sequence_rows),
        "case_count": len(rows), "elapsed_seconds": summary["elapsed_seconds"],
        "summary_sha256": sha256_file(OUTPUT / "summary.json"),
    }, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sequence"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
