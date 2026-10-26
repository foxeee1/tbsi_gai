#!/usr/bin/env python3
"""Measure sequence-held-out S1 offset action utility on frozen train states."""

import csv
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAME = "TBSI_E_MHSA_FULL_4090D_seed42"
EXP = ROOT / "output/experiments/M1_S1_TRUE_ACTION_UTILITY_AUDIT_20261015"
FEATURES = ROOT / "output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz"
CAPTURE = ROOT / "output/experiments/M1_BOUNDED_READOUT_PREDICTOR_20261014_retry1/capture_rows.csv"
TRAIN_STATES = ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/training_locked/train_side_states.csv"
TRAIN_SUMMARY = ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/training_locked/training_summary.json"
PREDICTOR = ROOT / "experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json"
SEQUENCE_LIST = ROOT / "output/experiments/predictor_sequence_list.txt"
CHECKPOINT = ROOT / "output/experiments" / NAME / "checkpoints/TBSITrack_ep0015.pth.tar"
CONFIG = ROOT / "experiments/tbsi_track" / (NAME + ".yaml")
CONFIG_SHA = "fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e"
CHECKPOINT_SHA = "1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef"
TRACE_DIRS = [
    ROOT / "output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run/trace",
    ROOT / "output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank24/trace",
    ROOT / "output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank48/trace",
]
RHO = 0.5
UTILITY_THRESHOLD = 0.08


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def decode(network, tracker, response, size, offset, state, shape, resize):
    box = network.box_head.cal_bbox(response, size, offset).view(-1, 4).mean(0)
    box = box * (tracker.params.search_size / resize)
    cx = state[0] + 0.5 * state[2]
    cy = state[1] + 0.5 * state[3]
    half = 0.5 * tracker.params.search_size / resize
    xywh = [float(box[0] + cx - half - 0.5 * box[2]),
            float(box[1] + cy - half - 0.5 * box[3]),
            float(box[2]), float(box[3])]
    from lib.utils.box_ops import clip_box
    return clip_box(xywh, shape[0], shape[1], margin=10)


def fit_ridge(x, y, alpha=10.0):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = np.clip((x - mean) / scale, -8.0, 8.0)
    target_mean = y.mean(axis=0)
    weight = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]),
                             z.T @ (y - target_mean))
    return mean, scale, target_mean, weight


def predict_ridge(model, x):
    mean, scale, target_mean, weight = model
    z = np.clip((x - mean) / scale, -8.0, 8.0)
    return target_mean + z @ weight


def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def sequence_bootstrap(rows, field, n_boot=10000, seed=3407):
    by_seq = {}
    for row in rows:
        by_seq.setdefault(row["sequence"], []).append(float(row[field]))
    seqs = sorted(by_seq)
    means = np.asarray([np.mean(by_seq[s]) for s in seqs], dtype=np.float64)
    rng = np.random.RandomState(seed)
    sampled = rng.randint(0, len(seqs), size=(n_boot, len(seqs)))
    values = means[sampled].mean(axis=1)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def summarize_arm(rows, use_field):
    chosen = [r for r in rows if r[use_field]]
    values = np.asarray([float(r["predicted_action_iou_delta"]) for r in chosen])
    all_values = np.asarray([float(r["predicted_action_iou_delta"]) if r[use_field] else 0.0 for r in rows])
    harmful = values < 0.0
    return {
        "coverage": float(len(chosen) / len(rows)),
        "selected": len(chosen),
        "selected_mean_iou_delta": float(values.mean()) if len(values) else 0.0,
        "harmful_rate_selected": float(harmful.mean()) if len(values) else 0.0,
        "mean_iou_delta_per_state_with_noop": float(all_values.mean()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(EXP))
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing nonempty output directory: {}".format(out))
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("protected M1 config/checkpoint hash mismatch")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TBSI_INFER_FP16"] = "0"
    os.environ["TBSI_CHECKPOINT_OVERRIDE"] = str(CHECKPOINT)
    os.environ.pop("TBSI_BOUNDED_READOUT_PREDICTOR", None)
    os.environ.pop("TBSI_READOUT_POLICY", None)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data = np.load(str(FEATURES), allow_pickle=False)
    captures = {(r["sequence"], int(r["frame"])): r for r in read_csv(CAPTURE)}
    state_rows = {(r["sequence"], int(r["frame"])): r for r in read_csv(TRAIN_STATES)}
    summary = json.loads(TRAIN_SUMMARY.read_text())
    oof = {(r["sequence"], int(r["frame"])): r for r in summary["oof_predictions"]}
    traces = {}
    for trace_root in TRACE_DIRS:
        for path in trace_root.glob("*/frames.jsonl"):
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    traces[(path.parent.name, int(record["frame_id"]))] = record

    from lib.test.evaluation.lasherdataset import LasHeRDataset
    from lib.test.parameter.tbsi_track import parameters
    from lib.test.tracker.tbsi_track import TBSITrack
    from lib.train.data.processing_utils import sample_target

    seqmap = {seq.name: seq for seq in LasHeRDataset("trainingset", str(SEQUENCE_LIST)).get_sequence_list()}
    params = parameters(NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, "m1_s1_true_action_utility_audit")
    network = tracker.network.eval()

    sequence = data["sequence"].astype(str)
    frame = data["frame"].astype(int)
    sample_keys = list(zip(sequence, frame))
    if len(sample_keys) != 144 or len(set(sequence)) != 72:
        raise RuntimeError("unexpected frozen train state manifest")
    if set(sample_keys) != set(captures) or set(sample_keys) != set(state_rows) or set(sample_keys) != set(oof):
        raise RuntimeError("feature/capture/state/OOF key sets differ")

    x = np.asarray([[float(state_rows[k]["peak_margin"])] for k in sample_keys], dtype=np.float64)
    y = np.asarray([[float(captures[k]["offset_dx"]), float(captures[k]["offset_dy"])]
                    for k in sample_keys], dtype=np.float64)
    sequences = np.asarray(sequence)
    predicted_delta = np.zeros_like(y)
    for held in sorted(set(sequence)):
        test = sequences == held
        train = ~test
        predicted_delta[test] = predict_ridge(fit_ridge(x[train], y[train]), x[test])

    full_model = json.loads(PREDICTOR.read_text())["models"]["offset_delta"]
    full_mean = np.asarray(full_model["mean"], dtype=np.float64)
    full_scale = np.asarray(full_model["scale"], dtype=np.float64)
    full_bias = np.asarray(full_model["bias"], dtype=np.float64)
    full_weight = np.asarray(full_model["weight"], dtype=np.float64)
    full_z = np.clip((x[:, 0] - full_mean[0]) / full_scale[0], -8.0, 8.0)
    in_sample_delta = full_bias[None, :] + full_z[:, None] * full_weight[None, :]

    risk_policy = json.loads((ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/training_locked/policy_selective_risk.json").read_text())
    motion_policy = json.loads((ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/training_locked/policy_motion_innovation.json").read_text())
    readout = json.loads(PREDICTOR.read_text())
    utility_model = readout["models"]["offset_utility"]
    mu_u, scale_u = utility_model["mean"][0], utility_model["scale"][0]

    output_rows = []
    parity_errors = []
    with torch.no_grad():
        for idx, key in enumerate(sample_keys):
            seq, fid = key
            trace = traces.get(key)
            if trace is None or not trace.get("gt_valid", False):
                raise RuntimeError("missing/invalid trace for {}:{}".format(seq, fid))
            image_paths = seqmap[seq].frames
            rgb = cv2.imread(image_paths[0][fid], cv2.IMREAD_COLOR)
            tir = cv2.imread(image_paths[1][fid], cv2.IMREAD_COLOR)
            if rgb is None or tir is None:
                raise RuntimeError("missing image for {}:{}".format(seq, fid))
            image = np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)
            state = np.asarray(trace["tracker_state_before"], dtype=np.float64)
            gt = np.asarray(trace["gt_box"], dtype=np.float64)
            _, resize, _ = sample_target(image, state, params.search_factor, output_sz=params.search_size)
            fused = torch.as_tensor(data["fused"][idx].transpose(2, 0, 1)[None],
                                    dtype=torch.float32, device="cuda")
            score, _, size_map, offset_map = network.box_head(fused)
            response = tracker.output_window * score
            flat = response.flatten(1)[0].float()
            peak = int(torch.argmax(flat).item())
            px, py = peak % 16, peak // 16
            peak_margin = float((torch.topk(flat, k=2).values.diff().abs().item()))
            if abs(peak_margin - float(state_rows[key]["peak_margin"])) > 2e-4:
                raise RuntimeError("peak-margin mismatch for {}:{}".format(seq, fid))
            base_box = decode(network, tracker, response, size_map, offset_map, state, image.shape, resize)
            base_iou = iou(base_box, gt)
            parity_errors.append(abs(base_iou - float(captures[key]["base_iou"])))

            delta = torch.as_tensor(predicted_delta[idx] * RHO, device=offset_map.device, dtype=offset_map.dtype)
            changed = offset_map.clone()
            changed[:, :, py, px] = offset_map[:, :, py, px] + delta
            action_box = decode(network, tracker, response, size_map, changed, state, image.shape, resize)
            action_iou = iou(action_box, gt)
            action_delta = action_iou - base_iou

            z_u = np.clip((float(state_rows[key]["peak_margin"]) - mu_u) / scale_u, -8.0, 8.0)
            predicted_utility = float(utility_model["bias"] + z_u * utility_model["weight"][0])
            base_eligible = predicted_utility > UTILITY_THRESHOLD
            oof_row = oof[key]
            risk_probability = float(oof_row["risk_probability"])
            innovation = float(oof_row["normalized_motion_innovation"])
            crop = np.asarray(trace["current_crop_xywh"], dtype=np.float64)
            ix = max(0.0, min(crop[0] + crop[2], gt[0] + gt[2]) - max(crop[0], gt[0]))
            iy = max(0.0, min(crop[1] + crop[3], gt[1] + gt[3]) - max(crop[1], gt[1]))
            gt_area = max(float(gt[2] * gt[3]), 1e-12)
            crop_fraction = ix * iy / gt_area
            target = y[idx]
            predicted = predicted_delta[idx]
            norm_pred = float(np.linalg.norm(predicted))
            norm_target = float(np.linalg.norm(target))
            cosine = float(np.dot(predicted, target) / max(norm_pred * norm_target, 1e-12))
            output_rows.append({
                "sequence": seq, "frame": fid, "event_label": str(data["label"][idx]),
                "base_iou": base_iou, "predicted_action_iou": action_iou,
                "predicted_action_iou_delta": action_delta,
                "gt_oracle_offset_iou_delta": float(captures[key]["offset_utility"]),
                "base_offset_utility_estimate": predicted_utility,
                "peak_margin": float(state_rows[key]["peak_margin"]),
                "risk_probability_oof": risk_probability,
                "motion_innovation_oof": innovation,
                "offset_dx_oof": float(predicted[0]), "offset_dy_oof": float(predicted[1]),
                "offset_applied_magnitude_grid": float(RHO * norm_pred),
                "gt_offset_dx": float(target[0]), "gt_offset_dy": float(target[1]),
                "gt_offset_magnitude_grid": norm_target,
                "offset_direction_cosine_to_gt_correction": cosine,
                "gt_crop_area_fraction": crop_fraction,
                "gt_fully_inside_crop": bool(crop_fraction >= 1.0 - 1e-6),
                "s1_eligible": bool(base_eligible),
                "selective_risk_admitted": bool(base_eligible and risk_probability <= float(risk_policy["max_harm_probability"])),
                "motion_innovation_admitted": bool(base_eligible and innovation <= float(motion_policy["max_normalized_innovation"])),
                "in_sample_offset_dx": float(in_sample_delta[idx, 0]),
                "in_sample_offset_dy": float(in_sample_delta[idx, 1]),
            })

    max_parity = max(parity_errors) if parity_errors else float("inf")
    if max_parity > 1e-3:
        raise RuntimeError("no-op baseline parity failed: max abs IoU error {}".format(max_parity))

    metrics = {}
    for arm, field in [("S1_base", "s1_eligible"),
                       ("selective_risk", "selective_risk_admitted"),
                       ("motion_innovation", "motion_innovation_admitted")]:
        metrics[arm] = summarize_arm(output_rows, field)
        selected = [r for r in output_rows if r[field]]
        metrics[arm]["sequence_bootstrap_95ci_selected_mean_utility"] = sequence_bootstrap(
            selected, "predicted_action_iou_delta") if selected else None
        metrics[arm]["sequence_bootstrap_95ci_mean_utility_per_state"] = sequence_bootstrap(
            [{**r, "utility_for_bootstrap": float(r["predicted_action_iou_delta"]) if r[field] else 0.0}
             for r in output_rows], "utility_for_bootstrap")

    harmful = np.asarray([float(r["predicted_action_iou_delta"]) < 0.0 for r in output_rows])
    risk_score = np.asarray([float(r["risk_probability_oof"]) for r in output_rows])
    motion_score = np.asarray([float(r["motion_innovation_oof"]) for r in output_rows])
    eligible = np.asarray([r["s1_eligible"] for r in output_rows], dtype=bool)
    eligible_harmful = harmful[eligible]
    observed = {
        "n_states": len(output_rows),
        "n_sequences": len(set(r["sequence"] for r in output_rows)),
        "actual_predicted_offset_action_harmful_auc_risk_score": auc_binary(harmful, risk_score),
        "actual_predicted_offset_action_harmful_auc_motion_innovation": auc_binary(harmful, motion_score),
        "harmful_rate_among_s1_eligible": float(eligible_harmful.mean()) if len(eligible_harmful) else None,
        "harmful_rate_among_all_oof_offset_actions": float(harmful.mean()),
        "no_op_max_abs_iou_parity_error": max_parity,
        "mean_offset_direction_cosine": float(np.mean([r["offset_direction_cosine_to_gt_correction"] for r in output_rows])),
        "median_offset_direction_cosine": float(np.median([r["offset_direction_cosine_to_gt_correction"] for r in output_rows])),
        "direction_cosine_positive_fraction": float(np.mean([r["offset_direction_cosine_to_gt_correction"] > 0 for r in output_rows])),
        "mean_applied_offset_magnitude_grid": float(np.mean([r["offset_applied_magnitude_grid"] for r in output_rows])),
        "mean_gt_offset_magnitude_grid": float(np.mean([r["gt_offset_magnitude_grid"] for r in output_rows])),
        "mean_gt_crop_area_fraction": float(np.mean([r["gt_crop_area_fraction"] for r in output_rows])),
        "fraction_gt_fully_inside_crop": float(np.mean([r["gt_fully_inside_crop"] for r in output_rows])),
        "mean_gt_oracle_offset_iou_delta": float(np.mean([r["gt_oracle_offset_iou_delta"] for r in output_rows])),
        "gt_oracle_offset_positive_fraction": float(np.mean([r["gt_oracle_offset_iou_delta"] > 0 for r in output_rows])),
        "actual_vs_gt_oracle_utility_sign_disagreement_fraction": float(np.mean([
            (r["predicted_action_iou_delta"] > 0) != (r["gt_oracle_offset_iou_delta"] > 0)
            for r in output_rows])),
        "arm_metrics": metrics,
    }
    for key, label in [("gt_fully_inside_crop", "fully_inside_crop"),
                        ("s1_eligible", "s1_eligible"),
                        ("selective_risk_admitted", "selective_risk_admitted"),
                        ("motion_innovation_admitted", "motion_innovation_admitted")]:
        group = [r for r in output_rows if r[key]]
        observed[label + "_n"] = len(group)
        observed[label + "_mean_actual_offset_utility"] = float(np.mean([r["predicted_action_iou_delta"] for r in group])) if group else None
        observed[label + "_harmful_fraction"] = float(np.mean([r["predicted_action_iou_delta"] < 0 for r in group])) if group else None

    with (out / "per_state_true_action_utility.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (out / "summary.json").write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
    manifest = {
        "id": "M1-S1-TRUE-ACTION-UTILITY-AUDIT-20261015",
        "parent": "M1-S1-ADMISSION-METHODS-FULL244-20261014",
        "question": "Do deployment-visible admission scores identify harmful sequence-held-out S1 offset actions when utility is measured from the actual predicted offset vector?",
        "method": "Leave-one-sequence-out ridge offset-vector fit from the frozen 144 states; replay no-op and predicted S1 offset on the same cached fused feature, crop, state, and CENTER head; compute decoded IoU delta.",
        "dataset": "LasHeR train-side fixed-state cohort; 72 sequences, 144 states",
        "seed": 42,
        "precision": "FP32",
        "checkpoint_sha256": sha256(CHECKPOINT),
        "config_sha256": sha256(CONFIG),
        "features_sha256": sha256(FEATURES),
        "capture_rows_sha256": sha256(CAPTURE),
        "risk_policy_sha256": sha256(ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/training_locked/policy_selective_risk.json"),
        "motion_policy_sha256": sha256(ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/training_locked/policy_motion_innovation.json"),
        "s1_predictor_sha256": sha256(PREDICTOR),
        "trace_dirs": [str(p.relative_to(ROOT)) for p in TRACE_DIRS],
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(), "workers": 0},
        "status": "COMPLETE",
        "limits": ["fixed-state one-step utility, not closed-loop tracking", "states are a preselected diagnostic cohort", "single M1 checkpoint and seed"],
        "results": observed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(observed, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
