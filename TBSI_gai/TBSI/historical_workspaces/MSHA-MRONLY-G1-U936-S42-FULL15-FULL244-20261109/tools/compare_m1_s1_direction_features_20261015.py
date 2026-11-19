#!/usr/bin/env python3
"""Compare direction features for the frozen M1-S1 offset action."""

import csv
import hashlib
import json
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
EXP = ROOT / "output/experiments/M1_S1_DIRECTION_FEATURES_20261015"
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
ALPHA = 10.0


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
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def decode(network, tracker, response, size, offset, state, shape, resize):
    box = network.box_head.cal_bbox(response, size, offset).view(-1, 4).mean(0)
    box = box * (tracker.params.search_size / resize)
    cx, cy = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    half = 0.5 * tracker.params.search_size / resize
    xywh = [float(box[0] + cx - half - 0.5 * box[2]),
            float(box[1] + cy - half - 0.5 * box[3]), float(box[2]), float(box[3])]
    from lib.utils.box_ops import clip_box
    return clip_box(xywh, shape[0], shape[1], margin=10)


def fit_ridge(x, y):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = np.clip((x - mean) / scale, -8.0, 8.0)
    target_mean = y.mean(axis=0)
    weight = np.linalg.solve(z.T @ z + ALPHA * np.eye(z.shape[1]), z.T @ (y - target_mean))
    return mean, scale, target_mean, weight


def predict(model, x):
    mean, scale, target_mean, weight = model
    return target_mean + np.clip((x - mean) / scale, -8.0, 8.0) @ weight


def seq_bootstrap(rows, field, n_boot=10000, seed=3407):
    groups = {}
    for row in rows:
        groups.setdefault(row["sequence"], []).append(float(row[field]))
    seq_values = np.asarray([np.mean(groups[s]) for s in sorted(groups)], dtype=np.float64)
    rng = np.random.RandomState(seed)
    draw = rng.randint(0, len(seq_values), size=(n_boot, len(seq_values)))
    boot = seq_values[draw].mean(axis=1)
    return [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]


def summarize(rows, arm):
    eligible = [r for r in rows if r["s1_eligible"]]
    selected = [r for r in eligible if abs(float(r[arm + "_applied_magnitude_grid"])) > 0]
    values = np.asarray([float(r[arm + "_iou_delta"]) for r in rows], dtype=np.float64)
    selected_values = np.asarray([float(r[arm + "_iou_delta"]) for r in selected], dtype=np.float64)
    cosine = [float(r[arm + "_direction_cosine"]) for r in rows]
    return {
        "coverage_s1_eligible": len(selected) / max(1, len(rows)),
        "selected_states": len(selected),
        "mean_iou_delta_per_state_including_noop": float(values.mean()) if len(values) else 0.0,
        "sequence_bootstrap_95ci_mean_iou_delta_per_state": seq_bootstrap(
            [{"sequence": r["sequence"], "v": float(r[arm + "_iou_delta"]) if r["s1_eligible"] else 0.0}
             for r in rows], "v"),
        "selected_mean_iou_delta": float(selected_values.mean()) if len(selected_values) else 0.0,
        "selected_harmful_rate": float(np.mean(selected_values < 0)) if len(selected_values) else 0.0,
        "direction_cosine_mean_vs_gt_correction": float(np.mean(cosine)),
        "direction_cosine_positive_fraction": float(np.mean(np.asarray(cosine) > 0)),
        "mean_applied_magnitude_grid": float(np.mean([float(r[arm + "_applied_magnitude_grid"]) for r in rows])),
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
                    rec = json.loads(line)
                    traces[(path.parent.name, int(rec["frame_id"]))] = rec

    from lib.test.evaluation.lasherdataset import LasHeRDataset
    from lib.test.parameter.tbsi_track import parameters
    from lib.test.tracker.tbsi_track import TBSITrack
    from lib.train.data.processing_utils import sample_target

    seqmap = {seq.name: seq for seq in LasHeRDataset("trainingset", str(SEQUENCE_LIST)).get_sequence_list()}
    params = parameters(NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, "m1_s1_direction_feature_comparison")
    network = tracker.network.eval()
    sequence = data["sequence"].astype(str)
    frame = data["frame"].astype(int)
    keys = list(zip(sequence, frame))
    if len(keys) != 144 or len(set(sequence)) != 72:
        raise RuntimeError("unexpected frozen train state manifest")
    if set(keys) != set(captures) or set(keys) != set(state_rows) or set(keys) != set(oof):
        raise RuntimeError("feature/capture/state/OOF key sets differ")

    base_x = np.asarray([[float(state_rows[k]["peak_margin"])] for k in keys], dtype=np.float64)
    direction = np.asarray([[float(captures[k]["offset_dx"]), float(captures[k]["offset_dy"])] for k in keys])
    arm_x = {"A_peak_margin": base_x, "B1_peak_offset_map": None, "B2_top2_relative_xy": None}
    extracted = []
    output_rows = []
    parity_errors = []
    readout = json.loads(PREDICTOR.read_text())
    utility_model = readout["models"]["offset_utility"]
    risk_policy = json.loads((ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/training_locked/policy_selective_risk.json").read_text())
    motion_policy = json.loads((ROOT / "output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/training_locked/policy_motion_innovation.json").read_text())

    with torch.no_grad():
        for i, key in enumerate(keys):
            seq, fid = key
            trace = traces.get(key)
            if trace is None or not trace.get("gt_valid", False):
                raise RuntimeError("missing/invalid trace for {}:{}".format(seq, fid))
            image_paths = seqmap[seq].frames
            rgb = cv2.imread(image_paths[0][fid], cv2.IMREAD_COLOR)
            tir = cv2.imread(image_paths[1][fid], cv2.IMREAD_COLOR)
            if rgb is None or tir is None:
                raise RuntimeError("missing image for {}:{}".format(seq, fid))
            image = np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)
            state = np.asarray(trace["tracker_state_before"], dtype=np.float64)
            gt = np.asarray(trace["gt_box"], dtype=np.float64)
            _, resize, _ = sample_target(image, state, params.search_factor, output_sz=params.search_size)
            fused = torch.as_tensor(data["fused"][i].transpose(2, 0, 1)[None], dtype=torch.float32, device="cuda")
            score, _, size_map, offset_map = network.box_head(fused)
            response = tracker.output_window * score
            flat = response.flatten(1)[0].float()
            top = torch.topk(flat, k=2).indices
            py1, px1 = int(top[0] // 16), int(top[0] % 16)
            py2, px2 = int(top[1] // 16), int(top[1] % 16)
            peak_margin = float(torch.topk(flat, k=2).values.diff().abs().item())
            if abs(peak_margin - float(state_rows[key]["peak_margin"])) > 2e-4:
                raise RuntimeError("peak-margin mismatch for {}:{}".format(seq, fid))
            b1 = offset_map[0, :, py1, px1].float().cpu().numpy()
            b2 = np.asarray([px2 - px1, py2 - py1], dtype=np.float64)
            extracted.append((b1, b2))
            base_box = decode(network, tracker, response, size_map, offset_map, state, image.shape, resize)
            base_iou = iou(base_box, gt)
            parity_errors.append(abs(base_iou - float(captures[key]["base_iou"])))
            util = utility_model
            z = np.clip((peak_margin - util["mean"][0]) / util["scale"][0], -8.0, 8.0)
            eligible = bool(util["bias"] + z * util["weight"][0] > UTILITY_THRESHOLD)
            oof_row = oof[key]
            base_row = {
                "sequence": seq, "frame": fid, "event_label": str(data["label"][i]),
                "base_iou": base_iou, "gt_offset_dx": float(direction[i, 0]), "gt_offset_dy": float(direction[i, 1]),
                "gt_oracle_offset_iou_delta": float(captures[key]["offset_utility"]), "peak_margin": peak_margin,
                "risk_probability_oof": float(oof_row["risk_probability"]),
                "motion_innovation_oof": float(oof_row["normalized_motion_innovation"]),
                "s1_eligible": eligible,
                "selective_risk_admitted": bool(eligible and float(oof_row["risk_probability"]) <= float(risk_policy["max_harm_probability"])),
                "motion_innovation_admitted": bool(eligible and float(oof_row["normalized_motion_innovation"]) <= float(motion_policy["max_normalized_innovation"])),
            }
            output_rows.append(base_row)

    arm_x["B1_peak_offset_map"] = np.column_stack([base_x, np.asarray([x[0] for x in extracted])])
    arm_x["B2_top2_relative_xy"] = np.column_stack([base_x, np.asarray([x[1] for x in extracted])])
    max_parity = max(parity_errors) if parity_errors else float("inf")
    if max_parity > 1e-3:
        raise RuntimeError("no-op baseline parity failed: {}".format(max_parity))

    sequences = np.asarray(sequence)
    summaries = {}
    for arm, features in arm_x.items():
        pred = np.zeros_like(direction, dtype=np.float64)
        for held in sorted(set(sequence)):
            test = sequences == held
            pred[test] = predict(fit_ridge(features[~test], direction[~test]), features[test])
        for i, key in enumerate(keys):
            seq, fid = key
            trace = traces[key]
            state = np.asarray(trace["tracker_state_before"], dtype=np.float64)
            gt = np.asarray(trace["gt_box"], dtype=np.float64)
            image_paths = seqmap[seq].frames
            rgb = cv2.imread(image_paths[0][fid], cv2.IMREAD_COLOR)
            tir = cv2.imread(image_paths[1][fid], cv2.IMREAD_COLOR)
            image = np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)
            _, resize, _ = sample_target(image, state, params.search_factor, output_sz=params.search_size)
            fused = torch.as_tensor(data["fused"][i].transpose(2, 0, 1)[None], dtype=torch.float32, device="cuda")
            score, _, size_map, offset_map = network.box_head(fused)
            response = tracker.output_window * score
            base_iou = float(output_rows[i]["base_iou"])
            delta = torch.as_tensor(pred[i] * RHO, device=offset_map.device, dtype=offset_map.dtype)
            changed = offset_map.clone()
            if output_rows[i]["s1_eligible"]:
                flat = response.flatten(1)[0].float()
                peak = int(torch.argmax(flat).item())
                px, py = peak % 16, peak // 16
                changed[:, :, py, px] += delta
            action_box = decode(network, tracker, response, size_map, changed, state, image.shape, resize)
            action_iou = iou(action_box, gt)
            cosine = float(np.dot(pred[i], direction[i]) / max(np.linalg.norm(pred[i]) * np.linalg.norm(direction[i]), 1e-12))
            output_rows[i][arm + "_delta_x_oof"] = float(pred[i, 0])
            output_rows[i][arm + "_delta_y_oof"] = float(pred[i, 1])
            output_rows[i][arm + "_applied_magnitude_grid"] = float(RHO * np.linalg.norm(pred[i])) if output_rows[i]["s1_eligible"] else 0.0
            output_rows[i][arm + "_iou"] = action_iou if output_rows[i]["s1_eligible"] else base_iou
            output_rows[i][arm + "_iou_delta"] = action_iou - base_iou if output_rows[i]["s1_eligible"] else 0.0
            output_rows[i][arm + "_direction_cosine"] = cosine
        summaries[arm] = summarize(output_rows, arm)
        seq_deltas = {}
        for row in output_rows:
            seq_deltas.setdefault(row["sequence"], []).append(float(row[arm + "_iou_delta"]))
        summaries[arm]["per_sequence_macro_mean_iou_delta"] = float(np.mean([np.mean(v) for v in seq_deltas.values()]))
        summaries[arm]["per_sequence_mean_iou_delta"] = {s: float(np.mean(v)) for s, v in sorted(seq_deltas.items())}

    # Paired sequence bootstrap against A, preserving sequence as the sampling unit.
    seqs = sorted(set(sequence))
    paired = {}
    by_seq = {s: [r for r in output_rows if r["sequence"] == s] for s in seqs}
    rng = np.random.RandomState(20261015)
    picks = rng.randint(0, len(seqs), size=(10000, len(seqs)))
    for arm in ("B1_peak_offset_map", "B2_top2_relative_xy"):
        seq_delta = np.asarray([
            np.mean([float(r[arm + "_iou_delta"]) - float(r["A_peak_margin_iou_delta"]) for r in by_seq[s]])
            for s in seqs
        ])
        boot = seq_delta[picks].mean(axis=1)
        paired[arm + "_minus_A"] = {
            "sequence_macro_mean_delta_iou": float(seq_delta.mean()),
            "sequence_bootstrap_95ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "sequences_improved": int((seq_delta > 0).sum()), "sequences_degraded": int((seq_delta < 0).sum()),
        }

    with (out / "per_state_direction_features.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    result = {"n_states": len(output_rows), "n_sequences": len(seqs), "s1_eligible_states": int(sum(r["s1_eligible"] for r in output_rows)),
              "no_op_max_abs_iou_parity_error": max_parity, "arms": summaries, "paired_vs_A": paired}
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = {
        "id": "M1-S1-DIRECTION-FEATURES-20261015", "parent": "M1-S1-TRUE-ACTION-UTILITY-AUDIT-20261015",
        "hypothesis": "Adding spatial direction evidence to peak-margin input improves held-sequence-out offset action direction and decoded utility.",
        "design": {"A": "peak_margin scalar", "B1": "peak_margin + CENTER offset-map x/y at response argmax", "B2": "peak_margin + top-1 to top-2 response displacement x/y",
                   "one_changed_variable": "predictor input features", "ridge_alpha": ALPHA, "rho": RHO, "S1_utility_threshold": UTILITY_THRESHOLD,
                   "fit": "leave-one-sequence-out over same 72 train sequences / 144 fixed states", "admission": "same frozen S1 utility gate for all arms", "fallback": "no-op"},
        "dataset": "LasHeR train-side fixed-state cohort; 72 sequences, 144 states", "seed": 42, "precision": "FP32",
        "checkpoint_sha256": sha256(CHECKPOINT), "config_sha256": sha256(CONFIG), "feature_cache_sha256": sha256(FEATURES),
        "code_sha256": sha256(Path(__file__)), "runtime": {"python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(), "workers": 0},
        "limitations": ["fixed-state one-step action utility only; not closed-loop tracking", "preselected 72-sequence diagnostic cohort", "single protected M1 checkpoint/seed"],
        "results": result}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
