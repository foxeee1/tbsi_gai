#!/usr/bin/env python3
"""Measure the causal effect of a single S1-vs-M1 internal state commit."""

import contextlib
import csv
import hashlib
import io
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
CONFIG = ROOT / "experiments/tbsi_track" / (NAME + ".yaml")
CHECKPOINT = ROOT / "output/experiments" / NAME / "checkpoints/TBSITrack_ep0015.pth.tar"
PREDICTOR = ROOT / "experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json"
POLICY = ROOT / "experiments/artifacts/m1_readout_independent_20261014.json"
SOURCE_LIST = ROOT / "output/experiments/predictor_sequence_list.txt"
PREVIOUS_LIST = ROOT / "output/experiments/M1_S1_CLOSED_LOOP_RISK_DIAGNOSTIC_20261015/sequence_manifest.txt"
OUT = ROOT / "output/experiments/M1_S1_SINGLE_COMMIT_EFFECT_20261015"
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


def log(message):
    with (OUT / "run.log").open("a") as stream:
        stream.write(message + "\n")
    print(message, flush=True)


def image_pair(paths):
    visible = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    infrared = cv2.imread(paths[1], cv2.IMREAD_COLOR)
    if visible is None or infrared is None or visible.shape[:2] != infrared.shape[:2]:
        raise RuntimeError("invalid RGB/TIR pair: {} {}".format(paths[0], paths[1]))
    return np.concatenate((visible, infrared), axis=2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def choose_events(action_rows, n_frames):
    usable = [r for r in action_rows if r["action"] != "noop" and
              int(r["frame_id"]) + 30 < n_frames]
    chosen = []
    for target_fraction in (0.25, 0.70):
        low, high = target_fraction - 0.15, target_fraction + 0.15
        candidates = [r for r in usable if low <= int(r["frame_id"]) / float(n_frames) <= high]
        if not candidates:
            continue
        target = target_fraction * n_frames
        event = min(candidates, key=lambda r: (abs(int(r["frame_id"]) - target), int(r["frame_id"])))
        if all(abs(int(event["frame_id"]) - int(old["frame_id"])) >= 35 for old in chosen):
            chosen.append(event)
    return sorted(chosen, key=lambda r: int(r["frame_id"]))


def first_recovery(values, threshold=0.5, run=5):
    for i in range(max(0, len(values) - run + 1)):
        if all(x >= threshold for x in values[i:i + run]):
            return i + 1
    return None


def clustered_interval(rows, field, seed=3407, n_boot=10000):
    groups = {}
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            groups.setdefault(row["sequence"], []).append(float(value))
    seqs = sorted(groups)
    if len(seqs) < 2:
        return {"sequences": len(seqs), "ci95": None}
    means = np.asarray([np.mean(groups[s]) for s in seqs], dtype=np.float64)
    rng = np.random.RandomState(seed)
    sampled = rng.randint(0, len(seqs), size=(n_boot, len(seqs)))
    boot = means[sampled].mean(axis=1)
    return {"sequences": len(seqs), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])]}


def summarize(rows):
    result = {"events": len(rows), "sequences": len(set(r["sequence"] for r in rows)),
              "event_output_parity_max_abs": max((r["event_output_max_abs"] for r in rows), default=None),
              "pre_event_state_parity_max_abs": max((r["pre_state_max_abs"] for r in rows), default=None),
              "intervention_state_delta_nonzero_fraction": float(np.mean([
                  r["state_commit_delta_l2"] > 1e-8 for r in rows])) if rows else None,
              "by_horizon": {}}
    immediate = [r["immediate_readout_delta_s1_minus_m1"] for r in rows
                 if r.get("immediate_readout_delta_s1_minus_m1") is not None]
    result["immediate_readout_effect"] = {
        "valid_events": len(immediate),
        "mean_iou_delta_s1_minus_m1": float(np.mean(immediate)) if immediate else None,
        "median_iou_delta_s1_minus_m1": float(np.median(immediate)) if immediate else None,
        "positive_event_fraction": float(np.mean(np.asarray(immediate) > 0)) if immediate else None,
        "sequence_cluster_bootstrap": clustered_interval(rows, "immediate_readout_delta_s1_minus_m1"),
    }
    for horizon in (10, 30):
        key = "delta_commit_minus_fallback_h{}".format(horizon)
        values = [float(r[key]) for r in rows if r[key] is not None]
        result["by_horizon"][str(horizon)] = {
            "valid_events": len(values),
            "mean_iou_delta": float(np.mean(values)) if values else None,
            "median_iou_delta": float(np.median(values)) if values else None,
            "positive_event_fraction": float(np.mean(np.asarray(values) > 0)) if values else None,
            "sequence_cluster_bootstrap": clustered_interval(rows, key),
        }
    for action in sorted(set(r["action"] for r in rows)):
        subset = [r for r in rows if r["action"] == action]
        result.setdefault("by_action", {})[action] = {
            "events": len(subset), "sequences": len(set(r["sequence"] for r in subset)),
            "mean_delta_h10": float(np.mean([r["delta_commit_minus_fallback_h10"] for r in subset
                                               if r["delta_commit_minus_fallback_h10"] is not None]))
                              if any(r["delta_commit_minus_fallback_h10"] is not None for r in subset) else None,
            "mean_delta_h30": float(np.mean([r["delta_commit_minus_fallback_h30"] for r in subset
                                               if r["delta_commit_minus_fallback_h30"] is not None]))
                              if any(r["delta_commit_minus_fallback_h30"] is not None for r in subset) else None,
        }
    result["mean_recovery_frame_commit"] = float(np.mean([r["recovery_frame_commit"] for r in rows
                                                             if r["recovery_frame_commit"] is not None])) if any(
        r["recovery_frame_commit"] is not None for r in rows) else None
    result["mean_recovery_frame_fallback"] = float(np.mean([r["recovery_frame_fallback"] for r in rows
                                                              if r["recovery_frame_fallback"] is not None])) if any(
        r["recovery_frame_fallback"] is not None for r in rows) else None
    return result


def main():
    global OUT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=12)
    parser.add_argument("--max-events-per-sequence", type=int, default=2, choices=[1, 2])
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--reuse-events", default="")
    args = parser.parse_args()
    OUT = Path(args.output)
    for key, path in (("config", CONFIG), ("checkpoint", CHECKPOINT),
                      ("predictor", PREDICTOR), ("policy", POLICY)):
        if sha(path) != EXPECTED[key]:
            raise RuntimeError("{} hash mismatch: {}".format(key, path))
    if OUT.exists():
        raise RuntimeError("refusing to overwrite output directory {}".format(OUT))
    source = [x.strip() for x in SOURCE_LIST.read_text().splitlines() if x.strip()]
    previous = [x.strip() for x in PREVIOUS_LIST.read_text().splitlines() if x.strip()]
    if len(source) != 72 or len(set(source)) != 72 or not set(previous).issubset(set(source)):
        raise RuntimeError("frozen source or prior diagnostic manifest contract failed")
    candidates = [x for x in source if x not in set(previous)]
    names = sorted(candidates, key=lambda n: hashlib.sha256(n.encode("utf-8")).hexdigest())[:args.sequences]
    if len(names) != args.sequences:
        raise RuntimeError("not enough disjoint train-side sequences")
    OUT.mkdir(parents=True)
    manifest_file = OUT / "sequence_manifest.txt"
    manifest_file.write_text("\n".join(names) + "\n")
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["TBSI_INFER_FP16"] = "0"
    os.environ["TBSI_CHECKPOINT_OVERRIDE"] = str(CHECKPOINT)
    os.environ["TBSI_BOUNDED_READOUT_PREDICTOR"] = str(PREDICTOR)
    os.environ["TBSI_READOUT_POLICY"] = str(POLICY)
    os.environ["TBSI_READOUT_STATE_MODE"] = "coupled"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from lib.test.evaluation.lasherdataset import LasHeRDataset
    from lib.test.parameter.tbsi_track import parameters
    from lib.test.tracker.tbsi_track import TBSITrack
    dataset = LasHeRDataset("trainingset", str(manifest_file)).get_sequence_list()
    if [s.name for s in dataset] != names:
        raise RuntimeError("resolved sequence order differs from manifest")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
    run_command = [sys.executable, "-u", str(Path(__file__).resolve().relative_to(ROOT.parents[1])),
                   "--sequences", str(args.sequences), "--max-events-per-sequence", str(args.max_events_per_sequence),
                   "--output", str(OUT)]
    if args.reuse_events:
        run_command.extend(["--reuse-events", args.reuse_events])
    manifest = {
        "id": "M1-S1-SINGLE-COMMIT-EFFECT-20261015", "status": "RUNNING",
        "parent": "M1-S1-OUTPUT-ONLY-STATE-FULL244-20261015",
        "hypothesis": "A single S1-state commit at an actual correction event has a measurable causal effect on future tracking when pre-event history and event output are fixed.",
        "intervention": "same S1 event output; one branch commits corrected state, paired branch commits the same-frame M1 state; both resume coupled S1 state updates on the next frame",
        "dataset": "LasHeR trainingset; {} SHA256-selected whole sequences from source72 excluding the prior 12-sequence branch diagnostic".format(len(names)),
        "source_manifest": str(SOURCE_LIST), "source_manifest_sha256": sha(SOURCE_LIST),
        "excluded_prior_manifest": str(PREVIOUS_LIST), "excluded_prior_manifest_sha256": sha(PREVIOUS_LIST),
        "sequence_manifest": str(manifest_file), "sequence_manifest_sha256": sha(manifest_file),
        "sequences": names, "sequence_count": len(names),
        "event_selection": "non-noop S1 actions nearest 25% and 70% sequence positions within fixed +/-15% windows; no GT used",
        "labels": "same-frame IoU(S1 corrected output)-IoU(M1 raw output) plus paired future IoU(commit)-IoU(fallback), next 10/30 frames",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
        "config": str(CONFIG), "config_sha256": sha(CONFIG),
        "predictor": str(PREDICTOR), "predictor_sha256": sha(PREDICTOR),
        "policy": str(POLICY), "policy_sha256": sha(POLICY),
        "runner_sha256": sha(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT.parents[1])).decode().strip(),
        "command": run_command,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "gpu": gpu,
                        "precision": "FP32", "tracker_processes": 1, "workers": 0},
        "events": [], "elapsed_seconds": None,
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log("START paired single-commit experiment sequences={} gpu={}".format(len(dataset), gpu))

    selected = {}
    if args.reuse_events:
        event_manifest = json.loads(Path(args.reuse_events).read_text())
        if not event_manifest:
            raise RuntimeError("reused event manifest is empty")
        valid_names = set(names)
        for event in event_manifest:
            if event["sequence"] not in valid_names or event["action"] == "noop":
                raise RuntimeError("reused event manifest has invalid sequence/action")
            selected.setdefault(event["sequence"], []).append(event)
        for name in names:
            selected.setdefault(name, [])
        log("REUSE events={} source={}".format(len(event_manifest), args.reuse_events))
    else:
        discovery_dir = OUT / "discovery_actions"
        discovery_dir.mkdir()
        os.environ["TBSI_READOUT_AUDIT_DIR"] = str(discovery_dir)
        discover_params = parameters(NAME)
        discover_params.checkpoint = str(CHECKPOINT)
        discover_params.debug = 0
        with contextlib.redirect_stdout(io.StringIO()):
            discoverer = TBSITrack(discover_params, "single_commit_discovery")
        for si, seq in enumerate(dataset, 1):
            discoverer.initialize(image_pair((seq.frames[0][0], seq.frames[1][0])),
                                  {"init_bbox": np.asarray(seq.ground_truth_rect[0]).tolist(), "sequence_name": seq.name})
            for frame_idx in range(1, len(seq.frames[0])):
                image = image_pair((seq.frames[0][frame_idx], seq.frames[1][frame_idx]))
                discoverer.track(image, {"sequence_name": seq.name, "frame_num": frame_idx})
            audit = discovery_dir / seq.name / "actions.jsonl"
            action_rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
            selected[seq.name] = choose_events(action_rows, len(seq.frames[0]))[:args.max_events_per_sequence]
            log("DISCOVERY {}/{} {} actions={} selected={}".format(
                si, len(dataset), seq.name, sum(r["action"] != "noop" for r in action_rows),
                [int(x["frame_id"]) for x in selected[seq.name]]))
        del discoverer
        torch.cuda.empty_cache()
        event_manifest = [{"sequence": name, "frame_id": int(action["frame_id"]), "action": action["action"],
                           "peak_margin": float(action["peak_margin"]),
                           "normalized_entropy": float(action["normalized_entropy"]),
                           "top5_margin": float(action["top5_margin"]),
                           "offset_utility": float(action["offset_utility"]),
                           "size_utility": float(action["size_utility"])}
                          for name in names for action in selected[name]]
    os.environ["TBSI_READOUT_AUDIT_DIR"] = ""
    (OUT / "selected_events.json").write_text(json.dumps(event_manifest, indent=2, sort_keys=True) + "\n")
    manifest["events"] = event_manifest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not event_manifest:
        manifest["status"] = "FAILED-NO-CORRECTION-EVENTS"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise RuntimeError("no eligible non-noop correction events found")

    params_a, params_b = parameters(NAME), parameters(NAME)
    params_a.checkpoint = params_b.checkpoint = str(CHECKPOINT)
    params_a.debug = params_b.debug = 0
    with contextlib.redirect_stdout(io.StringIO()):
        commit = TBSITrack(params_a, "single_commit_corrected_state")
        fallback = TBSITrack(params_b, "single_commit_m1_state")
    commit.network.eval()
    fallback.network.eval()
    rows = []
    t0 = time.time()
    jsonl = OUT / "paired_event_effects.jsonl"
    for si, seq in enumerate(dataset, 1):
        def get_frame(k):
            return image_pair((seq.frames[0][k], seq.frames[1][k]))
        for action in selected[seq.name]:
            event_frame = int(action["frame_id"])
            init_info = {"init_bbox": np.asarray(seq.ground_truth_rect[0]).tolist(), "sequence_name": seq.name}
            commit.initialize(get_frame(0), init_info)
            fallback.initialize(get_frame(0), init_info)
            prefix_max = 0.0
            for frame_idx in range(1, event_frame):
                image = get_frame(frame_idx)
                info = {"sequence_name": seq.name, "frame_num": frame_idx}
                out_a = commit.track(image, info)["target_bbox"]
                out_b = fallback.track(image, info)["target_bbox"]
                prefix_max = max(prefix_max, float(np.max(np.abs(np.asarray(out_a) - np.asarray(out_b)))))
                if prefix_max > 1e-5:
                    raise RuntimeError("S1 prefix mismatch before event {}:{} max={}".format(
                        seq.name, event_frame, prefix_max))
            pre_state_max = float(np.max(np.abs(np.asarray(commit.state) - np.asarray(fallback.state))))
            if pre_state_max > 1e-6:
                raise RuntimeError("pre-event tracker states differ: {}:{} {}".format(seq.name, event_frame, pre_state_max))
            event_info = {"sequence_name": seq.name, "frame_num": event_frame}
            event_image = get_frame(event_frame)
            fallback.readout_state_mode = "output_only"
            event_a = commit.track(event_image, event_info)["target_bbox"]
            event_b = fallback.track(event_image, event_info)["target_bbox"]
            fallback.readout_state_mode = "coupled"
            event_diff = float(np.max(np.abs(np.asarray(event_a) - np.asarray(event_b))))
            if event_diff > 1e-5:
                raise RuntimeError("event output parity failed {}:{} max={}".format(seq.name, event_frame, event_diff))
            state_delta = float(np.linalg.norm(np.asarray(commit.state, dtype=np.float64) -
                                               np.asarray(fallback.state, dtype=np.float64)))
            future_a, future_b = [], []
            gt_at_event = np.asarray(seq.ground_truth_rect[event_frame], dtype=np.float64).tolist()
            event_iou = iou(event_a, gt_at_event)
            raw_m1_event_box = [float(x) for x in fallback.state]
            raw_m1_event_iou = iou(raw_m1_event_box, gt_at_event)
            for frame_idx in range(event_frame + 1, min(len(seq.frames[0]), event_frame + 31)):
                image = get_frame(frame_idx)
                info = {"sequence_name": seq.name, "frame_num": frame_idx}
                box_a = commit.track(image, info)["target_bbox"]
                box_b = fallback.track(image, info)["target_bbox"]
                gt = np.asarray(seq.ground_truth_rect[frame_idx], dtype=np.float64).tolist()
                future_a.append(iou(box_a, gt))
                future_b.append(iou(box_b, gt))
            row = {
                "sequence": seq.name, "event_frame": event_frame, "action": action["action"],
                "peak_margin": float(action["peak_margin"]), "normalized_entropy": float(action["normalized_entropy"]),
                "top5_margin": float(action["top5_margin"]), "offset_utility": float(action["offset_utility"]),
                "size_utility": float(action["size_utility"]), "event_output_max_abs": event_diff,
                "prefix_output_max_abs": prefix_max, "pre_state_max_abs": pre_state_max,
                "state_commit_delta_l2": state_delta, "event_iou": event_iou,
                "raw_m1_event_box": raw_m1_event_box, "raw_m1_event_iou": raw_m1_event_iou,
                "immediate_readout_delta_s1_minus_m1": event_iou - raw_m1_event_iou,
                "future_frames": len(future_a),
                "future_h10_iou_commit": float(np.mean(future_a[:10])) if len(future_a) >= 10 else None,
                "future_h10_iou_fallback": float(np.mean(future_b[:10])) if len(future_b) >= 10 else None,
                "delta_commit_minus_fallback_h10": (float(np.mean(np.asarray(future_a[:10]) - np.asarray(future_b[:10])))
                                                       if len(future_a) >= 10 else None),
                "future_h30_iou_commit": float(np.mean(future_a)) if len(future_a) >= 30 else None,
                "future_h30_iou_fallback": float(np.mean(future_b)) if len(future_b) >= 30 else None,
                "delta_commit_minus_fallback_h30": (float(np.mean(np.asarray(future_a) - np.asarray(future_b)))
                                                       if len(future_a) >= 30 else None),
                "low_iou_count_commit_h30": sum(x < 0.1 for x in future_a) if len(future_a) >= 30 else None,
                "low_iou_count_fallback_h30": sum(x < 0.1 for x in future_b) if len(future_b) >= 30 else None,
                "recovery_frame_commit": first_recovery(future_a),
                "recovery_frame_fallback": first_recovery(future_b),
                "future_iou_commit": future_a, "future_iou_fallback": future_b,
            }
            rows.append(row)
            with jsonl.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            log("EVENT {}/{} {}:{} action={} state_delta={:.4f} d10={} d30={}".format(
                si, len(dataset), seq.name, event_frame, action["action"], state_delta,
                row["delta_commit_minus_fallback_h10"], row["delta_commit_minus_fallback_h30"]))

    summary = summarize(rows)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    columns = [k for k, v in rows[0].items() if not isinstance(v, list)]
    with (OUT / "paired_event_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{k: v for k, v in r.items() if k in columns} for r in rows])
    manifest.update({"status": "DIAGNOSTIC-COMPLETE", "event_count": len(rows),
                     "elapsed_seconds": time.time() - t0,
                     "outputs": ["selected_events.json", "paired_event_effects.csv", "paired_event_effects.jsonl", "metrics.json", "run.log"]})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log("FINISH events={} sequences={} elapsed={:.1f}s".format(
        summary["events"], summary["sequences"], manifest["elapsed_seconds"]))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
