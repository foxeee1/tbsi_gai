#!/usr/bin/env python3
"""Evaluate S1 outputs while keeping the internal tracker state on M1 boxes."""

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("TBSI_PYTHON", "/root/autodl-tmp/conda_envs/tbsi/bin/python")
CONFIG = "TBSI_E_MHSA_FULL_4090D_seed42"
CHECKPOINT = ROOT / "output/experiments" / CONFIG / "checkpoints/TBSITrack_ep0015.pth.tar"
CONFIG_PATH = ROOT / "experiments/tbsi_track" / (CONFIG + ".yaml")
PREDICTOR = ROOT / "experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json"
POLICY = ROOT / "experiments/artifacts/m1_readout_independent_20261014.json"
FULL_OUT = ROOT / "output/experiments/M1_S1_OUTPUT_ONLY_STATE_FULL244_20261015"
RESULT_ROOT = ROOT / "output/test/tracking_results/tbsi_track"
EXPECTED_CONFIG_SHA = "fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e"
EXPECTED_CHECKPOINT_SHA = "1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef"
EXPECTED_PREDICTOR_SHA = "b97bf2fe4d576a447f6cabe62a7523a2276f6837ffcc8761714b181119d79c6c"
EXPECTED_POLICY_SHA = "ff4cbef73341eae20c726911f916b25cd80b6a61458bba0a1e5196004840c098"
M1_RUNID = "1030"
S1_RUNID = "1031"
FULL_RUNID = "1044"
PREFLIGHT_RUNID = "1047"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_stage(command, env, log_path):
    started = time.time()
    with Path(log_path).open("w") as stream:
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return time.time() - started


def names_from(path):
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from lib.test.evaluation.lasherdataset import LasHeRDataset

    if (sha(CONFIG_PATH) != EXPECTED_CONFIG_SHA or sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(PREDICTOR) != EXPECTED_PREDICTOR_SHA or sha(POLICY) != EXPECTED_POLICY_SHA):
        raise RuntimeError("protected M1 config/checkpoint or frozen S1 artifact hash mismatch")
    if not Path(PYTHON).is_file():
        raise RuntimeError("Python executable not found: {}".format(PYTHON))
    if not (RESULT_ROOT / (CONFIG + "_" + M1_RUNID)).is_dir() or not (RESULT_ROOT / (CONFIG + "_" + S1_RUNID)).is_dir():
        raise RuntimeError("matched M1/S1 parent prediction directories are missing")

    dataset = LasHeRDataset("testingset")
    all_sequences = list(dataset.sequence_list)
    if len(all_sequences) != 244 or len(set(all_sequences)) != 244:
        raise RuntimeError("expected 244 official sequences")
    run_id = PREFLIGHT_RUNID if args.preflight else FULL_RUNID
    out = FULL_OUT / "preflight_1047" if args.preflight else FULL_OUT / "full244"
    result_dir = RESULT_ROOT / (CONFIG + "_" + run_id)
    if (out.exists() and any(out.iterdir())) or result_dir.exists():
        raise RuntimeError("refusing to overwrite existing output: {} or {}".format(out, result_dir))
    sequences = all_sequences[:1] if args.preflight else all_sequences
    out.mkdir(parents=True, exist_ok=True)
    sequence_file = out / "sequence_list.txt"
    sequence_file.write_text("\n".join(sequences) + "\n")

    import platform
    import torch
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
    manifest = {
        "id": "M1-S1-OUTPUT-ONLY-STATE-PREFLIGHT-20261015" if args.preflight else "M1-S1-OUTPUT-ONLY-STATE-FULL244-20261015",
        "parent": "M1-S1-FULL244-CONFIRM-20261014",
        "hypothesis": "Keeping the internal tracker state on the uncorrected M1 box will remove S1 remaining162 regressions if they are mainly caused by state feedback, while preserving S1 as the reported output.",
        "one_changed_variable": "state commit: corrected S1 output is returned, while raw M1 box updates self.state for the next crop",
        "seed": 42, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
        "config": str(CONFIG_PATH), "config_sha256": sha(CONFIG_PATH),
        "predictor": str(PREDICTOR), "predictor_sha256": sha(PREDICTOR),
        "policy": str(POLICY), "policy_sha256": sha(POLICY),
        "tracker_code": str(ROOT / "lib/test/tracker/tbsi_track.py"),
        "tracker_code_sha256": sha(ROOT / "lib/test/tracker/tbsi_track.py"),
        "prechange_snapshot": str(ROOT / "experiments/snapshots/tbsi_track_before_s1_output_only_state_20261015.py"),
        "prechange_snapshot_sha256": sha(ROOT / "experiments/snapshots/tbsi_track_before_s1_output_only_state_20261015.py"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT.parents[1])).decode().strip(),
        "dataset": "LasHeR official testingset; excludes repository SKIP_SEQUENCES",
        "dataset_path": dataset.base_path, "sequence_manifest": str(sequence_file),
        "sequence_manifest_sha256": sha(sequence_file), "sequences": len(sequences),
        "scope": "single-sequence contract preflight" if args.preflight else "official full-244 tracking",
        "precision": "FP32", "training": "none; frozen train-side S1 predictor and policy",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda, "gpu": gpu,
                         "workers": 0, "tracker_processes": 1},
        "baseline_prediction_dirs": {"M1": str(RESULT_ROOT / (CONFIG + "_" + M1_RUNID)),
                                      "S1": str(RESULT_ROOT / (CONFIG + "_" + S1_RUNID))},
        "status": "RUNNING", "stages": {},
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    env = os.environ.copy()
    env.update({"TBSI_RPP_TEST_LIST": str(sequence_file), "TBSI_INFER_FP16": "0",
                "TBSI_CHECKPOINT_OVERRIDE": str(CHECKPOINT), "TBSI_BOUNDED_READOUT_PREDICTOR": str(PREDICTOR),
                "TBSI_READOUT_POLICY": str(POLICY), "TBSI_READOUT_STATE_MODE": "output_only",
                "TBSI_READOUT_AUDIT_DIR": str(out / "action_audit"),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    if args.preflight:
        env["TBSI_M1_TRACE_DIR"] = str(out / "trace")
    command = [PYTHON, "-u", "tracking/test.py", "tbsi_track", CONFIG,
               "--runid", run_id, "--dataset_name", "rpp_lasher_test",
               "--threads", "0", "--num_gpus", "1"]
    manifest["command"] = command
    manifest["stages"]["tracking"] = {"status": "RUNNING", "log": str(out / "tracking.log"),
                                        "result_dir": str(result_dir)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("START state=output_only sequences={} runid={}".format(len(sequences), run_id), flush=True)
    elapsed = run_stage(command, env, out / "tracking.log")
    found = {p.stem for p in result_dir.glob("*.txt") if not p.name.endswith("_time.txt")}
    if found != set(sequences):
        raise RuntimeError("prediction coverage mismatch: expected {}, got {}".format(len(sequences), len(found)))
    manifest["stages"]["tracking"].update({"status": "COMPLETE", "seconds": elapsed,
                                             "prediction_files": len(found)})
    manifest["status"] = "PREFLIGHT-COMPLETE" if args.preflight else "TRACKING-COMPLETE"

    if args.preflight:
        seq = sequences[0]
        old_s1 = RESULT_ROOT / (CONFIG + "_" + S1_RUNID) / (seq + ".txt")
        new_out = result_dir / (seq + ".txt")
        old_s1_rows = old_s1.read_text().splitlines()
        new_rows = new_out.read_text().splitlines()
        if len(old_s1_rows) < 2 or len(new_rows) < 2:
            raise RuntimeError("preflight result files must include initialization and one tracked frame")
        old_first_tracked = old_s1_rows[1].strip()
        new_first_tracked = new_rows[1].strip()
        trace_file = out / "trace" / seq / "frames.jsonl"
        trace_rows = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
        if len(trace_rows) < 2:
            raise RuntimeError("trace did not capture two frames for state-commit check")
        if old_first_tracked != new_first_tracked:
            raise RuntimeError("first tracked reported box differs from S1 coupled control")
        baseline_rows = np.loadtxt(RESULT_ROOT / (CONFIG + "_" + M1_RUNID) / (seq + ".txt"), dtype=float, ndmin=2)
        if len(baseline_rows) < 2:
            raise RuntimeError("M1 result file must include initialization and one tracked frame")
        baseline_first = baseline_rows[1]
        internal_first = np.asarray(trace_rows[0]["tracker_state_after"], dtype=float)
        internal_second_before = np.asarray(trace_rows[1]["tracker_state_before"], dtype=float)
        # Tracker result files quantize xywh to integer pixels; compare at that precision.
        if not np.allclose(internal_first, baseline_first[:4], rtol=0, atol=1.0):
            raise RuntimeError("first internal committed state does not match quantized M1 first output")
        if not np.allclose(internal_second_before, baseline_first[:4], rtol=0, atol=1.0):
            raise RuntimeError("second-frame crop state does not follow quantized M1 first output")
        manifest["contract_check"] = {"sequence": seq, "first_tracked_reported_box_matches_s1": True,
                                       "first_internal_state_matches_m1": True,
                                       "second_frame_state_before_matches_m1_first_box": True,
                                       "comparison_precision": "saved tracker outputs are integer-quantized; internal float state allowed within 1 pixel",
                                       "traced_frames": len(trace_rows)}
        manifest["status"] = "PREFLIGHT-PASSED"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest["contract_check"], indent=2), flush=True)
        return

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("FINISH output-only tracking seconds={} predictions={}".format(elapsed, len(found)), flush=True)

    from lib.test.analysis.extract_results import extract_results
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    os.environ["TBSI_RPP_TEST_LIST"] = str(sequence_file)
    official_dataset = get_dataset("rpp_lasher_test")
    trackers = [Tracker("tbsi_track", CONFIG, "rpp_lasher_test", M1_RUNID),
                Tracker("tbsi_track", CONFIG, "rpp_lasher_test", S1_RUNID),
                Tracker("tbsi_track", CONFIG, "rpp_lasher_test", FULL_RUNID)]
    report_name = "M1_S1_OUTPUT_ONLY_STATE_FULL244_20261015"
    eval_data = extract_results(trackers, official_dataset, report_name, skip_missing_seq=False)
    overlap = np.asarray(eval_data["ave_success_rate_plot_overlap"], dtype=np.float64)
    center = np.asarray(eval_data["ave_success_rate_plot_center"], dtype=np.float64)
    center_norm = np.asarray(eval_data["ave_success_rate_plot_center_norm"], dtype=np.float64)
    metric_arrays = {"AUC": overlap.mean(axis=2) * 100.0, "PR": center[:, :, 20] * 100.0,
                     "NPR": center_norm[:, :, 20] * 100.0, "OP50": overlap[:, :, 10] * 100.0,
                     "OP75": overlap[:, :, 15] * 100.0}
    names = eval_data["sequences"]
    if len(names) != 244 or overlap.shape[:2] != (244, 3):
        raise RuntimeError("official evaluator did not return 244 sequences x 3 arms")
    arm_labels = ["M1", "S1", "S1_OUTPUT_ONLY"]
    rows = []
    for i, name in enumerate(names):
        row = {"sequence": name}
        for metric, values in metric_arrays.items():
            for j, arm in enumerate(arm_labels):
                row[arm + "_" + metric] = float(values[i, j])
            row["OUTPUT_ONLY_MINUS_M1_" + metric] = float(values[i, 2] - values[i, 0])
            row["OUTPUT_ONLY_MINUS_S1_" + metric] = float(values[i, 2] - values[i, 1])
        rows.append(row)
    with (out / "per_sequence_official.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    low_file = ROOT / "output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/s1_full62/sequence_list.txt"
    normal_file = ROOT / "output/experiments/M1_SINGLE_SIGNAL_NORMAL20_20261012/s1_full20/sequence_list.txt"
    low, normal = names_from(low_file), names_from(normal_file)
    name_set = set(names)
    if len(low) != 62 or len(normal) != 20 or low & normal or not (low | normal) <= name_set:
        raise RuntimeError("frozen low-AUC/normal cohort contract mismatch")
    cohorts = {"low_auc62": low, "normal20_sentinel": normal,
               "remaining162": name_set - low - normal, "all244": name_set}
    summaries = {"n_sequences": len(names), "metrics": {}, "cohorts": {}}
    for metric, values in metric_arrays.items():
        summaries["metrics"][metric] = {}
        for j, arm in enumerate(arm_labels):
            summaries["metrics"][metric][arm] = float(values[:, j].mean())
        summaries["metrics"][metric]["output_only_minus_m1"] = float((values[:, 2] - values[:, 0]).mean())
        summaries["metrics"][metric]["output_only_minus_s1"] = float((values[:, 2] - values[:, 1]).mean())
        delta = values[:, 2] - values[:, 1]
        summaries["metrics"][metric]["output_only_vs_s1_wins_losses_ties"] = [
            int((delta > 1e-8).sum()), int((delta < -1e-8).sum()), int((np.abs(delta) <= 1e-8).sum())]
    rng = np.random.RandomState(3407)
    for cohort, members in cohorts.items():
        ids = [i for i, name in enumerate(names) if name in members]
        entry = {"n_sequences": len(ids), "metrics": {}}
        for metric, values in metric_arrays.items():
            diff_m1 = values[ids, 2] - values[ids, 0]
            diff_s1 = values[ids, 2] - values[ids, 1]
            boot = diff_s1[rng.randint(0, len(ids), size=(10000, len(ids)))].mean(axis=1)
            entry["metrics"][metric] = {
                "M1": float(values[ids, 0].mean()), "S1": float(values[ids, 1].mean()),
                "S1_OUTPUT_ONLY": float(values[ids, 2].mean()),
                "output_only_minus_m1": float(diff_m1.mean()), "output_only_minus_s1": float(diff_s1.mean()),
                "output_only_minus_s1_sequence_bootstrap_95ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            }
        summaries["cohorts"][cohort] = entry
    summaries["official_eval_data"] = str(ROOT / "output/test/result_plots" / report_name / "eval_data.pkl")
    (out / "official_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    (out / "stages" ).mkdir(exist_ok=True)
    audit_dir = out / "persistence_audit"
    audit_command = [PYTHON, "-u", "tools/m1_official_metric_and_persistence_audit.py", "--repo", str(ROOT),
                     "--pred-dir", str(result_dir), "--baseline-dir", str(RESULT_ROOT / (CONFIG + "_" + M1_RUNID)),
                     "--dataset", "rpp_lasher_test", "--output-dir", str(audit_dir)]
    audit_env = os.environ.copy()
    audit_env["TBSI_RPP_TEST_LIST"] = str(sequence_file)
    elapsed_audit = run_stage(audit_command, audit_env, out / "persistence_audit.log")
    manifest["stages"]["official_evaluation"] = {"status": "COMPLETE", "sequences": len(names),
                                                   "summary": str(out / "official_summary.json")}
    manifest["stages"]["persistence_audit"] = {"status": "COMPLETE", "seconds": elapsed_audit,
                                                 "output": str(audit_dir)}
    manifest["status"] = "COMPLETE / ANALYSIS-RECORDED"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries["metrics"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
