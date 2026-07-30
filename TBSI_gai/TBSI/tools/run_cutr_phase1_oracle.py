import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CUTR phase-1 oracle evaluation.")
    parser.add_argument("--config", default="vitb_256_tbsi_32x1_1e4_lasher_15ep_sot",
                        help="Tracker config yaml name.")
    parser.add_argument("--dataset", default="mini_lasher_test",
                        help="Evaluation dataset name.")
    parser.add_argument("--runid", default="cutr_oracle_phase1_sotpretrain",
                        help="Result suffix used to isolate outputs.")
    parser.add_argument("--checkpoint", default="",
                        help="Checkpoint override. Defaults to the available SOT pretrained model.")
    parser.add_argument("--eta", type=float, default=0.10,
                        help="RGB/TIR action step size.")
    parser.add_argument("--gamma", type=float, default=0.08,
                        help="Temporal action step size.")
    parser.add_argument("--prior-sigma", type=float, default=0.35,
                        help="Center prior sigma for the temporal action.")
    parser.add_argument("--threads", type=int, default=1,
                        help="Tracking threads. Use 1 for safety.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs exposed to the test runner.")
    parser.add_argument("--sequence", default="",
                        help="Optional single-sequence smoke test.")
    return parser.parse_args()


def aggregate_oracle_logs(log_root: Path):
    frame_counter = Counter()
    sequence_summary = {}
    total_frames = 0

    for path in sorted(log_root.glob("*.jsonl")):
        seq_counter = Counter()
        best_ious = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                action = record["selected_action"]
                seq_counter[action] += 1
                frame_counter[action] += 1
                best_ious.append(float(record["best_iou"]))
                total_frames += 1
        if best_ious:
            sequence_summary[path.stem] = {
                "frames": len(best_ious),
                "mean_best_iou": sum(best_ious) / len(best_ious),
                "action_counts": dict(seq_counter),
            }

    return {
        "total_frames": total_frames,
        "action_counts": dict(frame_counter),
        "action_ratios": {
            k: (v / total_frames if total_frames > 0 else 0.0)
            for k, v in frame_counter.items()
        },
        "sequence_summary": sequence_summary,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    tracking_root = repo_root / "tracking"
    output_root = repo_root / "output" / "diagnostics" / "cutr_phase1" / args.runid
    oracle_log_root = output_root / "oracle_logs"
    output_root.mkdir(parents=True, exist_ok=True)
    oracle_log_root.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint.strip()
    if not checkpoint:
        checkpoint = str(repo_root / "pretrained_models" / "TBSITrack_SOT.pth.tar")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TBSI_ORACLE_MODE"] = "1"
    env["TBSI_ORACLE_ETA"] = str(args.eta)
    env["TBSI_ORACLE_GAMMA"] = str(args.gamma)
    env["TBSI_ORACLE_PRIOR_SIGMA"] = str(args.prior_sigma)
    env["TBSI_ORACLE_LOG_ROOT"] = str(oracle_log_root)
    env["TBSI_CHECKPOINT_OVERRIDE"] = checkpoint

    test_cmd = [
        sys.executable, "tracking/test.py", "tbsi_track", args.config,
        "--runid", args.runid,
        "--dataset_name", args.dataset,
        "--threads", str(args.threads),
        "--num_gpus", str(args.num_gpus),
    ]
    if args.sequence:
        test_cmd.extend(["--sequence", args.sequence])

    test_log = output_root / "test_stdout.log"
    with test_log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(test_cmd, cwd=str(repo_root), env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"Tracking failed, see {test_log}")

    analysis_log = output_root / "analysis_stdout.log"
    if args.sequence:
        analysis_log.write_text("Skipped full-dataset analysis for single-sequence smoke test.\n", encoding="utf-8")
    else:
        analysis_cmd = [
            sys.executable, "tracking/analysis_results.py",
            "--tracker_name", "tbsi_track",
            "--tracker_param", args.config,
            "--dataset_name", args.dataset,
            "--runid", args.runid,
        ]
        with analysis_log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(analysis_cmd, cwd=str(repo_root), env=env, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise SystemExit(f"Analysis failed, see {analysis_log}")

    summary = {
        "config": args.config,
        "dataset": args.dataset,
        "runid": args.runid,
        "checkpoint": checkpoint,
        "eta": args.eta,
        "gamma": args.gamma,
        "prior_sigma": args.prior_sigma,
        "sequence": args.sequence or None,
        "oracle_log_root": str(oracle_log_root),
    }
    summary.update(aggregate_oracle_logs(oracle_log_root))

    summary_path = output_root / "oracle_action_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "status": "ok",
        "test_log": str(test_log),
        "analysis_log": str(analysis_log),
        "summary_path": str(summary_path),
        "total_frames": summary["total_frames"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
