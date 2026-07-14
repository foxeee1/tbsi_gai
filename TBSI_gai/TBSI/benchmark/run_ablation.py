"""
Ablation experiments for post-fusion temporal token stage 2.
Each: train → fast test via API (100-seq subset) → report.
"""
import os, sys, json, time, subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PRJ = os.path.join(os.path.dirname(__file__), "..")
LOG_DIR = os.path.join(PRJ, "output/logs")
os.makedirs(LOG_DIR, exist_ok=True)

EXPERIMENTS = [
    {
        "name": "A_8ep",
        "config": "vitb_256_tbsi_ablation_8ep",
        "epochs": 8,
        "desc": "More epochs (8ep) — test convergence",
    },
    {
        "name": "B_bn128",
        "config": "vitb_256_tbsi_ablation_bn128",
        "epochs": 3,
        "desc": "Bottleneck 128 — test info loss",
    },
    {
        "name": "C_rs03",
        "config": "vitb_256_tbsi_ablation_rs03",
        "epochs": 3,
        "desc": "Residual init 0.3 — test if 0.1 too conservative",
    },
]

# Also need baseline 4ep_stage2 full result for comparison
BASELINES = {
    "sprint_da_ch_full": 55.83,   # DA baseline (4ep)
    "4ep_stage2_full": 54.94,     # stage2 full test (already done)
}

def run_cmd(cmd, logfile=None):
    """Run command, tee to logfile, return success."""
    print(f"  RUN: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PRJ)
    if logfile:
        with open(logfile, 'w') as f:
            f.write(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return False
    return result.stdout

def evaluate_fast(config_name):
    """Evaluate on 100-seq subset via API, return AUC."""
    from benchmark.bm_evaluate import evaluate
    metrics = evaluate(
        tracker_name="tbsi_track",
        config_name=config_name,
        dataset_name="lasher_test",
        subset_size=100,
        threads=0,
        num_gpus=1,
        force_eval=True,
        method="api",
    )
    return metrics.get("SR", 0.0)

def main():
    results = {}

    for exp in EXPERIMENTS:
        name = exp["name"]
        config = exp["config"]
        epochs = exp["epochs"]
        desc = exp["desc"]

        print(f"\n{'='*60}")
        print(f"EXP {name}: {desc}")
        print(f"  config={config}, epochs={epochs}")
        print(f"{'='*60}")

        # Step 1: Clean checkpoints
        ckpt_dir = f"output/checkpoints/train/tbsi_track/{config}"
        if os.path.exists(ckpt_dir):
            import shutil
            shutil.rmtree(ckpt_dir)
            print(f"  Cleaned: {ckpt_dir}")

        # Step 2: Train
        print(f"  Training {epochs} epochs...")
        train_log = f"{LOG_DIR}/train_{name}.log"
        start = time.time()
        cmd = [sys.executable, "tracking/train.py",
               "--script", "tbsi_track",
               "--config", config,
               "--save_dir", "./output",
               "--mode", "single"]
        out = run_cmd(cmd, train_log)
        if not out:
            print(f"  SKIP {name}: training failed")
            results[name] = 0.0
            continue
        train_time = (time.time() - start) / 60
        print(f"  Training done: {train_time:.1f} min")

        # Step 3: Fast test (100 seq)
        print(f"  Fast testing (100 seq)...")
        test_log = f"{LOG_DIR}/test_{name}.log"
        start = time.time()
        try:
            auc = evaluate_fast(config)
            test_time = (time.time() - start) / 60
            print(f"  AUC (100-seq) = {auc:.2f}, test_time={test_time:.1f} min")
            results[name] = auc
        except Exception as e:
            print(f"  Test failed: {e}")
            results[name] = 0.0

        # Save intermediate results
        with open(f"{LOG_DIR}/ablation_results.json", 'w') as f:
            json.dump({"baselines": BASELINES, "results": results}, f, indent=2)

    # Final report
    print(f"\n{'='*60}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Exp':<20} {'AUC(100seq)':<15} {'vs DA baseline':<15}")
    print(f"{'-'*50}")
    for name, auc in results.items():
        vs = auc - BASELINES["sprint_da_ch_full"]
        print(f"{name:<20} {auc:<15.2f} {vs:<+15.2f}")

    # Save
    with open(f"{LOG_DIR}/ablation_results.json", 'w') as f:
        json.dump({"baselines": BASELINES, "results": results}, f, indent=2)
    print(f"\nSaved to {LOG_DIR}/ablation_results.json")

if __name__ == "__main__":
    main()
