"""
方向A/方向C 对比实验运行器
自动: Medium-Channel 训练评测 → Medium-Compensate 训练评测 → 对比报告
"""
import os, sys, json, time, subprocess

PROJECT = "/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON = "/root/autodl-tmp/conda_envs/tbsi/bin/python"
LOGDIR = os.path.join(PROJECT, "benchmark", "ledgers")
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
os.environ["OMP_NUM_THREADS"] = ""
os.environ["PYTHONPATH"] = f"{PROJECT}:{os.environ.get('PYTHONPATH', '')}"
os.makedirs(LOGDIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_cmd(cmd, desc, log_name):
    log(f"{desc} 开始...")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT)
    elapsed = (time.time() - t0) / 60
    path = os.path.join(LOGDIR, log_name)
    with open(path, "w") as f:
        f.write(result.stdout[-3000:] + "\n---STDERR---\n" + result.stderr[-1000:])
    if result.returncode != 0:
        log(f"[✗] {desc} 失败 ({elapsed:.0f}min): {result.stderr[-200:]}")
        return False
    log(f"[✓] {desc} 完成 ({elapsed:.0f}min)")
    return True

def run_eval(config_name, desc):
    log(f"{desc} 评测...")
    r = subprocess.run([PYTHON, "-c", f"""
import sys, json
sys.path.insert(0, '{PROJECT}')
from benchmark.bm_evaluate import evaluate_level
from benchmark.bm_config import MEDIUM
from collections import OrderedDict
MEDIUM['config_name'] = '{config_name}'
MEDIUM['test_subset_size'] = 100
m = evaluate_level(MEDIUM, threads=4, num_gpus=1, force_eval=True)
print('METRICS:', json.dumps(m))
"""], capture_output=True, text=True)
    path = os.path.join(LOGDIR, f"eval_{config_name}.log")
    with open(path, "w") as f:
        f.write(r.stdout + "\n---STDERR---\n" + r.stderr[:500])
    for line in r.stdout.split("\n"):
        if line.startswith("METRICS:"):
            metrics = json.loads(line[8:])
            log(f"[✓] {desc}: SR={metrics.get('SR','?')} PR={metrics.get('PR','?')} NPR={metrics.get('NPR','?')}")
            return metrics
    log(f"[✗] {desc} 评测失败")
    return None

# ====== 实验1: Medium-Channel (方向A) ======
log("=" * 55)
log("实验1: Medium-Channel (方向A) - 10ep x 36k")
log("=" * 55)
t1 = time.time()
ok = run_cmd(
    f"unset OMP_NUM_THREADS && {PYTHON} lib/train/run_training.py --script tbsi_track --config vitb_256_tbsi_medium_channel --save_dir ./output",
    "Channel 训练", "train_medium_channel.log"
)
if not ok:
    log("Channel 训练失败，尝试重置检查点...")
    import shutil
    shutil.rmtree(os.path.join(PROJECT, "output", "checkpoints"), ignore_errors=True)
    ok = run_cmd(
        f"unset OMP_NUM_THREADS && {PYTHON} lib/train/run_training.py --script tbsi_track --config vitb_256_tbsi_medium_channel --save_dir ./output",
        "Channel 训练 (重试)", "train_medium_channel_retry.log"
    )
if not ok: sys.exit(1)
m1 = run_eval("vitb_256_tbsi_medium_channel", "Channel")
if not m1: sys.exit(1)
log(f"Channel 总耗时: {(time.time()-t1)/60:.0f}min")

# ====== 实验2: Medium-Compensate (方向C) ======
log("")
log("=" * 55)
log("实验2: Medium-Compensate (方向C) - 10ep x 36k")
log("=" * 55)
t2 = time.time()
ok = run_cmd(
    f"unset OMP_NUM_THREADS && {PYTHON} lib/train/run_training.py --script tbsi_track --config vitb_256_tbsi_medium_compensate --save_dir ./output",
    "Compensate 训练", "train_medium_compensate.log"
)
if not ok: sys.exit(1)
m2 = run_eval("vitb_256_tbsi_medium_compensate", "Compensate")
if not m2: sys.exit(1)
log(f"Compensate 总耗时: {(time.time()-t2)/60:.0f}min")

# ====== 对比报告 ======
log("")
log("=" * 55)
log("  方向A vs 方向C 对比结果")
log("=" * 55)
log(f"")
log(f"{'指标':<15} {'方向A(Channel)':>15} {'方向C(Compensate)':>15}")
log(f"{'-'*15:<15} {'-':>15} {'-':>15}")
log(f"{'SR':<15} {m1.get('SR','?'):>15} {m2.get('SR','?'):>15}")
log(f"{'OP50':<15} {m1.get('OP50','?'):>15} {m2.get('OP50','?'):>15}")
log(f"{'OP75':<15} {m1.get('OP75','?'):>15} {m2.get('OP75','?'):>15}")
log(f"{'MeanIoU':<15} {m1.get('MeanIoU','?'):>15} {m2.get('MeanIoU','?'):>15}")
log(f"{'PR':<15} {m1.get('PR','?'):>15} {m2.get('PR','?'):>15}")
log(f"{'NPR':<15} {m1.get('NPR','?'):>15} {m2.get('NPR','?'):>15}")
log(f"")
log(f"Baseline (L1 Sprint, load时 SOT): SR=54.72, PR=68.95, NPR=65.02")
log(f"原始 3.2 (L1 Sprint): SR=56.37, PR=71.22, NPR=67.27")
log(f"")

# Save comparison
comparison = {
    "experiment": "3.2 DA Modes Comparison (Medium: 10ep x 36k)",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "Direction_A_Channel": m1,
    "Direction_C_Compensate": m2,
    "baseline": {"SR": 54.72, "PR": 68.95, "NPR": 65.02},
    "original_32_spatial": {"SR": 56.37, "PR": 71.22, "NPR": 67.27},
}
with open(os.path.join(LOGDIR, "da_modes_comparison.json"), "w") as f:
    json.dump(comparison, f, indent=2, default=str)
log(f"对比结果已保存: {LOGDIR}/da_modes_comparison.json")
log("=" * 55)
