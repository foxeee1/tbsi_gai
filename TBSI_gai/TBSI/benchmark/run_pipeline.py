"""TBSI 三级流水线基线：全自动 L1→L2→L3 (IN1K 权重)
- 配置: vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k
- L1/L2 使用 TBSITrack_IN1K.pth.tar 微调, SOT_PRETRAIN=False (全模型训练)
- L3 使用原始 IN1K 配置 (deit backbone 预训练)
"""
import os, sys, json, time, subprocess, shutil, yaml
from datetime import datetime

PROJECT = "/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON = "/root/autodl-tmp/conda_envs/tbsi/bin/python"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
os.environ["OMP_NUM_THREADS"] = ""
os.environ["PYTHONPATH"] = f"{PROJECT}:{os.environ.get('PYTHONPATH', '')}"

LOGDIR = os.path.join(PROJECT, "benchmark", "ledgers")
os.makedirs(LOGDIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGFILE = os.path.join(LOGDIR, f"pipeline_{TIMESTAMP}.log")

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")

def update_config(config_name, spe=None, val_spe=None, epochs=None, bs=None):
    """更新 YAML 配置。不修改 batch_size 以外的训练超参。"""
    path = os.path.join(PROJECT, "experiments", "tbsi_track", f"{config_name}.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    changes = []
    if spe:
        old = cfg["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"]
        cfg["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"] = spe
        changes.append(f"SPE={old}→{spe}")
    if val_spe:
        old = cfg["DATA"]["VAL"]["SAMPLE_PER_EPOCH"]
        cfg["DATA"]["VAL"]["SAMPLE_PER_EPOCH"] = val_spe
        changes.append(f"VAL_SPE={old}→{val_spe}")
    if epochs:
        old = cfg["TRAIN"]["EPOCH"]
        cfg["TRAIN"]["EPOCH"] = epochs
        changes.append(f"EP={old}→{epochs}")
    if bs:
        old = cfg["TRAIN"]["BATCH_SIZE"]
        cfg["TRAIN"]["BATCH_SIZE"] = bs
        changes.append(f"BS={old}→{bs}")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    if changes:
        log(f"  {config_name}: {', '.join(changes)}")

def run_training(config_name, desc):
    """运行训练"""
    log(f"{desc} 开始...")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, "lib/train/run_training.py",
         "--script", "tbsi_track", "--config", config_name,
         "--save_dir", os.path.join(PROJECT, "output")],
        capture_output=True, text=True, cwd=PROJECT
    )
    elapsed = (time.time() - t0) / 60
    # Save log tail
    log_path = os.path.join(LOGDIR, f"train_{config_name}_{TIMESTAMP}.log")
    with open(log_path, "w") as f:
        f.write(result.stdout[-3000:] + "\n---STDERR---\n" + result.stderr[-2000:])
    if result.returncode != 0:
        log(f"[✗] {desc} 失败 ({elapsed:.0f}min): {result.stderr[-300:]}")
        return False
    log(f"[✓] {desc} 完成 ({elapsed:.0f}min)")
    return True

def run_eval(config_name, level_cfg, desc):
    """运行评测"""
    log(f"{desc} 评测...")
    subset = level_cfg.get("test_subset_size")
    r = subprocess.run([PYTHON, "-c", f"""
import sys; sys.path.insert(0,'{PROJECT}')
from benchmark.bm_evaluate import evaluate_level
from benchmark.bm_config import LEVEL{level_cfg['level']}
m = evaluate_level(LEVEL{level_cfg['level']}, threads=6, num_gpus=1, force_eval=True)
print('METRICS:', __import__('json').dumps(m))
"""], capture_output=True, text=True)
    log_path = os.path.join(LOGDIR, f"eval_{config_name}_{TIMESTAMP}.log")
    with open(log_path, "w") as f:
        f.write(r.stdout + "\n---STDERR---\n" + r.stderr[-1000:])
    for line in r.stdout.split("\n"):
        if line.startswith("METRICS:"):
            metrics = json.loads(line[8:])
            log(f"[✓] {desc}: SR={metrics.get('SR','?')} PR={metrics.get('PR','?')} "
                f"NPR={metrics.get('NPR','?')}")
            return metrics
    log(f"[✗] {desc} 评测失败: {r.stderr[-200:]}")
    return None

# =========================================
# Pipeline
# =========================================
log("=" * 55)
log("TBSI 三级流水线基线 (IN1K 权重, bs=32)")
log(f"开始: {datetime.now().isoformat()}")
log("=" * 55)

# 清理之前的测试缓存
for p in [os.path.join(PROJECT, "output", "test")]:
    if os.path.exists(p): shutil.rmtree(p)

# ========== LEVEL 1: SPRINT ==========
log("\n" + "=" * 55)
log("LEVEL 1: SPRINT")
log("配置: 4ep × 12000 smpl, bs=32 (数据量 = Full 的 20%)")
log("测试: 50 seq 子集")
log("预计: ~8-10 min")
log("=" * 55)

# L1: 20% 数据量
update_config("vitb_256_tbsi_sprint", spe=12000, val_spe=5000)
t1 = time.time()
if not run_training("vitb_256_tbsi_sprint", "L1 训练"): sys.exit(1)
l1 = run_eval("vitb_256_tbsi_sprint",
              {"config_name": "vitb_256_tbsi_sprint", "test_subset_size": 50, "level": 1},
              "L1")
if not l1: sys.exit(1)
l1_t = (time.time()-t1)/60

# ========== LEVEL 2: VERIFY ==========
log("\n" + "=" * 55)
log("LEVEL 2: VERIFY")
log("配置: 8ep × 30000 smpl, bs=32 (数据量 = Full 的 50%)")
log("测试: 全量 245 seq")
log("预计: ~40-50 min")
log("=" * 55)

# L2: 50% 数据量
update_config("vitb_256_tbsi_verify", spe=30000, val_spe=10000)
t2 = time.time()
if not run_training("vitb_256_tbsi_verify", "L2 训练"): sys.exit(1)
l2 = run_eval("vitb_256_tbsi_verify",
              {"config_name": "vitb_256_tbsi_verify", "test_subset_size": None, "level": 2},
              "L2")
if not l2: sys.exit(1)
l2_t = (time.time()-t2)/60

# ========== LEVEL 3: FULL ==========
log("\n" + "=" * 55)
log("LEVEL 3: FULL")
log("配置: 15ep × 60000 smpl, bs=32 (原始 IN1K 配置)")
log("测试: 全量 245 seq")
log("预计: ~3-4 h")
log("=" * 55)

# L3: 原始配置, 不修改
t3 = time.time()
if not run_training("vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k", "L3 训练"): sys.exit(1)
l3 = run_eval("vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k",
              {"config_name": "vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k", "test_subset_size": None, "level": 3},
              "L3")
if not l3: sys.exit(1)
l3_t = (time.time()-t3)/60

# ========== 保存 ==========
ledger = {
    "session_id": TIMESTAMP,
    "created_at": datetime.now().isoformat(),
    "current_level": 3, "is_completed": True, "total_iterations": 3,
    "baselines": {"level1": l1, "level2": l2, "level3": l3},
    "iterations": [
        {"id": "baseline_l1", "level": 1,
         "description": "[baseline] L1 Sprint (IN1K, 4ep×12k_smpl, bs=32, 50seq)",
         "metrics": l1, "gate_passed": True},
        {"id": "baseline_l2", "level": 2,
         "description": "[baseline] L2 Verify (IN1K, 8ep×30k_smpl, bs=32, 245seq)",
         "metrics": l2, "gate_passed": True},
        {"id": "baseline_l3", "level": 3,
         "description": "[baseline] L3 Full (IN1K, 15ep×60k_smpl, bs=32, 245seq)",
         "metrics": l3, "gate_passed": True},
    ],
    "timing_min": {"L1": round(l1_t), "L2": round(l2_t), "L3": round(l3_t),
                   "total": round(l1_t+l2_t+l3_t)},
}
with open(os.path.join(LOGDIR, "bm_ledger.json"), "w") as f:
    json.dump(ledger, f, indent=2)
log(f"[✓] 基线已保存")

log("\n" + "=" * 55)
log("  三级流水线 全部完成!")
log("=" * 55)
log(f"  L1 Sprint ({l1_t:.0f}min): SR={l1['SR']} PR={l1['PR']} NPR={l1['NPR']}")
log(f"  L2 Verify ({l2_t:.0f}min): SR={l2['SR']} PR={l2['PR']} NPR={l2['NPR']}")
log(f"  L3 Full   ({l3_t:.0f}min): SR={l3['SR']} PR={l3['PR']} NPR={l3['NPR']}")
log(f"  总计: {(l1_t+l2_t+l3_t)/60:.1f}h")
log("=" * 55)
