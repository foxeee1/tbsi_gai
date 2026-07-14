"""
实验1: LayerDA — TBSILayer内逐层退化调制 (L1 Sprint, ~50min)
实验2: PixelDA — 像素级质量估计替代全局池化 (L1 Sprint, ~50min)
顺序运行，实验结果对比
"""
import os, sys, json, time, subprocess
PROJECT = "/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON = "/root/autodl-tmp/conda_envs/tbsi/bin/python"
LOGDIR = os.path.join(PROJECT, "benchmark", "ledgers")
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
os.environ["OMP_NUM_THREADS"] = ""; os.environ["PYTHONPATH"] = f"{PROJECT}:{os.environ.get('PYTHONPATH', '')}"
os.makedirs(LOGDIR, exist_ok=True)
log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_cmd(cmd, desc, log_name):
    log(f"{desc} 开始..."); t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT)
    t = (time.time()-t0)/60
    with open(os.path.join(LOGDIR, log_name), "w") as f:
        f.write(r.stdout[-2000:] + "\n---STDERR---\n" + r.stderr[-1000:])
    if r.returncode != 0: log(f"[✗] {desc} 失败 ({t:.0f}min)"); return False
    log(f"[✓] {desc} 完成 ({t:.0f}min)"); return True

def run_eval(config_name, desc):
    log(f"{desc} 评测...")
    r = subprocess.run([PYTHON, "-c", f"""
import sys, json; sys.path.insert(0, '{PROJECT}')
from benchmark.bm_evaluate import evaluate_level
from benchmark.bm_config import LEVEL1; from collections import OrderedDict
LEVEL1['config_name'] = '{config_name}'; LEVEL1['test_subset_size'] = 100
m = evaluate_level(LEVEL1, threads=4, num_gpus=1, force_eval=True)
print('METRICS:', json.dumps(m))
"""], capture_output=True, text=True)
    with open(os.path.join(LOGDIR, f"eval_{config_name}.log"), "w") as f:
        f.write(r.stdout + "\n---STDERR---\n" + r.stderr[:500])
    for line in r.stdout.split("\n"):
        if line.startswith("METRICS:"):
            m = json.loads(line[8:]); log(f"[✓] {desc}: SR={m.get('SR','?')} PR={m.get('PR','?')} NPR={m.get('NPR','?')}")
            return m
    log(f"[✗] {desc} 评测失败"); return None

results = {}

# ====== 实验1: LayerDA ======
log("="*55); log("实验1: LayerDA — 层内退化调制 (L1 Sprint)"); log("="*55)
t1 = time.time()
if run_cmd(f"unset OMP_NUM_THREADS && {PYTHON} lib/train/run_training.py --script tbsi_track --config vitb_256_tbsi_sprint_layerda --save_dir ./output",
           "LayerDA 训练", "train_layerda.log"):
    results['layerda'] = run_eval("vitb_256_tbsi_sprint_layerda", "LayerDA")
log(f"LayerDA 总耗时: {(time.time()-t1)/60:.0f}min")

# ====== 实验2: PixelDA ======
log(""); log("="*55); log("实验2: PixelDA — 像素级质量估计 (L1 Sprint)"); log("="*55)
t2 = time.time()
if run_cmd(f"unset OMP_NUM_THREADS && {PYTHON} lib/train/run_training.py --script tbsi_track --config vitb_256_tbsi_sprint_pixelda --save_dir ./output",
           "PixelDA 训练", "train_pixelda.log"):
    results['pixelda'] = run_eval("vitb_256_tbsi_sprint_pixelda", "PixelDA")
log(f"PixelDA 总耗时: {(time.time()-t2)/60:.0f}min")

# ====== 对比 ======
log(""); log("="*55); log("  实验对比结果"); log("="*55)
r1 = results.get('layerda', {}); r2 = results.get('pixelda', {})
log(f"{'指标':<15} {'原始3.2':>10} {'LayerDA':>10} {'PixelDA':>10}")
log(f"{'-'*15:<15} {'-':>10} {'-':>10} {'-':>10}")
log(f"{'SR':<15} {'56.37':>10} {str(r1.get('SR','?') if r1 else '?'):>10} {str(r2.get('SR','?') if r2 else '?'):>10}")
log(f"{'PR':<15} {'71.22':>10} {str(r1.get('PR','?') if r1 else '?'):>10} {str(r2.get('PR','?') if r2 else '?'):>10}")
log(f"{'NPR':<15} {'67.27':>10} {str(r1.get('NPR','?') if r1 else '?'):>10} {str(r2.get('NPR','?') if r2 else '?'):>10}")
comp = {"experiment": "LayerDA vs PixelDA", "layerda": r1, "pixelda": r2,
        "baseline_32": {"SR": 56.37, "PR": 71.22, "NPR": 67.27}}
with open(os.path.join(LOGDIR, "da_experiments_comparison.json"), "w") as f:
    json.dump(comp, f, indent=2, default=str)
log(""); log(f"对比保存: {LOGDIR}/da_experiments_comparison.json"); log("="*55)
