"""
方向1 (stat): 特征统计自适应校准 — L1 Sprint
方向2 (gauss): 可学习高斯中心先验 — L1 Sprint
方向3 (agate): 注意力输出门控 — L1 Sprint
"""
import os, sys, json, time, subprocess
P = "/root/autodl-tmp/TBSI_gai/TBSI"
PY = "/root/autodl-tmp/conda_envs/tbsi/bin/python"
LD = os.path.join(P, "benchmark", "ledgers")
os.chdir(P); sys.path.insert(0, P)
os.environ["OMP_NUM_THREADS"] = ""; os.environ["PYTHONPATH"] = f"{P}:{os.environ.get('PYTHONPATH','')}"
os.makedirs(LD, exist_ok=True)
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run_cmd(cmd, desc, ln):
    log(f"{desc} 开始..."); t0=time.time()
    r=subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=P)
    t=(time.time()-t0)/60
    with open(os.path.join(LD,ln),"w") as f: f.write(r.stdout[-2000:]+"\n---STDERR---\n"+r.stderr[-1000:])
    if r.returncode!=0: log(f"[✗] {desc} 失败 ({t:.0f}min): {r.stderr[-200:]}"); return False
    log(f"[✓] {desc} 完成 ({t:.0f}min)"); return True

def run_eval(cfg, desc):
    log(f"{desc} 评测...")
    r=subprocess.run([PY,"-c",f"""
import sys,json; sys.path.insert(0,'{P}')
from benchmark.bm_evaluate import evaluate_level
from benchmark.bm_config import LEVEL1; from collections import OrderedDict
LEVEL1['config_name']='{cfg}'; LEVEL1['test_subset_size']=100
m=evaluate_level(LEVEL1, threads=4, num_gpus=1, force_eval=True)
print('METRICS:',json.dumps(m))
"""], capture_output=True, text=True)
    with open(os.path.join(LD,f"eval_{cfg}.log"),"w") as f: f.write(r.stdout+"\n---\n"+r.stderr[:500])
    for l in r.stdout.split("\n"):
        if l.startswith("METRICS:"):
            m=json.loads(l[8:]); log(f"[✓] {desc}: SR={m.get('SR','?')} PR={m.get('PR','?')} NPR={m.get('NPR','?')}")
            return m
    log(f"[✗] {desc} 评测失败"); return None

results={}

for name, cfg in [("D1-stat","vitb_256_tbsi_sprint_d1_stat"),("D2-gauss","vitb_256_tbsi_sprint_d2_gauss"),("D3-agate","vitb_256_tbsi_sprint_d3_agate")]:
    log(""); log("="*55); log(f"{name}: {cfg}"); log("="*55)
    t0=time.time()
    if run_cmd(f"unset OMP_NUM_THREADS && {PY} lib/train/run_training.py --script tbsi_track --config {cfg} --save_dir ./output", f"{name} 训练", f"train_{cfg}.log"):
        results[name]=run_eval(cfg, name)
    log(f"{name} 耗时: {(time.time()-t0)/60:.0f}min")

log(""); log("="*55); log("  方向1/2/3 对比结果"); log("="*55)
log(f"{'指标':<12} {'D1(stat)':>10} {'D2(gauss)':>12} {'D3(agate)':>12} {'原始3.2':>10}")
log(f"{'-'*12:<12} {'-':>10} {'-':>12} {'-':>12} {'-':>10}")
for k in ['SR','PR','NPR']:
    log(f"{k:<12} {str(results.get('D1-stat',{}).get(k,'?') if results.get('D1-stat') else '?'):>10} "
        f"{str(results.get('D2-gauss',{}).get(k,'?') if results.get('D2-gauss') else '?'):>12} "
        f"{str(results.get('D3-agate',{}).get(k,'?') if results.get('D3-agate') else '?'):>12} {'56.37/71.22/67.27':>10}" if k=='SR' else
        f"{k:<12} {str(results.get('D1-stat',{}).get(k,'?') if results.get('D1-stat') else '?'):>10} "
        f"{str(results.get('D2-gauss',{}).get(k,'?') if results.get('D2-gauss') else '?'):>12} "
        f"{str(results.get('D3-agate',{}).get(k,'?') if results.get('D3-agate') else '?'):>12}")
with open(os.path.join(LD,"d123_comparison.json"),"w") as f: json.dump(results,f,indent=2,default=str)
log(f"对比保存: {LD}/d123_comparison.json"); log("="*55)
