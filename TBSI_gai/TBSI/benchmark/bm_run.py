"""
TBSI 二级训练测试基准 - 主编排器
=================================

  L1 Sprint (≤1h) → 快速验证 + 复合/胜率/无损三重门控
       ↓ 通过（全部满足）
  L2 Full   (~6h) → 全量训练确认最终结果

门控条件 (L1→L2, 缺一不可):
  1. 复合指标 ≥ +1.0%
  2. 胜率 > 55% (过半序列提升)
  3. 无核心指标降幅 > 1.0%

用法:
  python benchmark/bm_run.py --init                   # 初始化 + L1 基线
  python benchmark/bm_run.py --run --desc "修改说明"  # 训练+测试+门控
  python benchmark/bm_run.py --status                 # 状态
"""

import os, sys, json, time, argparse, subprocess
from datetime import datetime
prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from benchmark.bm_config import (
    LEVELS, LEVEL1, LEVEL2,
    PROJECT_ROOT, OUTPUT_DIR, LEDGERS_DIR, LEDGER_FILENAME,
    get_checkpoint_path,
    GATE_COMPOSITE_DELTA, GATE_WIN_RATE, GATE_MAX_SINGLE_DEGRADE,
)


# ============================================================
# 会话管理
# ============================================================
def _ledger_path() -> str:
    os.makedirs(LEDGERS_DIR, exist_ok=True)
    return os.path.join(LEDGERS_DIR, LEDGER_FILENAME)

def init_ledger() -> dict:
    now = datetime.now()
    ledger = {
        "session_id": now.strftime("%Y%m%d_%H%M%S"),
        "created_at": now.isoformat(),
        "current_level": 1,
        "baselines": {},
        "iterations": [],
        "advancements": [],
        "total_iterations": 0,
        "is_completed": False,
    }
    _save(ledger)
    return ledger

def _save(ledger: dict):
    path = _ledger_path()
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    print(f"  [✓] 会话已保存: {path}")

def _load() -> dict:
    path = _ledger_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================
# 训练
# ============================================================
def train_level(level_cfg: dict, force: bool = False) -> bool:
    ckpt = get_checkpoint_path(level_cfg["config_name"], level_cfg["epochs"])
    if os.path.exists(ckpt) and not force:
        print(f"  [=] Checkpoint 已存在: {ckpt}\n      加 --force 重新训练")
        return True

    cmd = (
        f"unset OMP_NUM_THREADS && "
        f"{sys.executable} tracking/train.py "
        f"--script tbsi_track --config {level_cfg['config_name']} "
        f"--save_dir {OUTPUT_DIR} --mode single"
    )
    print(f"  训练: {level_cfg['desc']}")
    start = time.time()
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    elapsed = (time.time() - start) / 60
    if result.returncode != 0:
        print(f"  [✗] 训练失败 (rc={result.returncode})")
        return False
    print(f"  [✓] 训练完成, 耗时 {elapsed:.0f} min")
    return True


def get_time_estimate(level_cfg: dict) -> str:
    n = level_cfg["epochs"] * level_cfg["sample_per_epoch"]
    if n <= 50000:
        return "~40-60 min 训练 + ~10 min 测试"
    elif n <= 300000:
        return "~2-3 h 训练 + ~20 min 测试"
    else:
        return "~5-6 h 训练 + ~20 min 测试"


# ============================================================
# 评测
# ============================================================
def evaluate_level(level_cfg: dict, force_eval: bool = False) -> dict:
    from benchmark.bm_evaluate import evaluate_level as _eval
    cfg_name = level_cfg["config_name"]
    subset = level_cfg.get("test_subset_size")
    print(f"  [评测] {cfg_name}")
    print(f"  [测试集] {'子集 ' + str(subset) + ' seq' if subset else '全量 245 seq'}")

    metrics = _eval(level_cfg, threads=6, num_gpus=1, force_eval=force_eval)

    if "Composite" not in metrics:
        from benchmark.bm_evaluate import compute_composite
        metrics["Composite"] = compute_composite(metrics)

    print(f"  [指标] SR={metrics.get('SR','N/A')}  OP50={metrics.get('OP50','N/A')}  "
          f"OP75={metrics.get('OP75','N/A')}  MeanIoU={metrics.get('MeanIoU','N/A')}")
    print(f"         PR={metrics.get('PR','N/A')}  NPR={metrics.get('NPR','N/A')}  "
          f"Composite={metrics.get('Composite','N/A')}")
    return metrics


# ============================================================
# 严格门控 (L1→L2, 全部满足才通过)
# ============================================================
def gating_decision(new: dict, baseline: dict, ledger: dict = None) -> dict:
    """
    三重门控，缺一不可:
      1. 复合指标 ≥ +1.0%
      2. 胜率 > 55%
      3. 无核心指标降幅 > 1.0%
    """
    result = {"passed": False, "details": {}, "overall_message": "", "win_analysis": {}}

    if not baseline:
        result["passed"] = True
        result["overall_message"] = "无基线 → 自动设为基准"
        return result

    new_c = new.get("Composite", 0)
    base_c = baseline.get("Composite", 0)
    composite_delta = round(new_c - base_c, 2)

    detail_lines = []
    checks = []

    # --- 检查1: 复合指标 ---
    c_pass = composite_delta >= GATE_COMPOSITE_DELTA
    result["details"]["Composite"] = {
        "baseline": base_c, "new": new_c, "delta": composite_delta,
        "threshold": f"≥+{GATE_COMPOSITE_DELTA}", "pass": c_pass,
    }
    status = "✓" if c_pass else "✗"
    detail_lines.append(f"  [{status}] Composite: {base_c} → {new_c} (Δ={composite_delta:+.2f}, 需≥+{GATE_COMPOSITE_DELTA})")
    checks.append(c_pass)

    # --- 检查2: 核心指标不降 ---
    core_pass = True
    for key in ["SR", "PR", "NPR"]:
        nv = new.get(key, 0)
        bv = baseline.get(key, 0)
        d = round(nv - bv, 2)
        degraded = d < GATE_MAX_SINGLE_DEGRADE
        if degraded:
            core_pass = False
        result["details"][key] = {
            "baseline": bv, "new": nv, "delta": d,
            "threshold": f"≥{GATE_MAX_SINGLE_DEGRADE}", "pass": not degraded,
        }
        status = "✓" if not degraded else "✗"
        detail_lines.append(f"  [{status}] {key}: {bv} → {nv} (Δ={d:+.2f}, 下限{GATE_MAX_SINGLE_DEGRADE})")
    checks.append(core_pass)

    # --- 检查3: 胜率 ---
    win = compute_win_rate(new, baseline)
    result["win_analysis"] = win
    win_pass = win.get("win_rate", 0) >= GATE_WIN_RATE
    result["details"]["WinRate"] = {
        "new": f"{win.get('win_rate', 0)}%",
        "mean_delta": win.get("mean_delta", 0),
        "threshold": f">{GATE_WIN_RATE}%", "pass": win_pass,
    }
    status = "✓" if win_pass else "✗"
    detail_lines.append(f"  [{status}] WinRate: {win.get('win_rate',0)}% ({win.get('wins',0)}W/{win.get('losses',0)}L)"
                        f" 均值Δ={win.get('mean_delta',0):+.3f}, 需>{GATE_WIN_RATE}%")
    checks.append(win_pass)

    # --- 输出 ---
    for line in detail_lines:
        print(line)

    result["passed"] = all(checks)
    if result["passed"]:
        result["overall_message"] = (
            f"三重门控全部通过 → L1 通过! "
            f"Composite+{composite_delta:+.2f}, 胜率{win.get('win_rate',0)}%, 无损"
        )
    else:
        failed = [c for c in checks if not c]
        msgs = []
        if not c_pass: msgs.append(f"复合{composite_delta:+.2f}(需≥+{GATE_COMPOSITE_DELTA})")
        if not core_pass: msgs.append("核心指标降幅超标")
        if not win_pass: msgs.append(f"胜率{win.get('win_rate',0)}%(需>{GATE_WIN_RATE}%)")
        result["overall_message"] = (
            f"L1 未通过: {'; '.join(msgs)}。继续迭代改进。"
        )

    return result


def compute_win_rate(new_metrics: dict, baseline_metrics: dict) -> dict:
    from benchmark.bm_evaluate import compute_seq_win_rate
    win = compute_seq_win_rate(new_metrics, baseline_metrics)

    print(f"  [胜率] {win.get('win_rate', 0)}% ({win.get('wins',0)}胜/"
          f"{win.get('losses',0)}负/{win.get('ties',0)}平)")
    print(f"  [平均Δ] {win.get('mean_delta', 0):+.3f}%  (std={win.get('std_delta', 0):.3f})")
    if win.get("best_seq"):
        print(f"  [最好] {win['best_seq']} Δ={win.get('best_delta',0):+.2f}")
    if win.get("worst_seq"):
        print(f"  [最差] {win['worst_seq']} Δ={win.get('worst_delta',0):+.2f}")
    return win


# ============================================================
# 主迭代逻辑
# ============================================================
def check_git_modifications() -> dict:
    try:
        diff = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        staged = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        files = set()
        for line in diff.stdout.split("\n") + staged.stdout.split("\n"):
            if "|" in line:
                f = line.split("|")[0].strip()
                if f:
                    files.add(f)
        diff_text = diff.stdout[:5000] + staged.stdout[:5000]
        return {"changed_files": sorted(files), "current_commit": commit.stdout.strip(),
                "diff_detail": diff_text, "has_changes": len(files) > 0}
    except Exception as e:
        return {"changed_files": [], "current_commit": "", "diff_detail": str(e), "has_changes": False}


def cmd_run(level: int = None, desc: str = "", force: bool = False):
    ledger = _load()
    if not ledger:
        print("[!] 无会话. 请先运行 --init")
        return

    if level is None:
        level = ledger["current_level"]
    elif level > ledger["current_level"] and not force:
        print(f"[!] Level {level} 未解锁 (当前 L{ledger['current_level']}), 用 --force 跳过")
        return

    lv_cfg = LEVELS[level - 1]
    iter_id = f"iter_{ledger['total_iterations'] + 1:03d}"

    print("=" * 64)
    print(f"  TBSI 基准 · {iter_id} · Level {level} ({lv_cfg['name'].upper()})")
    print(f"  {lv_cfg['desc']}")
    print(f"  预计: {get_time_estimate(lv_cfg)}")
    print("=" * 64)

    # --- Step 0: change check ---
    print(f"\n[0] 修改: \"{desc}\"")
    git_info = check_git_modifications()
    changed = git_info["changed_files"]
    if len(changed) > 2 and not force:
        print(f"\n[!] 检测到 {len(changed)} 个文件修改 (上限 2):")
        for f in changed:
            print(f"    {f}")
        if input("  继续? (y/N): ").strip().lower() != "y":
            print("  已取消")
            return
    if changed:
        print(f"  修改文件: {changed}")

    # --- Step 1: Train ---
    print(f"\n[1] 训练 ...")
    if not train_level(lv_cfg, force=force):
        return

    # --- Step 2: Evaluate ---
    print(f"\n[2] 评测 ...")
    metrics = evaluate_level(lv_cfg)

    # --- Step 3: Gate ---
    print(f"\n[3] 门控判断 ...")
    bl_key = f"level{level}"
    baseline = ledger.get("baselines", {}).get(bl_key, {})

    if not baseline:
        print(f"  尚无 Level {level} 基准 → 本次结果设为基准")
        ledger.setdefault("baselines", {})[bl_key] = metrics
        gate = {"passed": True, "details": {}, "overall_message": "基准已建立", "win_analysis": {}}
    else:
        gate = gating_decision(metrics, baseline)

    # Record
    record = {
        "id": iter_id, "level": level,
        "timestamp": datetime.now().isoformat(),
        "description": desc,
        "changed_files": changed,
        "git_commit": git_info.get("current_commit", ""),
        "metrics": {k: v for k, v in metrics.items() if k != "seq_metrics"},
        "gate_passed": gate["passed"],
        "gate_details": gate["details"],
        "win_analysis": gate.get("win_analysis", {}),
    }
    if "seq_metrics" in metrics:
        seq_path = os.path.join(LEDGERS_DIR, f"seq_metrics_{iter_id}.json")
        with open(seq_path, "w") as f:
            json.dump(metrics["seq_metrics"], f, indent=1)

    ledger["iterations"].append(record)
    ledger["total_iterations"] += 1

    # --- Step 4: Advance? ---
    print(f"\n[4] 决策: {'✓ 通过' if gate['passed'] else '✗ 未通过'}")
    print(f"  {gate['overall_message']}")

    if gate["passed"] and level < 2:
        next_lv = level + 1
        next_key = f"level{next_lv}"
        if next_key not in ledger.get("baselines", {}):
            print(f"\n  [→] 需要建立 Level {next_lv} 基准")
            if input(f"  现在训练并评测? (Y/n): ").strip().lower() != "n":
                next_cfg = LEVELS[next_lv - 1]
                train_level(next_cfg)
                next_metrics = evaluate_level(next_cfg)
                if not next_metrics:
                    return
                ledger.setdefault("baselines", {})[next_key] = next_metrics
                if "seq_metrics" in next_metrics:
                    sp = os.path.join(LEDGERS_DIR, f"baseline_level{next_lv}_seq.json")
                    with open(sp, "w") as f:
                        json.dump(next_metrics["seq_metrics"], f, indent=1)
                print(f"  Level {next_lv} 基准建立: "
                      f"SR={next_metrics.get('SR','N/A')}, "
                      f"Composite={next_metrics.get('Composite','N/A')}")
        ledger["current_level"] = next_lv
        ledger["advancements"].append({
            "from": level, "to": next_lv,
            "timestamp": datetime.now().isoformat(),
            "trigger_iteration": iter_id,
            "metrics_snapshot": {k: v for k, v in metrics.items() if k != "seq_metrics"},
        })
        print(f"\n  [✓] 已推进到 Level {next_lv}")
    elif gate["passed"] and level == 2:
        print(f"\n  [★] Level 2 完成! 全部验证通过!")
        ledger["is_completed"] = True
    else:
        print(f"\n  [-] 停留在 Level {level}，继续迭代改进")

    _save(ledger)
    _print_summary(record, gate, changed, ledger)


def _print_summary(record: dict, gate: dict, changed: list, ledger: dict):
    m = record["metrics"]
    print("\n" + "=" * 64)
    print("  迭代摘要")
    print("=" * 64)
    print(f"  ID      : {record['id']}")
    print(f"  Level   : {record['level']} ({LEVELS[record['level']-1]['name']})")
    print(f"  描述    : {record['description']}")
    print(f"  文件    : {changed or '(无)'}")
    print(f"  ──────────────────────────")
    print(f"  SR      : {m.get('SR','N/A'):>7}    OP50: {m.get('OP50','N/A'):>6}   OP75: {m.get('OP75','N/A'):>6}")
    print(f"  MeanIoU : {m.get('MeanIoU','N/A'):>7}    PR  : {m.get('PR','N/A'):>6}   NPR : {m.get('NPR','N/A'):>6}")
    print(f"  Composite: {m.get('Composite','N/A'):>6}")
    print(f"  有效    : {m.get('valid_count','?')}/{m.get('total_count','?')} 序列")
    w = record.get("win_analysis", {})
    if w:
        print(f"  胜率    : {w.get('win_rate','N/A')}% ({w.get('wins',0)}W/{w.get('losses',0)}L)")
        print(f"  Δ均值   : {w.get('mean_delta','N/A'):+}")
    print(f"  ──────────────────────────")
    print(f"  门控    : {'✓ PASS' if gate['passed'] else '✗ FAIL'}")
    print(f"  {gate['overall_message'][:70]}")
    lvl = ledger["current_level"]
    print(f"  状态    : {'★ ALL COMPLETE' if ledger['is_completed'] else f'→ Level {lvl}'}")
    print("=" * 64)


# ============================================================
# 初始化
# ============================================================
def cmd_init():
    print("=" * 64)
    print("  TBSI 二级基准 · 初始化")
    print("  1. 生成配置 → 2. 修复路径 → 3. 建立 L1 Sprint 基线")
    print("=" * 64)

    if _load():
        if input("\n存在会话, 覆盖? (y/N): ").strip().lower() != "y":
            print("取消")
            return

    print("\n[1/3] 生成配置文件 ...")
    from benchmark.bm_generate_configs import generate_all_configs
    generate_all_configs()

    print("\n[2/3] 修复测试路径 ...")
    _fix_env()

    print("\n[3/3] 建立 Level 1 (Sprint) 基准 ...")
    print(f"  配置: {LEVEL1['desc']}")
    print(f"  预计: {get_time_estimate(LEVEL1)}")
    if input("  开始? (Y/n): ").strip().lower() == "n":
        l = init_ledger()
        print(f"\n会话已创建 (ID: {l['session_id']})")
        return

    train_level(LEVEL1)
    metrics = evaluate_level(LEVEL1)
    l = init_ledger()
    l["baselines"]["level1"] = metrics
    if "seq_metrics" in metrics:
        sp = os.path.join(LEDGERS_DIR, "baseline_level1_seq.json")
        with open(sp, "w") as f:
            json.dump(metrics["seq_metrics"], f, indent=1)
    _save(l)

    print(f"\n[✓] Level 1 基准: SR={metrics.get('SR','N/A')} "
          f"Composite={metrics.get('Composite','N/A')}")
    print(f"  接下来: 修改代码 → python benchmark/bm_run.py --run --desc \"修改说明\"")


def _fix_env():
    local_py = os.path.join(PROJECT_ROOT, "lib", "test", "evaluation", "local.py")
    if not os.path.exists(local_py):
        print("  [!] local.py 不存在")
        return False
    with open(local_py) as f:
        content = f.read()
    old = None
    for line in content.split("\n"):
        if "check_dir" in line and "=" in line:
            old = line.split("=")[-1].strip().strip("'\"")
            break
    if old and old != OUTPUT_DIR:
        content = content.replace(f"settings.check_dir = '{old}'",
                                  f"settings.check_dir = '{OUTPUT_DIR}'")
        with open(local_py, "w") as f:
            f.write(content)
        print(f"  [✓] check_dir: {old} → {OUTPUT_DIR}")
    else:
        print(f"  [=] check_dir 无需修改: {old}")
    return True


# ============================================================
# 状态
# ============================================================
def cmd_status():
    l = _load()
    if not l:
        print("[!] 无会话, 运行 --init")
        return

    print("=" * 64)
    print("  TBSI 基准 · 状态")
    print("=" * 64)
    print(f"  会话    : {l['session_id']}")
    print(f"  创建    : {l['created_at'][:19]}")
    print(f"  当前    : Level {l['current_level']} ({LEVELS[l['current_level']-1]['name']})")
    print(f"  迭代    : {l['total_iterations']}")
    print(f"  完成    : {'★' if l.get('is_completed') else '○'}")

    print(f"\n  ── 基线指标 ──")
    for i in [1, 2]:
        bk = f"level{i}"
        lm = LEVELS[i-1]
        if bk in l.get("baselines", {}):
            bm = l["baselines"][bk]
            print(f"  L{i} {lm['name']:>7}: SR={bm.get('SR','-'):>6}  "
                  f"PR={bm.get('PR','-'):>6}  NPR={bm.get('NPR','-'):>6}  "
                  f"Composite={bm.get('Composite','-'):>6}")
        else:
            print(f"  L{i} {lm['name']:>7}: (未建立)")

    print(f"\n  ── 迭代历史 (最近 8) ──")
    for it in l.get("iterations", [])[-8:]:
        m = it.get("metrics", {})
        mk = "✓" if it.get("gate_passed") else "✗"
        print(f"  [{mk}] {it['id']} L{it['level']}  "
              f"SR={m.get('SR','-'):>6}  Comp={m.get('Composite','-'):>6}  "
              f"{it['description'][:50]}")

    if l.get("advancements"):
        print(f"\n  ── 推进 ──")
        for a in l["advancements"]:
            print(f"  L{a['from']} → L{a['to']} ({a['timestamp'][:19]})")

    print(f"\n  下一步: python benchmark/bm_run.py --run --desc \"修改说明\"")


def cmd_reset():
    if input("确定重置? (y/N): ").strip().lower() != "y":
        return
    p = _ledger_path()
    if os.path.exists(p):
        os.remove(p)
    for f in os.listdir(LEDGERS_DIR):
        if f.startswith("seq_metrics") or f.startswith("baseline"):
            os.remove(os.path.join(LEDGERS_DIR, f))
    print("已重置. 可通过 --init 重新初始化")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="TBSI 二级基准 (复合指标+胜率+无损三重门控)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true", help="初始化 + L1 基线")
    ap.add_argument("--run", action="store_true", help="运行迭代 (训练+测试+门控)")
    ap.add_argument("--status", action="store_true", help="查看状态")
    ap.add_argument("--reset", action="store_true", help="重置")
    ap.add_argument("--advance", action="store_true", help="推进到下一级")
    ap.add_argument("--level", type=int, choices=[1, 2], help="指定级别")
    ap.add_argument("--desc", type=str, default="", help="修改描述")
    ap.add_argument("--force", action="store_true", help="强制运行")
    args = ap.parse_args()

    if args.init:
        cmd_init()
    elif args.run:
        cmd_run(args.level, args.desc, args.force)
    elif args.status:
        cmd_status()
    elif args.reset:
        cmd_reset()
    elif args.advance:
        l = _load()
        if not l:
            print("[!] 无会话")
            return
        if l["current_level"] < 2:
            l["current_level"] += 1
            _save(l)
            print(f"[✓] 推进到 Level {l['current_level']}")
        else:
            print("[!] 已是最高级")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
