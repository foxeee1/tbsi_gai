"""
TBSI 基准 - 多指标评测模块
============================
提取 7+1 项指标 (SR/OP50/OP75/MeanIoU/PR/NPR/Composite/FPS)
支持每序列分析和胜率计算。

用法:
    from benchmark.bm_evaluate import evaluate
    metrics = evaluate("tbsi_track", config_name, subset_size=100)
"""

import os, sys, re, json, subprocess
prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.append(prj_path)

import torch
import numpy as np


# ============================================================
# 核心: 从 eval_data 提取完整多指标
# ============================================================

def compute_all_metrics(eval_data: dict) -> dict:
    """
    从 extract_results 返回的 eval_data 中提取全部指标。

    返回:
        {
            "SR": float, "OP50": float, "OP75": float, "MeanIoU": float,
            "PR": float, "NPR": float,
            "seq_metrics": {seq_name: {"AUC": float, "IoU": float, "Prec": float}, ...},
            "valid_count": int, "total_count": int,
        }
    """
    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    valid_n = valid.long().sum().item()
    total_n = len(eval_data["sequences"])

    # success/overlap: [N, T, K] where T=tracker(1), K=threshold
    sr = torch.tensor(eval_data["ave_success_rate_plot_overlap"])  # [N, 1, K]
    avg_iou = torch.tensor(eval_data["avg_overlap_all"])           # [N, 1]
    pr = torch.tensor(eval_data["ave_success_rate_plot_center"])   # [N, 1, K]
    npr = torch.tensor(eval_data["ave_success_rate_plot_center_norm"])

    thresh_overlap = torch.tensor(eval_data["threshold_set_overlap"])
    thresh_center = torch.tensor(eval_data["threshold_set_center"])
    thresh_ncenter = torch.tensor(eval_data["threshold_set_center_norm"])

    # --- Aggregate metrics ---
    # AUC = mean over thresholds (of success rate averaged over sequences)
    sr_valid = sr[valid, 0, :]     # [valid_N, K]
    auc_seq = sr_valid.mean(dim=1)  # per-sequence AUC
    auc_global = auc_seq.mean().item() * 100.0

    # OP50 / OP75 at specific thresholds
    idx50 = (thresh_overlap == 0.50).nonzero(as_tuple=True)[0]
    idx75 = (thresh_overlap == 0.75).nonzero(as_tuple=True)[0]
    op50 = sr_valid[:, idx50].mean().item() * 100.0 if idx50.numel() > 0 else 0.0
    op75 = sr_valid[:, idx75].mean().item() * 100.0 if idx75.numel() > 0 else 0.0

    # MeanIoU = average overlap per frame avg (over sequences)
    iou_valid = avg_iou[valid, 0]
    mean_iou = iou_valid.mean().item() * 100.0  # scale to 0-100

    # Precision @ 20px
    idx20 = (thresh_center == 20.0).nonzero(as_tuple=True)[0]
    pr_valid = pr[valid, 0, :]
    prec = pr_valid[:, idx20].mean().item() * 100.0 if idx20.numel() > 0 else 0.0

    # Norm Precision @ 0.20
    idx20n = (thresh_ncenter == 0.20).nonzero(as_tuple=True)[0]
    npr_valid = npr[valid, 0, :]
    nprec = npr_valid[:, idx20n].mean().item() * 100.0 if idx20n.numel() > 0 else 0.0

    # --- Per-sequence metrics ---
    seq_names = eval_data["sequences"]
    seq_metrics = {}
    for i, (sname, is_valid) in enumerate(zip(seq_names, valid.tolist())):
        if is_valid:
            sr_i = torch.tensor(eval_data["ave_success_rate_plot_overlap"][i][0])  # [K]
            iou_i = torch.tensor(eval_data["avg_overlap_all"][i][0])
            pr_i = torch.tensor(eval_data["ave_success_rate_plot_center"][i][0])
            seq_metrics[sname] = {
                "AUC": round(sr_i.mean().item() * 100.0, 2),
                "IoU": round(iou_i.item() * 100.0, 2),
                "Prec": round(pr_i[idx20].item() * 100.0, 2) if idx20.numel() > 0 else 0.0,
                "valid": True,
            }
        else:
            seq_metrics[sname] = {"AUC": 0.0, "IoU": 0.0, "Prec": 0.0, "valid": False}

    # Round aggregate
    return {
        "SR": round(auc_global, 2),
        "OP50": round(op50, 2),
        "OP75": round(op75, 2),
        "MeanIoU": round(mean_iou, 2),
        "PR": round(prec, 2),
        "NPR": round(nprec, 2),
        "seq_metrics": seq_metrics,
        "valid_count": valid_n,
        "total_count": total_n,
    }


def compute_seq_win_rate(new_metrics: dict, baseline_metrics: dict) -> dict:
    """
    对比新方法和基线的每序列 AUC, 计算胜率/负率/平率。

    Returns:
        {"win_rate": float, "wins": int, "losses": int, "ties": int,
         "mean_delta": float, "std_delta": float,
         "best_seq": str, "worst_seq": str, "sequences": [...]}
    """
    if "seq_metrics" not in new_metrics or "seq_metrics" not in baseline_metrics:
        return {"win_rate": 0, "wins": 0, "losses": 0, "ties": 0}

    new_seqs = new_metrics["seq_metrics"]
    base_seqs = baseline_metrics["seq_metrics"]

    wins = losses = ties = 0
    deltas = []
    seq_details = []

    for sname in new_seqs:
        if sname not in base_seqs:
            continue
        nv = new_seqs[sname]
        bv = base_seqs[sname]
        if not nv.get("valid", False) or not bv.get("valid", False):
            continue

        delta = nv["AUC"] - bv["AUC"]
        deltas.append(delta)
        if delta > 0.3:    # >0.3% → win (beyond noise threshold)
            wins += 1
            label = "win"
        elif delta < -0.3:
            losses += 1
            label = "loss"
        else:
            ties += 1
            label = "tie"
        seq_details.append({"seq": sname, "delta": round(delta, 2), "label": label})

    total = wins + losses + ties
    if total == 0:
        return {"win_rate": 0, "wins": 0, "losses": 0, "ties": 0, "mean_delta": 0, "std_delta": 0}

    deltas_arr = np.array(deltas)
    seq_details.sort(key=lambda x: x["delta"], reverse=True)

    return {
        "win_rate": round(wins / total * 100, 1),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_delta": round(float(deltas_arr.mean()), 3),
        "std_delta": round(float(deltas_arr.std()), 3),
        "best_seq": seq_details[0]["seq"] if seq_details else "",
        "best_delta": seq_details[0]["delta"] if seq_details else 0,
        "worst_seq": seq_details[-1]["seq"] if seq_details else "",
        "worst_delta": seq_details[-1]["delta"] if seq_details else 0,
        "sequences": seq_details[:10],  # 只保留 top-10
    }


def compute_composite(metrics: dict) -> float:
    """加权复合指标，防单指标投机"""
    from benchmark.bm_config import COMPOSITE_WEIGHTS
    score = 0.0
    total_w = 0.0
    for key, w in COMPOSITE_WEIGHTS.items():
        if key in metrics and metrics[key] is not None:
            score += w * metrics[key]
            total_w += w
    return round(score / total_w, 2) if total_w > 0 else 0.0


# ============================================================
# API 评测
# ============================================================

def evaluate_via_api(tracker_name: str, config_name: str,
                     dataset_name: str = "lasher_test",
                     subset_size: int = None, threads: int = 6,
                     num_gpus: int = 1, force_eval: bool = False) -> dict:
    """通过 Python API 运行评测，返回完整指标。"""
    from lib.test.evaluation import get_dataset, Tracker
    from lib.test.evaluation.running import run_dataset
    from lib.test.analysis.extract_results import extract_results

    # Load dataset
    dataset = get_dataset(dataset_name)
    total_seqs = len(dataset)
    if subset_size and subset_size < total_seqs:
        dataset = dataset[:subset_size]
        print(f"  [数据集] {subset_size}/{total_seqs} 序列子集")
    else:
        print(f"  [数据集] 全量 {total_seqs} 序列")

    tracker = Tracker(tracker_name, config_name, dataset_name, run_id=None)
    trackers = [tracker]

    # Clear cached results if forced
    if force_eval:
        from lib.test.evaluation.environment import env_settings
        eval_cache = os.path.join(env_settings().result_plot_path,
                                  f"{config_name}_bm_eval", "eval_data.pkl")
        if os.path.exists(eval_cache):
            os.remove(eval_cache)
            print(f"  [清除缓存] {eval_cache}")
        # Also remove raw tracking results so they get regenerated
        results_dir = tracker.results_dir
        if os.path.exists(results_dir):
            import shutil
            shutil.rmtree(results_dir)
            print(f"  [清除缓存] raw results: {results_dir}")

    # Run tracking
    print(f"  [评测中] threads={threads}, gpus={num_gpus} ...")
    run_dataset(dataset, trackers, debug=0, threads=threads, num_gpus=num_gpus)

    # Extract metrics
    report_name = f"{config_name}_bm_eval"
    eval_data = extract_results(trackers, dataset, report_name)

    # Full metrics computation
    metrics = compute_all_metrics(eval_data)

    # Composite
    metrics["Composite"] = compute_composite(metrics)

    # Print summary
    print(f"  [结果] SR={metrics['SR']} OP50={metrics['OP50']} OP75={metrics['OP75']} "
          f"MeanIoU={metrics['MeanIoU']}")
    print(f"         PR={metrics['PR']} NPR={metrics['NPR']} "
          f"Composite={metrics['Composite']}")
    print(f"         有效序列: {metrics['valid_count']}/{metrics['total_count']}")

    return metrics


# ============================================================
# CLI 方式 (全量测试)
# ============================================================

def run_test_cmd(tracker_name: str, config_name: str, dataset_name: str = "lasher_test",
                 threads: int = 6, num_gpus: int = 1) -> str:
    cmd = [sys.executable, "tracking/test.py", tracker_name, config_name,
           "--dataset_name", dataset_name, "--threads", str(threads),
           "--num_gpus", str(num_gpus)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=prj_path)
    if result.returncode != 0:
        raise RuntimeError(f"Test failed for {config_name}: {result.stderr[:500]}")
    return result.stdout + result.stderr


def run_analysis_cmd(tracker_name: str, config_name: str, dataset_name: str = "lasher_test",
                     runid: int = None) -> str:
    cmd = [sys.executable, "tracking/analysis_results.py",
           "--tracker_name", tracker_name, "--tracker_param", config_name,
           "--dataset_name", dataset_name]
    if runid is not None:
        cmd += ["--runid", str(runid)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=prj_path)
    if result.returncode != 0:
        raise RuntimeError(f"Analysis failed for {config_name}: {result.stderr[:500]}")
    return result.stdout + result.stderr


def parse_analysis_stdout(stdout: str) -> dict:
    """从 analysis_results.py 输出中解析 SR/PR/NPR"""
    metrics = {}
    auc_m = re.search(r'AUC\s*\|[\s\d.]+', stdout)
    prec_m = re.search(r'Precision\s*\|[\s\d.]+', stdout)
    nprec_m = re.search(r'Norm Precision\s*\|[\s\d.]+', stdout)

    def last_num(text):
        nums = re.findall(r'\d+\.\d+', text)
        return float(nums[-1]) if nums else 0.0

    if auc_m:
        metrics["AUC"] = last_num(auc_m.group())
        auc_nums = re.findall(r'\d+\.\d+', auc_m.group())
        metrics["SR"] = float(auc_nums[0]) if len(auc_nums) >= 1 else 0
        metrics["OP50"] = float(auc_nums[1]) if len(auc_nums) >= 2 else 0
        metrics["OP75"] = float(auc_nums[2]) if len(auc_nums) >= 3 else 0
    if prec_m:
        metrics["PR"] = last_num(prec_m.group())
    if nprec_m:
        metrics["NPR"] = last_num(nprec_m.group())
    return metrics


# ============================================================
# 统一入口
# ============================================================

def evaluate(tracker_name: str, config_name: str, dataset_name: str = "lasher_test",
             subset_size: int = None, threads: int = 6, num_gpus: int = 1,
             force_eval: bool = False, method: str = "api") -> dict:
    """统一评测入口。method='api' 支持子集评测和完整指标。"""
    if method == "api" or subset_size is not None:
        return evaluate_via_api(tracker_name, config_name, dataset_name,
                                subset_size, threads, num_gpus, force_eval)
    else:
        run_test_cmd(tracker_name, config_name, dataset_name, threads, num_gpus)
        stdout = run_analysis_cmd(tracker_name, config_name, dataset_name)
        metrics = parse_analysis_stdout(stdout)
        return metrics


def evaluate_level(level_cfg: dict, threads: int = 6, num_gpus: int = 1,
                   force_eval: bool = False) -> dict:
    """按级别配置评测"""
    return evaluate(
        tracker_name="tbsi_track",
        config_name=level_cfg["config_name"],
        dataset_name="lasher_test",
        subset_size=level_cfg.get("test_subset_size"),
        threads=threads, num_gpus=num_gpus,
        force_eval=force_eval,
        method="api",
    )
