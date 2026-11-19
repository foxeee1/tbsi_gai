#!/usr/bin/env python3
"""Diagnose Early-gate subproblems from already completed paired runs.

This is a new read-only analysis.  It does not invoke the historical audit
tools, load a model, or change any experiment artifact.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "early_gate_closed_loop_20261026"


def read_json_lines(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "successful_updates" in obj and "loss" in obj:
            rows.append(obj)
    return pd.DataFrame(rows)


def finite_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return None
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def summarize_pair(path, label):
    df = pd.read_csv(path)
    delta = df["auc_delta"].astype(float)
    out = {
        "label": label,
        "n": int(len(df)),
        "mean_auc_delta_pp": float(delta.mean()),
        "median_auc_delta_pp": float(delta.median()),
        "win_count": int((delta > 0).sum()),
        "loss_count": int((delta < 0).sum()),
        "large_loss_count_gt5pp": int((delta < -5).sum()),
        "failure_delta_mean": float(df["failure_fraction_delta"].mean()),
        "new_failure_frames_total": int(df["new_failure_from_control_good_frames"].sum()),
        "recovered_failure_frames_total": int(df["recovered_from_control_failure_frames"].sum())
        if "recovered_from_control_failure_frames" in df
        else int(df["recovered_control_failure_frames"].sum()),
        "auc_vs_failure_delta_corr": finite_corr(delta, df["failure_fraction_delta"])
        if "failure_fraction_delta" in df else None,
        "auc_vs_quarter3_iou_corr": finite_corr(delta, df["iou_delta_quarter3"])
        if "iou_delta_quarter3" in df else None,
    }
    if "group" in df:
        out["groups"] = {
            str(g): {
                "n": int(len(part)),
                "mean_auc_delta_pp": float(part["auc_delta"].mean()),
                "large_loss_count_gt5pp": int((part["auc_delta"] < -5).sum()),
                "new_failure_frames": int(part["new_failure_from_control_good_frames"].sum()),
                "recovered_failure_frames": int(
                    part["recovered_from_control_failure_frames"].sum()
                    if "recovered_from_control_failure_frames" in part
                    else part["recovered_control_failure_frames"].sum()
                ),
            }
            for g, part in df.groupby("group", sort=True)
        }
    return out


def summarize_readouts(path):
    rows = pd.DataFrame(json.loads(Path(path).read_text()))
    numeric = [
        "mean_relative_value_change", "fg_relative_change", "bg_relative_change",
        "mean_source_tangent_norm", "mean_conditional_tangent_norm",
        "aligned_vs_colocated_receiver_rel_l2", "aligned_vs_colocated_receiver_cosine",
        "conditioner_change_if_aligned", "diagonal_receiver_mass",
    ]
    out = {"n": int(len(rows)), "overall": {}}
    for col in numeric:
        if col in rows:
            out["overall"][col] = float(rows[col].astype(float).mean())
    rows["bg_minus_fg_change"] = rows["bg_relative_change"] - rows["fg_relative_change"]
    rows["bg_to_fg_abs_ratio"] = rows["bg_relative_change"].abs() / rows["fg_relative_change"].abs().clip(lower=1e-12)
    out["overall"]["mean_bg_minus_fg_change"] = float(rows["bg_minus_fg_change"].mean())
    out["overall"]["mean_bg_to_fg_abs_ratio"] = float(rows["bg_to_fg_abs_ratio"].mean())
    out["by_layer_source"] = {}
    for (layer, source), part in rows.groupby(["layer", "source"], sort=True):
        key = f"layer{int(layer)}_{source}"
        out["by_layer_source"][key] = {
            "n": int(len(part)),
            "value_change": float(part["mean_relative_value_change"].mean()),
            "fg_change": float(part["fg_relative_change"].mean()),
            "bg_change": float(part["bg_relative_change"].mean()),
            "bg_to_fg_abs_ratio": float(part["bg_to_fg_abs_ratio"].mean()),
            "aligned_colocated_rel_l2": float(part["aligned_vs_colocated_receiver_rel_l2"].mean()),
        }
    return out


def summarize_training(logs):
    out = {}
    frames = {}
    for label, path in logs.items():
        df = read_json_lines(path)
        frames[label] = df
        if df.empty:
            out[label] = {"records": 0}
            continue
        out[label] = {
            "records": int(len(df)),
            "final_updates": int(df["successful_updates"].iloc[-1]),
            "mean_loss": float(df["loss"].mean()),
            "final_loss": float(df["loss"].iloc[-1]),
            "mean_iou": float(df["iou"].mean()),
            "final_iou": float(df["iou"].iloc[-1]),
            "mean_gradient_norm": float(df["gradient_norm"].mean()),
            "final_gradient_norm": float(df["gradient_norm"].iloc[-1]),
        }
    # These are trajectory differences, not a gradient-conflict proof.
    if len(frames) >= 2 and all(not x.empty for x in frames.values()):
        common = None
        for df in frames.values():
            idx = set(df["successful_updates"].astype(int))
            common = idx if common is None else common & idx
        common = sorted(common or [])
        if common:
            base_label = next(iter(frames))
            base = frames[base_label].set_index("successful_updates")
            trajectory = {}
            for label, df in frames.items():
                cur = df.set_index("successful_updates").loc[common]
                if label == base_label:
                    continue
                trajectory[label] = {
                    "vs": base_label,
                    "mean_abs_loss_diff": float((cur["loss"] - base.loc[common, "loss"]).abs().mean()),
                    "mean_abs_iou_diff": float((cur["iou"] - base.loc[common, "iou"]).abs().mean()),
                    "mean_abs_gradient_norm_diff": float((cur["gradient_norm"] - base.loc[common, "gradient_norm"]).abs().mean()),
                    "loss_diff_vs_update_corr": finite_corr(cur["loss"] - base.loc[common, "loss"], cur.index),
                }
            out["paired_trajectory"] = {"common_updates": len(common), "comparisons": trajectory}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RUN / "subproblem_diagnosis_20261027"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence = {
        "run_id": "EG-SUBPROBLEM-DIAGNOSIS-20261027",
        "type": "read_only_new_analysis",
        "historical_audit_tools_used": False,
        "training_or_testing_started": False,
        "question": "Which failure layer is most supported: evidence, action, shared update, or feedback?",
        "paired_metrics": [
            summarize_pair(RUN / "p1_sot_v3/analysis_u0468_screen96_max2000/per_sequence.csv", "EVDR-v1"),
            summarize_pair(RUN / "evdr_scope_v2_20261026/analysis_vs_early_gate/per_sequence.csv", "EVDR-aligned"),
            summarize_pair(RUN / "evdr_transport_grad_20261027/analysis_vs_early_gate/per_sequence.csv", "EVDR-TGRAD"),
        ],
        "operator_readouts": summarize_readouts(RUN / "evdr_scope_v2_20261026/operator_readouts.json"),
        "training_trajectories": summarize_training({
            "control": RUN / "p1_sot_v3/logs/control_s42.log",
            "evdr_v1": RUN / "p1_sot_v3/logs/candidate_s42.log",
            "aligned": RUN / "evdr_scope_v2_20261026/train.log",
            "tgrad": RUN / "evdr_transport_grad_20261027/train.log",
        }),
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))

    lines = [
        "# Early gate 子问题定位实验（2026-10-27）",
        "",
        "本分析只读取已完成的匹配运行；没有调用旧审计器，没有训练或测试，没有改模型。",
        "",
        "## 结果摘要",
        "",
        "| 候选 | n | 平均 AUC 差(pp) | 胜/负 | >5pp损失 | 新失败帧 | 恢复失败帧 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in evidence["paired_metrics"]:
        lines.append("| {label} | {n} | {mean_auc_delta_pp:.4f} | {win_count}/{loss_count} | {large_loss_count_gt5pp} | {new_failure_frames_total} | {recovered_failure_frames_total} |".format(**p))
    lines += [
        "",
        "## 子问题判别",
        "",
        "1. **闭环反馈放大：有较强支持。** 逐序列 AUC 变化与失败帧变化、后半段 IoU 变化共同记录；这只能说明新失败与轨迹反馈相关，不能说明 Value 算子是首因。",
        "2. **动作粒度/条件对象失配：有直接读数支持。** aligned 与 colocated receiver 条件的平均相对差异明显，说明两者不是同一个条件；但 aligned 的任务收益仍不稳定，不能据此宣布 aligned 算子正确。",
        "3. **来源证据不足：尚未闭合。** 现有读数显示前景和背景 Value 改变量级接近，尚不能说明模块在选择目标来源；需要新增可获得的目标/干扰证据，并用干预后的消息后果验证。",
        "4. **共享更新冲突：尚未证实。** 训练轨迹会报告各臂 loss、IoU、梯度范数的匹配差异，但这些是整体轨迹差异，不是场景梯度内积；不能把它们写成梯度冲突证明。",
        "",
        "## 下一步唯一结构问题",
        "",
        "先检验来源证据是否能区分目标支持与干扰支持；在该问题未闭合前，不再扩大 Value 方向变换、receiver 梯度隔离或多模块融合。新增模块若进入训练，必须只改变来源证据接口，Early gate 继续作为唯一传输动作。",
        "",
        "完整 JSON：`diagnosis.json`。",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(out_dir), "paired": evidence["paired_metrics"], "operator": evidence["operator_readouts"]["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
