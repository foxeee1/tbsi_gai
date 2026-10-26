#!/usr/bin/env python3
"""Finalize official metrics for the completed output-only state run."""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from run_m1_s1_output_only_state_20261015 import (  # noqa: E402
    CONFIG, FULL_OUT, FULL_RUNID, M1_RUNID,
)

RESULT_ROOT = ROOT / "output/test/tracking_results/tbsi_track"
OUT = FULL_OUT / "full244"
RESULT_DIR = RESULT_ROOT / (CONFIG + "_" + FULL_RUNID)
SEQUENCE_FILE = OUT / "sequence_list.txt"
MANIFEST_FILE = OUT / "manifest.json"
REPORT_NAME = "M1_S1_OUTPUT_ONLY_STATE_FULL244_20261015"
PYTHON = "/root/autodl-tmp/conda_envs/tbsi/bin/python"


def names_from(path):
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def main():
    if not MANIFEST_FILE.is_file() or not SEQUENCE_FILE.is_file():
        raise RuntimeError("completed tracking manifest or sequence manifest is missing")
    manifest = json.loads(MANIFEST_FILE.read_text())
    sequences = [x.strip() for x in SEQUENCE_FILE.read_text().splitlines() if x.strip()]
    found = {p.stem for p in RESULT_DIR.glob("*.txt") if not p.name.endswith("_time.txt")}
    if manifest.get("status") not in ("TRACKING-COMPLETE", "COMPLETE / ANALYSIS-RECORDED") or len(sequences) != 244 or found != set(sequences):
        raise RuntimeError("run1044 is not a complete, manifest-matched official 244 prediction set")

    os.environ["TBSI_RPP_TEST_LIST"] = str(SEQUENCE_FILE)
    from lib.test.analysis.extract_results import extract_results
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker

    dataset = get_dataset("rpp_lasher_test")
    trackers = [Tracker("tbsi_track", CONFIG, "rpp_lasher_test", M1_RUNID),
                Tracker("tbsi_track", CONFIG, "rpp_lasher_test", "1031"),
                Tracker("tbsi_track", CONFIG, "rpp_lasher_test", FULL_RUNID)]
    eval_data = extract_results(trackers, dataset, REPORT_NAME, skip_missing_seq=False)
    overlap = np.asarray(eval_data["ave_success_rate_plot_overlap"], dtype=np.float64)
    center = np.asarray(eval_data["ave_success_rate_plot_center"], dtype=np.float64)
    center_norm = np.asarray(eval_data["ave_success_rate_plot_center_norm"], dtype=np.float64)
    metric_arrays = {"AUC": overlap.mean(axis=2) * 100.0, "PR": center[:, :, 20] * 100.0,
                     "NPR": center_norm[:, :, 20] * 100.0, "OP50": overlap[:, :, 10] * 100.0,
                     "OP75": overlap[:, :, 15] * 100.0}
    names = eval_data["sequences"]
    if len(names) != 244 or overlap.shape[:2] != (244, 3):
        raise RuntimeError("official evaluator did not return 244 sequences x 3 arms")

    labels = ["M1", "S1", "S1_OUTPUT_ONLY"]
    rows = []
    for i, name in enumerate(names):
        row = {"sequence": name}
        for metric, values in metric_arrays.items():
            for j, label in enumerate(labels):
                row[label + "_" + metric] = float(values[i, j])
            row["OUTPUT_ONLY_MINUS_M1_" + metric] = float(values[i, 2] - values[i, 0])
            row["OUTPUT_ONLY_MINUS_S1_" + metric] = float(values[i, 2] - values[i, 1])
        rows.append(row)
    with (OUT / "per_sequence_official.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    low = names_from(ROOT / "output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/s1_full62/sequence_list.txt")
    normal = names_from(ROOT / "output/experiments/M1_SINGLE_SIGNAL_NORMAL20_20261012/s1_full20/sequence_list.txt")
    name_set = set(names)
    if len(low) != 62 or len(normal) != 20 or low & normal or not (low | normal) <= name_set:
        raise RuntimeError("frozen low-AUC/normal cohort contract mismatch")
    cohorts = {"low_auc62": low, "normal20_sentinel": normal,
               "remaining162": name_set - low - normal, "all244": name_set}
    summary = {"n_sequences": 244, "metrics": {}, "cohorts": {}}
    for metric, values in metric_arrays.items():
        per_arm = {label: float(values[:, j].mean()) for j, label in enumerate(labels)}
        delta_s1 = values[:, 2] - values[:, 1]
        summary["metrics"][metric] = {
            **per_arm,
            "output_only_minus_m1": float((values[:, 2] - values[:, 0]).mean()),
            "output_only_minus_s1": float(delta_s1.mean()),
            "output_only_vs_s1_wins_losses_ties": [int((delta_s1 > 1e-8).sum()),
                                                    int((delta_s1 < -1e-8).sum()),
                                                    int((np.abs(delta_s1) <= 1e-8).sum())],
        }
    rng = np.random.RandomState(3407)
    for cohort, members in cohorts.items():
        ids = [i for i, name in enumerate(names) if name in members]
        entry = {"n_sequences": len(ids), "metrics": {}}
        for metric, values in metric_arrays.items():
            diff_m1, diff_s1 = values[ids, 2] - values[ids, 0], values[ids, 2] - values[ids, 1]
            boot = diff_s1[rng.randint(0, len(ids), size=(10000, len(ids)))].mean(axis=1)
            entry["metrics"][metric] = {
                "M1": float(values[ids, 0].mean()), "S1": float(values[ids, 1].mean()),
                "S1_OUTPUT_ONLY": float(values[ids, 2].mean()),
                "output_only_minus_m1": float(diff_m1.mean()),
                "output_only_minus_s1": float(diff_s1.mean()),
                "output_only_minus_s1_sequence_bootstrap_95ci": [float(np.quantile(boot, .025)),
                                                                    float(np.quantile(boot, .975))],
            }
        summary["cohorts"][cohort] = entry
    summary["official_eval_data"] = str(ROOT / "output/test/result_plots" / REPORT_NAME / "eval_data.pkl")
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    audit_dir = OUT / "persistence_audit"
    audit_command = [PYTHON, "-u", "tools/m1_official_metric_and_persistence_audit.py",
                     "--repo", str(ROOT), "--pred-dir", str(RESULT_DIR),
                     "--baseline-dir", str(RESULT_ROOT / (CONFIG + "_" + M1_RUNID)),
                     "--dataset", "rpp_lasher_test", "--output-dir", str(audit_dir)]
    audit_env = os.environ.copy()
    audit_env["TBSI_RPP_TEST_LIST"] = str(SEQUENCE_FILE)
    audit_started = time.time()
    with (OUT / "persistence_audit.log").open("w") as stream:
        subprocess.run(audit_command, cwd=str(ROOT), env=audit_env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)

    manifest["stages"]["official_evaluation"] = {"status": "COMPLETE", "sequences": 244,
                                                   "summary": str(OUT / "official_summary.json")}
    manifest["stages"]["persistence_audit"] = {"status": "COMPLETE",
                                                  "seconds": time.time() - audit_started,
                                                  "output": str(audit_dir)}
    manifest["analysis_recovery"] = {
        "initial_attempt": "FAILED: evaluation subprocess scope omitted TBSI_RPP_TEST_LIST",
        "recovery": "read-only official evaluation rerun using the frozen 244-sequence manifest; tracking was not repeated",
    }
    manifest["analysis_code"] = str(ROOT / "tools/finalize_m1_s1_output_only_state_20261015.py")
    manifest["analysis_code_sha256"] = subprocess.check_output(
        ["sha256sum", str(ROOT / "tools/finalize_m1_s1_output_only_state_20261015.py")]).decode().split()[0]
    manifest["status"] = "COMPLETE / ANALYSIS-RECORDED"
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    preflight_history = [
        (FULL_OUT / "preflight/manifest.json", "1045", "FAILED-PREFLIGHT / FLOAT-VS-QUANTIZED-ASSERTION",
         "State comparison tolerance was tighter than integer-quantized saved tracker output."),
        (FULL_OUT / "preflight_1046/manifest.json", "1046", "FAILED-PREFLIGHT / FRAME-ALIGNMENT-ASSERTION",
         "Compared initialization row with first tracked frame; corrected to align tracked rows."),
        (FULL_OUT / "preflight_1047/manifest.json", "1047", "PREFLIGHT-PASSED",
         "First tracked returned box matches S1; committed state and next-frame input match M1 within output quantization."),
    ]
    for path, run_id, status, note in preflight_history:
        if path.is_file():
            prior = json.loads(path.read_text())
            prior["status"] = status
            prior["preflight_note"] = note
            prediction = RESULT_ROOT / (CONFIG + "_" + run_id) / (sequences[0] + ".txt")
            prior["tracking_completed"] = prediction.is_file()
            if prior["tracking_completed"]:
                prior["stages"]["tracking"].update({"status": "COMPLETE", "prediction_files": 1})
            path.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n")

    report = [
        "# M1 S1 输出修正与状态更新解耦（2026-10-15）", "",
        "## 实验设计", "",
        "以受保护 M1 epoch15 为底座，冻结 S1 predictor/policy 与 readout 动作。候选仍返回 S1 修正框，",
        "但内部 `self.state` 使用未修正的 M1 框更新；因此只改变状态提交路径。全量 LasHeR official testingset，244/244，FP32。", "",
        "结果属于同一官方测试集上的机制诊断，不能作为独立泛化确认。", "",
        "## 总体指标", "",
        "| 指标 | M1 | S1 耦合更新 | S1 输出、M1 状态 | 候选-M1 | 候选-S1 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ["AUC", "PR", "NPR", "OP50", "OP75"]:
        x = summary["metrics"][metric]
        report.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} |".format(
            metric, x["M1"], x["S1"], x["S1_OUTPUT_ONLY"],
            x["output_only_minus_m1"], x["output_only_minus_s1"]))
    report += ["", "## Cohort AUC", "", "| Cohort | n | M1 | S1 | 输出解耦 | 候选-M1 | 候选-S1 |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for cohort in ["low_auc62", "normal20_sentinel", "remaining162", "all244"]:
        e = summary["cohorts"][cohort]
        x = e["metrics"]["AUC"]
        report.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} |".format(
            cohort, e["n_sequences"], x["M1"], x["S1"], x["S1_OUTPUT_ONLY"],
            x["output_only_minus_m1"], x["output_only_minus_s1"]))
    report += ["", "## 逐序列与持久性", "",
               "完整 244 序列、五项指标及候选相对 M1/S1 的差值见 `output/experiments/M1_S1_OUTPUT_ONLY_STATE_FULL244_20261015/full244/per_sequence_official.csv`。",
               "候选相对 S1 的 AUC 序列胜/负/平为 108/131/5。持续失效审计中，候选/M1 的失效序列为 135/136，失效段为 427/430，最长失效段均为 1494 帧；整体持久性变化很小。", "",
               "remaining162 中候选相对 S1 的五项变化及序列 bootstrap 95% CI：", "",
               "| 指标 | 候选-S1 | 95% CI |", "|---|---:|---:|"]
    remaining_metrics = summary["cohorts"]["remaining162"]["metrics"]
    for metric in ["AUC", "PR", "NPR", "OP50", "OP75"]:
        x = remaining_metrics[metric]
        report.append("| {} | {:+.4f} | [{:+.4f}, {:+.4f}] |".format(
            metric, x["output_only_minus_s1"], *x["output_only_minus_s1_sequence_bootstrap_95ci"]))
    report += ["", "## 归因结论", "",
               "支持：S1 的内部状态反馈是 remaining162 退化的重要来源。状态提交改回 M1 后，remaining162 五项指标都高于 S1，AUC 从相对 M1 `-1.0470` 收敛到 `-0.0865`；但 OP75 仍低于 M1 `0.3816`。",
               "不支持：对所有序列全局使用 M1 状态。low-AUC62 的 S1 收益基本全部消失，候选 overall AUC 也低于 M1 `0.0707`。这说明状态反馈同时承载了低分序列的收益，不能简单禁用。",
               "下一步应做单变量的选择性状态提交/回退：仅当动作在连续证据下可靠时将修正框提交为下一帧状态，否则保留 M1 状态；保持 readout 输出动作不变。提交准则只用 train-side 拟合，再用独立固定序列筛选，不能按该 full-test 结果调阈值。", "",
               "当前实验不支持转向候选级融合：这里只能说明状态路径影响收益/退化，尚未证明正确目标只存在于单模态候选。", "",
               "## 归因边界", "",
               "输出框仍是 S1 框，所以本实验不测试内部 M1 状态对外部评价框的质量；状态反馈结论限于当前 frozen predictor/policy 与该测试集。",
               "首个汇总尝试因评测进程未继承 `TBSI_RPP_TEST_LIST` 失败；修复后只重跑只读评测和审计，未重复跟踪。", ""]
    report_path = ROOT / "experiments/reports/m1_s1_output_only_state_full244_20261015.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"metrics": summary["metrics"], "cohorts": summary["cohorts"],
                      "report": str(report_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
