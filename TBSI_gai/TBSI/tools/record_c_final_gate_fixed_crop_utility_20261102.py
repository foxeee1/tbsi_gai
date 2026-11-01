#!/usr/bin/env python3
"""Register every C final-gate fixed-crop diagnostic attempt."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNROOT = ROOT / "output/experiments"
CHECKPOINT = RUNROOT / "EG_C_EARLY_LR_DROP_U936_FULL15_FULL244_S42_20261101/C/train/checkpoint_u7020.pth.tar"
CONFIG = ROOT / "experiments/tbsi_track/EG_C_EARLY_LR_DROP_U936_SOT_S42_20261101.yaml"
CASES = ROOT / "experiments/early_gate_closed_loop_20261026/c_final_gate_utility_20261102/cases.json"
REPORT = ROOT / "experiments/early_gate_closed_loop_20261026/c_final_gate_utility_20261102/REPORT.md"


def record(run_id, commit, output_name, status, conclusion, gpu_hours="NA", notes=""):
    return {
        "ID": run_id,
        "Date": "20261102",
        "Parent": "C early-LR-drop u7020 full244; fixed-checkpoint mechanism diagnostic",
        "CommitOrSnapshot": "code commit " + commit + "; exact source details in output manifest; C run snapshot under EG_C_EARLY_LR_DROP_U936_FULL15_FULL244_S42_20261101/C/source_at_launch",
        "Config": str(CONFIG) + " sha256=159ff0e31ea1e55dab5b3ac229e7472b050ea98ab257b3da581feb1c77a44f8d",
        "Seed": "42",
        "Dataset": "LasHeR testingset; 8 predeclared sequences x frame 50; case manifest " + str(CASES),
        "EvalScope": "Same saved C state at frame 49 and same current crop; native final gate versus g=1; GT post-hoc only; no full test",
        "AUC": "NA", "PR": "NA", "NPR": "NA", "OP50": "NA", "OP75": "NA", "FPS": "NA",
        "GPU": "1x NVIDIA GeForce RTX 4090 D 24 GiB",
        "HardwareConfig": "Python 3.8.20; PyTorch 2.0.0+cu118; CUDA 11.8; FP32; tracker threads=0; torch threads=1; OpenCV threads=0; serial single process",
        "GPUHours": gpu_hours,
        "Checkpoint": str(CHECKPOINT) + " sha256=8742d0cd456084e6bd2cb39e8524d1fa5e5b33c9cf7f1eba72ce3bb4cb29148e",
        "AssumptionOrHypothesis": "The trained C source-gain action may change immediate localization relative to an unconditional g=1 cross-message action on fixed states.",
        "Conclusion": conclusion,
        "Status": status,
        "Evidence": str(RUNROOT / output_name),
        "Notes": notes,
    }


def main():
    output4 = RUNROOT / "EG_C_FINAL_GATE_FIXED_CROP_UTILITY_RETRY4_20261102"
    completed = json.loads((output4 / "completed.json").read_text())
    completed3 = json.loads((RUNROOT / "EG_C_FINAL_GATE_FIXED_CROP_UTILITY_RETRY3_20261102/completed.json").read_text())
    cases = completed["cases"]
    deltas = [item["branches"][0]["iou"] - item["branches"][1]["iou"] for item in cases]
    mean_delta = sum(deltas) / len(deltas)
    mean_abs = sum(abs(value) for value in deltas) / len(deltas)
    rows = [
        record("EG-C-GATE-UTILITY-INITIAL-20261102", "f3d1a7d", "EG_C_FINAL_GATE_FIXED_CROP_UTILITY_20261102",
               "FAILED-SOURCE-PARITY / NO-INTERVENTION-RESULT",
               "Mutable active source diverged from the C full244 trajectory at carcominginlight frame 24; no branch result interpreted.",
               notes="Strict parity retained. Evidence: first output manifest and this run's launch traceback. No training."),
        record("EG-C-GATE-UTILITY-SNAPSHOT-RETRY1-20261102", "7461d24", "EG_C_FINAL_GATE_FIXED_CROP_UTILITY_RETRY1_20261102",
               "FAILED-SNAPSHOT-DEPENDENCY / ZERO-TRACKING",
               "Abbreviated exact_parent_code snapshot lacked early_conditioned_msha_gate.py; stopped during import before inference.",
               notes="No model forward or intervention. The missing dependency was later recovered from a complete same-day source snapshot."),
        record("EG-C-GATE-UTILITY-PREFIX-RETRY2-20261102", "7461d24", "EG_C_FINAL_GATE_FIXED_CROP_UTILITY_RETRY2_20261102",
               "FAILED-PREFIX-PARITY / NO-INTERPRETATION",
               "Complete model-source overlay still diverged from saved free-running trajectory at carcominginlight frame 24; strict parity was not relaxed.",
               notes="No intervention result interpreted. This motivated restoring the saved t-1 tracker box directly; C temporal tokens and online template updates are disabled."),
        record("EG-C-GATE-UTILITY-REPLAY-RETRY3-20261102", "1bc00a5", "EG_C_FINAL_GATE_FIXED_CROP_UTILITY_RETRY3_20261102",
               "DIAGNOSTIC-COMPLETE / ACTION-STRENGTH-UNCAPTURED / SUPERSEDED",
               "Saved-state fixed-crop native-versus-g=1 predictions completed and native output passed saved C box parity; gate action magnitudes were not included.",
               gpu_hours="%.6f" % (completed3["elapsed_seconds"] / 3600.0),
               notes="Results retained as an intermediate diagnostic and superseded by retry4, which adds per-layer action-strength capture."),
        record("EG-C-GATE-UTILITY-FIXED-CROP-RETRY4-20261102", "61334da", "EG_C_FINAL_GATE_FIXED_CROP_UTILITY_RETRY4_20261102",
               "DIAGNOSTIC-COMPLETE / FIXED-CROP-UTILITY-NOT-SUPPORTED-ON-8-CASES",
               "8/8 pairs had identical state/crop hashes and native saved-box parity. Mean signed delta IoU g(native)-g(1)=%+.6f; mean absolute delta=%.6f. Native log-gain was nonzero, but immediate localization differences were tiny; no propagation or population claim." % (mean_delta, mean_abs),
               gpu_hours="%.6f" % (completed["elapsed_seconds"] / 3600.0),
               notes="Full per-case results and limitations: " + str(REPORT) + ". No training or full test."),
    ]

    table = ROOT / "experiments/experiment_asset_registry.csv"
    with table.open(newline="", encoding="utf-8") as stream:
        existing = list(csv.DictReader(stream))
        fields = list(existing[0])
    by_id = {row["ID"]: index for index, row in enumerate(existing)}
    for row in rows:
        if row["ID"] in by_id:
            existing[by_id[row["ID"]]] = row
        else:
            existing.append(row)
    temp_table = table.with_suffix(".csv.tmp")
    with temp_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
    temp_table.replace(table)

    markdown = ROOT / "experiments/实验资产数据库.md"
    text = markdown.read_text(encoding="utf-8")
    marker = "<!-- EG-C-GATE-UTILITY-FIXED-CROP-20261102 BEGIN -->"
    end_marker = "<!-- EG-C-GATE-UTILITY-FIXED-CROP-20261102 END -->"
    body = [marker, "## C final gate fixed-crop utility diagnostic (2026-11-02)", "",
            "完整逐帧结果、执行合同和解释边界见 `" + str(REPORT) + "`。所有失败尝试均保留，最终有效结果为 RETRY4。", "",
            "| ID | 状态 | 结果 / 结论 | 证据 |", "|---|---|---|---|"]
    for row in rows:
        body.append("| `%(ID)s` | %(Status)s | %(Conclusion)s | `%(Evidence)s` |" % row)
    body.extend(["", "GT 仅用于推理后评分。RETRY4 对比 native C 最终 gate（包含 search-context 残差）与将两路 logit 同时归零得到的 `g=1`；8 个固定 frame 的状态与 crop hash 一致。该筛选诊断不代表 full-set 或闭环传播结论。", end_marker])
    block = "\n".join(body)
    if marker in text:
        left = text.index(marker)
        right = text.index(end_marker, left) + len(end_marker)
        text = text[:left] + block + text[right:]
    else:
        text += "\n\n" + block + "\n"
    markdown.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
