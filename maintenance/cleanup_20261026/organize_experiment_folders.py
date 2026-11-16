#!/usr/bin/env python3
"""Organize TBSI experiments/output into active and deprecated folders.

This is a one-time, explicit allowlist cleanup. Deprecated experiment reports
and registry records are retained; non-report artifacts outside the allowlist
are removed after a SHA256 inventory is written.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "TBSI_gai" / "TBSI"
EXP = ROOT / "experiments"
OUT = ROOT / "output"
MAINT = Path(__file__).resolve().parent

BASE_CONFIGS = {
    "v1.0.1-单卡4090D-24G-等效全局128加速.yaml",
    "v1.0.3-单卡4090D-24G-等效全局128-seed3407.yaml",
    "TBSI_E_MHSA_FULL_4090D_seed42.yaml",
    "M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023.yaml",
    "M1_AMCM_E2_FULL244_20261017.yaml",
    "M1_AMCM_E2_10EP_FULL244_20261017.yaml",
    "M1_AMCM_E2_FULLDATA_10EP_20261018.yaml",
    "M1_AMCM_E2_FULLDATA_10EP_M1MATCH_20261018.yaml",
    "M1_AMCM_E2_SEED3407_10EP_FULL244_20261018.yaml",
}

ACTIVE_REPORTS = {
    "tbsi_e_mhsa_full_20260927.md",
    "m1_early_conditioned_gate_operator_sequence_analysis_20261023.md",
    "m1_early_conditioned_msha_gate_proxy_20261021.md",
    "m1_early_gate_bottleneck_audit_20261026.md",
    "m1_early_gate_candidate_checkpoint_ab_20261024.md",
    "m1_early_gate_control_repeatability_20261026.md",
    "m1_early_gate_sequence_scene_attribution_20261026.md",
    "m1_dcvr_early_gate_full244_sequence_comparison_20261026.md",
    "m1_amcm_e2_10ep_50seq_20261017.md",
    "m1_amcm_e2_expanded_proxy_20261017.md",
    "m1_amcm_e2_full244_comparison_20261018.md",
    "m1_amcm_e2_fulldata_10ep_full244_20261019.md",
    "m1_amcm_e2_per_sequence_diagnosis_20261018.md",
    "m1_amcm_e2_seed3407_confirmation_20261018.md",
}

ACTIVE_SCRIPTS = {
    "analyze_m1_early_conditioned_gate_full244_20261023.py",
    "analyze_m1_amcm_e2_seed_confirmation_20261018.py",
    "analyze_amcm_e2_fulldata.py",
}

ACTIVE_RUNS = {
    "TBSI_E_MHSA_FULL_4090D_seed42",
    "v1.0.1-单卡4090D-24G-等效全局128加速",
    "v1.0.3-单卡4090D-24G-等效全局128-seed3407",
    "M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023",
    "M1_AMCM_E2_FULL244_20261017",
    "M1_AMCM_E2_10EP_FULL244_20261017",
    "M1_AMCM_E2_FULLDATA_10EP_20261018",
    "M1_AMCM_E2_FULLDATA_10EP_M1MATCH_20261018",
    "M1_AMCM_E2_SEED3407_10EP_FULL244_20261018",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_report(source: Path, destination_root: Path, prefix: str) -> None:
    relative = source.relative_to(ROOT)
    destination = destination_root / prefix / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)


def prepare_dirs(parent: Path) -> tuple[Path, Path]:
    active = parent / "主实验"
    deprecated = parent / "过去废弃实验"
    active.mkdir(parents=True, exist_ok=True)
    deprecated.mkdir(parents=True, exist_ok=True)
    return active, deprecated


def main() -> None:
    active_exp, old_exp = prepare_dirs(EXP)
    active_out, old_out = prepare_dirs(OUT)
    removed_rows: list[dict[str, str]] = []
    experiment_reports = [
        p for p in EXP.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".tex", ".pdf"}
        and "主实验" not in p.relative_to(EXP).parts
        and "过去废弃实验" not in p.relative_to(EXP).parts
    ]
    output_reports = [
        p for p in OUT.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".tex", ".pdf"}
        and "主实验" not in p.relative_to(OUT).parts
        and "过去废弃实验" not in p.relative_to(OUT).parts
    ]

    # Preserve all historical report files, including reports embedded in run
    # folders, before removing non-allowlisted artifacts.
    for report in experiment_reports:
        if report.name in ACTIVE_REPORTS:
            destination = active_exp / "报告" / report.name
        else:
            destination = old_exp / "报告" / report.relative_to(EXP)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if report.resolve() != destination.resolve():
            shutil.copy2(report, destination)

    for report in output_reports:
        copy_report(report, old_out, "报告")

    # Keep the asset registry as the historical report index.
    for name in ("实验资产数据库.md", "experiment_asset_registry.csv", "registry.csv"):
        path = EXP / name
        if path.exists():
            shutil.copy2(path, old_exp / name)

    # Retain exactly the selected experiment configs and analysis scripts.
    config_dir = active_exp / "配置"
    config_dir.mkdir(exist_ok=True)
    for path in EXP.rglob("*.yaml"):
        if (
            path.name in BASE_CONFIGS
            and "主实验" not in path.relative_to(EXP).parts
            and "过去废弃实验" not in path.relative_to(EXP).parts
        ):
            shutil.copy2(path, config_dir / path.name)

    script_dir = active_exp / "分析脚本"
    script_dir.mkdir(exist_ok=True)
    for path in ROOT.rglob("*.py"):
        if path.name in ACTIVE_SCRIPTS and not any(
            parent in {"主实验", "过去废弃实验"} for parent in path.parts
        ):
            shutil.copy2(path, script_dir / path.name)

    # Keep full experiment directories for M1, the requested Early gate and
    # AMCM E2 full-train/full-test runs, plus the two baseline runs.
    run_root = OUT / "experiments"
    active_runs_root = active_out / "实验"
    active_runs_root.mkdir(exist_ok=True)
    if run_root.exists():
        for run in sorted(run_root.iterdir()):
            if run.is_dir() and run.name in ACTIVE_RUNS:
                shutil.move(str(run), str(active_runs_root / run.name))

    # Retain only selected tracker prediction directories.
    tracking_root = OUT / "test" / "tracking_results" / "tbsi_track"
    active_tracking = active_out / "测试结果" / "tracking_results" / "tbsi_track"
    if tracking_root.exists():
        for result in sorted(tracking_root.iterdir()):
            if not result.is_dir():
                continue
            name = result.name
            keep = (
                name in ACTIVE_RUNS
                or name.startswith("M1_AMCM_E2_FULL244_20261017")
                or name.startswith("M1_AMCM_E2_10EP_FULL244_20261017")
                or name.startswith("M1_AMCM_E2_FULLDATA_10EP_20261018")
                or name.startswith("M1_AMCM_E2_SEED3407_10EP_FULL244_20261018")
                or name.startswith("M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023")
            )
            if keep:
                destination = active_tracking / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(result), str(destination))

    # Retain only the final numbered checkpoint per selected run. If a run has
    # no numbered checkpoint, keep a uniquely named final/latest checkpoint.
    for run in active_runs_root.iterdir():
        if not run.is_dir():
            continue
        checkpoints = [
            p for p in run.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pth", ".tar"}
        ]
        numbered = []
        for checkpoint in checkpoints:
            import re
            match = re.search(r"(?:ep|epoch)[_-]?(\d+)", checkpoint.name, re.IGNORECASE)
            if match:
                numbered.append((int(match.group(1)), checkpoint))
        if numbered:
            final_epoch = max(epoch for epoch, _ in numbered)
            keep_checkpoints = {p for epoch, p in numbered if epoch == final_epoch}
        else:
            named_final = [p for p in checkpoints if "final" in p.name.lower() or "latest" in p.name.lower()]
            keep_checkpoints = set(named_final) if len(named_final) == 1 else set()
        for checkpoint in checkpoints:
            if checkpoint not in keep_checkpoints:
                removed_rows.append({
                    "path": str(checkpoint.relative_to(ROOT)),
                    "size_bytes": str(checkpoint.stat().st_size),
                    "sha256": sha256(checkpoint),
                })
                checkpoint.unlink()

    # Record hashes for checkpoints that are intentionally not retained.
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pth", ".tar"}:
            rel = str(path.relative_to(ROOT))
            keep = any(part in ACTIVE_RUNS for part in path.parts)
            if keep:
                continue
            removed_rows.append({"path": rel, "size_bytes": str(path.stat().st_size), "sha256": sha256(path)})

    inventory = MAINT / "folder_cleanup_removed_checkpoints.csv"
    with inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(removed_rows)

    # Remove everything else below experiments/output after report extraction
    # and active-run relocation. Never touch TBSI/external_repos.
    for parent, keep_dirs in ((EXP, {"主实验", "过去废弃实验"}), (OUT, {"主实验", "过去废弃实验"})):
        for child in list(parent.iterdir()):
            if child.name in keep_dirs:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    # Prune non-report content from the deprecated folders; the registry files
    # above remain as indexes, and reports are preserved with relative paths.
    for parent in (old_exp, old_out):
        for path in sorted(parent.rglob("*"), reverse=True):
            if path.is_file() and path.suffix.lower() not in {".md", ".tex", ".pdf", ".csv"}:
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    (active_exp / "README.md").write_text(
        "# 主实验\n\n保留 M1、两份 4090D 基线、Early-conditioned MSHA gate full15 和 AMCM E2 全量实验。"
        "各组配置、结果、日志、权重及相关分析材料集中于本目录。\n\n"
        "注意：v1.0.3 配置与日志存在，但整理前未在 output 中发现对应 checkpoint。\n",
        encoding="utf-8",
    )
    (old_exp / "README.md").write_text(
        "# 过去废弃实验\n\n仅保留历史实验报告和资产登记索引；非保留实验的配置、权重、日志、预测和分析中间产物已清理。\n",
        encoding="utf-8",
    )
    (active_out / "README.md").write_text(
        "# 主实验输出\n\n保留 M1、两份基线、Early-conditioned MSHA gate full15 与 AMCM E2 全量组的训练/测试产物。\n",
        encoding="utf-8",
    )
    (old_out / "README.md").write_text(
        "# 过去废弃实验输出\n\n此处仅保留历史实验报告；非保留组的权重、日志、预测与中间分析产物已清理。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
