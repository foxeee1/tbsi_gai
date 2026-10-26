# TCSI Strategy v1：P0 机制审计

日期：2026-09-27  
状态：`DIAGNOSTIC` / `STOP_BEFORE_TRAINING`

本轮读取既有 `fast_fusion_snapshot`，不训练、不改模型、不修改 canonical YAML。输入共 3,420 行、60 条序列、层 3/6/9。

| Top-K | RGB->TIR mass | TIR->RGB mass | 双向均值 |
|---:|---:|---:|---:|
| 8 | 0.200934 | 0.201798 | 0.201366 |
| 16 | 0.359596 | 0.360885 | 0.360240 |
| 32 | 0.621865 | 0.623928 | 0.622897 |

全局 attention entropy 均值为 `4.793552`，cross-modal cosine delta 均值为 `0.003960`。集合分解 sanity 通过，但既有 snapshot 没有逐 head/query 的 `idx_rgb` 和 `idx_tir`，因此无法计算 exact shared/complementary/invalid ratio，也无法验证 GT 对齐的 support supervision。训练准入为 `STOP_BEFORE_TRAINING`，不能声称该方法优于 vanilla Top-K。

若继续，下一步只能新增 forward probe，保存 support indices、GT-aligned foreground/background mask、shared/complementary ratio 和参数量统计。

证据：`tools/analyze_tcsi_strategy_v1.py`；`output/experiments/tcsi_strategy_v1_p0_audit/analysis.json`；`output/experiments/tcsi_strategy_v1_p0_audit/analysis.md`；输入 SHA256 `80a03a82f53a81feec463ce2e7f12064ece25ee6068aa7366c5fc4f2c4f1b7db`。
