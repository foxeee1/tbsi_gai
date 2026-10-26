# M1 固定状态单模态候选可用性审计（2026-10-15）

## 结论

在完全相同的 M1 crop、tracker state、GT、CENTER head 和 decoder 下，单独 RGB/TIR 候选本身都低于 M1 fused，但候选并集存在明确的定位上限：GT 选优的 `best-of-three` 平均 IoU 为 `0.68778`，相对 M1 fused `+0.04396`，序列聚类 bootstrap 95% CI 为 `[+0.02057, +0.05269]`。因此，**候选级路由值得进入下一步 OOF selector 实验**；这 `+4.40` 个 IoU 点是 oracle 上限，不是可部署涨点。

## 性能对比

| 候选 | 平均 IoU | 中位数 IoU | 相对 M1 fused 平均 ΔIoU | 95% 序列聚类 CI | 正增益状态 | 负增益状态 |
|---|---:|---:|---:|---:|---:|---:|
| M1 fused | 0.64382 | 0.69920 | 0 | [0, 0] | 0/144 | 0/144 |
| RGB-only | 0.56484 | 0.58004 | -0.07899 | [-0.10730, -0.03524] | 47/144 | 95/144 |
| TIR-only | 0.45748 | 0.46535 | -0.18635 | [-0.23365, -0.14641] | 26/144 | 117/144 |
| Best-of-three oracle | 0.68778 | 0.71826 | **+0.04396** | **[+0.02057, +0.05269]** | 64/144 | 0/144 |

候选覆盖情况：RGB-only 胜过 fused 的状态为 `47/144`，TIR-only 为 `26/144`，至少一个单模态胜过 fused 的状态为 `64/144`；best-of-three 在所有 `144/144` 状态上不低于 fused。按序列均值统计，oracle 有正增益的序列 `42/72`，RGB-only 有正增益 `19/72`，TIR-only 有正增益 `8/72`。

## 逐序列对比

完整 72 条序列表已写入：

`output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015_retry2/per_sequence_candidates.csv`

该表包含每条序列的状态数、M1/RGB/TIR/oracle 平均 IoU、相对 M1 的平均变化，以及 RGB/TIR 胜出状态数。典型互补序列包括 `blackmanleft`（TIR-only 相对 fused `+0.40310`）、`treeboy`（RGB-only `+0.21813`）、`rightestblkboy2`（oracle `+0.26703`）；也存在 fused 已明显更优的序列，例如 `girlgoleft`，两种单模态均退化，oracle 无增益。因此必须学习条件化选择，不能固定切换到某一模态。

## 合同检查与限制

- 144/144 fixed states，72 条序列；feature/capture/trace key 完全一致，无重复。
- M1 fused IoU 与已有 capture 最大绝对差 `4.61e-05`。
- RGB/TIR 特征重新经 M1 fusion projection 的最大重组差 `0.00394`。
- FP32、单进程、0 workers、RTX 4090 D；耗时约 `158.0 s`。
- 这是 train-side fixed-state audit，不是闭环跟踪，也不是 official test。
- best-of-three 使用 GT 选候选，只能证明候选集合存在信息，不能证明 selector 可达到该上限。
- 单模态直接解码的负收益不能简单解释为模态无用，因为它们没有经过重新训练的单模态 head；本实验只证明“候选集合/当前解码合同”的互补性。

## 归因与下一步

结论标签：**支持下一步单变量检验**候选级选择器；**不支持**直接采用 RGB-only 或 TIR-only；**尚未支持**全量训练或 official test。

下一步冻结三候选生成路径，只训练或拟合一个 sequence-held-out selector，输入仅使用推理可见信号：三候选峰值/响应统计、候选间框中心与尺寸分歧、M1 fused 置信度和历史 state 一致性；输出 `fused / rgb / tir / noop`。先在同一 144 状态上做 leave-one-sequence-out，要求 selector 相对 fused 的 sequence-cluster CI 下界高于零，并同时检查 remaining162/low-AUC62/normal20 分层；未达到条件时不接入闭环、不做 full-244。

## 产物

- Runner：`tools/audit_m1_modality_candidate_availability_20261015.py`
- 最终指标：`output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015_retry2/metrics.json`
- 逐状态：`output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015_retry2/per_state_candidates.csv`
- 逐序列：`output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015_retry2/per_sequence_candidates.csv`
- Provenance：`output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015_retry2/manifest.json`
- 早期聚合失败记录：`output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015/failed_attempts.md`
