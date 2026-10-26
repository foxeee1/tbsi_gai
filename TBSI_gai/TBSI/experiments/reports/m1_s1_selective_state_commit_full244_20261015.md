# M1 S1 选择性状态提交（2026-10-15）

## 实验设计

Parent 为受保护 M1 epoch15；S1 readout predictor 与 independent action policy 固定。唯一变化是在内部 `self.state` 写回时，用 train-side selective-risk 分数决定提交 S1 修正框或回退到同帧 M1 框。对外返回框仍使用 S1 readout 结果。风险策略阈值 `0.2008109703` 未按 test 调整。

先运行 LasHeR 官方首序列合同预检 run1048：首个报告框与 S1 完全一致，303 帧 state commit 决策全部符合冻结阈值，14 帧提交修正状态、289 帧回退 M1。随后对官方 testingset 244/244 FP32 串行跟踪 run1049。总跟踪耗时 6565.76 秒，约 33.34 帧/秒；评测、逐序列指标与 persistence audit 均完成。

固定资产：M1 YAML SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`；M1 epoch15 checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`；S1 action predictor/policy SHA256 `b97bf2fe4d576a447f6cabe62a7523a2276f6837ffcc8761714b181119d79c6c` / `ff4cbef73341eae20c726911f916b25cd80b6a61458bba0a1e5196004840c098`；train-side risk policy SHA256 `551b077bb5db4ddb33875a8102c40e6d896b937bb9dff5c23025c30f935377e7`。

## 总体指标

| 指标 | M1 | S1 | S1输出/M1状态 | 选择性状态 | 候选-M1 | 候选-S1 | 候选-输出/M1状态 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AUC | 56.2077 | 56.4599 | 56.1370 | 55.7994 | -0.4083 | -0.6605 | -0.3376 |
| PR | 70.2073 | 70.5443 | 70.1870 | 69.7968 | -0.4105 | -0.7475 | -0.3901 |
| NPR | 66.1980 | 66.5330 | 66.0424 | 65.7167 | -0.4813 | -0.8163 | -0.3257 |
| OP50 | 68.3339 | 68.7700 | 68.2718 | 67.8358 | -0.4981 | -0.9341 | -0.4359 |
| OP75 | 47.1349 | 47.1488 | 46.8499 | 46.6885 | -0.4465 | -0.4603 | -0.1614 |

选择性候选相对 S1 的逐序列 AUC 胜/负/平为 102/141/1；相对 M1 为 104/132/8；相对输出/M1-state 组为 111/115/18。五项总体指标均低于 M1、S1 及输出/M1-state 组。

## Cohort 与提交覆盖

| Cohort | n | M1 AUC | S1 AUC | 输出/M1-state AUC | 选择性 AUC | 选择性-M1 | 选择性-S1 | state commit 覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low-AUC62 | 62 | 22.1797 | 25.9071 | 22.1448 | 23.2699 | +1.0902 | -2.6372 | 12.33% |
| normal20 | 20 | 85.8051 | 85.8081 | 85.7506 | 85.6160 | -0.1891 | -0.1921 | 3.21% |
| remaining162 | 162 | 65.5768 | 64.5298 | 65.4903 | 64.5679 | -1.0089 | +0.0381 | 7.34% |

remaining162 的候选相对 S1 五项差值 AUC/PR/NPR/OP50/OP75 为 `+0.0381/+0.1914/+0.0437/+0.0207/+0.0973`，序列 bootstrap 95% CI 分别为 `[-1.2167,+1.3435]`、`[-1.4977,+1.8127]`、`[-1.5056,+1.6807]`、`[-1.5412,+1.6017]`、`[-0.9552,+1.2836]`，均跨零。相对输出/M1-state 组，remaining162 AUC 仍低 `0.9224`。normal20 候选相对 S1 AUC 的序列 bootstrap 95% CI 为 `[-0.4602,-0.0110]`，显示该固定小组存在退化信号。

全量共 219,048 帧，其中 18,954 帧提交修正状态、200,094 帧回退到 M1，覆盖率 8.65%；冻结阈值规则逐帧违规数为 0。训练 OOF 覆盖率为 15.28%，测试覆盖明显更低。逐序列覆盖见 `output/experiments/M1_S1_SELECTIVE_STATE_COMMIT_FULL244_20261015/full244/state_commit_by_sequence.csv`。

## 归因结论

**当前实验排除**：按 same-state offset utility 训练的风险概率与阈值，不能作为有效的多帧 state-feedback harm 代理。该规则只放行 8.65% 帧；剩余序列 AUC 与 S1 近似且区间宽、与输出/M1-state 组相比仍明显更低；正常组 AUC 也低于 S1。总体五项指标全部回落到 M1 以下。

**尚未排除**：经闭环训练、直接以未来若干帧状态伤害为标签的状态回退器是否有效。当前候选不证明延迟提交、按序列/阶段门控或其他闭环风险特征无效。不得在这 244 条测试序列上调阈值后重报为独立验证。

下一步不应扩大融合算子，也不应围绕当前阈值做 test-side sweep。先在 LasHeR train-side 构造状态分叉闭环样本：同一帧并行维护 S1-state 与 M1-state rollout，监督未来固定 horizon 的定位/失效差值；按序列留出校准，再在新的固定诊断序列验证。若该风险目标在 train-heldout 上仍不能排序未来伤害，停止选择性 state gate，回到状态反馈机制本身（如短延迟提交或状态恢复策略）的单变量检验。

## 证据与限制

逐序列官方指标：`output/experiments/M1_S1_SELECTIVE_STATE_COMMIT_FULL244_20261015/full244/per_sequence_official.csv`；指标与 cohort bootstrap：`official_summary.json`；逐帧阈值核验和覆盖：`state_commit_audit.json` 与 `state_commit_by_sequence.csv`；持续失败审计：`persistence_audit/official_metric_and_persistence.json`。候选/M1 持续失败序列数均为 136；持续失败段为 433/430，最长段 1495/1494 帧，未显示恢复改善。

这是同一官方测试集上的机制诊断，不是独立泛化确认。风险器原标签是固定状态下的 same-state offset utility `<= 0`，并非未来帧状态伤害；因此负结果定位为“该风险目标/阈值不适配状态提交”，不能外推为所有风险门控策略无效。
