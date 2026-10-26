# M1 Query-conditioned Cross-Value fixed-state pretest

日期：2026-10-24

## 研究问题

验证 Early-conditioned MSHA gate 的第一优先级改动：让 cross-modal Value 残差同时依赖接收 query、来源 key 和早期三元证据，是否能在不改变原生 QK attention 的情况下改变目标候选排序。

## 实现

新增 `QueryConditionedCrossValue`，每个 `tbsi_e` MSHA 层、两个方向各一个低秩变换：

`delta_ij = U GELU(D [q_i, k_j, e_j])`

`e_j = [max_s cos(t_rgb_j,s_rgb), max_s cos(t_tir_j,s_tir), cos(t_rgb_j,t_tir_j)]`

`delta` 按来源 Value token 范数限制为最多 0.10 倍，新增 `up` 投影零初始化。Q/K、原生 attention、CENTER、框回放和模板更新均未修改。默认 feature gate 关闭；本次仅通过环境变量启用诊断分支，加载受保护 M1 checkpoint，不写回 M1 权重。

## 合同检查

- `py_compile`：通过。
- 随机张量 forward：通过。
- 单 batch backward：4 个 query-conditioned 参数张量均有 finite 非零梯度。
- save/load：输出最大绝对差 `0.0`。
- M1 checkpoint strict load：初次发现新增参数缺失，已在测试入口加入仅对 `query_conditioned_` 参数的诊断白名单；其他缺失或 unexpected key 仍 fatal。
- 28-event 输出：112 行，4 个条件臂，3 个 MSHA 层；无 NaN。

## 固定状态设计

使用预先选定的 28 个 train-side 事件：14 `matched_good`、14 `severe_onset`。同一 tracker state、同一 crop、同一 RNG 下比较：

1. `M1`：候选分支关闭；
2. `QUERY_CONDITIONED`：正常 query/key/evidence 条件；
3. `QUERY_SHUFFLED`：仅将 receiver query 循环移位；
4. `EVIDENCE_SHUFFLED`：仅将 source evidence 循环移位。

非零 probe 权重由固定 seed 生成，用于测量算子表达能力和信息依赖，不是训练后权重。

## 结果

四臂固定状态平均 IoU：

| 条件 | mean fixed-state IoU |
|---|---:|
| M1 | 0.554583 |
| QUERY_CONDITIONED | 0.554577 |
| QUERY_SHUFFLED | 0.554585 |
| EVIDENCE_SHUFFLED | 0.554577 |

相对 M1 的 fixed-state IoU 差：

| 条件 | mean | median | 正向事件 | 负向事件 |
|---|---:|---:|---:|---:|
| QUERY_CONDITIONED | -0.0000068 | +0.0000257 | 15 | 12 |
| QUERY_SHUFFLED | +0.0000017 | +0.0000086 | 16 | 11 |
| EVIDENCE_SHUFFLED | -0.0000069 | +0.0000167 | 15 | 12 |

28 个事件中，`raw_gt_rank` 三个候选臂相对 M1 均为 0 次改变。`raw_gt_hard_bg_margin` 平均变化为：正常 `-1.74e-5`，query shuffled `-5.39e-5`，evidence shuffled `-1.89e-5`。候选确实改变了下游 feature：正常条件 foreground/background query delta ratio 约 `1.10`，query shuffled 约 `1.06`，但没有传导为目标峰排序改变。

## 结论

`CONTRACT-PASSED / REPRESENTATION-CHANGE-PASSED / TARGET-RANK-MECHANISM-NOT-SUPPORTED / DO-NOT-TRAIN`

候选证明了 pairwise query-conditioned Value 分支可执行、可反向传播、能产生方向性 feature 改变；但在当前 M1 checkpoint 和随机 probe 权重下，目标 rank、peak 和 fixed-state IoU 没有改善，且打乱条件没有形成可区分的定位结果。因此不能进入训练测试，也不能把该算子写成已验证的定位改进。

## 归因与下一步

这次结果排除了“模块未执行、梯度断开、保存加载错误”三类解释，但尚未区分两种可能：

1. 当前训练监督不能驱动 query-conditioned 残差学习目标排序；
2. 只改 Value 内容而不改 attention support，仍不足以改变 CENTER 候选竞争。

下一步若继续，只允许做一个低成本诊断：使用同一 28-event 状态，加入受控的 target-aligned probe 或小幅 alpha 扫描，检查 CENTER 对 pairwise message 的敏感区间；不得直接启动训练或继续增大 MLP。若在受控幅度范围内 raw rank 仍完全不变，应停止该候选并转向保留空间位置的 token transport。

证据目录：`output/experiments/M1_MSHA_QUERY_CONDITIONED_VALUE_FIXED28_20261024/`。
