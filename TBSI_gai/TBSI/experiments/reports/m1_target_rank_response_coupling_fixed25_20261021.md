# M1 TargetRankGate CENTER response coupling 诊断

实验 ID：`M1-TARGET-RANK-GATE-RESPONSE-COUPLING-FIXED25-B003-20261021`  
目的：在相同 gate、template、crop、state 和 frame 下，验证 target score 是否能直接改变 CENTER 的决策排序。

## 唯一改变量

候选只打开 CENTER response coupling：

\[
R' = R + 0.03\frac{S}{\operatorname{std}(S)}
\]

其中 `S` 是已有 `TargetRankGate` score，`R` 是 CENTER 原始 response。control 使用同一 gate checkpoint，仅关闭 coupling。backbone、TBSI、fusion、CENTER 权重、template、crop 和 tracker state 完全匹配。

## 结果

证据：`output/experiments/M1_TARGET_RANK_GATE_RESPONSE_COUPLING_FIXED25_B003_20261021_RETRY3/{summary.json,fixed_state_paired.csv}`。

| 指标 | 结果 |
|---|---:|
| gate GT-background score margin delta | `0`（gate 相同） |
| CENTER target-background margin delta 平均 | `+0.03503` |
| CENTER target-background margin delta 中位数 | `+0.03512` |
| CENTER margin 上升事件 | `18/25` |
| CENTER GT rank delta 平均 | `+0.36` |
| CENTER GT rank 改善事件 | `1/25` |
| CENTER peak-GT margin delta 平均 | `+0.00218` |

## 结论

直接将 gate score 加到 CENTER response 可以明显改变 response margin，但没有稳定改善 GT rank，反而在平均 rank 上退化。这说明当前 score 的空间排序与 CENTER 的峰值选择目标并不一致；简单的 additive coupling 会制造新的局部峰，不能作为最终方法，也不应继续只搜索 `beta` 强度。

该结果支持停止当前 `TargetRankGate + response coupling` 组合的短训和 beta 搜索。若继续做方法，应先改变 score 的监督和空间结构：使用前景 prototype、top-k hard-background、Gaussian/soft target map，并让 loss 直接约束 CENTER response 的 GT rank。否则应将该分支保留为失败诊断证据。

本实验是 25 个固定状态事件的机制诊断，不是 full-244 性能结论。
