# M1 S1 后续实验前两步结果（2026-10-14）

## 目的与协议

按序完成两项判别实验：先审计 S1 首次介入与逐帧 IoU 变化的时序关系；再用 train-side 固定状态校准熵阈值，并在锁定阈值后做闭环对照。所有闭环臂使用受保护 M1 epoch15 checkpoint、同一 LasHeR union82 manifest（低 AUC 62 + normal 20）、当前 tracker、FP32 串行评测；未训练模型。M1 YAML 和 checkpoint 的 SHA256 分别为 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e` 与 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。环境为 1×RTX 4090 D 24 GiB、Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1。

有效比较为同环境当前 M1 run1027、S1 run1028 和仅增加熵准入阈值的 entropy run1026。另有 run1025 同时启用 peak-margin 与 entropy gate，属于混杂筛查，不用于熵的独立归因。评测范围是 proxy union82，不是 LasHeR full-244。

## 实验一：首次介入审计

82 序列共 68,380 帧。S1 序列宏平均 AUC 相对当前代码 M1 为 +2.8283。对 30 条 S1 AUC 下降序列检查首次介入帧：8 条 IoU 下降、18 条持平、4 条上升。`carcomingfromlight` 是首次动作即下降的风险实例。汇总的首次 offset 动作帧平均 ΔIoU 为 +0.00804，正值占 23.65%；no-op 帧平均 ΔIoU 为 +0.03820，正值占 26.48%。

这只是帧对齐描述：首次介入后两个 tracker 的状态和后续 crop 已分叉，不能把这些差异解释为同一状态下的动作因果效应。首次介入审计能定位值得复查的序列和时点，但没有证明首动作是序列最终退化的来源。

## 实验二：熵门控校准与闭环对照

train-side 72 序列、144 固定状态上校准得到 `max_entropy=0.9998925924301147`。原有 utility gate 在校准状态上的覆盖为 58.33%，覆盖样本 harmful rate 为 0%；熵值集中于 0.99943 至 0.99989，未显示额外风险区分力。该阈值随后固定用于 test-side 闭环，未按测试结果调参。

| Cohort | S1 对 M1 AUC / PR / NPR / OP50 / OP75 | Entropy 对 M1 AUC / PR / NPR / OP50 / OP75 | Entropy 对 S1 AUC 差及序列 bootstrap 95% CI |
|---|---|---|---|
| Low-AUC 62 | +3.73 / +5.11 / +4.70 / +5.08 / +3.06 | +2.58 / +3.76 / +3.09 / +3.30 / +1.69 | -1.15 `[-2.59, +0.15]` |
| Normal 20 | +0.03 / +0.02 / +0.08 / +0.01 / -0.02 | +0.06 / +0.06 / +0.15 / +0.03 / +0.00 | +0.03 `[0.00, +0.07]` |
| Union 82 | +2.83 / +3.87 / +3.57 / +3.84 / +2.31 | +1.97 / +2.86 / +2.37 / +2.50 / +1.28 | -0.86 `[-1.95, +0.14]` |

括号内区间跨零（normal 的下界为 0），因此 entropy 相对 S1 的 AUC 差没有显示稳健统计分离。动作计数方面，S1 为 37,648 offset / 30,650 no-op；entropy 为 34,337 offset / 33,961 no-op，即 offset 介入保留 91.2%。熵门控没有明显降低 normal cohort 的介入（normal offset 从 492 降至 478，保留 97.2%），却在 low-AUC cohort 将介入从 37,156 降至 33,859（保留 91.1%），并伴随低 AUC cohort 各项平均收益下降。

## 归因与决定

- **已排除**：本次校准的熵阈值没有展现出比现有 S1 admission 更有效的正常序列风险过滤能力；闭环平均结果还损失了 low-AUC 收益。熵门控不晋级。
- **暂不支持**：不能由首次介入帧审计断言“第一次修正造成最终退化”，因为干预后状态不再匹配。
- **支持下一步单变量检验**：S1 作为 union82 proxy 候选保留；若继续推进，先在固定 checkpoint、冻结策略下做 full-244 一次确认，并重点记录 normal 退化序列、首次介入前后状态差异和逐序列配对指标。该确认仍是单 seed 方法筛查，不能单独形成稳健涨点结论。

run1025 同时使用 peak-margin 与 entropy gate，变量混杂，登记为 `CONFOUNDED / NO ATTRIBUTION`。首次分析脚本把逗号分隔 GT 当制表符读取而失败；修正解析后重跑，原失败输出与 retry 证据均保留。未启动训练；本报告不主张 full-244 性能。

## 证据路径

- 首次介入逐帧及逐序列审计：`output/experiments/M1_S1_FIRST_TWO_STEPS_20261014_same_env_audit/`
- train-side 校准与三臂闭环逐序列结果：`output/experiments/M1_S1_FIRST_TWO_STEPS_20261014_step2_retry1/`
- 混合门控筛查及首次解析失败记录：`output/experiments/M1_S1_FIRST_TWO_STEPS_20261014/`
- 分析脚本：`tools/run_m1_s1_first_two_steps_20261014.py`（SHA256 前缀 `39fd6a01`）
