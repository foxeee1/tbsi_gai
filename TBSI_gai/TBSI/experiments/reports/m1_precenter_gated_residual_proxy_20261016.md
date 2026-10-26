# M1 pre-CENTER gated residual proxy（2026-10-16）

## 研究问题

测试简洁的 `pre-CENTER gated residual fusion`：只在 CENTER 解码前修改 fused feature，不提交新 tracker state，检查能否保持 normal 序列稳定并改善融合表示。

## 合同

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch-15 checkpoint。
- Candidate：`M1_PRECENTER_GATED_RESIDUAL_PROXY_20261016`，epoch-3，只有 `precenter_gated_residual.*` 可训练。
- 结构：`fused + sigmoid(gate(context)) * tanh(residual(context)) * 0.10`，residual projection zero-init，初始为 identity；不改 decoder，不改 tracker state。
- 数据：LasHeR normal20 control，固定清单 SHA256 `635905661a3be4ea8e9a9a5a28c01268d36137800c0a73b21f2b83b884023a2b`。
- 评测：FP32、串行、单 GPU、同一 tracker/evaluator；M1 run `3003`，candidate run `3002`。
- 代码 commit：`09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`。

## 结果

| arm | AUC | Precision | Norm Precision | mean IoU |
|---|---:|---:|---:|---:|
| M1 | 85.8051 | 98.3701 | 98.7767 | 87.5839 |
| pre-CENTER residual | 85.7663 | 98.2955 | 98.8287 | 87.5743 |
| candidate - M1 | -0.0388 | -0.0745 | +0.0520 | -0.0095 |

逐序列 AUC：8 胜、12 负，median delta `-0.0565`。最大正向序列为 `blueboy +0.3896`、`minibus +0.3052`、`darkcarturn +0.2835`；最大负向序列为 `mototurneast -0.5074`、`bikeboyturntimes -0.4115`、`whitecarcomeinrain -0.2835`。完整记录在 `output/experiments/M1_PRECENTER_GATED_RESIDUAL_PROXY_20261016/paired_normal20.json`。

## 额外运行

同一 candidate 的 RPP-v1 60 序列评测在完成 4/60 后因超长序列中止；结果目录和日志保留，不纳入性能结论。

## 结论与路由

结论：`INCONCLUSIVE / NO-GO-FOR-FULL-TRAINING`。合同、梯度和代理可拟合性通过；normal20 没有 AUC/IoU 增益，只有极小的 Norm Precision 正差，尚不足以证明 CENTER 前融合修正有效，也没有证据支持进入 full-244。

下一步只做一个最小机制检验：在已定义的 severe-onset/remaining-like 固定状态 cohort 上比较 M1 与 candidate 的 CENTER top-1、GT-cell recall 和 fixed-state IoU，并读取 gate/residual 统计。如果困难 cohort 仍无一致改善，停止该模块；如果只在困难 cohort 改善，再考虑一次同 parent 的短训确认，保留 no-op/identity fallback。

## 固定状态机制审计结果

在同一 crop、同一 `tracker_state_before`、同一 123 个事件上重放 M1 与 candidate：

| split | M1 fixed-state IoU | candidate | delta | M1 GT-cell top-1 recall | candidate |
|---|---:|---:|---:|---:|---:|
| all (123) | 0.62283 | 0.62081 | -0.00202 | 21.14% | 20.33% |
| matched-good (60) | 0.79318 | 0.79253 | -0.00064 | 20.00% | 20.00% |
| pre-onset (31) | 0.68605 | 0.68421 | -0.00184 | 38.71% | 38.71% |
| severe-onset (32) | 0.24217 | 0.23741 | -0.00476 | 6.25% | 3.125% |

结论升级为 `REFUTED / STOP-PRECENTER-RESIDUAL`：该 residual 没有改善 CENTER 选择或 severe-onset fixed-state 定位，且在最关键的 severe-onset cohort 上略有恶化。停止该结构的短训、参数调节和 full-244 评测。剩余增量应回到“中心选择/候选可达性”的直接机制，优先研究低复杂度的中心重排或 crop 可达性补偿，而不是继续扩大 feature residual。

## 中心候选重排审计

在同一 123 个固定状态上，对 M1 和 candidate 的 top-8 候选做离线分析：

| arm / split | top-1 IoU | top-8 oracle IoU | center-prior IoU | margin<0.005 + no-op |
|---|---:|---:|---:|---:|
| M1 / all | 0.6228 | 0.7346 | 0.6708 | 0.6237 |
| M1 / matched-good | 0.7932 | 0.8352 | 0.7824 | 0.7838 |
| M1 / severe-onset | 0.2422 | 0.5475 | 0.4444 | 0.2602 |
| candidate / all | 0.6208 | 0.7328 | 0.6699 | 0.6328 |
| candidate / severe-onset | 0.2374 | 0.5457 | 0.4452 | 0.2836 |

top-8 oracle 说明中心候选仍有明显可达性上限；但中心先验会伤害 matched-good，margin `<0.005` 的 no-op 版本也没有可靠的全体收益。结论为 `CENTER-ORACLE-HEADROOM-SUPPORTED / SIMPLE-RERANKER-NOT-SUPPORTED`。不把 oracle 当作可实现涨点，也不把当前中心先验接入闭环。若继续，只允许做一次独立训练/验证的候选重排器，并必须使用序列留出和 no-op 目标；否则应停止该路线。

## 轻量候选重排器 OOF

在 M1 的 top-8 候选上拟合一个 ridge utility ranker，输入仅为 `rank/x/y/score/margin/center_dist/size/offset`，预测 utility 不为正时执行 no-op。按序列哈希固定划分 47 条训练序列和 13 条留出序列，共 29 个 held-out 状态。

| 指标 | held-out 结果 |
|---|---:|
| sequence-macro selected utility | +0.0471 IoU |
| sequence-macro oracle utility | +0.0848 IoU |
| oracle capture | 55.5% |
| positive / negative sequences | 7 / 2 |
| abstention rate | 13.8% |
| severe-onset mean utility | +0.2204（8 状态） |
| matched-good mean utility | +0.0056（13 状态） |

结论：`OOF-MECHANISM-SIGNAL-SUPPORTED / CLOSED-LOOP-NOT-YET-VALIDATED`。候选效用信号比固定中心先验更有希望，尤其在 severe-onset；但样本仍小、单一划分、存在负收益序列，不能据此宣称最终涨点。下一步只做一次小规模闭环验证：冻结该 ranker、明确 no-op，使用独立 severe-onset 与 normal-control 序列，比较 M1 与 ranker 的逐序列结果；通过后再考虑短训或 full test。
