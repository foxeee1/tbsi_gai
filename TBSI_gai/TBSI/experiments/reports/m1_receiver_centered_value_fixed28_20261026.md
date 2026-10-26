# M1 receiver-centered cross-Value residual fixed28 机制筛查（2026-10-26）

## 目的

验证一个比 Early-conditioned MSHA gate 更细的单变量假设：保持原生 MSHA 的 Q/K 和 attention 分布不变，只依据早期 RGB/TIR template-search 对应关系，为每个接收 template token 构造 source-specific transport；从跨模态 Value 消息中减去前景先验均值，加入按原生 cross-attention mass 限幅的有界 residual。

四臂为 `NATIVE`、`RECEIVER_CENTERED_CORRECT`、`RECEIVER_CENTERED_SHIFTED` 和 `RECEIVER_CENTERED_ROW_SHUFFLED`。后两者分别破坏搜索空间对应关系或接收端关系行，以区分真实关系信号与同幅度扰动。

## 协议

- parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15 checkpoint
- 28 个固定 train-side calibration event：14 `matched_good` + 14 `severe_onset`
- FP32，串行固定状态回放，逐臂恢复同一 tracker state 和 RNG
- 模板更新、query-conditioned probe、FCSV-R、cross-support 全部关闭
- 仅 receiver-centered 路径由环境变量切换；未训练、未写 checkpoint、未运行 full benchmark

## 结果

| arm | mean fixed-state IoU | 相对 native | wins / losses |
|---|---:|---:|---:|
| NATIVE | 0.5188073 | 0 | - |
| RECEIVER_CENTERED_CORRECT | 0.5187978 | -0.0000095 | 13 / 14 |
| RECEIVER_CENTERED_SHIFTED | 0.5189727 | +0.0001654 | 17 / 10 |
| RECEIVER_CENTERED_ROW_SHUFFLED | 0.5189269 | +0.0001196 | 15 / 12 |

correct 在 `matched_good` 的平均变化为 `+0.0000223`，在 `severe_onset` 为 `-0.0000412`。三个非 native 臂的输出都到达 `center_input`；correct 三层平均 residual ratio 约 `0.0272`，因此不是开关未执行或残差为零。correct 不能胜过两个破坏关系的对照，定位差异主要是微小幅度变化，未出现稳定的 CENTER 纠错链条。

## 判定

**MECHANISM-SCREEN-COMPLETE / RECEIVER-SPECIFIC-EVIDENCE-NOT-SUPPORTED / NO-TRAINING**。

本结果只排除“当前早期对应关系 + 前景均值中心化 + 固定有界残差”这一具体合同在 M1 epoch15 固定状态上的可用性；不证明所有接收条件 Value 建模都无效。由于正确关系不优于 shifted 和 row-shuffled，继续调 `rho`、`alpha` 或 rank 会变成幅度搜索，不能形成新的因果证据，当前不进入 quick3 训练。

## 下一步

回到已有 Early gate 的最小可归因优化，优先做同一 cross-Value 路径的**训练侧可学习内容变换**，而不是继续固定状态手工路由：零初始化、共享的 source/counterpart 差分低秩 adapter，保留 native Q/K；先完成 zero-init parity、单 batch 梯度、16 样本可拟合性，再按 quick3 配对合同训练。若该可学习变换仍不能在同一合同下产生目标/困难负峰的可重复改善，则停止 MSHA Value 细化，保留 Early gate 作为最终小论文方案。

证据目录：`output/experiments/M1_EARLY_GATE_RECEIVER_CENTERED_SMOKE_20261026/`、`output/experiments/M1_EARLY_GATE_RECEIVER_CENTERED_FIXED28_20261026/`。
