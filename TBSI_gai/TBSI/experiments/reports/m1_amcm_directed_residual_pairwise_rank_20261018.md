# AMCM directed residual pairwise top-background ranking（2026-10-18）

## Protocol

在 `M1_AMCM_DR_CENTER_RANK_PROXY_20261018` 基础上只改变 ranking loss：使用 GT cell 与当前最高背景 cell 的 pairwise hinge，替代 top-8 背景 cell 平均约束。directed residual、KL reliability supervision、冻结范围、3 epoch 训练和 difficult5 测试均保持不变。

## Results

| 方法 | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 65.09 | 76.05 | 53.54 | 85.65 | 76.43 |
| Pairwise rank | 65.32 | 76.29 | 53.76 | 86.19 | 77.42 |
| 差值 | +0.23 | +0.24 | +0.22 | +0.54 | +0.99 |

逐序列 AUC（candidate - M1）：`manoncall -0.30`、`boyss +1.04`、`girldownstairfromlight +0.43`、`bike2left +0.01`、`girlinrain +0.22`。

## Peak trace

| 序列 | 有效帧 | rank 改善/不变/变差 | 平均 rank 变化 |
|---|---:|---:|---:|
| `manoncall` | 520 | 14 / 481 / 25 | +0.100 |
| `boyss` | 327 | 28 / 278 / 21 | -0.003 |
| `girldownstairfromlight` | 1137 | 57 / 1011 / 69 | +1.403 |
| `bike2left` | 256 | 5 / 242 / 9 | +0.301 |
| `girlinrain` | 311 | 0 / 311 / 0 | 0.000 |

正值表示 GT rank 变差。pairwise loss 带来初步的 difficult5 指标改善，尤其是 `boyss`，但没有在所有目标序列上改善 GT rank；`girldownstairfromlight` 和 `bike2left` 的峰值排序仍恶化。因此当前证据支持“top-background 监督比 top-8 平均监督更有用”，但尚不足以支持“已解决错误背景峰”。

## Decision

结论：`PRELIMINARY-POSITIVE / MECHANISM-INCOMPLETE / NOT-PROMOTED`。下一步应固定 pairwise 形式，加入只在 crop 内目标存在且 M1 top-1 错误帧启用的 mask，并继续使用固定状态 trace 验证 rank 翻转，而不是同时改变结构或状态更新。

