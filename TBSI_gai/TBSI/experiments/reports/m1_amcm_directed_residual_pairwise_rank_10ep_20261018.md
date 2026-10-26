# AMCM directed residual pairwise rank 10-epoch follow-up（2026-10-18）

在 3 epoch pairwise top-background 结果基础上，仅将训练时长改为 10 epoch；每 epoch 仍为 256 samples，seed、数据、loss、模块和 difficult5 测试协议不变。

| 方法 | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 65.09 | 76.05 | 53.54 | 85.65 | 76.43 |
| Pairwise 3 epoch | 65.32 | 76.29 | 53.76 | 86.19 | 77.42 |
| Pairwise 10 epoch | 64.97 | 75.84 | 53.83 | 85.55 | 76.66 |

10 epoch 相对 M1 为 AUC `-0.12`、OP50 `-0.21`、OP75 `+0.29`、Precision `-0.10`、Norm Precision `+0.23`；相对 pairwise 3 epoch AUC `-0.35`。

逐序列 AUC（10 epoch - M1）：`manoncall -0.70`、`boyss -0.13`、`girldownstairfromlight +0.32`、`bike2left -0.07`、`girlinrain +0.11`。

逐帧 GT rank 变化（candidate-M1）显示：`manoncall` 平均 `+0.083`，`boyss +0.138`，`girldownstairfromlight +1.981`，`bike2left +0.094`，`girlinrain 0`。10 epoch 没有改善主要错误峰序列，且 `girldownstairfromlight` 的排序退化扩大。

结论：`LONGER-TRAINING-NOT-SUPPORTED`。3 epoch 的小幅收益没有随训练时长保持，当前瓶颈不是单纯欠训练；更可能是训练样本分布与 difficult5 闭环失败帧不匹配，或 pairwise 目标在正常帧上造成过度调整。停止继续增加 epoch，下一步应改为 failure-frame 条件采样/掩码，并保持短训作为对照。

