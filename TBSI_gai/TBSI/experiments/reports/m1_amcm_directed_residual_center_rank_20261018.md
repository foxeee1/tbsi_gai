# AMCM directed residual + CENTER ranking supervision（2026-10-18）

## Protocol

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15。
- Candidate：`M1_AMCM_DR_CENTER_RANK_PROXY_20261018`，在 directed residual 上仅打开现有 `CENTER_RANK_LOSS`；其余结构、KL reliability supervision、冻结范围、数据和训练进度不变。
- 训练：LasHeR train，256 samples/epoch，3 epochs，seed42，FP32，batch2，worker1。
- 测试：与前一轮完全相同的 RPP difficult5 五序列，FP32 闭环；同时保存逐帧峰值 trace。

## Training evidence

`Loss/center_rank` 在训练中持续非零，约 `0.0002--0.0015`；AMCM 16/16 参数仍有梯度，说明 ranking loss 已进入 AMCM 的反向路径。验证 IoU 为 `0.67032`，与不加 ranking loss 的 directed residual 结果 `0.67032` 基本相同。

## Tracking comparison

| 方法 | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 65.09 | 76.05 | 53.54 | 85.65 | 76.43 |
| Directed residual + rank | 64.62 | 75.58 | 54.42 | 85.08 | 75.88 |
| Candidate - M1 | -0.47 | -0.47 | +0.88 | -0.57 | -0.55 |

逐序列 AUC（candidate - M1）：`manoncall -1.26`、`boyss -0.11`、`girldownstairfromlight -0.64`、`bike2left +0.06`、`girlinrain +0.02`。

## Peak trace

| 序列 | 有效帧 | rank 改善/不变/变差 | 平均 rank 变化 |
|---|---:|---:|---:|
| `manoncall` | 493 | 5 / 465 / 23 | +0.434 |
| `boyss` | 327 | 19 / 280 / 28 | +0.061 |
| `girldownstairfromlight` | 1143 | 64 / 1000 / 79 | +1.186 |
| `bike2left` | 256 | 4 / 247 / 5 | +0.141 |
| `girlinrain` | 311 | 0 / 311 / 0 | 0.000 |

正值表示 candidate 的 GT rank 变差。ranking loss 没有在目标序列上稳定产生 top-1 纠正，且 `girldownstairfromlight` 的 rank 明显恶化。

## Decision

结论：`OBJECTIVE-PATH-VALID / CURRENT-RANK-CONTRACT-NOT-SUPPORTED`。这次实验排除了“完全没有把峰值目标传给 AMCM”的解释，但不支持继续直接增大该 loss。当前实现把 GT heatmap 最大 cell 与所有低热图背景 cell 做统一 top-k hinge，未区分 crop 内真实目标候选、Hann 后峰值和前一帧状态风险。下一步应改为固定状态下的 GT-cell 对当前 top-background 的 pairwise margin，并只在 crop 内目标存在且 M1 top-1 错误的帧采样，避免正常帧和不可达帧主导梯度。

