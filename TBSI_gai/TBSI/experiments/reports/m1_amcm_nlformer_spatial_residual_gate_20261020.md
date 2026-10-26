# M1 AMCM NLformer Residual Route Follow-up (2026-10-20)

## 单一假设

上一轮把模板匹配图输入 NLformer，但输出 residual 仍近似空间均匀。本轮只改变 AMCM 内 residual 的空间幅度：对 RGB/TIR 模板匹配图做温度为 `0.07` 的空间 softmax、归一化到均值 1，再分别乘到 NLformer residual 的 RGB/TIR 通道组。NLformer 主体、pre-fusion 注入、CENTER head、外部 loss 和训练数据均不变。

## 训练

- 配置：`M1_AMCM_NLFORMER_SPATIAL_RESIDUAL_GATE_QUICK15_3EP_20261020.yaml`
- 公开 SOT 预训练初始化，未使用 M1 checkpoint
- LasHeR train，512 samples/epoch × 3 epochs，batch 2，FP32，`NUM_WORKER=4`
- 仅 AMCM 可训练；alpha 训练后为 `0.01833`
- checkpoint：`output/experiments/M1_AMCM_NLFORMER_SPATIAL_RESIDUAL_GATE_QUICK15_3EP_20261020/checkpoints/TBSITrack_ep0003.pth.tar`

## 固定 10 帧结果

固定相同 crop、template 和 tracker state 的 M1 失败事件：

- 平均 IoU：`0.30133 -> 0.25263`，delta `-0.04870`
- IoU 提升 5/10，退化 5/10
- GT rank 改善 1/10；top-1 转回 GT 0 次；top-1 离开 GT 1 次
- RGB residual target/background ratio：`0.31463`
- TIR residual target/background ratio：`0.26308`

代表性结果：

| 序列 | IoU delta | GT rank | margin |
|---|---:|---:|---:|
| `7rightorangegirl` | -0.0632 | 1 | 0.2131 -> 0.0684 |
| `ab_bolstershaking` | -0.2068 | 1 -> 2 | 0.0930 -> -0.0159 |
| `blkboydown` | -0.2730 | 6 -> 7 | -0.0772 -> -0.0625 |
| `boyss` | +0.1566 | 5 -> 3 | -0.1655 -> -0.1360 |
| `umbrella` | -0.2370 | 3 -> 18 | -0.0381 -> -0.2289 |

## 归因

**已确认**：直接注入空间路由确实改变了 residual 的空间分布，已经不再是上一轮的约 `1.0` 均匀比值。

**已排除**：这种变化不是目标选择性。固定失败帧中 target/background 比显著小于 1，说明当前模板匹配图在困难 crop 中更强地放大了背景 residual，反而扩大错误峰影响。

**当前瓶颈**：模板匹配图本身在失败帧不是可靠的目标先验。把它直接作为 residual route 会把错误模板相似度硬编码进输出，导致 margin 下降和状态漂移。问题已经从“输出空间均匀”收窄为“输入模板匹配的目标/背景可分性不足，且现有 loss 没有校正该路由方向”。

## 决策

`NOT-PROMOTED`，不做 50 序列或 full training。继续增大 route 强度会放大错误背景，不能作为下一步。下一步若继续，只能在 AMCM 内增加一个最小的**双模态一致性校验**：模板匹配只有在 RGB/TIR 同时支持时才允许路由 residual；先在同一 10 帧检查 route 的 target/background 比和 margin。若一致性校验仍无法超过 1，再停止 NLformer 路线并回到数据/监督合同审计。

证据：`output/experiments/M1_AMCM_NLFORMER_SPATIAL_RESIDUAL_GATE_FIXED10_DIAG_20261020/per_event.csv`。
