# N-gram Local Fusion + Target/Background Margin 实验

## 唯一改变量

在 `M1_AMCM_NGRAM_QUICK15_3EP_20261020` 基础上只打开现有
`TRAIN.AMCM_REPR_LOSS`：weight `0.5`、margin `0.05`、hard-background top-k `8`。
该损失对 AMCM 输出的 RGB/TIR 表征与模板原型计算目标区域最高相似度和背景最高相似度之间的 hinge margin，未改变 N-gram 结构、`tbsi_fuse_search`、CENTER head、数据和推理状态。

## 训练

配置：`M1_AMCM_NGRAM_MARGIN_QUICK15_3EP_20261020.yaml`。
`Loss/amcm_repr` 训练中从约 `0.069` 降到 `0.040`，说明 margin loss 已进入图并被优化。验证 IoU 第 2 epoch 约 `0.744`，第 3 epoch 约 `0.650`，仍有短训过拟合迹象。

## 25 事件闭环

| 组别 | 平均 ΔIoU | 胜/负/平 |
|---|---:|---:|
| failure (20) | +0.04609 | 9 / 7 / 4 |
| control (5) | +0.00297 | 1 / 4 / 0 |
| all (25) | +0.03747 | 10 / 11 / 4 |

增益由 `whitesuvturn` +0.924、`leftgirl'swhitebag` +0.367、`middrillmaster` +0.322、`girl\`sblkbag` +0.174 等少数事件显著贡献；`boy\`sheadingreycol` -0.422、`rightcar-chongT` -0.394 仍严重退化。中位数 ΔIoU 为 0，尚不能称为稳定提升。

## 固定 10 帧结构诊断

- 目标/背景 residual 比中位数：RGB `0.988`，TIR `1.065`；相比无 margin 的 `1.281/1.162`，目标选择性没有改善。
- residual 空间 CV 中位数：RGB `0.630`，TIR `0.758`；空间分布更平滑。
- 相对 residual RMS：RGB `0.0451`，TIR `0.0667`，比无 margin 版本提升约两个数量级，说明 margin 确实解决了“修正过小”。
- 固定状态平均 IoU：`+0.00576`，8/10 帧上升、2/10 下降。
- GT rank 仍为 `0/10` 改善，top-1 没有翻转；margin 只是略有增加，尚未改变中心峰排序。

结构文件：`output/experiments/M1_AMCM_NGRAM_MARGIN_STRUCTURE_AUDIT_20261020/paired_per_sequence.csv`。

## 梯度方向

加入 margin loss 后的 actor-loss 梯度审计：

- RGB GT 区域 `d·dL/dfeature` 平均 `+4.42e-4`，仅 `4/10` 帧为降低 loss 的方向；
- TIR GT 区域平均 `+3.71e-3`，`0/10` 帧为降低 loss 的方向；
- hard background 的 TIR 方向平均 `-2.75e-3`，说明模块更倾向压低背景，但同时也在提高 GT 区域 loss。

梯度文件：`output/experiments/M1_AMCM_NGRAM_MARGIN_GRADIENT_AUDIT_20261020/per_sequence.csv`。

## 结论

已支持：margin loss 能把 N-gram residual 从近似不可见的幅度激活，并在固定困难帧上带来小幅结构改善。

尚未支持：目标 residual 的方向与 tracking objective 一致，或能稳定改善 GT rank/峰 margin。当前损失存在“背景抑制有效、目标增强方向错误”的模态不对称，且整体增益被少数恢复序列主导。

下一步仍只改一个变量：固定 margin loss，加入 RGB/TIR 对称的 GT residual 方向约束或把 margin 计算改为直接使用 `tbsi_fuse_search` 后的融合响应；先在同一 10 帧验证 TIR 的正向梯度是否消失，再决定是否扩大训练范围。
