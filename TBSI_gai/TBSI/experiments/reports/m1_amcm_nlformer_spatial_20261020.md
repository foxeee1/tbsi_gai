# M1 AMCM NLformer Spatial Template Follow-up (2026-10-20)

## 研究问题

NLformer AMCM 的上一轮诊断显示：三分支有梯度，但残差在目标和强背景位置几乎等幅。该实验只改变 AMCM 内的模板条件：把原来的模板全局均值条件，替换为 RGB/TIR 搜索 token 对模板 token 的逐位置余弦匹配图，并做空间均值去除。NLformer 三分支、`tbsi_fuse_search` 注入位置、CENTER head、外部 loss、训练数据和优化器均保持不变。

## 合同与训练

- 配置：`M1_AMCM_NLFORMER_SPATIAL_QUICK15_3EP_20261020.yaml`
- 初始化：`pretrained_models/TBSITrack_SOT_Pretrained.pth.tar`；未加载 M1 checkpoint
- LasHeR train，512 samples/epoch，3 epochs，batch 2，FP32，`NUM_WORKER=4`
- 训练模式：仅 `AMCM_RELIABILITY` 可训练；AMCM 输出仍在 `tbsi_fuse_search` 前注入
- checkpoint：`output/experiments/M1_AMCM_NLFORMER_SPATIAL_QUICK15_3EP_20261020/checkpoints/TBSITrack_ep0003.pth.tar`
- 运行时修复：AMCM 内 Woodbury 的 2x2 求逆改为显式解析式，避免 PyTorch 1.9 的 cuSPARSE `linalg.inv` 兼容错误；数学计算保持等价

## 50 序列闭环结果

同一 E2 冻结 50 序列清单、FP32 推理和 M1 epoch15 参考结果：

| 指标 | M1 | Spatial NLformer | 变化 |
|---|---:|---:|---:|
| 平均逐帧 IoU | 0.54668 | 0.51887 | -0.02781 |
| success proxy AUC | 0.54747 | 0.51083 | -0.03664 |
| 序列提升/退化 | - | 18 / 32 | - |

该 AUC 是逐帧 IoU 阈值 proxy，不是官方 full-test AUC。结果仅用于候选淘汰和机制判断，不能作为严格 matched M1 性能结论。

明显提升：`3rdfatboy` +0.5966、`4thboywithwhite` +0.1226、`belowdarkgirl` +0.0695、`3rdgrouplastboy` +0.0635。

明显退化：`10runone` -0.6329、`ab_bolstershaking` -0.2741、`7rightorangegirl` -0.2129、`basketballathand` -0.1734、`ab_rightlowerredcup_quezhen` -0.1608。

## 固定失败帧机制诊断

固定 10 个 M1 失败事件，使用相同 template、crop 和 tracker state：

- 平均固定帧 IoU：`0.30133 -> 0.25372`，delta `-0.04761`
- IoU 提升 5/10，退化 5/10
- GT rank 改善 1/10；top-1 转为 GT 0 次；top-1 离开 GT 1 次
- residual target/background ratio：RGB 均值 `1.0016`，TIR 均值 `1.0025`
- NLformer spatial condition 权重训练后范数 `0.0644`，`alpha` 为 `0.0156`，说明该分支不是未更新；但输出 residual 仍没有目标选择性

最具代表性的失败：`umbrella` 的 GT rank `3 -> 19`，margin `-0.038 -> -0.255`；`ab_bolstershaking` rank `1 -> 2`，margin `0.093 -> -0.028`。`boyss` IoU 上升但 margin 仍为负，属于解码峰变化，不能解释为目标/背景可分性改善。

## 归因结论

**已排除**：问题仅仅是“AMCM 没有空间模板输入”；新增空间匹配图后，训练权重和 alpha 都非零，但目标/背景 residual 比仍约为 1。

**暂不支持**：NLformer 三分支在当前 AMCM 输出合同下能稳定改善 crop 内目标峰竞争。50 序列总体退化，且高基线序列损失集中。

**当前瓶颈**：模板匹配图只作为 NLformer 内部加性条件，经过 GroupNorm、注意力和最终 `tanh` residual 投影后，未被强制保留为输出空间选择性；现有 CENTER/GIoU/L1 loss 只约束最终框，不能直接要求 residual 在 GT 增强、背景抑制。

## 下一步决策

该候选不晋级 full training。若继续，只允许一个 AMCM 内部变量：让模板匹配图直接调制 residual 的空间幅度（保留 NLformer 内容分支），并在同一固定 10 帧先检查 residual ratio、margin 和 rank；不能继续增大训练轮数或叠加新的 loss 来掩盖当前输出合同问题。若直接空间调制仍无法使 residual ratio 明显偏离 1，则应停止 NLformer 路线，转回接口/监督合同审计。

证据：

- `output/experiments/M1_AMCM_NLFORMER_SPATIAL_QUICK15_3EP_20261020/analysis_50seq/`
- `output/test/tracking_results/tbsi_track/M1_AMCM_NLFORMER_SPATIAL_QUICK15_3EP_20261020_50SEQ/`
- `output/experiments/M1_AMCM_NLFORMER_SPATIAL_FIXED10_DIAG_20261020/per_event.csv`
- `output/experiments/M1_AMCM_NLFORMER_SPATIAL_QUICK15_3EP_20261020/logs/`
