# M1 N-gram Local Fusion AMCM 替换实验

## 核心机理

N-gram Local Fusion 在 AMCM 边界内对 RGB/TIR search feature 分别进行 3x3、5x5、7x7 depthwise 局部 token 聚合，经 1x1 混合得到局部上下文 `L_rgb/L_tir`。模板 token 与每个 search token 做归一化匹配得到 `s_rgb/s_tir`，再拼接

`E = [L_rgb, L_tir, |L_rgb-L_tir|, s_rgb, s_tir]`。

双流 message 网络从 `E` 生成有界 residual，减去空间均值后由另一模态的模板匹配图路由：

`d_rgb = route(s_tir) * tanh(M_rgb(E)) * alpha`，
`d_tir = route(s_rgb) * tanh(M_tir(E)) * alpha`，
`out_rgb = search_rgb + d_rgb`，`out_tir = search_tir + d_tir`。

`out_rgb/out_tir` 在原有 `tbsi_fuse_search(cat(out_rgb,out_tir))` 前注入；最终 residual 投影零初始化，未训练时与 parent 完全一致。该模块不读取历史框或运动状态，也不改变模板更新逻辑。

## 快速训练合同

- 配置：`M1_AMCM_NGRAM_QUICK15_3EP_20261020.yaml`
- parent：M1 epoch15，checkpoint 只读
- LasHeR train 512 samples/epoch，3 epochs，batch 2，FP32，`NUM_WORKER=4`
- 仅 AMCM 参数可训练，固定 CENTER head、loss、optimizer 和 `tbsi_fuse_search`
- checkpoint：`output/experiments/M1_AMCM_NGRAM_QUICK15_3EP_20261020/checkpoints/TBSITrack_ep0003.pth.tar`

修复一次实现合同问题后，训练日志不再出现零梯度参数。第 2 epoch validation IoU 约 0.745，第 3 epoch 降至约 0.650，说明该 3 epoch proxy 存在过拟合风险。

## 闭环性能（25 个固定事件）

| 组别 | 事件数 | 平均事件 IoU 变化 | 胜/负/平 |
|---|---:|---:|---:|
| failure | 20 | -0.02033 | 7 / 7 / 6 |
| control | 5 | +0.00135 | 1 / 4 / 0 |
| all | 25 | -0.01600 | 8 / 11 / 6 |

最大退化：`rightcar-chongT` -0.4084，`boyunder2baskets` -0.0769。较大改善：`ab_bikeoccluded` +0.0257、`leftgirl'swhitebag` +0.0228、`boy\`sheadingreycol` +0.0214。结果是固定事件短训 proxy，不是 full-244 主结果。

## 空间结构与融合传递

固定 10 个低 AUC/困难事件，报告见 `output/experiments/M1_AMCM_NGRAM_STRUCTURE_AUDIT_20261020/paired_per_sequence.csv`。

- residual 具有一定空间结构，但幅度很小：RGB/TIR 相对 RMS 中位数约 `3.77e-4/3.04e-4`。
- 目标/强背景 residual 比中位数为 RGB `1.28`、TIR `1.16`；空间 CV 中位数约 `0.92/0.88`。因此不是完全均匀缩放，但目标选择性很弱且不稳定。
- 固定 10 帧 GT rank 改善 `0/10`，top-1 没有翻转；margin 变化在 `1e-6` 量级。
- 用 paired snapshots 计算 `||Δfused|| / ||Δinput||` 的平均传递比约 `2.65`，说明 `tbsi_fuse_search` 没有把 residual 完全截断；但 `Δfused` 的绝对量仍很小，无法改变峰竞争。

## loss 梯度方向

使用项目原始 GIoU+L1+CENTER-focal actor loss，在固定 10 事件上对 residual 做 `d · dL/dfeature` 审计，结果见 `output/experiments/M1_AMCM_NGRAM_GRADIENT_AUDIT_20261020/per_sequence.csv`。

- GT 区域平均方向：RGB `+3.52e-6`，TIR `+1.63e-6`；正值表示该 residual 一阶上增加 tracking loss。
- GT 区域朝降低 loss 的帧数：RGB `5/10`，TIR `4/10`。
- residual 与 GT 梯度余弦均值：RGB `0.0104`，TIR `0.0023`，接近正交。
- 因此训练信号确实能到达 N-gram 投影，但现有 tracking loss 没有稳定地把 residual 推向目标增强/背景抑制方向。

## 结论与下一步

当前实现满足输入、注入和梯度可达性，但未满足“目标/背景分离改善、峰 margin/GT rank 同向、闭环可重复增益”。主要瓶颈不是 `tbsi_fuse_search` 截断，而是 residual 过小且 tracking loss 对目标 residual 的方向不一致；短训还存在第 3 epoch 过拟合。N-gram 在当前 AMCM 合同下暂不晋级，不能据此宣称该论文机制无效。

下一步只允许一个单变量：保留同一 N-gram 结构和 pre-fusion 注入，增加直接作用于目标峰/强背景峰 margin 的训练约束，先在同一固定 10 事件上复查 `d·dL`、margin 和 GT rank；若方向仍接近零，再暂停更换模块，审计 search crop/标签坐标与 CENTER loss 的可导路径。
