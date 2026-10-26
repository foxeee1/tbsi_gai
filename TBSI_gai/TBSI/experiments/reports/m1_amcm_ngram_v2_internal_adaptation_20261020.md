# N-gram Local Fusion V2 内部适应性改造

## 改动范围

只修改 `AMCMLiteReliability` 的 N-gram 分支：将原来的 sigmoid 匹配路由改为模板匹配超额路由

`r = clamp(ReLU(sim - mean(sim)) / mean(ReLU(sim - mean(sim))), 0, 3)`。

RGB residual 由 TIR 超额匹配路由，TIR residual 由 RGB 超额匹配路由。其余 N-gram 局部聚合、margin loss、pre-fusion 注入、`tbsi_fuse_search`、CENTER head 和推理闭环不变。模式名为 `ngram_local_fusion_v2`。

## 合同检查

- 输入仍包含 RGB/TIR search feature 和模板 token matching；输出仍在 `tbsi_fuse_search` 前注入。
- 最终 residual 投影保持零初始化，关闭或未训练时 parent parity 为 0。
- 所有 18 个 AMCM 参数在训练日志中均可获得梯度。

## 25 事件闭环

| 组别 | 平均 ΔIoU | 胜/负 |
|---|---:|---:|
| failure (20) | +0.00323 | 7 / 8 |
| control (5) | -0.00862 | 1 / 4 |
| all (25) | +0.00086 | 8 / 12 |

V2 修复了两个上一版严重退化：`rightcar-chongT` 从 `-0.394` 变为 `+0.167`，`boy\`sheadingreycol` 从 `-0.422` 变为 `+0.016`。但 `basketball849` 退化 `-0.498`，`whitesuvturn` 从 `+0.924` 变为 `0`，整体收益基本消失。

## 固定 10 帧结构诊断

- residual 相对 RMS：RGB `0.104`，TIR `0.101`，幅度充分。
- 但由于 ReLU 超额路由，多个 GT 区域 residual 为 0；目标/背景比中位数 RGB/TIR 为 `0/0`，均值被背景零值分母造成的极端值污染。
- residual 空间 CV 中位数 RGB/TIR `1.23/1.37`，显示高度稀疏。
- 固定状态 IoU Δ `+0.00135`，6/10 上升。
- GT rank 改善 `0/10`，top-1 没有翻转；margin 中位数增加，但没有转化为正确峰归属。

## 梯度诊断

- GT 区域 `d·dL/dfeature` 平均 RGB `-5.67e-5`、TIR `-5.42e-5`，方向均值首次转为降低 loss。
- 但 RGB 只有 `2/10`、TIR 只有 `2/10` 帧有非零负向信号；其余帧因路由为零而没有有效梯度。
- 因此“平均方向正确”是由少数激活帧造成，不能证明训练信号在多个困难序列上稳定驱动目标修正。

## 晋级判定

`NOT PROMOTED`。V2 满足输入、注入、zero-gate 和梯度可达性，但不满足目标/背景结构稳定、GT rank/top-1 改善和多序列闭环一致性。失败原因是硬 ReLU 超额路由过稀疏：它抑制了背景，却也切断了匹配图未超过均值时的目标梯度。

下一步若继续，只允许在模块内部把硬 ReLU 换成带非零下限的平滑超额路由，并保持相同实验合同；否则应停止 N-gram 路线，回到模板匹配信号本身是否能覆盖目标的诊断。
