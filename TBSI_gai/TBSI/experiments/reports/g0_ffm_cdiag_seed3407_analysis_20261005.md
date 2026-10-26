# G0 Cdiag seed 3407 确认实验报告

## 结论

本次唯一授权的 Cdiag seed 3407 短训已完成。训练合同与 seed 42 相同，仅改变训练 seed：index 5 输出、index 6 TBSI 前、152 条训练序列、1 epoch、672 batch、原 tracking loss、FP32、24 条固定 rescue 开发序列。结果没有复现 seed 42 的正向全轨迹增益，Cdiag 线索封存，不进入 full training。

## 配对结果（相对同一 C0）

| 指标 | seed 42 | seed 3407 |
|---|---:|---:|
| mean paired ΔAUC | `+0.017827` | `-0.017403` |
| median paired ΔAUC | `+0.001223` | `-0.000084` |
| 10% paired-delta trimmed mean | `+0.019752` | `-0.004387` |
| mean paired ΔIoU | `+0.018191` | `-0.017702` |
| AUC wins / losses | `14 / 10` | `11 / 13` |
| AUC regression <= -0.05 | 3 | 5 |
| largest gain | `+0.233047` | `+0.237888` |
| largest loss | `-0.124744` | `-0.451951` |

seed 42 审计中去掉最大获益序列后仍为 `+0.008469`，但 bootstrap 95% CI 为 `[-0.007975,+0.046862]`；seed 3407 的均值与稳健 trimmed mean 均转负，说明 seed 42 的正向结果不稳定，不能升级为 Cdiag 的可靠收益。

## 困难窗口

固定 24 个 rescue 事件的 `[t,t+30]` 窗口中，seed 3407 Cdiag mean IoU 为 `0.208316`，C0 为 `0.207197`，恢复率均为 `20/24`。窗口均值只有 `+0.001119`，没有形成稳定救援证据。

## 关门合同

seed 3407 训练 checkpoint 的 gamma0 与 C0 逐框 parity 通过：24/24 序列长度一致，最大绝对框差 `0.0`。因此 seed 3407 的负结果来自 residual 开启后的实际作用，而不是关门路径破坏 M1。

## 最终路线决策

- FFM spatial：停止，已由前一轮困难窗口退化证实。
- Cdiag：停止；两个 seed 在同一开发集上的方向不一致，且 seed 3407 有 5 条严重退化。
- 不跑第三 seed、full-244、位置搜索、gate 搜索或新模块。
- 结论边界：index 5 的低扰动跨模态残差曾在 seed 42 开发运行中出现正向线索，但未通过随机性确认；它不是稳定方法收益或论文主张。

## 证据

- 审计：`output/experiments/g0_ffm_20261005/cdiag_audit.json`
- seed 42 gamma0：`output/experiments/g0_ffm_20261005/cdiag_gamma0_parity.json`
- seed 3407 全轨迹：`output/experiments/g0_ffm_cdiag_seed3407_20261005/analysis.json`
- seed 3407 困难窗口：`output/experiments/g0_ffm_cdiag_seed3407_20261005/window_analysis.json`
- seed 3407 runner：`output/experiments/g0_ffm_cdiag_seed3407_20261005/runner.log`
