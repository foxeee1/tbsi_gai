# AMCM directed residual 梯度与峰值归因（2026-10-18）

## 实验身份

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15，seed 42。
- Candidate：`M1_AMCM_DR_DIAG2_PROXY_20261018`，3 epoch、256 samples/epoch、batch 2、FP32、worker 1，仅训练 `amcm_reliability.*`。
- 训练 checkpoint：`output/experiments/M1_AMCM_DR_DIAG2_PROXY_20261018/checkpoints/TBSITrack_ep0003.pth.tar`。
- 固定状态事件：M1 candidate-recall 记录中的 123 个事件、60 个训练序列；两臂使用相同 crop、template、`state_before`。
- 闭环 trace：RPP difficult5 五个序列，FP32，逐序列独立执行；trace 位于 `output/experiments/M1_AMCM_DR_DIAG2_PROXY_20261018/difficult5_trace_{dr2,m12}/`。

## 梯度与特征流

`amcm_gradient_flow.csv` 中 24 个采样点均为 16/16 个参数有非零梯度。梯度范数约为 0.043--0.100，部分采样达到配置的 `GRAD_CLIP_NORM=0.1`，因此不存在“模块没有被优化”的证据。

`feature_changes.csv` 显示 directed residual 的概率从约 `0.50/0.50` 逐步偏到约 `0.45/0.55`；RGB/TIR message 范数约 6--9，实际 residual 范数约 3--5，而输入范数约 190--240，残差是小幅、非零的局部修正。训练监督损失从 `0.01870` 降到 `0.01042`，验证 IoU 从 `0.65301` 到 `0.67032`。

因此已支持两点：计算图和梯度流通；可靠性分支确实产生了空间修正。尚未支持的是修正方向能稳定改变 CENTER 的目标峰排序。

## 固定状态峰值归因

123 个事件中，Hann 后 GT rank 的均值为 `1.2195 -> 1.2195`，`123/123` 个事件 rank 不变；平均 raw peak 到 GT 距离仅 `1.0745 -> 1.0593`。这说明在同一 crop 和同一 state 下，残差主要改变连续分数，未改变离散 top-1 排序。

## difficult5 逐序列闭环

| 序列 | 有效帧 | top-1 改变帧 | GT rank 改善/不变/变差 | 平均 rank 变化（candidate-M1） |
|---|---:|---:|---:|---:|
| `manoncall` | 520 | 97 | 17 / 475 / 28 | +0.069 |
| `boyss` | 327 | 59 | 26 / 271 / 30 | +0.440 |
| `girldownstairfromlight` | 1148 | 192 | 75 / 1006 / 67 | +0.071 |
| `bike2left` | 256 | 8 | 6 / 243 / 7 | -0.168 |
| `girlinrain` | 311 | 6 | 0 / 311 / 0 | 0.000 |

这里的 rank 变化只统计 GT 与 crop 有效重叠且诊断字段完整的帧；正值表示 candidate 的 GT 排名变差。`manoncall` 和 `boyss` 是目标问题最集中的序列，但 candidate 没有持续把背景峰翻转为目标峰；`bike2left` 的改善很小，不能代表稳定机制。

## 结论与下一步

结论为 `GRADIENT-AND-RESIDUAL-VALID / PEAK-RERANKING-NOT-SUPPORTED`。当前瓶颈不再是“模块没有学习”，更像是监督目标与最终 CENTER 峰排序仍不一致，或 residual 幅度不足以跨越背景峰 margin。下一步应只做一个变量：在相同 directed residual 接口上加入直接的 GT-vs-top-background ranking/utility 监督，并在固定状态下预注册 top-1 翻转率和 rank 改善门槛；不要同时解冻 head 或改变 crop/state 更新。

