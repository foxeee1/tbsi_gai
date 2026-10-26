# CFVI P1 Full Matched M1 Report

## 结论

状态：`VALID-FULL-MATCHED-NEGATIVE`。

固定 Haar-CFVI、仅在 location 9 注入的 P1 已按与 TBSI-E M1 相同的 SOT 初始化、完整训练集、15 epoch、seed 42 和 full-244 FP32 测试协议完成。P1 没有相对匹配 M1 产生稳定增益，因此停止该路线的 M2、第二 seed 和新的频率结构扩展。

本结果是当前仓库 244 条测试序列的匹配协议结果，不宣称官方 245 条 LasHeR 测试划分结果。

## 主指标

| 指标 | P1 | 匹配 M1 | P1 - M1 |
|---|---:|---:|---:|
| AUC | 56.20 | 56.21 | -0.01 |
| OP50 | 68.38 | 68.33 | +0.05 |
| OP75 | 47.01 | 47.13 | -0.12 |
| Precision | 70.39 | 70.21 | +0.18 |
| Norm Precision | 66.43 | 66.20 | +0.23 |

两组均为 244/244 序列完成评测。P1 的 AUC 差异为四舍五入后的 -0.01，不能解释为方法增益。

## 逐序列配对分析

- mean delta：`+0.000214`
- median delta：`-0.001110`
- 5% trimmed mean：`-0.001812`
- sequence bootstrap 95% CI：`[-0.012137, +0.012972]`
- win rate：`47.95%`；loss rate：`52.05%`
- 去掉最大 3 个 gain 后均值：`-0.007313`
- 去掉最大 3 个 loss 后均值：`+0.005739`
- large gain (delta >= 0.05)：`7.79%`
- large loss (delta <= -0.05)：`9.84%`

最大收益序列为 `ab_bikeoccluded` (+0.6667)、`3rdfatboy` (+0.5751)、`3bike1` (+0.5728)；最大退化序列为 `1blackteacher` (-0.6382)、`carcominginlight` (-0.4258)、`whiteridingbike` (-0.2670)。收益和退化均较分散，整体配对区间跨 0。

## 注入诊断

P1 分支确实参与了 forward：初始 lambda 为 RGB/TIR `0.001/0.001`，epoch-15 末约为 `0.0066895/-0.0099199`。按 `||lambda * delta_V|| / ||V_before||` 计算，整体均值 rho 为 RGB `0.0098585`、TIR `0.0147617`，末期约为 RGB `0.01098`、TIR `0.01688`。因此“分支没有执行”不是本轮阴性结果的解释；但非零注入也没有带来可重复的 full-run 性能收益。

## 协议与可复现信息

- 配置：`experiments/tbsi_track/CFVI_P1_FULL_MATCHED_M1_seed42.yaml`
- 初始化：`TBSITrack_SOT_Pretrained.pth.tar`
- seed：42；训练 15 epoch；batch 16；梯度累积 8；sample/epoch 59904
- 测试：244 序列，FP32，serial，`TBSI_INFER_FP16=0`
- P1 checkpoint：`output/experiments/CFVI_P1_FULL_MATCHED_M1_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
- P1 eval：`output/experiments/cfvi_p1_full_matched_m1_20261001/P1/analysis/eval_data.pkl`
- M1 eval：`output/experiments/cfvi_p1_full_matched_m1_20261001/M1_reference/analysis/eval_data.pkl`
- 逐序列差值：`output/experiments/cfvi_p1_full_matched_m1_20261001/paired_delta.csv`
- pipeline log：`output/experiments/cfvi_p1_full_matched_m1_20261001/full_pipeline.log`
- CFVI diagnostics：`output/experiments/CFVI_P1_FULL_MATCHED_M1_seed42/logs/cfvi_diagnostics.jsonl`
- 配置 SHA256：`83ed1b09a92486bc6ba1b8daf1876be1917dc40517c200630d58cc0886018c2a`
- 代码快照：`09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9 + working-tree`

## 限制与后续决定

本轮是单 seed，且当前仓库测试清单为 244 条；proxy-60 已被用于此前选型，不能作为独立确认集。由于严格匹配 full 协议下 P1 持平且胜率低于 50%，不继续 M2、第二 seed、full confirmation 或新的频带门控。若未来重启频率方向，应先提出独立的等参数普通空间域对照，并预注册新的未参与选型的确认序列；本报告不把 Haar 频率互补机制视为已验证贡献。
