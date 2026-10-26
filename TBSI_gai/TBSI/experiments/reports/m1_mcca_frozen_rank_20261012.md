# M1 MCCA 冻结适配器排序监督实验

## 实验目的

验证此前归因提出的假设：MCCA 负收益部分来自训练目标没有直接约束推理时的 Hann 加窗峰竞争。保持 M1 主干、融合器、CENTER head、MCCA 结构、学习率和冻结合同不变，仅增加与在线选峰一致的目标峰-困难背景峰排序损失。

## 改动与合同

排序损失为 `L_rank = lambda * [margin - max(S_H_target) + max(S_H_background)]_+`，其中 `S_H = score_map * Hann`，正区域为 GT 覆盖网格，负区域为 GT 外扩 1 token 后的网格；`lambda=0.5`、`margin=0.05`。仅 6 个 MCCA adapter 参数更新，M1 权重和 CENTER head 冻结，3 epochs、252 个 optimizer steps、FP32、seed 42。配置为 `M1_MCCA_FROZEN_RANK_3EP_seed42`。

## 训练检查

- 训练完成，无 NaN 或中断。
- `rank_valid_fraction=1.0`。
- `Loss/mcca_rank` 约由 `0.0087` 降至 `0.0062`，说明新增监督确实产生了有效梯度和优化信号。
- MCCA 残差没有被置零；这不是“模块未接通”或“loss 恒为零”的失败。

## 预注册 96 序列配对结果

| 指标 | 冻结 M1 | 排序监督 MCCA | 差值 | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| AUC | 65.6715 | 65.0052 | -0.6664 | [-2.2977, 0.4564] |
| PR | 79.1348 | 78.3024 | -0.8324 | [-2.7006, 0.5016] |
| NPR | 76.7483 | 75.9846 | -0.7637 | [-2.6018, 0.5174] |
| OP50 | 79.4736 | 78.7133 | -0.7603 | [-2.6145, 0.5507] |
| OP75 | 60.1086 | 59.2197 | -0.8888 | [-2.6872, 0.3013] |

AUC 逐序列为 52 胜、44 负；中位差 `+0.0631`，但去掉最大获益序列后的均值差为 `-0.8508`。候选 FPS `65.10`，冻结 M1 FPS `73.25`，约下降 11.1%。

## 归因结论

这次实验排除了“完全没有针对问题加监督”这一解释，但没有证明排序监督能解决问题：

1. **优化链路正常**：排序项可计算、梯度有效、loss 下降，MCCA 参数确实被训练。
2. **代理目标与闭环目标仍不一致**：排序项只在训练 crop 上约束单帧 Hann 后正负峰 margin；推理时错误由离散 argmax、模板/搜索状态更新和早期偏差累积共同决定。它没有直接约束跨帧恢复、候选召回、offset/size 读出或闭环状态稳定性。
3. **监督区域可能不等于真实困难负峰**：GT 外扩 1 token 的背景集合不一定包含实际在线竞争峰，因此 rank loss 下降不代表实际峰排序正确。
4. **冻结 M1 时自由度有限**：adapter 同时要改变失配帧选峰并保持正常帧行为，容易出现局部 margin 改善而整体闭环不稳定。
5. **速度代价仍存在**：准确率未改善时，约 11% FPS 降低不能接受为 full training 的理由。

## 决策

记为 `COMPLETE / RANK-OBJECTIVE-NOT-SUFFICIENT`。不启动 full training，不继续调 `lambda`、margin、dilation 或 MCCA 方向。保留 checkpoint、日志、96 序列预测和配对分析作为归因资产。当前证据支持“监督目标与闭环任务仍有接口错位”，不支持“只要给 MCCA 加排序 loss 就能稳定涨点”。

证据：配置 `experiments/tbsi_track/M1_MCCA_FROZEN_RANK_3EP_seed42.yaml`；checkpoint `/autodl-fs/data/M1_MCCA_FROZEN_RANK_3EP_20261011/output/experiments/M1_MCCA_FROZEN_RANK_3EP_seed42/checkpoints/TBSITrack_ep0003.pth.tar`；配对指标 `output/experiments/M1_MCCA_FROZEN_RANK_3EP_20261011/metrics/paired.json`；日志 `output/experiments/M1_MCCA_FROZEN_RANK_3EP_20261011/logs/RANK_eval.log`。
