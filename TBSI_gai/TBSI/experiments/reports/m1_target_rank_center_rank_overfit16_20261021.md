# M1 TargetRankGate + CENTER rank loss 16-sample overfit

实验 ID：`M1-TARGET-RANK-GATE-CENTER-RANK-OVERFIT16-20261021`

## 目的

验证目标条件 score 和 CENTER response coupling 是否能在极小固定训练集上同时过拟合。若监督链正确，`target_rank` 与 `center_rank` 应明显下降，且 GT rank 应改变。

## 设置

- LasHeR train/val：每 epoch 16 samples，batch 2，5 epochs，FP32，worker 4。
- M1 backbone、TBSI、fusion 和 CENTER 主体冻结。
- 只训练 `TargetRankGate`，打开有界 response coupling，`beta=0.03`。
- 同时打开 target score ranking loss 与 CENTER GT/hard-background pairwise rank loss；两者 top-k=3，Hann-aware CENTER rank loss。
- 首次未限幅 coupling 在 epoch 2 出现 NaN；随后将 score coupling 改为 `tanh(score/std(score))`，重新运行稳定完成 5 epochs。

## 结果

证据目录：`output/experiments/M1_TARGET_RANK_GATE_CENTER_RANK_OVERFIT16_20261021/`。

- 有界 coupling 训练完成，无 NaN。
- `Loss/target_rank` 全程约 `0.0248--0.0252`，没有明显下降。
- `Loss/center_rank` 约 `0.018--0.041` 波动，没有显示持续下降。
- 验证 IoU 从约 `0.78` 到 `0.68` 附近波动，未形成小样本拟合趋势。

## 结论

**CONTRACT-STABLE / OVERFIT-FAIL / NOT-PROMOTED**。

有界 coupling 修复了数值稳定性，但监督链仍不能在 16 个样本上学习目标/背景排序。因此当前问题不是训练步数不足，也不是单纯的 response coupling 强度，而是 score 的目标构造和负样本定义与 CENTER 的空间决策不一致。停止 50-event 或 full-244 训练。

若继续方法路线，下一次必须改变训练信号本身：使用模板前景 prototype、Gaussian/soft target map 和动态 top-k hard-background，并记录训练 batch 的 GT rank；不得继续只调 beta、margin 或 epoch。
