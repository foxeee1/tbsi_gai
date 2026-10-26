# MHSA Failure Analysis v1：阶段 B

## 运行范围

- M1：`TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15，FP32 tracker。
- 数据：LasHeR test 244 序列。
- 采样：每序列固定取前 64 帧以内，三层联合模板 MHSA；这是有界机制诊断，不是 full tracking metric rerun。
- 输出：46,716 条 layer-frame 统计，全部与阶段 A 配对表按 `seq/frame` 成功合并。
- parity：attention gate disabled/enabled `max_abs_diff=0.0`。

## 结果

H1“跨模态过融合”要求退化组同时表现为更高 cross-mass、更高模态不一致和更高背景泄漏。实际 hard regression 与 gain 组在三层上的差异很小，且方向不一致；例如 layer 0 的 RGB→TIR cross-mass 为 0.005514 vs 0.005379，template disagreement 为 0.367037 vs 0.370936，不能形成 H1 支持证据。

H2“融合不足”没有得到稳定支持。H3“交互尺度不匹配”也没有直接统计证据；所有序列级相关性的绝对 Spearman 最大约 0.10，主要指标与 ΔIoU 的关联接近零。

## 结论与下一步

当前数据只说明 M1 的收益/退化具有样本异质性，不能把该异质性归因于 cross-mass、attention entropy、模态一致性或背景泄漏。停止直接训练 Search-Conditioned Multi-Range Template Fusion，也不启动第二 seed 或 full confirmation。若继续该方向，下一步只能是预注册的定向干预（cross-modal edge scaling 或固定局部拓扑）验证因果性；否则应停止 MHSA 改进线。

证据：`output/experiments/mhsa_failure_analysis_v1/attention_stats_bounded/`、`output/experiments/mhsa_failure_analysis_v1/failure_cases/`。

## 阶段 C 补充

对同一 244 序列前 64 帧提取 15,816 条 RGB/TIR 图像质量、目标面积、运动和尺度变化记录。common-failure 组的目标面积较小、运动和尺度变化较大，但序列级 Pearson 与 ΔIoU 仍较弱：面积 -0.048、运动 -0.034、尺度 -0.041、RGB Laplacian -0.100。该结果描述数据难度，不足以支持可靠性门控或多尺度融合训练。

证据：`output/experiments/mhsa_failure_analysis_v1/image_quality/`。
