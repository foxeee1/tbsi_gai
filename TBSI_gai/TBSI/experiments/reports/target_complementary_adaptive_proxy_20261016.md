# B2：目标条件互补分支与 noop 门控

## 实验合同

以 M1 epoch15 为 parent，固定 difficult5，3 epoch、256 samples/epoch、batch 2、FP32、1 worker，仅训练 `TargetComplementaryAdaptiveFusion`。原 MSHA fused feature 是主支路，RGB/TIR 差分是候选互补支路；门控输入为目标相似度与模态分歧，门控 bias=-4，默认接近 noop。

## 结果

| 方法 | AUC | Precision | Norm Precision | 平均 IoU | 相对 M1 AUC |
|---|---:|---:|---:|---:|---:|
| M1 | 66.7115 | 89.3621 | 78.0039 | 0.67690 | 0 |
| B 原始 dynamic local | 64.7427 | 85.5197 | 75.3398 | 0.65686 | -1.9689 |
| B2 complementary + admission | 65.0782 | 86.1117 | 76.8295 | 0.66056 | -1.6333 |

逐序列 AUC：

| 序列 | M1 | B2 | B2-M1 |
|---|---:|---:|---:|
| manoncall | 68.9425 | 68.8511 | -0.0914 |
| boyss | 54.0952 | 45.9302 | -8.1651 |
| girldownstairfromlight | 46.5000 | 47.5838 | +1.0839 |
| bike2left | 79.6183 | 78.6548 | -0.9635 |
| girlinrain | 84.4017 | 84.3712 | -0.0305 |

## 结论

B2 相对 B 有小幅改善，说明显式 noop/admission 能减少无条件局部残差造成的伤害；但 `boyss` 仍是稳定退化点，且 B2 仍低于 M1 1.63 AUC。当前证据支持“门控设计方向比原始动态局部算子合理”，不支持进入全量训练。

下一步应只做 `boyss` 的固定 crop、模态候选和门控误放行诊断，确认退化来自错误互补方向还是状态闭环放大。暂不叠加 A、B、B2，也不增加更深局部卷积。
