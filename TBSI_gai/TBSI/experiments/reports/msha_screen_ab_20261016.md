# A/B 最小筛选：训练期融合适配器

## 合同

Parent 为受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15，固定 difficult5：`manoncall`、`boyss`、`girldownstairfromlight`、`bike2left`、`girlinrain`。两臂均为 3 epoch、256 samples/epoch、batch 2、FP32、1 worker，只训练新增模块，M1 backbone 和 CENTER head 冻结。

## 结果

| 方法 | AUC | Precision | Norm Precision | 平均 IoU | 相对 M1 AUC |
|---|---:|---:|---:|---:|---:|
| M1 | 66.7115 | 89.3621 | 78.0039 | 0.67690 | 0 |
| A: MSHA reliability | 63.7488 | 84.3545 | 75.2190 | 0.64665 | -2.9627 |
| B: target dynamic local | 64.7427 | 85.5197 | 75.3398 | 0.65686 | -1.9688 |

逐序列 AUC：

| 序列 | M1 | A | A-M1 | B | B-M1 |
|---|---:|---:|---:|---:|---:|
| manoncall | 68.9425 | 69.3355 | +0.3930 | 68.9242 | -0.0183 |
| boyss | 54.0952 | 45.7905 | -8.3048 | 45.3714 | -8.7238 |
| girldownstairfromlight | 46.5000 | 40.0047 | -6.4953 | 45.6822 | -0.8178 |
| bike2left | 79.6183 | 79.3033 | -0.3150 | 79.3033 | -0.3150 |
| girlinrain | 84.4017 | 84.3101 | -0.0916 | 84.4322 | +0.0305 |

## 归因

两臂都完成了 checkpoint 保存，且新增参数从零初始化后发生了非零更新，说明计算图和训练合同有效。A 的二阶可靠性输入没有转化为可泛化的局部修正，B 的目标条件局部算子能缓解 `girldownstairfromlight` 的退化，但两者都显著伤害 `boyss`，且未超过 M1。

结论为 `暂不支持进入全量训练`。A 直接淘汰；B 仅保留为“目标条件局部算子有局部机制迹象”的探索证据。由于这是 5 序列、单 seed、3 epoch proxy，不能据此宣称全量负结论；若继续，只应先针对 `boyss` 的局部失效做固定 crop/同状态诊断，而不是堆叠 A+B。

证据目录：

- `output/experiments/M1_MSHA_RELIABILITY_PROXY_20261016/`
- `output/experiments/M1_TARGET_DYNAMIC_LOCAL_PROXY_20261016/`
- `output/test/tracking_results/tbsi_track/M1_MSHA_RELIABILITY_PROXY_20261016/`
- `output/test/tracking_results/tbsi_track/M1_TARGET_DYNAMIC_LOCAL_PROXY_20261016/`
