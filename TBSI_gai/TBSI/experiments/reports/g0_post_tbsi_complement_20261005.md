# G0：首次模板桥接后的 RGB/TIR search 互补审计

日期：2026-10-05  
Parent：`TBSI-E-MHSA-FULL-20260927`  
性质：冻结 M1 的只读信息审计；不训练 FFM，不修改预测路径，不进入 full-244。  
主位置：第一次 TBSI（index 0，对应代码 location 3）完成后，block index 5 输出、第二次 TBSI（index 1，对应代码 location 6）之前。

## 1. 假设和实现合同

本实验检验：第一次模板桥接之后，RGB/TIR search 是否仍存在可利用的、目标相关的空间互补。冻结 epoch-15 M1，在固定 LasHeR train proxy 的 24 条序列上选取 2 个固定事件，共 48 个事件。GT 只在前向完成后生成 token 标签。

诊断接口只在 `diagnostic=True` 时返回以下 search token：

- `post_tbsi_rgb/tir`：第一次 TBSI 完成后的 search token；
- `post_block_rgb/tir`：block index 5 完成后的 search token，亦为第二次 TBSI 前的候选位置。

默认 forward 不启用这些返回；没有改动 checkpoint、预测 head、crop、模板或 TBSI 计算。证据：[g0_post_tbsi_complement.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/g0_post_tbsi_complement.py)。

## 2. 合同和覆盖审计

| 项目 | 结果 |
|---|---:|
| 事件数 | 48 |
| 目标有效 token 事件 | 43/48（89.58%） |
| diagnostic 与普通 tracker parity | 通过 |
| parity 事件 | 6 |
| 最大逐框差 | 0.0 |
| checkpoint | M1 epoch-15 |
| checkpoint SHA256 | `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef` |
| 推理精度 | FP32，`TBSI_INFER_FP16=0` |

5 个事件没有有效目标 token，未进入 token 相似度汇总；没有悄悄当作背景事件。

## 3. 对照定义

- `correct`：同一事件、同一空间 token 的 RGB/TIR 特征对应；报告 token 对角相似度和目标 token AP；
- `event_shuffle`：按固定循环顺序使用另一事件的 TIR 特征，维度和分布保留；
- `spatial_shift`：同一事件 TIR token 沿固定水平方向平移一个 token；
- `XY_only`：仅按 search crop 中心距离排序的位置先验。

AP 和目标/背景 margin 都是前向后的冻结特征统计，不是训练后的可学习 probe。`event_shuffle` 和 `spatial_shift` 只用于判断正确对应关系是否有可测增量。

## 4. 主要结果

| 特征位置 | correct 对角 margin | correct token AP | XY-only AP |
|---|---:|---:|---:|
| post TBSI | -0.09053 | 0.07475 | 0.82045 |
| post block 5 | -0.08109 | 0.07712 | 0.82045 |

post-TBSI 对照差值：

| 对照 | margin 差 | AP 差 |
|---|---:|---:|
| correct - event shuffle | -0.04678 | -0.01640 |
| correct - spatial shift | +0.00292 | -0.00199 |

序列级 bootstrap（24 条序列为单位）：

| 配对差 | 均值 | 95% CI |
|---|---:|---:|
| correct - event shuffle，margin | -0.04479 | [-0.06108, -0.02779] |
| correct - event shuffle，AP | -0.01690 | [-0.02752, -0.00744] |
| correct - spatial shift，margin | +0.00243 | [-0.00858, +0.01442] |
| correct - spatial shift，AP | -0.00180 | [-0.00652, +0.00237] |

正确 RGB/TIR 对应没有相对错配和空间错位产生稳定的目标定位增量；相反，冻结相似度统计中跨事件错配的结果更高。XY-only AP 远高于视觉对应 AP，说明固定事件上的主要可读信号仍是 crop 位置先验，而不是已被证实的目标条件互补。

## 5. 固定条件分层

预先固定的中心阈值为目标中心距 crop 中心不超过 2 个 token，M1 难例阈值为 event-frame IoU 小于 0.5。43 个有效事件全部落在 `centered_le_2_tokens` 且全部属于 `easy_iou_ge_0.5`：

- centered/off-center：`43/0`；
- easy/hard：`43/0`。

因此本次固定 24 序列事件没有提供偏心或 M1 已错误的有效子组，不能据此声称对困难事件完成验证。该覆盖缺口是实验限制，不通过事后调阈值修补。

## 6. 机器决策

预注册的继续条件要求正确配对同时超过 event shuffle 和 spatial shift，且 margin 差超过 `0.01`、AP 差超过 `0.03`。结果未满足：

```text
G0_STOP_NO_RELIABLE_COMPLEMENT
```

这不是“证明 FFM 不可学习”。冻结相似度不能排除一个训练后的跨注意力读出器学习到新关系；但当前没有足够的目标条件增量证据支持移植 FFM、训练 Cdiag/Cspatial 或把复杂空间交互称为新贡献。下一步不进入 P0/P1，也不跑 full-244；若未来重新开启，应先建立覆盖困难/偏心/低 IoU 条件的独立 G0，并重新登记新的实验合同。

## 7. 证据路径

- 汇总：[summary.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_post_tbsi_complement_20261005/summary.json)
- 序列级分析：[analysis.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_post_tbsi_complement_20261005/analysis.json)
- 事件指标：[event_metrics.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_post_tbsi_complement_20261005/event_metrics.json)
- parity：[parity.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_post_tbsi_complement_20261005/parity.json)
- 导出脚本：[g0_post_tbsi_complement.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/g0_post_tbsi_complement.py)
- 分析脚本：[g0_post_tbsi_analyze.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/g0_post_tbsi_analyze.py)
