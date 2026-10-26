# Hann-aware failure-conditioned AMCM fixed-state trace

## 目的

验证 difficult5 结果是否来自固定 crop 内的 CENTER 背景峰纠正。M1 和当前候选使用相同帧、相同模板、相同 `tracker_state_before`，只替换网络 checkpoint；不把任一 arm 的预测反馈给另一 arm。

## 协议

- 序列：`manoncall`、`boyss`。
- 事件：894 个（`manoncall` 520，`boyss` 374）；rank 有效事件 847 个。
- 精度：FP32。
- 记录：固定状态 IoU、Hann 后 GT rank、GT/background margin、峰距、AMCM reliability probability、residual norm。
- M1：受保护 epoch15 checkpoint。
- 候选：`M1_AMCM_DR_HANN_FAILURE_PAIRWISE_PROXY_20261018` epoch3 checkpoint。

## 结果

| 序列 | 事件 | 固定状态 ΔIoU | IoU 改善/退化 | rank 改善/不变/变差 | M1 rank → 候选 rank |
|---|---:|---:|---:|---:|---:|
| `manoncall` | 520 | +0.00034 | 312 / 195 | 3 / 507 / 10 | 1.2673 → 1.2827 |
| `boyss` | 374 | +0.00194 | 142 / 155 | 2 / 322 / 3 | 2.5963 → 2.5994 |
| 合计 | 894 | +0.00101 | 454 / 350 | 5 / 829 / 13 | 1.7804 → 1.7910 |

## 解释

固定 crop 中，候选几乎没有改变离散峰排序，GT rank 反而略差；因此 difficult5 的 `boyss` AUC +9.71 不能归因于“AMCM 稳定把背景 top-1 翻回 GT”。候选可能通过连续框回归、峰值附近的亚像素偏移或闭环状态轨迹产生收益。`manoncall` 的闭环退化也不是固定状态下的大幅峰值灾难，说明需要继续检查状态反馈和框更新，而不是继续扩大 rank loss。

## 判定

`FIXED-STATE-PEAK-CORRECTION-NOT-SUPPORTED / CLOSED-LOOP-OR-REGRESSION-MECHANISM-REMAINS`。

下一步只做闭环逐帧对齐：比较 M1 与候选在首次分叉帧的 `state_after`、预测框 IoU、峰值连续坐标和下一帧 crop 中心；不再增加训练轮数，也不再修改 CENTER rank loss。

## 证据

- 工具：`tools/run_m1_amcm_hann_failure_fixed_state_trace_20261018.py`
- 输出：`output/experiments/M1_AMCM_DR_HANN_FAILURE_PAIRWISE_PROXY_20261018/fixed_state_signal_audit_manoncall_boyss/`
- 主要文件：`paired_events.csv`、`per_event.csv`、`summary.json`
