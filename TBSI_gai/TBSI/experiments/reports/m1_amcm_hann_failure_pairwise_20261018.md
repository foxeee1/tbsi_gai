# AMCM directed residual: Hann-aware failure-conditioned pairwise ranking

## 研究问题

前一轮 pairwise rank 3 epoch 在 difficult5 上只有小幅增益，10 epoch 反而回落。该实验检验剩余假设：训练 ranking loss 使用 raw score map，且普通训练帧与推理时的错误背景峰分布不匹配。

## 唯一改变量

固定 directed residual AMCM、`tbsi_fuse_search` 前注入位置、GT-cell KL reliability supervision、仅训练 `amcm_reliability.*`、seed42、LasHeR train 256 samples/epoch、3 epoch、FP32、batch2、worker1。只将 rank 目标改为：

\[
s'=s_{raw}\odot H_{hann},\quad
s^+ = \max_{u: y(u)\ge 0.5}s'(u),\quad
s^- = \max_{u:y(u)<0.5}s'(u),
\]
\[
L_{rank}=\mathbf{1}[\arg\max(s')\notin GT]\,[m-s^+ + s^-]_+ .
\]

failure mask 使用 detached 当前 top-1，避免条件本身成为梯度捷径。推理和训练的 CENTER 选择合同因此一致，且只在当前背景峰胜出时施加辅助梯度。

## difficult5 结果

| 方法 | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 65.09 | 76.05 | 53.54 | 85.65 | 76.43 |
| 本实验 | 66.84 | 78.27 | 54.97 | 89.21 | 78.75 |
| 差值 | +1.75 | +2.22 | +1.43 | +3.56 | +2.32 |

逐序列 AUC / 相对 M1：

| 序列 | AUC | ΔAUC | ΔPrecision |
|---|---:|---:|---:|
| `manoncall` | 68.18 | -1.26 | -2.01 |
| `boyss` | 54.96 | +9.71 | +14.43 |
| `girldownstairfromlight` | 46.45 | -0.10 | +0.04 |
| `bike2left` | 79.91 | +0.17 | +0.24 |
| `girlinrain` | 84.72 | +0.23 | +0.10 |

## 机制证据

- 训练 loss 中 Hann-aware CENTER 标记持续为 1；rank loss 非零，说明 failure-conditioned 路径被实际使用。
- 16/16 AMCM 参数在每个记录步有非零梯度，梯度裁剪后仍接近有效上限；不存在“模块未更新”的解释。
- 总体提升高度集中于 `boyss`，`manoncall` 退化；因此不能把 +1.75 AUC 写成普遍解决背景峰。
- 本轮没有同步导出固定状态 rank 翻转 trace，不能声称已证明 GT 峰在闭环中稳定压过背景峰。

## 结论与下一步

结论为 `SUPERVISION-CONTRACT-HYPOTHESIS-SUPPORTED / GENERALIZATION-INCOMPLETE / NOT-PROMOTED`：监督目标与推理峰值合同的对齐值得保留，failure 条件可能解释了 boyss 的恢复，但序列间不稳定性和闭环机制仍未区分。下一步只允许做同一 checkpoint 的固定状态与逐帧 peak trace，并在不改结构的情况下复核 `manoncall` 的退化来源；不继续增加 epoch或解冻 CENTER head。

## 资产

- 配置：`experiments/tbsi_track/M1_AMCM_DR_HANN_FAILURE_PAIRWISE_PROXY_20261018.yaml`
- checkpoint SHA256：`e452ecfb3b84e561b14bf655c85070f7532a9069250b51a31bbd1eaf22b75226`
- 配置 SHA256：`c040d6f56a8dc3f870cfac8e5ae7dca02d4e299e283956dd8a8256696739bd33`
- 训练日志：`output/experiments/M1_AMCM_DR_HANN_FAILURE_PAIRWISE_PROXY_20261018/logs/`
- difficult5 结果：`output/test/tracking_results/tbsi_track/M1_AMCM_DR_HANN_FAILURE_PAIRWISE_PROXY_20261018/`

## 权重清理

删除了已被后续实验取代的中间 checkpoint：E2 3 epoch proxy、E2 first-30 ep5、directed residual center-rank、directed residual diagnostics、directed residual pre-rank proxy、pairwise 10 epoch。保留 M1、pairwise 3 epoch parent、当前候选、E2 full-244 两个 seed 的 checkpoint；其日志、结果和报告未删除。
