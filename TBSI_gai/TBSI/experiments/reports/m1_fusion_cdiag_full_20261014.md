# M1 post-TBSI Cdiag 局部融合残差：全量训练测试报告

## 结论

在固定 M1（seed=42、epoch 15）上，仅训练 4 个 post-TBSI Cdiag residual 参数，并完成 LasHeR full train 15 epochs 与 full-244 官方测试。相对 M1，AUC 下降 0.0274 个百分点，Precision 下降 0.0075，Norm Precision 上升 0.1000，OP50 上升 0.0899，OP75 上升 0.0502。该结果不支持 Cdiag feature-side residual 带来整体涨点，也没有接近预期的 1% 最终提升。

这不是“模块没有训练起来”：4 个 residual 参数均有非零梯度，zero-grad count 为 0/4，gamma 和跨模态映射权重均发生了明显更新。更符合证据的解释是，局部特征修整没有稳定改变 M1 的中心候选选择、crop 状态反馈和长期恢复闭环。因而停止继续扩展当前 Cdiag feature residual 结构。

## 实验合同

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15，seed 42。
- Candidate：`M1_FUSION_CDIAG_FULL_15EP_B64_W4_seed42`，独立输出目录。
- 训练数据：LasHeR full train，`SAMPLE_PER_EPOCH=59904`，15 epochs。
- 测试：LasHeR `lasher_test`，244/244 序列，官方 `analysis_results.py` 和独立 persistence audit 交叉核对。
- 训练合同：`TRAIN.CPLAIN_ONLY=True`，冻结 M1，仅训练 `gamma_rgb`、`gamma_tir`、`rgb_from_tir.weight`、`tir_from_rgb.weight`。
- 有效 batch：128（batch 64，grad accumulation 2）。候选使用 worker=4 配置；manifest 中保留了实际运行资源偏差记录。

## Full-244 五指标对比

| 指标 | M1 baseline | Cdiag candidate | Delta (candidate - M1) |
|---|---:|---:|---:|
| AUC / Success | 56.2077 | 56.1804 | -0.0274 |
| Precision | 70.2073 | 70.1997 | -0.0075 |
| Norm Precision | 66.1980 | 66.2979 | +0.1000 |
| OP50 | 68.3339 | 68.4238 | +0.0899 |
| OP75 | 47.1349 | 47.1851 | +0.0502 |

官方审计还显示：mean valid overlap `+0.000115`，frame failure IoU<0.5 从 `0.373657` 降到 `0.371436`，IoU<0.2 从 `0.296144` 降到 `0.295464`。但 persistent failure sequence 数仍为 `136`，persistent segment 从 `430` 增至 `442`；最长失败段由 `1494` 降至 `1124`。因此局部帧统计有轻微改善，却没有转化为整体 Success 或稳定恢复收益。

逐序列 AUC：candidate 胜 115、负 126、平 3；此前摘要中的胜负方向曾因审计字段标签解释错误而写反，已按 `M1=candidate`、`B0=baseline` 修正。平均序列 delta 为 `-0.02735` 个百分点，与总体 AUC 的轻微下降一致。

完整逐序列和逐帧输出变化分析见 `experiments/reports/m1_fusion_cdiag_full_sequence_analysis_20261014.md`，明细位于实验目录下的 `analysis_full_sequence/`。

## 归因证据

- 末期 total grad norm 约 `0.0965`，backbone 记录约 `0.0448`，4/4 residual 参数均有梯度，未见计算图断路。
- 末期 `gamma_rgb≈+0.0947`、`gamma_tir≈-0.1321`；RGB/TIR residual delta norm 约 `1230/1570`，说明模块实际学习到了非零修正。
- 因此当前结果可以排除“zero-gate、参数未更新、梯度完全断路”作为首要原因。
- 结合此前 fixed-state action-space 诊断：feature-side residual utility 接近 0，而 readout score/offset/size 仍有固定状态 headroom；但 readout top-K 当前部署信号在 held-out 上为负，尚未具备准入条件。
- 早期 82 条 proxy cohort 上 Cdiag 的 +2.8206 AUC 没有在 full-244 复现，应视为 cohort 筛选偏差或泛化不足，不能作为方法收益。

## 决策与下一步

本实验状态：`COMPLETE / NO-MATERIAL-FULL-TEST-GAIN`。不再扩大 Cdiag feature-side residual 或把 proxy 增益写成可泛化结论。若仍以约 1% 最终提升为目标，下一条实验线必须直接作用于 center/readout 与 crop-state recovery 的联合闭环，并先解决可部署信号：优先做带 no-op/abstain 的 readout utility 信号校准，再进行单变量短训和 full-244 验证。只有在 held-out 信号稳定为正、且能减少 persistent failure 的情况下，才进入新的联合模块训练。

## 证据路径

- 实验目录：`output/experiments/M1_FUSION_CDIAG_FULL_15EP_B64_W4_retry3_20261013/`
- Checkpoint：`training/experiments/M1_FUSION_CDIAG_FULL_15EP_B64_W4_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
- 官方审计：`official_audit/official_metric_and_persistence.json`
- 逐序列审计：`official_audit/per_sequence_official.csv`
- 梯度、loss、内部信号、特征变化：`training/.../gradient_flow.csv`、`loss_components.csv`、`internal_signals.csv`、`feature_changes.csv`
- 配置：`experiments/tbsi_track/M1_FUSION_CDIAG_FULL_B64_W4_seed42.yaml`
