# M1 Query-conditioned Cross-Value quick training test

日期：2026-10-24

## 目的

验证固定状态未通过后，query-conditioned cross-Value 分支能否从 tracking loss 中学习，而不是用随机 probe 直接判断其定位价值。

## 训练合同

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15 checkpoint。
- Candidate：3 epoch，LasHeR train，512 samples/epoch，batch 2，FP32，worker 4，seed 42。
- 仅训练 3 个 `tbsi_e` 层中两个方向的 query-conditioned low-rank 参数；M1 主干、CENTER 和其他状态冻结。
- template update off。
- 测试：预先固定的 12 个 RPP-v2 序列，FP32、threads=0，同一 tracker 入口。

Candidate 的 12 个新增参数张量均为非零，checkpoint 总 query-conditioned 参数范数为 `6.5984`，说明分支确实更新。

## 训练观察

最后一个 epoch 的训练日志仍有正常梯度和 loss：最后记录 `Loss/total=0.89691`、`IoU=0.79786`。训练过程中没有 query 分支断图或保存失败。

一次最初的 control 运行使用了 `QUERY_CONDITIONED_CROSS_VALUE_ONLY=False`，实际成为全模型续训，已标为混杂证据，不用于 paired 结论。随后尝试“关闭分支但只训练分支”的 control，因所有候选参数无梯度而自然失败；最终严格 control 使用受保护 M1 epoch15 checkpoint，避免人为构造无梯度 control。

## 12 序列 paired 结果

| 指标 | M1 parent | Candidate | Candidate - M1 |
|---|---:|---:|---:|
| AUC/Success | 49.6556 | 49.5210 | -0.1346 |
| Precision | 63.8160 | 63.2018 | -0.6142 |
| Norm Precision | 68.3400 | 66.8859 | -1.4541 |
| Mean IoU | 0.4978 | 0.4964 | -0.0014 |

逐序列 AUC 有 6 个上升、6 个下降。`catbrown2bush` 上升 `+7.40`，但 `rightbottlecomes` 下降 `-11.04`，不存在稳定的场景级改善模式。该 12 序列结果是 quick screening，不是 full LasHeR 结论。

## 结论

`QUICK-COMPLETE / LEARNABILITY-PASSED / PAIRED-PERFORMANCE-NOT-SUPPORTED / STOP-CURRENT-VALUE-ONLY-ROUTE`

候选能够被监督驱动，说明此前固定状态不通过并非单纯的梯度或合同问题。但在严格 M1 parent 对照下，3 epoch candidate 的 AUC、Precision、Norm Precision 和 Mean IoU 均略低，且逐序列方向不稳定。因此当前 query-conditioned Value-only 设计没有足够证据进入 full training。

## 下一步

停止继续扩大该分支的训练量、rank 或残差幅度。若要继续 Early-conditioned MSHA，应改变信息流层级，优先验证保留空间位置的 token transport；该结果不支持通过延长训练把当前 Value-only 算子直接挽救为论文方法。

证据：

- candidate checkpoint：`output/experiments/M1_QUERY_CONDITIONED_CROSS_VALUE_QUICK3_20261024/checkpoints/TBSITrack_ep0003.pth.tar`
- candidate results：`output/test/tracking_results/tbsi_track/M1_QUERY_CONDITIONED_CROSS_VALUE_QUICK3_20261024_candidate_quick12/`
- parent results：`output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_query12_parent20261024/`
