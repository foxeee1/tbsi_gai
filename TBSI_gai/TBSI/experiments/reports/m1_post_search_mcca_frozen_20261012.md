# M1 post-search MCCA 冻结适配器验证

## 实验目的

验证 MCCA 的负收益是否主要来自模板消息注入位置。保持 M1 epoch15、CENTER head、原始损失和融合器不变，只把适配器移到 `tbsi_fuse_search` 输出之后、CENTER 之前；仅训练新增 `post_search_mcca` 参数 3 epochs。

## 合同

- Parent：`TBSI_E_MHSA_FULL_4090D_seed42` epoch15，受保护权重未覆盖。
- 数据：预注册 train-side validation96，序列级固定清单。
- 配置：`experiments/tbsi_track/M1_POST_SEARCH_MCCA_FROZEN_3EP_seed42.yaml`。
- 新模块：`lib/models/layers/post_search_mcca.py`，零初始化输出层，输入最终融合特征与 RGB/TIR 差异。
- 训练：3 epochs，252 optimizer steps，adapter-only，FP32，worker1，无 rank loss。

## 结果

| 指标 | 冻结 M1 | post-search MCCA | 差值 |
|---|---:|---:|---:|
| AUC | 65.6715 | 64.9903 | -0.6812 |
| PR | 79.1348 | 78.3673 | -0.7675 |
| NPR | 76.7483 | 75.9697 | -0.7787 |
| OP50 | 79.4736 | 78.6008 | -0.8729 |
| OP75 | 60.1086 | 59.3228 | -0.7857 |
| FPS | 73.25 | 71.53 | -2.35% |

AUC 逐序列 49 胜/47 负，95% bootstrap 区间 `[-2.5520, 0.6169]`；PR/NPR/OP 区间同样跨零。该版本没有形成正向点估计。

## 合同检查

- 新增适配器参数 12 个，参数 hash 已写入 `metrics/frozen_state_verification.json`。
- M1 参数张量无变化：`changed_non_adapter_tensors=0`。
- 训练过程中冻结模块的 BatchNorm `running_mean/var/num_batches_tracked` 发生 39 项运行态更新；这不是可训练参数更新，但说明冻结主干未完全切换到 eval 状态。后续冻结适配器实验必须显式冻结 BN 状态，否则会混入状态漂移。
- 训练日志无 NaN，适配器输出路径实际执行，checkpoint 可加载。

## 归因结论

将 MCCA 从模板消息位置移到最终 search 融合后仍不能改善 96 序列闭环，故“模板注入位置是主要负因”暂不支持。结合原始 MCCA、排序监督和 reliability gate 均为负，当前证据更支持以下共性瓶颈：

1. MCCA 产生的是逐 token 的局部特征修正，但训练损失通过 CENTER 热图、offset、size 和 GIoU 共同作用；没有稳定约束修正必须改善闭环选峰和后续状态。
2. 模块在单帧特征上有梯度和参数更新，但微小响应改变可能在 Hann 选峰和闭环状态更新中被放大，导致跨序列收益方向不一致。
3. 该候选不包含运动状态、候选召回或回归读出质量信息，因此无法直接解决已诊断的 top-K 召回与 offset/size 误差。
4. 本次还发现 frozen-adapter 训练合同存在 BN running-state 漂移；它需要修复后才能作为更严格的冻结归因对照，但不能把本次负结果直接归因于 BN，因为前述多个版本已有同方向负收益。

## 决策

`M1-POST-SEARCH-MCCA-FROZEN-3EP-20261012`：`COMPLETE / POST-SEARCH-POSITION-NOT-SUPPORTED`。

不进入 full training，不再扩展 MCCA 的模板/融合后位置、门控、rank loss 或学习率变体。保留代码、checkpoint、paired 结果和合同校验作为诊断资产。若恢复该线，唯一合理前提是先修复 BN eval 合同，并引入能直接对应候选召回或 CENTER 读出质量的监督/状态输入；那将是新的方法版本，不能继续称为当前 MCCA 的调参。

证据：`output/experiments/M1_POST_SEARCH_MCCA_FROZEN_3EP_20261012/metrics/paired.json`、`metrics/frozen_state_verification.json`、`output/test/tracking_results/tbsi_track/M1_POST_SEARCH_MCCA_FROZEN_3EP_seed42_853/`。
