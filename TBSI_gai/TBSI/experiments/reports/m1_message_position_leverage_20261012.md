# M1 消息注入位置杠杆诊断（2026-10-12）

## 结论

**位置判断：现有全链路证据最支持优先改进 CENTER 的联合选峰与 offset/size 读出，并把 crop 状态反馈作为闭环目标；在两个显式消息位置中，loc9 之后的搜索消息 S 比 loc9 模板跨模态消息 T 对最终分数排序更敏感。但本实验没有找到能稳定提高定位的 S 增益方向，因此暂不启动 S 模块训练。**

这一区分很重要：`S` 对删减更敏感，说明它是有杠杆的位置；不等于增加该消息就能改善定位。当前 32 个固定状态里，匹配幅度的小干预几乎不改变 top-1，而 S 增强时 target/background margin 的变化也没有稳定映射成更好的框。

## 实验身份与执行

- ID：`M1-MESSAGE-POSITION-LEVERAGE-20261012`；parent 为受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15。
- 固定 YAML SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`；checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，实验前后哈希一致。
- 独立 train-side cohort：12 条 outcome-blind 序列，从 47 条未用于所列 prior diagnostics 且 loader-compatible 的候选中按目标面积、相对运动、RGB亮度和TIR局部对比四个代理量平衡抽样。manifest SHA256 `fa0bdbf53760f433463ef75a54bb027943979c44be0ebeccc609dace0f6dd0ba`；sequence list SHA256 `7d9d0d582e63c8763e90254efcc47f51beeb738a766716feb91c48514b301022`。这些序列未用于此前列出的诊断，但可能在 M1 训练中出现过。
- 先对每条序列做 M1 闭环 trace，共 4,117 跟踪帧，耗时 140.61 秒。再从 baseline trace 固定同一模板、状态、图像和 crop，在 32 个状态上重放。状态组成：matched-good 4、pre-onset 4、severe-onset 4、good-only 8、reference 12。严重起点定义为 GT 仍在 crop 内且连续 3 帧 IoU `<0.5`；GT 只用于状态选择和事后评分。
- T：仅缩放 loc9 TBSI-E 联合模板 MHSA 的 post-softmax 双向 RGB↔TIR attention 子矩阵贡献，再过原 output projection。S：仅缩放紧接 loc9 后的普通 ViT `block[10]` 中 search-query 对 template-key/value 的 post-softmax attention 子矩阵贡献。两者都只改变计算出的消息，不改变状态和 crop。
- 首轮每个位置比较消息置零及 `0.95/1.05` 缩放。测得 T 的 post-projection 消息变化约 `2.88%`，S 约 `1.4–1.6%`，因此另做匹配幅度的 follow-up：T 保持 `0.95/1.05`，S 改为 `0.90/1.10`。实际消息输出变化中位数分别为 `2.878%` 和 `2.926%`。首轮未匹配结果保留在 `fixed_state/per_state.csv`，匹配对照在 `fixed_state_matched/per_state.csv`。
- 最大固定状态重放框 parity 误差 `0.0000763 px`。单卡 RTX 4090 D；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8、FP32、单进程。trace 与两轮固定状态诊断合计约 152.5 秒，即约 `0.0424 GPU小时`。无训练、无测试集推理、无 checkpoint 写入。

## 结果

指标均为同一状态相对 M1 baseline 的配对差。`GT rank` 表示 Hann score 下 GT cell 排名变化；负的 center/size/IoU 差分别表示误差降低或 IoU 降低，正负方向按指标定义解释。

| 位置 / 干预 | 实际消息输出变化 | Hann margin 中位差 | GT-cell中心误差中位差 | GT-cell size L1 中位差 | IoU 中位差 | top-1峰改变 |
|---|---:|---:|---:|---:|---:|---:|
| T 置零 | 57.56% | -0.00107 | +0.00028 token | -0.00201 token | +0.00041 | 0/32 |
| S 置零 | 29.26% | -0.01935 | -0.00125 token | -0.00086 token | +0.00332 | 1/32 |
| T 缩至 0.95 | 2.88% | -0.00005 | +0.00005 token | -0.00017 token | +0.00002 | 0/32 |
| S 缩至 0.90 | 2.93% | -0.00100 | -0.00039 token | -0.00024 token | +0.00026 | 1/32 |
| T 增至 1.05 | 2.88% | +0.00003 | -0.00001 token | +0.00018 token | 约 0 | 1/32 |
| S 增至 1.10 | 2.93% | +0.00108 | +0.00045 token | -0.00008 token | -0.00024 | 0/32 |

T 置零使 4/32 个 GT-cell rank 改变；S 置零使 11/32 改变。小幅匹配缩放时 GT-cell rank 变化至多 5/32，top-1 峰只在单个状态发生改变。S 增至 1.10 后，23/32 个状态的 target/background margin 上升，但 20/32 个状态 IoU 下降；margin 不足以作为最终框质量的替代指标。

在仅有 4 个 severe-onset 状态的小样本上，S=0.90 的 margin 中位数增加 `0.00130`，IoU 中位数减少 `0.00051`；S=1.10 的 margin 减少 `0.00071`，IoU 增加 `0.00075`。方向并不一致，无法把它解释为 S 应增强或抑制。严重组太小，只作探索性信号。

## 与既有链路证据合并

1. `M1-CANDIDATE-RECALL-READOUT-20261010`：严重起点 GT 精确格只有 `2/32` 在 top-1，但 `31/32` 在 top-16；即使 GT cell 选格，CENTER 中心误差中位仍为 `1.046 token`、size L1 为 `0.944 token`。说明选峰与选格后的回归读出同时受限。
2. `M1-CENTER-READOUT-FRESH-20261010`：新鲜 14 个 onset 上，GT-cell oracle 仅 `5/14` 把 IoU 提升超过 0.1；size L1 中位数从 `0.795` 变为 `0.854 token`。强制目标格不能单独解决框回归。
3. 这轮 T/S 干预补充了消息位置的因果敏感性：S 比 T 更影响最终 score 排序，但相同实际小扰动没有带来稳定峰值或框质量改善。它并未测试可学习的搜索条件门控、loc3/loc6，也未直接干预 CENTER head 参数或闭环状态更新。
4. 已有融合接口探针没有显示原 search fusion 一致抹掉目标信息；post-search MCCA 代理短训也未得到稳定提升，且有冻结 BN buffer 漂移合同问题。因此当前不把“再换一个融合器”排在 CENTER 联合读出和 crop 状态恢复之前。

## 结论边界与下一步

- **支持下一步单变量检验：**在固定 train-side 状态上，直接测试 CENTER 的 joint score/offset/size 一致性读出及其对真实选框的影响；报告 score 选峰、选格后框误差和闭环 crop-out/恢复率，不能只看 margin。
- **暂不支持：**把 S 整体放大、把 T 调弱或继续训练通用 MCCA。零消息干预只证明当前 M1 用到这些消息，不证明人为的单调增益有益。
- 若继续消息路线，应提出依赖当前 search 证据、能选择性改变 S 的机制，并先在新的 train-side 固定状态上证明它同时改善定位误差且不伤正常帧；否则不进入训练。
- 本实验为当前工作树 train-side replay，非归档逐帧复现、非 full test、非训练收益评估。固定状态反事实不会模拟干预后轨迹状态变化，所有 32 个状态来自同一个 M1 baseline 闭环。

## 产物

- 清单：`experiments/diagnostics/m1_message_position_20261012/manifest.json`
- 源码快照：`experiments/snapshots/m1_message_position_prechange_20261012.tar.gz`
- 脚本：`tools/build_m1_message_position_cohort_20261012.py`、`tools/diagnose_m1_message_position_20261012.py`、`tools/analyze_m1_message_position_20261012.py`
- 脚本 SHA256：cohort `0a67c5f82d067dd9e1553d1fa9ed4c584d1f974311d5e7bb2407bee9bab1b8b4`；diagnostic `136fb3f118ca9cfd3cebc4bbfdd006a8ad6a77e4d1e9abadd1b4ee2935f785dc`；analysis `81520a1a2ccf240e4f4c1e02501f42834cf2e9ad291819c51237378db7f420bb`。
- Trace：`output/experiments/M1_MESSAGE_POSITION_LEVERAGE_20261012/trace_run/`
- 固定状态逐条件数据：`output/experiments/M1_MESSAGE_POSITION_LEVERAGE_20261012/fixed_state/per_state.csv` 与 `fixed_state_matched/per_state.csv`
- 分析汇总：`output/experiments/M1_MESSAGE_POSITION_LEVERAGE_20261012/analysis/summary.json`
- 启动失败记录：`output/experiments/M1_MESSAGE_POSITION_LEVERAGE_20261012/startup_attempts.md`；两次均发生在样本前向前，未产生模型或资产变更。
