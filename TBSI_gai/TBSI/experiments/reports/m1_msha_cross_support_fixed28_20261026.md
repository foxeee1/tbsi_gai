# M1 MSHA Cross-Attention Support Fixed-State Diagnostic（2026-10-26）

## 目的与决策

本实验检验模板 MSHA 的跨模态 QK 交互是否因全局 token 支持而未利用局部空间对应。它比较原生注意力、同位置 3×3 cross-attention 窗口和等大小的半网格循环错位窗口。只改变一个 MSHA 层的 cross-attention logits 支持集；同模态注意力、Q/K/V 投影、Value、输出投影、CENTER/box head 和跟踪状态提交保持不变。

决策：`DIAGNOSTIC-COMPLETE / ALIGNED-OVER-SHIFTED-LOCALIZATION-NOT-SUPPORTED / PROTECTED-M1-REPLAY-PARITY-FAILED / NO-TRAINING`。本实验不支持把“固定 3×3 同位置 cross-attention”直接作为训练候选。对目标对应与错位窗口之间的定位差异，点估计偏向错位窗口，区间跨零；它排除的是该层、该强度的稀疏局部支持形式，不是所有 MSHA 输入建模或互补交互。

## 算子与对照

M1 在 `tbsi_e` 模板融合中拼接 RGB/TIR 模板 token，计算共享 Q/K/V，再对整个拼接序列做联合 softmax。诊断在 softmax 前只对 cross-modal logits 做 mask：

- `M1`: 不添加支持 mask，保持当前代码中的原生联合注意力。
- `ALIGNED_WINDOW`: 对每个 token 只开放另一模态同网格位置周围的环形 3×3 邻域，共 9 个 key。
- `SHIFTED_WINDOW`: 用相同的 9-key 窗口，把中心循环平移 4 个 token（8×8 网格的两个轴各平移 4），形成错位控制。

同模态 logits 不变；双向 cross-attention 使用转置对应窗口。两个局部臂每行均有相同数量的 cross keys，因此 `ALIGNED_WINDOW` 与 `SHIFTED_WINDOW` 是空间位置对照。与 M1 的比较额外包含支持数量从 64 降到 9 的影响，不能只解释为位置效应。

## 协议与 provenance

- Parent checkpoint：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15，SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- Config：受保护 YAML，SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`。
- 事件：既有 train-side calibration 28-event 集，14 matched-good、14 severe-onset；runner 实际读取的 sequence list `experiments/diagnostics/m1_center_readout_fresh_v1/sequence_list.txt` SHA256 `e1a976257d71969686c1dc0690596465d5e14d24366ecef567c58f8b129e65f6`。这不是 holdout，也已被此前分析使用。另一个 protocol 文件 `calibration_sequences.txt` 的 SHA256 是 `af81ee...`，但它不是本 runner 直接读取的文件。
- 执行：FP32、单 GPU、Torch threads=1、逐状态 serial replay，template update 关闭；相同状态和 RNG 下运行三臂。命令：

```bash
/root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/run_m1_msha_query_role_fixed_state_20261022.py \
  --cross-support-diagnostic --cross-support-layer 3 \
  --output output/experiments/M1_MSHA_CROSS_SUPPORT_DIAG_20261026_FIXED28_L3
```

- Environment：RTX 4090 D 24 GiB；Python 3.8.20；PyTorch 1.9.0+cu111；CUDA 11.1。
- 实验前源代码快照：`experiments/snapshots/M1_MSHA_CROSS_SUPPORT_DIAG_20261026/prechange_sources.tar.gz`，SHA256 `b8d46c5f2a7365a55ec96ba02ff4fc024e7d6b9a047e61ec791c025c538d4e16`。运行时源码包：`runtime_sources.tar.gz`，SHA256 `a2899a83a092524369d4b524d8657ee5589c3cf829eb0e31db692723cb552f65`。模型改动由 `TBSI_MSHA_CROSS_SUPPORT_MODE` 显式关闭，M1 臂使用 `off`。
- 本次运行中的 `tbsi_layer.py` SHA256 `d86b4407749e5e30fc08d6bee118b4e882b3b2e35f9b02468d3ef027544380c8`，runner SHA256 `bfe90c7bc54833f79b84f5df8eea000cacaf5010679847330a87ad09190cf1f0`。此前工作区已有未提交改动，当前两个文件运行前哈希分别为 `38314f27716887bccc03e18b0aa308864c16be63175802e68150d4dcf5ccc7c4` 与 `495398e0f82b6ffb1865131bfd32363bb2eb48d7fffc8358161076586817d296`；这与旧 fixed28 报告记载的 runner SHA256 `a1e2cb82512776abc448f10e43982ddd0f9398731218b6041b275bfe3fd5f1ed` 不同，代码 provenance 差异尚未逐项归因。
- 运行中实现诊断先后遇到两次层索引映射错误；两份 smoke 输出完整保留在 `...SMOKE_L3` 和 `...SMOKE_L3_RETRY1`。`...SMOKE_L3_RETRY2` 验证 layer3 命中且两种窗口各 9 个来源 token，随后才启动完整 28-event replay。

## 结果

所有 28 个事件的三臂 `state_before` 和 crop 字段在本次运行内部完全一致。定位对照如下，正值表示同位置窗口优于错位窗口：

| 读数 | matched-good（14） | severe-onset（14） | 28-event/sequence summary |
|---|---:|---:|---:|
| IoU：aligned − shifted，均值 | `+0.000366` | `-0.001525` | sequence macro `-0.000580`，14-sequence bootstrap 95% CI `[-0.003170,+0.001865]` |
| IoU：aligned 胜/平/负 | `8/0/6` | `3/1/10` | 11/1/16（以 `1e-9` 判平） |
| CENTER GT rank：aligned 对 shifted | `0/14/0` | `0/13/1` | 0 胜、27 平、1 负 |
| GT-hard-background margin：aligned − shifted，均值 | `-0.000561` | `+0.000075` | sequence macro `-0.000243`，95% CI `[-0.004359,+0.003529]` |

两种干预都确实改变了 MSHA 与定位输入：layer3 的 cross-attention overlap 从 M1 RGB/TIR query 的 `0.808/0.722` 降为 aligned 的 `0.294/0.271`、shifted 的 `0.311/0.285`；对应 cross mass 为 M1 `0.340/0.778`、aligned `0.100/0.427`、shifted `0.106/0.436`。对 M1 的差异主要反映 support 从 64 keys 缩到 9 keys；aligned 与 shifted 的 cross mass 接近，比较空间位置时更有解释性。CENTER 输入的平均 L2 变化为 aligned `7.879`、shifted `7.892`，但 target rank 没有随同位置关系改善。

## 受保护 M1 replay parity

当前 runner 的 `M1` 臂没有复现已存的 2026-10-23 同一 28-event M1 记录：28/28 decoded boxes 不完全相同，仅 1/28 `state_before` 完全相同；当前与旧输出的平均绝对 fixed-state IoU 差为 `0.05923`，最大 `0.60182`。因此，本结果不能与旧 M1 定位值直接比较，也不能宣称本轮“保护锚点 replay 通过”。

当前候选臂在本轮逐事件恢复同一个 M1 state/crop/RNG，所以 `ALIGNED_WINDOW` 与 `SHIFTED_WINDOW` 的成对差异仍是本次运行合同下的诊断观察；但底层 M1 feature/runner provenance 漂移原因未定位，结论应限定为该运行代码快照上的 28-event intervention。后续任何训练或论文主张前，必须先找到旧/当前 replay 的首个分叉，并恢复可验证的 M1 disabled parity。

## 归因与下一步

- 已确认：MSHA layer3 的跨模态注意力支持可以改变 cross mass、融合输出和 CENTER 输入。
- 当前干预暂不支持：将全局 cross support 替换为固定同位置局部窗口可改善 crop 内目标定位。严重 onset 中同位置 rank 没有胜过错位控制，IoU 均值偏负。
- 尚未区分：更柔性的关系偏置/输入表示是否有效；固定 3×3 局部窗口负向是否由局部支持过强造成；当前 replay 与旧 M1 的差异来自哪一处代码/数据/随机数路径。
- 决策：不做 layer6/9 扩展，也不把当前局部窗口送入训练。先审计 M1 replay parity，定位 `state_before` 的首个分叉；之后若仍研究 MSHA 中间建模，应由新反事实证据提出连续关系建模候选，并用等计算、等支持的错位控制复核。

## 输出与哈希

- 结果：`output/experiments/M1_MSHA_CROSS_SUPPORT_DIAG_20261026_FIXED28_L3/center_results.csv` SHA256 `e23d199e60f878c53c7c87c363eb94bd3cbb8f719a088376c546f2f9857e07c1`。
- Attention：同目录 `attention_diagnostics.jsonl` SHA256 `decc9087d8cb0cc8582b9f3f73d602fce288ec848464756fdc85b479d57f9807`。
- Feature / boundary deltas：同目录 `query_feature_deltas.csv`、`query_boundary_deltas.csv`、`branch_stats.csv`；run manifest: `run_manifest.json`。
- 本轮没有训练、优化器更新、checkpoint 写入、模板更新或 full benchmark 测试。
