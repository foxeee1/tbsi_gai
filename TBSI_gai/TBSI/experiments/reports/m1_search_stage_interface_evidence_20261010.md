# M1 搜索接口前后证据（2026-10-10）

## 回答

现有序列证据不支持“目标中心信息是在 `tbsi_fuse_search` 之后才被破坏”。在失败起点，`tbsi_fuse_search` 的输入（两路最终搜索特征，位于 TBSI/MHSA 影响后的 backbone 输出）已经出现目标可读性下降；融合输出的同容量线性探针可读性反而回升。因而当前最稳妥的判断是：

> 中心信息在搜索特征形成阶段已经不足，搜索融合没有显示出一致的二次破坏；剩余失效更可能发生在上游搜索表示/模板到搜索传递、CENTER head 读出或状态反馈。仅凭本结果不能把首因唯一归给模板 MHSA，也不能直接选择局部模块或门控模块。

## 序列证据

证据来自 LasHeR train-side 当前工作树 replay 的 72 条独立序列、104 个 held-out 帧，其中 16 条序列各有一个 `pre_onset` 和一个 `severe_onset` 事件帧。事件帧要求 GT 有效且目标完整位于当前 replay crop；每条序列按序列划分做 3-fold held-out 线性 1x1 probe。`rgb`/`tir`/`concat` 是 `tbsi_fuse_search` 输入，`fused` 是其输出。

| 事件 | 阶段 | GT-background margin 中位数 | 峰到 GT 中心距离中位数（token） | GT 区域为最高分比例 |
|---|---|---:|---:|---:|
| 匹配正常帧 | RGB | +7.687 | 1.220 | 0.625 |
| 匹配正常帧 | TIR | -1.116 | 1.597 | 0.500 |
| 匹配正常帧 | concat | +13.431 | 1.206 | 0.688 |
| 匹配正常帧 | fused | +10.989 | 0.918 | 0.750 |
| 失配前驱 | RGB | -11.647 | 4.971 | 0.000 |
| 失配前驱 | TIR | -3.045 | 2.612 | 0.250 |
| 失配前驱 | concat | -15.913 | 3.726 | 0.250 |
| 失配前驱 | fused | -4.377 | 4.072 | 0.188 |
| 严重失配起点 | RGB | -15.426 | 4.957 | 0.062 |
| 严重失配起点 | TIR | -8.486 | 2.889 | 0.125 |
| 严重失配起点 | concat | -18.220 | 3.303 | 0.062 |
| 严重失配起点 | fused | -6.038 | 4.946 | 0.062 |

相对 `concat`，`fused` 的目标背景 margin 在失配前驱和严重起点分别提高约 **11.54** 和 **12.18**；这与“融合把已有中心信息进一步抹掉”相反。与此同时，`fused` 在严重起点仍为负 margin，说明信息虽未被融合一致性破坏，仍不足以支持正确中心选择。探针只测张量中的可解码信息，不等价于原始 CENTER head 已经使用了这些信息。

同一事件的固定状态反事实也没有显示搜索融合发生大幅破坏：在 16 个匹配正常帧中，将目标区域特征衰减 25% 后，RGB/TIR 的最终 IoU 中位数仅从 0.733 降至 0.733/0.731；在 16 个严重起点中，RGB/TIR 25% 衰减后 IoU 中位数从 0.004 降至 0.001/0.001。该干预是离线 GT 引导压力测试，不能作为自然场景因果结论，但与“融合后才首次丢失中心”不一致。

## MHSA 侧的全量约束

244 序列、7,807 个梯度样本的全量诊断显示，loc9 RGB/TIR 目标 key 质量相对均匀模板基准约为 **2.23/2.32 倍**，说明后层 MHSA 没有普遍忽略目标；loc9 目标质量与序列 IoU 的中位 Spearman 仅约 **+0.089/+0.101**，跨模态消息范数约 **+0.001/+0.026**。因此它排除了“MHSA 路径整体失活”这一简单解释，却没有证明模板 MHSA 已将目标信息有效传到最终搜索读出。

全量归档失败链条仍是：目标在 crop 内时先出现中心响应/定位失配，之后闭环框偏移导致出界和持续低 IoU。这个顺序支持优先检查搜索表示、中心 head 和状态反馈；不能从归档框单独判定 MHSA、搜索融合或 head 的唯一首因。

## 决策

当前模块方向优先级为：

1. **先查搜索表示到 CENTER head 的读出与局部空间定位**，因为失败起点的 `concat` 已经不可读，且 `fused` 没有进一步恶化；
2. **搜索融合修正/可靠性门控暂不作为已证实首因**，只有在固定状态、温和模态干预下出现跨序列“输入可读、融合后下降”的重复证据时才立项；
3. **head 适配需要单独校准实验**，因为 `fused` 仍保留部分可读性，原 head 可能没有充分读出它；
4. **模板 MHSA 局部空间模块暂不能由本证据直接立项**，需要额外采集最后一次 MHSA 后搜索 token 或做同状态 MHSA 接口反事实，才能区分“上游 backbone 搜索表征不足”和“模板信息传递不足”。

这不是严格 M1 复现：当前工作树 replay 与受保护归档预测存在代码差异；probe 环境为 Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8。结果是机制诊断，不是训练收益、测试集方法选择或因果结论。

## 证据路径

- `output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/probe_final/summary.json`
- `output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/probe_final/probe_dev.csv`
- `output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/combined_final/per_arm.csv`
- `output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/combined_final/selected_features.npz`
- `output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/analysis/analysis_summary.json`

脚本：`tools/diagnose_m1_search_stage_20261009.py`（SHA256 `4b51c7a44a377b16b8ad7ca51d196cde9d704b7b2f078518927091d1cfc182f4`）、`tools/probe_m1_search_features_20261009.py`（SHA256 `9c9fdd3793ac8329f5b3fa98bfbc4ca94f0958033407b71b7724c71f8d6960d8`）。
