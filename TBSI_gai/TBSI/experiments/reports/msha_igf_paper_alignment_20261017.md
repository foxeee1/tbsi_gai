# 当前 MSHA 与 IGF 实现的论文对齐分析

## 结论

当前两者都不是论文方法的忠实复现。MSHA 是 M1 融合输出后的可靠性残差筛选器；IGF 是根据论文动机重建的模板到 search cross-attention pre-fusion。两者都保留 M1 的 CENTER head 和 tracker state，因此实验结果只能说明“当前工程插入合同”的效果。

## 对照表

| 项目 | 论文机制 | 当前 MSHA reliability adapter | 当前 IGF pre-fusion |
|---|---|---|---|
| 输入 | RGB/TIR 图像与模板桥交互；显式目标/照明条件 | 已融合 search feature、RGB search、TIR search | RGB/TIR search token；RGB/TIR template token |
| 位置 | 论文 IGF 融合路径，配合 GIE、CA/SA | `tbsi_fuse_search` 后、CENTER 前 | `tbsi_fuse_search` 前 |
| 核心算子 | illumination-guided weighting、跨模态交互、目标保持模板更新 | first-order cosine、channel-centered second-order agreement、CNN residual + sigmoid gate | search query 对拼接 template KV 的 multi-head attention + bounded residual |
| 模态权重 | 论文有显式 illumination/reliability 变量 | 没有显式 RGB/TIR 权重，只产生单一空间 gate | 没有显式 illumination lambda；两路 residual 由同一 attention 输出拆分 |
| 模板更新 | TPTU 候选验证和 target-preserved 更新 | 无 | 无；模板仍由原 tracker 提供 |
| 状态 | 论文闭环包含模板验证/更新 | 不改 tracker state | 不改 tracker state |
| 训练 | 论文完整训练与消融合同 | M1 冻结，仅训练 adapter proxy | M1 epoch15 初始化，仅训练 IGF 3 epoch proxy |

## 关键偏差

1. **MSHA 偏差**：代码中的 `MshaReliabilityAdapter` 并未实现论文的 GIE、illumination lambda、CA/SA 或 TPTU。它只使用 M1 已融合后的特征和两个模态分支做局部一致性估计，属于工程诊断 adapter。
2. **IGF 偏差**：`IGFPreFusion` 的 template-to-search attention 捕捉了“模板桥接 search”的方向，但没有 RGB 图像照明估计、`lambda/(1-lambda)` 加权、论文中的完整跨模态路径，也没有 TPTU。因此应称为 IGF-inspired approximation。
3. **共同偏差**：两者都没有改变 CENTER 的 peak/offset/size 联合读出，也没有修正错误 crop 的状态反馈；这正是当前 M1 已观察的系统级失败链条。

## 实验对照

- MSHA reliability proxy：difficult5 AUC `63.7488`，相对 M1 `-2.9627`；`boyss` 和 `girldownstairfromlight` 明显退化。结论是 `RELIABILITY-GATE-NOT-SUPPORTED`。
- IGF pre-fusion proxy：full-244 AUC `56.2896`，相对 M1 `+0.0819`，但低于 S1 `56.4599`；OP75 `+0.3033`，AUC/OP50 未同步改善。结论是 `SMALL-POSITIVE-VS-M1 / BELOW-S1`。

这两个结果不能直接比较“论文模块谁更好”：数据范围、插入位置和监督合同不同。可以比较的是机制信号：IGF 在当前接口上带来轻微局部收益，MSHA 的 agreement residual 在 difficult5 上没有稳定收益；二者都未证明能解决长时漂移。

## 下一步判断

若继续论文复现，应单独实现并验证 GIE/illumination weighting 与 TPTU，而不是在当前 MSHA 或 IGF adapter 上继续加 head。若目标是修复 M1 退化，优先做同 crop、同状态的 CENTER 目标 cell 命中、offset/size 误差和 residual ratio 诊断；当前证据不支持把论文模块缺失直接归因于退化首因。
