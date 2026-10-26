# Early Gate v1 逐序列与场景归因（2026-10-26）

> **历史方案，非当前执行计划（2026-10-26 清理标记）：** 本文原始数据、引用和推导保留；其中候选优先级、准入条件和下一步命令不再作为当前指令。当前范围见项目 `README_当前研究.md`；本标记不表示所有假设已被证伪，也不批准新训练。清理前原文见 Git `c0f1aff`。


> **设计状态更新：** 本报告原先将 FSCIR 列为固定状态机制候选，之后曾收敛为 `FCCR²`。考虑论文当前的 Value-only 边界和用户提供的近年相关工作对照，`FCCR²` 现降为历史备选，当前主候选为 FCSV-R simplex：仅在首帧前景来源先验上按双模态完整搜索分布的共支持重分配跨模态模板 Value。此公式变化没有新增实验或因果证据；机制筛查仍需先通过同进程 tensor 重复性与 `α=0` parity。完整设计见 [m1_dcvr_v1_v2_value_source_reallocation_20261026.md](m1_dcvr_v1_v2_value_source_reallocation_20261026.md)。

## 结论

**判定：`SEQUENCE-EVIDENCE-REVIEWED / FSCIR-MECHANISM-TEST-ELIGIBLE / CAUSAL-MECHANISM-NOT-ESTABLISHED / NO-TRAINING`。**

这次把 Early-conditioned MSHA gate-only full-244 逐序列结果与 M1 官方属性、目标尺度、相对运动和 M1 失败事件表合并核查。结果支持一个具体观察：gate 的收益与损失高度异质，且会在少数序列上造成很大的闭环差异；已测场景标签没有形成足以解释涨跌的稳定分组。它不支持“SA/TC 序列普遍需要某种对应建模”，也不能证明 FSCIR 已针对了真正首因。

因此，前一份 v1/v2 算子归因中 FSCIR 的状态需要收窄：它可以作为**唯一待证伪的 QKV 前关系建模候选**进入 train-side 固定状态机制检验；目前还不能称作已由逐序列结果选出的正确算子，也不据此进入训练。

## 来源与配对可信度

- 主比较使用 `M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/full244_per_sequence_m1_compare.csv`。该候选从受保护 M1 epoch15 权重出发，只训练 585 个 gate 参数 3 epoch；它不是 full15 joint candidate，也不是从预训练权重全网训练。
- 244/244 序列的 CSV 中 `m1_auc` 与 M1 归档属性表中的 `m1_auc` 完全相同；gate-only 报告的 M1 aggregate 也采用受保护参考指标。故该表可用于描述 gate-only checkpoint 与归档 M1 的逐序列关系。它仍是单 seed、历史输出表的分析，checkpoint 已不在该 run 目录，不能重跑复核；证据等级为可用的归档配对结果，不是可复现新结果。
- M1/候选逐序列 CSV 提供 AUC、Precision、Normalized Precision；不包含逐序列 OP50/OP75。下文不补造这些指标。
- Early gate full15 joint candidate 的 per-sequence 预测已被 2026-10-24 provenance audit 判为混合/来源不一致。本报告不使用其逐序列差值作归因。full15 的小幅点估计只作为独立方向性历史记录。
- M1 官方属性及几何场景值来自 244 条 M1 归档序列表；标签重叠，且对象尺度、运动、遮挡与背景等因素相互混杂。对属性分组只作探索性描述，不作因果解释或多重检验后的显著性声明。

## 全集与序列效应

gate-only full-244 AUC 为 55.99，受保护 M1 为 56.21，aggregate 差值为 `-0.22 pp`。在逐序列表中，AUC 差值均值为 `-0.213 pp`、中位数为 `+0.0068 pp`、标准差为 `8.71 pp`；122 条上升、115 条下降、7 条相同。序列 bootstrap 95% CI 为 `[-1.30,+0.88] pp`，包括零。逐序列 Precision 和 NPR 差值均值分别为 `-0.359` 与 `-0.180 pp`。

差值分布说明 aggregate 被两侧尾部抵消：17 条序列下降超过 5 AUC 点，合计 `-378.17`；11 条上升超过 5 点，合计 `+326.20`。因此不是“所有序列轻微变差”，也不是“广泛改善但均值被少数例外拉低”。对现有候选更贴切的描述是：大多数序列差异很小，少数轨迹出现明显的恢复或灾难性退化。

按 M1 AUC 四分位分层，最低四分位 61 条平均增加 `+2.54 pp`，其余三组分别为 `-1.94/-0.42/-1.04 pp`；但最低组序列仍是 32 胜、28 负、1 平，且 parent AUC 与差值相关仅 `r=-0.130`。这提示低基线序列可能有更大恢复空间，却不能作为可靠的逐序列选择器。

| 序列 | AUC 差 | M1 AUC | 已登记场景/几何线索 | 解释边界 |
|---|---:|---:|---|---|
| `10runone` | `-65.37` | 79.48 | SA/TC/BC；中位目标短边 44 px；相对运动 p90 0.083 | 原 M1 表现较好，却在 gate-only 轨迹中严重退化；历史时间审计显示约 frame 51 后持续分叉。不能把它概括为“只救低基线困难序列”。 |
| `bluegirlbiketurn` | `-34.98` | 59.49 | MB/TC；中位短边 28 px；相对运动 p90 0.086 | 中小目标与多个场景属性共存；历史轨迹约 frame 38 后出现持续差距。属性无法区分是哪种因素在起作用。 |
| `whiteridingbike` | `-26.63` | 40.29 | LR/LI/TC；中位短边 20 px；相对运动 p90 0.137 | 小目标/低照度/热交叉标签共现；不能将其单因归于分辨率或某一模态。 |
| `blktribikecome` | `+58.75` | 23.91 | LR/TC；中位短边 37 px；M1 严重失配帧多数已在近似 crop 外 | 这是低基线强恢复例，但 M1 已有出界，不能证明 gate 修复了首次 crop 内选峰。 |
| `3rdfatboy` | `+54.29` | 18.49 | SA/TC/BC/DEF；中位短边 45 px；相对运动 p90 0.092 | 与 `10runone` 同有 SA/TC/BC，却变化方向相反；标签不解释门控效应方向。 |
| `leftfarboycomingpicktheball` | `+42.39` | 19.46 | LR/LI/TC/DEF；中位短边 40 px；相对运动 p90 0.047 | 低基线恢复例，M1 严重帧同时包含 crop 内/外阶段；无法仅据 AUC 分清恢复发生在何时。 |

完整 244 条差值在 [gate-only paired per-sequence CSV](../../output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/full244_per_sequence_m1_compare.csv)；属性和 M1 失败几何在 [M1 per-sequence scene table](../../output/experiments/M1_FULL244_SCENE_MINING_20261009/per_sequence_scene.csv)。

## 场景切片：哪些规律存在，哪些不存在

| 官方/派生切片 | n | AUC 差均值 | 序列 bootstrap 95% CI | 胜/负 | 判断 |
|---|---:|---:|---:|---:|---|
| LR 低分辨率 | 68 | `+0.70` | `[-1.39,+3.34]` | 33/33 | 没有稳定方向。 |
| SA 相似外观 | 106 | `+0.13` | `[-1.73,+1.89]` | 58/47 | 组内正负都有，不能指定为 gate 的有效场景。 |
| TC 热交叉 | 147 | `+0.02` | `[-1.76,+1.84]` | 76/68 | 基本无平均变化，且两侧都有极端序列。 |
| LI 低照度 | 51 | `-0.38` | `[-2.51,+1.98]` | 25/26 | 无稳定方向。 |
| BC 复杂背景 | 77 | `-0.53` | `[-3.36,+2.18]` | 33/42 | 无稳定方向。 |
| DEF 形变标签 | 46 | `+2.64` | `[+0.32,+6.04]` | 32/13 | 值得在独立 train/dev 事件中复核；这是在多项重叠属性中探索出来的信号，不能视为确认性结论。 |

这些 bootstrap 是按序列重采样的描述区间；标签并不独立，且查看了多个标签。DEF 的正向切片只能生成假设，不能用于声称“形变导致收益”。尤其 SA/TC 既包含 `10runone` 这类大幅退化，也包含 `3rdfatboy` 这类大幅恢复，否定了按标签直接加一个固定跨模态门控的简单设计逻辑。

M1 自身归档失败事件提供了问题背景：至少 365/377 段持续 `IoU<0.2` 事件的起点，目标仍完整处于近似 search crop 内；220 段后来才完全出界，首次出界距事件起点中位约 19.5 帧。这确实支持“先有 crop 内定位失配，之后闭环漂移加重”的系统级问题定义。它没有说明首个失配发生在模板 MSHA，也没有标记门控候选在哪一帧修复/破坏目标身份。

## 对输入建模算子的含义

1. **已排除**：只用“低分辨率、SA、TC、低照度、复杂背景”等整序列标签来选择一个全局模态权重或对所有同类序列统一增强。gate-only 结果在这些组内均未显示稳定一致方向。
2. **支持进一步检验**：算子应在 token/candidate 层区分目标相关证据和竞争背景，而不是只从序列级标签或全局相似度推断可靠性。依据是相同属性组中的巨大反向效应，以及 M1 的 crop 内起始失配表型；这是设计需求，不是已证实的 MSHA 首因。
3. **FSCIR 的定位**：QKV 前软对应输入重构比继续调 Value gate 更贴近“改变关系形成”的用户目标，但现有逐序列结果没有测量对应矩阵是否正确、是否对准目标、是否压低 hard negative。因此 FSCIR 只能进入**固定状态机制筛查**，不能依据这些结果直接写成性能候选已成立。
4. **最小判别分析**：在未用于方法挑选的 LasHeR train-side 事件中，固定模板、真实前一状态和同一个 crop；按预先冻结的匹配正常/失配起点各选多个序列。对同一层比较原生 MSHA、正确前景来源软对应、等面积 shifted/decoy 来源三种输入。只分析 QK logits/注意力变化是否定位到目标前景、GT-cell 对 hard-negative 的 rank/margin、CENTER/offset/decoded IoU；并按序列汇总。候选与 decoy 若没有稳定的目标特异差异，停止 FSCIR；若只有 QK 改变而定位读数不变，也停止在该插入点继续加复杂度。

这一步先做固定状态分析，不需要重新测试 LasHeR test，也不使用上述 test 序列来选阈值或调算子。只有该机制签名跨序列出现、且候选明显优于等面积 decoy，才有证据说该输入建模和目标对应问题对上；之后再审计代码合同和训练复现条件。当前 quick3 control 仍有 `11.15 pp` AUC 重复差，post-augmentation tensor hash 未做；固定状态 M1 replay parity 也未解决，这些会阻止快速训练归因，但不妨碍先完成只读/固定状态机制判断。

## 可追溯资产

- gate-only full-244 逐序列表与汇总：`output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/full244_per_sequence_m1_compare.csv`；运行合同和总体/时序分析：`experiments/reports/m1_early_conditioned_msha_gate_proxy_20261021.md`。
- M1 逐序列属性/几何：`output/experiments/M1_FULL244_SCENE_MINING_20261009/per_sequence_scene.csv`；官方属性场景说明：`experiments/reports/m1_full244_scene_mining_20261009.md`。
- full15 预测来源审计：`output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023/analysis/recompute_retry1_20261024/provenance_audit.md`。
- FSCIR 先前理论设计：`experiments/reports/m1_msha_v1_v2_operator_attribution_20261026.md`。本报告将其结论由“设计准入已通过”收窄为“候选机制检验准入”，没有修改受保护 M1 或任何模型资产。
- 本报告只读取已有 CSV/报告并做离线分组、相关与序列 bootstrap；没有训练、跟踪推理、模型前向或权重写入。GPUHours `0`。
