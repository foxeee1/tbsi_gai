# ECLVR 前置机制审计：跨层 Value 证据是否可获得（2026-10-26）

> **2026-10-26 证据边界补充：** 后续[信息流与初始化审计](m1_early_gate_bottleneck_audit_20261026.md)收窄了本报告准入结论。前景/背景方向关联仅为描述性统计；未比较current/early V的同类结构，且centroid自包含、位置偏置等尚未排除。单个evidence与几何对齐的负Pearson相关不能否定非线性selector，前景DeltaV范数较低也不能作为可学习变换无效的必要证伪条件。原“无训练机制必须先通过才可短训”要求不再作通用门槛；原始数值与wrapper-vs-wrapper parity撤回继续保留。本轮未训练ECLVR，也没有建立其性能收益。

本记录先完成了静态合同审计，随后用独立 wrapper 做了只读模型前向观测。没有修改模型代码、配置、checkpoint 或测试输出；没有训练或运行正式跟踪评测。

## 1. 审计对象与问题

研究 parent 是 Early-conditioned MSHA gate full15；受保护参照是 M1 `TBSI_E_MHSA_FULL_4090D_seed42`。ECLVR 想回答一个比 scalar gate 更窄的问题：同一来源模板 token 在早层和当前 MSHA 层之间的 Value 内容差异，是否包含可由 Early evidence 选择的目标相关方向。

如果早层和当前层的差异没有稳定结构，低秩分支只是把旧 gate 重新包装；如果差异存在，才有理由继续做 zero-init、梯度和小样本可拟合性检查。

## 2. 已由源码确认的合同

### 2.1 token 位置可以做结构对齐

- ViT 的 template 网格固定为 `8x8=64` token，search 网格固定为 `16x16=256` token，embed dim 为 768，12 heads，head dim 为 64。
- `vit_tbsi_care.py` 在 block index 2 捕获一次 `early_template_rgb/tir` 和 `early_search_rgb/tir`，这四个张量被传给配置位置 `[3, 6, 9]` 的三个 TBSI-E 层。
- TBSI-E 的 `V2TemplateFusion.forward` 接收 `x_rgb_template` 和 `x_tir_template`，每层用自己的 `LayerNorm(768)` 和 `qkv: 768 -> 2304`。因此每个当前层的模板 token 数量和 patch 索引都与 block 2 的模板 token 对齐。

这只证明了**同一 patch 索引的结构对应**，不证明两个层的 token 仍表示同一语义成分。block 3 前的早层 token 尚未经过 TBSI，而 block 6/9 的当前模板已经经过前一个 TBSI 的跨模态融合和 ViT block；层间内容混合使“同索引相减”成为待检验假设，而不是事实。

### 2.2 ECLVR 的 Value 坐标可以定义，但存在分布错配

当前层的原生 Value 是：

```text
V_l = split_heads(W_V^l(LN_l(X_l)))
```

若要比较早层和当前层，最保守的可比形式是：

```text
V_e_to_l = split_heads(W_V^l(LN_l(X_e)))
DeltaV   = V_e_to_l - V_l
```

这使用同一当前层 `W_V^l` 和同一 head 切分，避免直接相减两个不同 `W_V` 坐标系的结果。但 `LN_l(X_e)` 只是数值标准化，不是学习到的跨层语义对齐；如果早层特征在当前层的子空间外，`DeltaV` 的范数会反映坐标错配而不是可保留内容。这个风险必须由统计区分，不能靠增加 rank 或残差上限解决。

### 2.3 Early gate 的证据不是新的内容监督

Early gate 的证据仍是每个模板 token 的三个量：RGB 搜索相似度最大值、TIR 搜索相似度最大值和 RGB/TIR 模板一致性。它只提供来源条件，当前没有目标/背景标签监督 `DeltaV` 的方向选择。因此 ECLVR 的第一道证伪条件必须是 `DeltaV` 自身在目标前景与背景 token 间有结构差异；不能只检查 MLP 或 gate 是否有梯度。

## 3. 已有资产能否支持这项分析

对当前 `TBSI_gai/TBSI/output` 下的 2768 个 `.npz` 产物做了只读 key inventory：没有一个产物包含 `early_template`、当前 TBSI 模板 token、`V_l` 或 `DeltaV`。已有 diagnostics 主要是：

- `early_rgb/early_tir`、`late_input_*`、`late_output_*` 等 search-side 特征；
- `pre/post/residual/fused_*` 的后续融合张量；
- response map、box、state 和序列级 CSV。

其中 `early_rgb/early_tir` 的导出工具只保存 search token；它没有保存 block 2 的模板 token，也没有在每个 TBSI 位置保存当前模板输入。因此不能从现有文件计算 ECLVR 所需的 `DeltaV`、低秩能量、前景/背景差异或与 Early evidence 的相关性。

这不是“证据为零”的机制结论，而是**当前资产不具备判别该机制的观测量**。把后续融合张量或 search token 当成模板 Value 替代，会混入接收端和闭环影响，无法形成因果归因。

## 4. 设计层面的新增风险

1. **早层被三个 TBSI 层共享。** 若 ECLVR 在 `[3,6,9]` 都直接使用 block 2 的特征，差异来源同时包含层深、前序 TBSI 和模态混合，难以解释。首轮 diagnostics 必须按 TBSI location 分开统计，不能把三层拼成一个平均值。
2. **零初始化会造成学习延迟。** 若残差写成 `gamma * U diag(a) z` 且 `gamma=0`，首步主要只有 `gamma` 得到梯度，`U/W_e/W_z` 的梯度可能为零或极小。合同检查需分别记录每个参数组的梯度，不能以“总 loss 能下降”替代。
3. **低秩基不是天然目标子空间。** 随机初始化的 `U` 会把早层差异投影到任意方向。若没有先证明 `DeltaV` 的有效秩和目标/背景差异，直接设置 rank=8/16 是容量选择，不是机制选择。
4. **Early evidence 可能只被重复利用。** 如果 `W_e e` 主导 `a`，ECLVR 退化为另一个 gate；必须单独报告 `W_z z` 与 `W_e e` 的贡献，并做 evidence shuffle 作为诊断对照。
5. **当前层模板已包含跨模态历史。** 在 block 6/9，`X_l` 不是独立单模态模板。若 `DeltaV` 的变化主要来自前序 TBSI 的模态混合，论文应称“跨层 Value 保留”而不能称“同模态早期目标恢复”。

## 5. 下一步唯一允许的分析

新增一个独立、只读的 diagnostics export，不改变默认 forward：

### 必须保存的张量

对每个固定 train-side state、每个 TBSI location `l∈{3,6,9}`、每个模态分别保存：

```text
X_early_template       [B,64,768]   # block 2
X_current_template     [B,64,768]   # 进入该 TBSI-E 的模板
V_early_to_current     [B,12,64,64]
V_current              [B,12,64,64]
early_evidence         [B,64,3]
template_fg_masks      [B,64]
```

只保存固定 train-side calibration states，不从 test 涨跌挑选事件；导出前必须做 diagnostic-off 与 diagnostic-on 的逐框 parity。导出过程不写权重、不更新 tracker state，输出使用新的独立目录和 manifest。

### 必须计算的统计

对每层、模态、head、token 报告：

- `||DeltaV|| / ||V_current||`、`cos(V_early_to_current,V_current)`；
- 前景与背景的均值、分位数和 bootstrap 区间；
- `DeltaV` 的奇异值谱、前 `r` 个方向能量占比；
- `W_z z` 与 `W_e e` 的能量占比（实现后才有）；
- Early evidence shuffle、template token shuffle 后上述统计的变化；
- 将 Value 通过原生 cross-attention 得到的实际 source message，按前景/背景来源分解，并记录范数与余弦，不能只记录 gate 均值。

### 判定标准

- **停止 ECLVR：** 三个 TBSI 层均无稳定前景/背景差异；`DeltaV` 的有效秩接近噪声；正确 evidence 与 shuffle 的统计不可区分；或差异完全由范数变化解释。
- **进入实现合同：** 至少一个位置显示跨序列可重复的前景/背景方向差异，且 evidence shuffle 会破坏该差异；同时 `DeltaV` 不是只由某一个模态或单个事件贡献。
- **进入训练：** 仅在上述机制通过后，才做 zero-init parity、保存加载、逐参数梯度、冻结状态不漂移和小样本拟合。任何 quick/full 指标都必须使用同一 SOT 起点、fresh optimizer 和独立配置。

## 6. 初始审计判断（已由第 8、9 节更新）

在 calibration 导出前，ECLVR 被列为待测单一候选；当时资产缺少早层/当前层模板 Value 张量，无法判断方向结构。第 8 节补入振幅统计，第 9 节再补单位方向与 Early evidence 选择性分析，现行判断以第 9.1 节为准：不做 ECLVR 原始差分注入，只保留一个固定状态的方向对比消息诊断。不得把第 6 节早期候选判断当作当前准入结论。

导出前的静态审计状态（历史记录，已由第 8、9 节取代）为：

```text
STATIC-CONTRACT-AUDIT-COMPLETE /
EVIDENCE-MISSING /
INSTRUMENTATION-ONLY /
NO-IMPLEMENTATION /
NO-TRAINING
```

当时的待办是补齐 Value 观测量；该待办已由第 8、9 节的缓存张量分析完成。当前下一步不是实现零初始化低秩分支，而是仅做固定状态下的实际 source-message counterfactual，准入条件见第 9.1 节。

## 8. calibration 导出结果与去留判断

使用预登记的 `calibration_sequences.txt`（48 条 train-side 序列、每条两个固定事件，共 96 事件），从 Early gate full15 checkpoint 做 FP32、template update off 的只读导出。wrapper 脚本哈希为 `1ac96e976f774bf6693db908a2c169f14182d8d98af9c205a0c2332bab3b8339`。后续代码审计发现 `parity_check()` 创建的 baseline 与 instrumented tracker 都通过 `make_tracker()` 安装了 capture wrapper，因此短帧最大框差 `0.0` 是 wrapper-vs-wrapper 比较，不是 diagnostic-off-vs-on parity；该 parity 证据撤回。wrapper 只读取张量、复制到 CPU 并调用原始 forward，已有导出仍可作描述性统计，但不能称为已完成独立 parity 验证。

### 8.1 前景/背景差异

`fg_delta_norm_mean - bg_delta_norm_mean` 的事件级 bootstrap 95% 区间如下：

| TBSI location | RGB mean diff / 95% CI | TIR mean diff / 95% CI | `DeltaV/CurrentV` | `rank-8` energy |
|---:|---:|---:|---:|---:|
| 3 | `-0.0980 [-0.1127,-0.0833]` | `-0.0208 [-0.0354,-0.0066]` | `0.432 / 0.458` | `0.431 / 0.452` |
| 6 | `-0.3564 [-0.3909,-0.3239]` | `-0.2160 [-0.2459,-0.1861]` | `0.830 / 0.857` | `0.418 / 0.451` |
| 9 | `-0.3494 [-0.3823,-0.3153]` | `-0.2325 [-0.2653,-0.2009]` | `1.058 / 1.022` | `0.477 / 0.490` |

所有层和模态的区间都低于 0，表示当前层间差异的**幅度**更集中在背景 token，而不是前景 token。该结果反驳“直接注入完整、未对齐的早层差分”的合同，但本身不能推出单位方向也没有前景结构。`DeltaV/CurrentV` 从浅层约 0.43--0.46 增长到深层约 1.02--1.06，说明简单的小残差还会面对明显的层间分布错配。

### 8.2 Early evidence 的稳定性

按事件分别计算 token 级 `DeltaV` 范数与三个 Early evidence 的 Pearson 相关。相关符号并不稳定：

- location 3 RGB 的三个 evidence 正相关事件比例为 `0.740/0.656/0.854`，但 TIR 为 `0.438/0.333/0.479`；
- location 6 RGB 为 `0.740/0.594/0.802`，TIR 为 `0.646/0.594/0.698`；
- location 9 RGB 为 `0.479/0.500/0.490`，TIR 为 `0.448/0.281/0.458`。

浅层 RGB 的正相关不能抵消 TIR 和深层的失稳；三项证据在深层几乎不能预测差分幅度。这不支持把 `W_e e` 作为 ECLVR 的主要选择信号。

### 8.3 决策

当前 ECLVR 的机制筛查结论为：

```text
RAW-DELTA-AMPLITUDE-FOREGROUND-SELECTIVITY-REFUTED /
LAYER-DISTRIBUTION-MISMATCH-SUPPORTED /
STOP-UNNORMALIZED-CROSSLAYER-INJECTION /
NO-IMPLEMENTATION /
NO-TRAINING
```

这只停止当前的“block2 特征经当前层 `W_V` 直接投影后做同索引、未归一化差分注入”合同，不否定所有 Value 内容建模。方向级复核见第 9 节；目前没有可部署的 Early-evidence 方向选择器，也没有性能证据。当前论文路线继续以 Early gate 的 Value 幅度调制为 parent，不把 ECLVR 写成已验证贡献。

## 7. 追溯

- 代码：[vit_tbsi_care.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/tbsi_track/vit_tbsi_care.py:387)、[tbsi_layer.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py:621)。
- 现有 search diagnostics：[g0_export_frozen_features.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/g0_export_frozen_features.py:120)。
- 设计候选：[m1_early_gate_value_crosslayer_literature_scout_20261026.md](/root/autodl-tmp/TBSI_gai/TBSI/experiments/reports/m1_early_gate_value_crosslayer_literature_scout_20261026.md)。
- 本次导出脚本：[analyze_m1_eclvr_crosslayer_preflight_20261026.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/analyze_m1_eclvr_crosslayer_preflight_20261026.py)；产物目录：`output/experiments/M1_ECLVR_CROSSLAYER_CALIB48_20261026/`。
- 本次导出未产生 checkpoint 或 tracking benchmark；使用 1 个 GPU 做了约 `0.27 GPUHours` 的只读前向观测。AUC/PR/NPR/OP50/OP75/FPS 均为 `NA`。

## 9. 单位方向与证据选择性复核（2026-10-26）

在不重跑模型的条件下，对同一 96 个 NPZ 做 CPU 离线复核。脚本 SHA256 为 `7347468453648096ff1e88f20d48b5af05bf366dab626ee1c0cc4f1e19e2a5aa`，权威输出为 `output/experiments/M1_ECLVR_DIRECTIONAL_ANALYSIS_CALIB48_20261026_RETRY2/`。输入 96 个 NPZ 的排序清单 SHA256 汇总为 `85527a5f873e48d7b5c9c35197d8b4eae2fca80cb45ea810e6543558417d299c`。首次扩展分析运行因脚本局部变量名错误在计算前景方向相关时退出；失败目录 `...CALIB48_20261026_RETRY1/` 保留，修正后 RETRY2 为权威结果。按 48 条序列聚类、每序列先平均两个事件，再做 10,000 次 bootstrap；指标是每个 head 的前景/背景平均单位 `DeltaV` 方向向量之差的 L2 范数，再对 head 求均值。另将 8x8 foreground mask 做 63 个非零环形平移，作为空间对照分布；它是描述性对照，不是独立置换检验。

| TBSI location | 模态 | 前景-背景方向间距 / 95% CI | 高于平移 mask 中位数的均值差 / 95% CI | 高于平移 mask 中位数的序列数 |
|---:|---|---:|---:|---:|
| 3 | RGB | `0.422 [0.401, 0.445]` | `0.087 [0.072, 0.104]` | `46/48` |
| 3 | TIR | `0.419 [0.391, 0.448]` | `0.103 [0.083, 0.125]` | `47/48` |
| 6 | RGB | `0.481 [0.458, 0.504]` | `0.121 [0.103, 0.139]` | `46/48` |
| 6 | TIR | `0.468 [0.440, 0.494]` | `0.126 [0.107, 0.146]` | `47/48` |
| 9 | RGB | `0.512 [0.493, 0.531]` | `0.164 [0.147, 0.182]` | `48/48` |
| 9 | TIR | `0.510 [0.489, 0.531]` | `0.171 [0.155, 0.188]` | `48/48` |

因此，幅度结论与方向结论不同：`DeltaV` 范数在背景更大，但单位差分方向按真实 foreground mask 分组后，比大多数平移 mask 显示更强的前景/背景区分，且该差异从 location 3 到 9 增大。这支持“存在可分析的方向结构”，但不能证明注入这些方向会改善跟踪。location 6/9 的背景方向一致性不低于前景，故方向间距也不能单独解读为目标方向更纯。

关键否证是现有 Early evidence 对方向的选择关系。`evidence[2]` 是 Early RGB/TIR 模板 token 一致性；它与前景 `DeltaV` 方向对齐度的序列平均相关在全部六个层/模态组合中为负，范围 `-0.242` 至 `-0.413`，各自的序列 bootstrap 95% CI 均低于 0。相比之下，RGB/TIR 搜索相似度（`evidence[0:2]`）与方向对齐的相关较弱、跨层模态不稳定。此处对每个事件先计算 64 token 的 Pearson 相关，再在序列内平均，最后按序列 bootstrap；因此是可预测性诊断，不是因果效应。

### 9.1 收敛判断

- **停止：** 直接注入未对齐、未归一化的跨层 `DeltaV`；同层 Value 重建与 receiver-centered residual 已有负/无效 quick 或固定状态证据，不重开。
- **暂留一个细分诊断候选：** 在现有固定状态上检验“foreground-mask 条件的方向对比/去背景方向”是否能改变实际跨模态 source message，并优于等面积平移 mask、随机方向和仅范数匹配对照。当前结果只证明方向结构与 mask 有关联，不授权实现训练模块。
- **不采用：** 直接用 Early RGB/TIR 一致性正向放大跨层方向；该量与方向对齐稳定负相关。也不把负相关未经留出验证就反转成新 gate。
- **准入下一步的必要条件：** 同一固定状态、无额外训练的 counterfactual 必须记录真实 cross-attention 下的 source message 前景/背景分解，并在按序列留出时显示方向对比优于空间平移/随机对照；否则停止跨层 Value 分支，保留 Early gate 作为主线。

本分析只新增 CPU 统计，未运行模型前向、跟踪推理、训练或评测；跟踪性能指标与 GPUHours 均为 `NA/0`。parity 撤回说明见第 8 节；如后续复用导出，必须补做真正的未包装/包装逐框对照后再声称导出 parity。
