# Early gate 后继路线：训练判别力、真实信息流与初始化覆盖审计

日期：2026-10-26。研究 parent 为 Early-conditioned MSHA gate，受保护研究参照为 M1。此次只核对现有源码、配置、日志和报告；没有新增模型前向、训练、跟踪测试或性能指标，没有修改模型行为。

## 1. 核心判断

**目前不能把持续不涨点归结为“MSHA 难以改进”或“算子太简单”。更直接的障碍是：部分候选并未按宣称的初始化和作用路径实施，而低步数短训也不足以可靠区分候选。** 最近 ECLVR 只有观测分析，但此前确实做过 Value reconstruction 等 quick3。问题不是完全没有短训，而是把信息不足的短训负结果和无训练统计当成过强的机制否证。

本次确认三个源码事实：

1. **构造器的零初始化被外层统一初始化覆盖。** Early gate 末层、Value reconstruction 的 up 层都存在这一问题；历史“zero-init parity passed”不能按字面成立。
2. **旧 reconstruction 同时改变同模态与跨模态 Value。** 它不是论文限定的“仅调整跨模态来源消息”。负结果不能定位到跨模态内容修正本身。
3. **TBSI-E 的 MSHA 输出只更新模板。** 默认分支立即返回原搜索 token，搜索特征通过后续普通 ViT 自注意力接收影响；没有执行 legacy TBSI 的完整 s2t/t2s 桥接链。

此外，旧 quick3 control 重复差为 11.15 pp，而本轮关注的目标增益为约 0.5 pp；这一个重复差不是总体噪声标准差，但足以说明单个 quick7 配对的正负号没有稳定排序能力。

## 2. 当前实际信息流

对应 `vit_tbsi_care.py:350,375,398,441,492`、`tbsi_layer.py:621,1481,1677,1937`、`tbsi_track.py:537,616`。

```text
RGB: [64 template tokens, 256 search tokens] ─┐
                                           ├─ 各模态共享权重的 ViT blocks
TIR: [64 template tokens, 256 search tokens] ─┘
                  │ block index 2 后缓存 early template/search
                  │ 每个 template token: RGB max-match、TIR max-match、一致性
                  │ 3 → 32 → 2 → 两个来源方向的 gate
                  ▼
每个 TBSI location 3 / 6 / 9（零起始索引）:
当前两模态 template → LN → Q,K,V → 联合 128-token attention
  RGB message = A_RR V_R + A_RT (g_T ⊙ V_T)
  TIR message = A_TT V_T + A_TR (g_R ⊙ V_R)
  → output projection → 加回原 template residual → 写回 template
  当前 search tokens 在此默认分支保持原值
                  ▼
下一 ViT block: template/search 联合自注意力
  以 search 为 query，读取更新后的 template 和原 search
                  ▼
最终 LayerNorm → 取两路 search → 拼接 → Conv + BN + ReLU
                  ▼
原 CENTER score/size/offset heads → 解码 → 下一帧 crop/state
```

这是**帧内模板表示交互**，不是在线替换模板图像的时序模板更新。原模板从 block 0 起已经与搜索 token 做联合自注意力，因此原 MSHA 的模板输入并非完全不含搜索条件。论文宜说“缺少对跨模态来源 Value 的显式条件控制”，不宜说“原网络无条件、不利用搜索信息”。

### 2.1 传导路径存在，但局部改变不等于定位效用

设原两路消息为上式，则跨模态内容增量的直接影响为：

`ΔM_R = A_RT [g_T ⊙ ΔV_T]`，`ΔZ_R = W_O ΔM_R`。

之后还要经过模板 residual、下一块的 LN/自注意力/MLP、最终搜索融合和定位头。必须记录实际 `ΔM`、`ΔZ`、下一块 search delta，才能判断变化在哪里衰减。**目前未证明 LN 消除了变化，也未证明 CENTER 是阻断首因。** 非零梯度或较大 Value 范数都不能代替对任务有益的变化。

scalar gate 是逐模板 token 的两个系数，不是只有全局两个常数。它不改变单个来源 Value 向量的方向，但不同 token 重加权可以改变总消息方向；联合训练时 W_V 也可适配。因此不能宣称 gate 完全不能改变信息内容，也不能把它等同于能生成新内容的 Value 重构。

## 3. 初始化覆盖：已经确认的实现问题

源码调用顺序：

```text
EarlyTemplateValueReconstruction.__init__:
  zeros_(up_rgb.weight), zeros_(up_tir.weight)
V2TemplateFusion.__init__:
  zeros_(early_gate[-1].weight/bias)
VisionTransformerTBSI.__init__:346
  self.init_weights(weight_init)
VisionTransformerTBSI.init_weights:554-561
  递归遍历子模块，仅跳过 PVLAdapter
_init_vit_weights:598-622
  对每个 nn.Linear: trunc_normal_(weight, std=.02)
```

统一初始化末尾只为 MessageAdapter、EarlyTargetMarginGate、EarlyUtilityRouter 恢复特殊初始化，没有恢复 EarlyTemplateValueReconstruction 或普通 early_gate。没有 checkpoint 覆盖这些新增参数时，构造器的零输出不保留。当前源文件 `tbsi_layer.py` 与 `vit_tbsi_care.py` 的 hash 见 snapshot；模型文件与旧 quick3 provenance 中记录的相应 hash 匹配。

历史 preflight 的 score/box 最大差 `5.47e-5 / 6.95e-5` 只说明小于其 `1e-4` 容差。源码已证明输出层非零，因此不能把差异全解释成 transpose/reshape 舍入，也不能把“接近 parent”写作“严格恒等”。同样，不应再用“up 为零导致第一步 down 梯度被阻断”解释该历史运行——其实际 up 并非零。

**边界：** 此问题证明实现不符合宣称，尚未证明其造成了 -4.39 pp 或所有失败。受保护 M1、已有 Early gate checkpoint 和历史默认路径不得被顺手重置。修复应只作用于新候选独立模式中的新增残差层，并在整个模型初始化和权重加载后检查实际参数；旧 gate 初始化行为先保留为明确 parent 条件。

## 4. 前置检查与正式训练没有共享同一配对保障

`preflight_m1_early_gate_value_recon_sot_20261026.py:117` 显式把 candidate 的公共 state 拷给 parent，原因是 SOT 权重不包含新增 TBSI-E 参数。这个措施在前置检查内有效。

但正式 `run_m1_early_gate_value_recon_quick3_20261026.sh` 两次独立调用训练入口；两份 YAML 的 INIT_CHECKPOINT 均为空，也没有公共初始化文件。candidate 在模块构造和统一初始化时额外消耗随机数。相同 seed 不保证后续公共 TBSI/gate 参数相同。脚本也没有设置 MATCHED_SEED；当前 train_script 的默认值为 None，且旧 provenance 没有保存该入口文件的 hash，因此不能用当前入口推定全部历史运行细节。

**已确认：** 正式短训未登记与 preflight 等价的公共初始化保障。**尚未实测：** 两臂 step-0 公共参数差异的具体大小、增强后输入是否相同、该差异解释多少性能下降。不能把修复过的 preflight 当成正式短训也已配对的证据。

## 5. 旧 Value 重构改动范围与论文机制不一致

`tbsi_layer.py:1329-1341` 先令 `v = concat(V_R+ΔV_R,V_T+ΔV_T)`，随后 `:1481-1492` 从这个 v 同时计算 self 与 cross。因此 RGB 消息实际为：

`M_R(old) = A_RR(V_R+ΔV_R) + A_RT[g_T ⊙ (V_T+ΔV_T)]`。

仅修改 cross-source Value 的版本应为：

`M_R(cross-only) = A_RR V_R + A_RT[g_T ⊙ (V_T+ΔV_T)]`。

TIR 对称。旧版额外项 `A_RR ΔV_R` 同样改变了本模态路径。它可能帮助，也可能破坏本来可用的表示；尚无 self-only/cross-only 对照来区分。因此旧版负结果不能证伪论文限定的 cross-only 机制。

### 5.1 输入看似复杂，但差分项不增加表达力

旧重构先拼接 `[s,t,s-t,e]` 再过单个线性 down 和 GELU。线性层前：

`W_s s + W_t t + W_d(s-t) + W_e e = (W_s+W_d)s + (W_t-W_d)t + W_e e`。

所以在这种位置显式拼差分没有增加函数表达能力，仅改变参数化和优化偏置。147600 参数、三组输入并不自动等于更强的交互建模。旧分支仍是同索引 token 的低秩映射，没有新增空间对应关系；但这些结构限制是否已成为性能瓶颈，需要有效训练区分，不能凭公式直接判死。

## 6. 现有短训与“可拟合”证据能支持什么

| 资产 | 实际观察 | 可支持与不可支持 |
|---|---|---|
| Value reconstruction quick3 | AUC 48.5779 → 44.1906，-4.3874 pp，7序列 | 当前运行负结果真实；初始化/作用范围/重复波动使结构性归因不成立 |
| control repeat A/B | AUC 45.5054 / 56.6525，差11.1471 pp | 显示单次 quick7 排序不可靠；不是总体方差估计，也不能直接归为 BN 或 CUDA |
| quick3 预算 | 每 epoch 512/16=32 microbatches，32/8=4更新，三轮12个更新边界 | AMP 实际成功 step 数仍需 scaler/skip 日志；不能称充分训练。M1完整日程约7020个更新边界 |
| preflight 梯度 | 新增12 tensors参加更新，up gradient非零 | 支持分支可微、能更新，不证明学到了目标相关内容 |
| 所谓16样本拟合 | 实际2个合成样本重复8步，拟合同结构教师 score/box，MSE 8.59e-9→1.24e-9 | 支持微小教师输出差可优化；不是真实GT跟踪过拟合，不证明真实监督足以驱动模块 |
| Early gate full15 | 历史AUC56.56；M1保护参考56.21 | 单臂点估计；旧逐序列来源失配、缺matched gate-off，尚不能独立归因给gate |
| DCVR-v1 full15 | AUC55.96；训练mask uniform fallback100%，测试0% | 输入不一致已确认；不能单独归因到Value算子无效 |

旧 reconstruction 报告关于“缺场景选择”“Early evidence不够可靠”的结论超过测量。不能仅从序列名字标注退化类型；需要具体帧或数据集属性。总loss接近也不能证明轨迹可重复。

近期 ECLVR 的单位方向/前景关联是描述性结构，不是定位效用。前景范数小于背景不意味着可学习校正无效；单个 evidence 与某个几何指标的负 Pearson 相关也不能否定非线性条件模型。原“未训练必须先涨点”的准入逻辑不适用于全部可学习算子，应收窄为实现与数值风险检查。

## 7. 推荐下一轮：复用同一重构器，区分作用路径

**不新增可靠度输入、专家、频域、head 或搜索分支。** 先验证被旧实现混在一起的两条路径。

| 臂 | Gate | 新重构器 | Self Value | Cross Value | 用途 |
|---|---|---|---|---|---|
| A | 同一Early gate parent | 关闭 | 原V | 原V | 直接parent |
| B | 同A | 与C完全相同 | V+ΔV | V+ΔV | 复核旧作用范围；使用修正后的共同训练合同 |
| C | 同A | 与B完全相同 | 原V | V+ΔV | 唯一主要干预：将重构限制在跨模态来源路径 |

保留已有 rank8、max_delta0.10、三个TBSI位置；不同时搜索rank、层位、额外loss。B/C复用同一套输入和网络，先不为了公式精简改变参数化。C-B识别写入路径，C-A判断相对parent效用。此设计不保证C优于B：若B更好，论文的“必须保护self路径”主张即不成立。

共同保障是实验工具修复，不是额外方法贡献：从一次SOT载入后导出公共未训练state；所有臂加载同一公共参数，新增重构器完成外层初始化后再清零up；隔离模块初始化RNG并统一post-init训练RNG；记录增强完成后的tensor hash。关闭新增分支时保留parent行为。真实batch拟合只用于发现断图/标签错误，不要求未训练模块先提升IoU。

### 有限预算的推进方式

1. 用一个真实训练batch完成构造后参数检查、写入路径检查、梯度和实际step记录；这一步不再开展大规模frozen机制筛查。
2. 建立新的短训版本：先每臂 **128个成功optimizer updates**（有效batch128约16384次样本呈现），保留128步checkpoint；如尚在学习且未发现实现错误，再把A/C延长到384步。步数只是预算，不声称收敛。LR先保持`1e-4`，这一窗口内不因旧的3epoch标签提前衰减；三臂同一schedule。记录AMP跳步和实际呈现数。
3. 保留quick7用于故障观察；排序改用事先固定的更大开发cohort，例如40序列，按与结果无关的确定性规则选取并保存manifest/hash。它仍是开发集。新协议需重新训练匹配A/B/C，不能接用旧quick3 control。
4. 同一已训练checkpoint，缓存相同crop/state，只开关新增残差，记录`ΔV→ΔM_cross→下一ViT的search delta→定位loss/IoU`；不要求每个事件都改善。训练期可加只读一阶效用`-<∂L/∂V_cross,ΔV>`，但它不是完整跟踪收益，也不改loss。
5. 如修复后A重复仍强烈分叉，优先区分输入hash、公共state和数值执行，暂停候选排序；不继续扩展模块。若C只增大消息而不改善相同状态的定位和更大cohort结果，暂不支持该算子。若C稳定胜A且优于B，才支持“跨模态内容修正＋保护self”的机制和后续full训练。

worker4训练仍为单任务5进程，串行运行；推理使用已登记精度/线程口径。此次没有启动任何一臂，也未创建训练配置。

## 8. 论文问题与证据边界

建议待验证的问题为：**在模板已经包含搜索上下文的条件下，如何修正跨模态模板消息中的不适宜内容，同时保留本模态可用表示？**

逻辑链应由实验逐项建立：原消息存在场景相关错误传播 → 幅度控制是否不足 → cross-only内容校正是否优于gate-only与self+cross校正 → 真实cross message改变 → 后续search与定位改善 → 闭环序列收益。当前已确认的是实现与判别力缺口，未确认“错误Value传播是M1首因”，也未确认“保护self一定提点”。不能用安防场景前缀代替与已有质量融合工作的机制区分。

当前结论分类：**已排除**历史up层严格零初始化和旧重构仅改cross的说法；**暂不支持**用旧quick3负结果否定内容修正；**尚未区分**初始化、self扰动、训练预算与表示限制各自的性能影响；**支持下一步单变量检验**上述A/B/C路径对照。

## 9. 证据与登记

- 源码/原表/旧报告快照：`experiments/snapshots/m1_early_gate_bottleneck_audit_20261026/before/`；校验表：同目录`source_manifest.json`。
- 旧短训：`experiments/reports/m1_early_gate_value_recon_quick3_20261026.md`；原始数值保持。
- control重复：`experiments/reports/m1_early_gate_control_repeatability_20261026.md`。
- 历史实现检查：`output/experiments/M1_EARLY_GATE_VALUE_RECON_SOT_PREFLIGHT_20261026/preflight.json`；保留原status，使用本审计更正其解释。
- provenance风险来源：`experiments/reports/dcvr_manuscript_draft_20261026.tex:169-178`；未重新运行或改写M1结果。
- Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`，工作树原有修改保留；软件/硬件为历史资产各自记录，无新运行环境声明。
- 本分析AUC/PR/NPR/OP50/OP75/FPS均NA，GPUHours=0；新ID `M1-EARLY-GATE-BOTTLENECK-AUDIT-20261026`，同步中文主表与CSV，并对原preflight、quick3和ECLVR方向结论加审计说明。
