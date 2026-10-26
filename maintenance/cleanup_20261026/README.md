# 项目清理清单与执行记录

此次任务：列出可移除代码和冗余报告，保存长期未提交的研究状态，整理当前上下文及 Git 工作树。没有选择新算子，没有训练/测试，没有删除权重、原始预测、数据集或实验日志。

## 1. 对长期没有稳定增益的判断

目前的证据不能支持一个统一的算法首因。可以确认的是：

- 多条实验功能混在共享构造器、forward、actor、tracker 与 optimizer 中，报告叙述与真实开关/作用路径有时不一致。这直接削弱方法归因。
- 历史 quick3 仅12个更新边界，quick7 control 两次跨度11.15 pp；按其正负决定0.5 pp量级候选的去留没有足够判别力。
- 部分所谓合同通过实际上只支持很窄的事实：构造器零初始化被全局初始化覆盖、两份合成输入拟合教师被称为16样本拟合、前置公共参数配对未沿用到正式训练。
- 旧候选建议、修正说明与后续建议同时保留在数百份报告中，却没有唯一当前入口，容易让过时计划重新进入执行。
- 这些事实说明研究流程存在重要混杂；尚未量化各自对模型性能的影响，不能因此保证清理后必然涨点，也不能据此否定所有历史候选。

之前助手对部分负结果的解释强于证据、对未训练算子设置过强的机制门槛，需要更正。这不应归咎于用户没有继续增加模块。

## 2. 已执行与未执行

| 项目 | 本轮动作 | 是否改变实验数值行为 |
|---|---|---|
| 长期未提交代码、配置、报告、启动器 | 本地checkpoint提交`c0f1aff`，完整清单见`files_before.json` | 无 |
| 无使用的标准库imports | 删除6个import语句 | 不改变张量/参数路径；未运行模型验证 |
| 255个上游参考源码文件 | 仅从当前Git索引移出，磁盘原位保留；归档分支可恢复 | 不改变磁盘源码 |
| 研究说明与启动脚本被全局ignore | 修正`.gitignore`，使规则/资产主表/脚本可追溯 | 无 |
| 当前研究上下文 | 新增项目README与子目录AGENTS，按需读取资产 | 无 |
| 7份已过时行动方案 | 加“历史方案，非当前执行计划”标记，原数据和路径保留 | 无 |
| 29项大段代码候选与1项保留重构器 | 列出准确位置和耦合项，**尚未删除** | 无 |
| 339份Markdown报告、149份YAML | 完成逐文件清单；没有按负指标、日期或未直接登记就删除 | 无 |
| master、旧分支、remote和历史提交 | 保留；无force push、reset或历史改写 | 无 |

本轮清理完成后仍保留嵌入式历史模块，**没有把主干宣称为已经最小化**。进一步删模块需要同时移除调用点、配置和训练分支，并核对受保护路径；单独删除一个文件会导致导入/历史配置失败。用户当前要求的具体删除清单已经完整产出。

## 3. 可直接核对的代码片段

下面所有行号统一对应清理前提交`c0f1aff`，避免import删除导致行号漂移。

| 文件 / 位置 | 片段或模块 | 清理意见 |
|---|---|---|
| `lib/models/tbsi_track/tbsi_track.py:4,5,7` | `import math`、`from operator import ipow`、`from typing import List` | 已删除；AST无名称读取 |
| `lib/models/tbsi_track/vit_tbsi_care.py:21,24,25` | `logging`、`OrderedDict`、`deepcopy` imports | 已删除；AST无名称读取 |
| `lib/models/layers/tbsi_layer.py:33–98` | `EarlyTargetMarginGate` | 可移出当前主线；连同构造、margin分支、配置和初始化特殊处理 |
| 同文件`:101–149` | `DiscriminativeEvidenceRouter` | 可移出当前主线；连同actor DER loss、mask输入和DER_ONLY路径 |
| 同文件`:191–242`、`:1347–1494` | `QueryConditionedCrossValue`与输出分派 | 可移出；后一个范围还包含原生else，**不能整段机械删除** |
| 同文件`:440–473` | `_condition_search_message` | 历史T2S探针；连同TBSILayer环境分支抽离 |
| 同文件`:476–535` | `_early_gate_token_reliability` | 历史token置信缩放；连同forward倍率处理抽离 |
| 同文件`:658–733` | FCSV-R来源重分配 | 历史环境开关；移到legacy实现或随对应runner整体归档 |
| 同文件`:842–918` | uncertainty实验 | 连同participation/message_gate及诊断处理 |
| 同文件`:1002–1057` | cross_scale反事实/collector | 适合移入独立分析路径；先保留受保护默认分支 |
| 同文件`:1495–1520` | PRIORITY2 capture | 可抽离诊断；保留原产物和读取脚本 |
| 同文件`:1523–1609` | receiver-centered Value residual | 历史fixed28路线；连同环境参数和runner归档 |
| 同文件`:152–188` | `EarlyTemplateValueReconstruction` | **暂不删除**；旧负结果归因已撤回，不因清理擅自否定候选 |

独立模块的下一批移除候选：

- 主融合/head旁路：`adaptive_post_fusion_router.py`、`precenter_gated_residual.py`、`target_rank_gate.py`、`amcm_reliability.py`、`msha_screen_adapters.py`。
- 搜索/模板额外分支：`spatial_token_bridge.py`、`search_conditioned_t2s_router.py`、`early_utility_router.py`、`early_search_residual.py`、`post_tbsi_cross_modal_residual.py`、`local_cross_modal_correspondence.py`。
- 时序/专家/外部对照：`causal_tam_adapter.py`、`masked_template_hmoe.py`、`pvl_adapter.py`。
- tracker注入历史路线：`single_signal_prefusion.py`、`local_signal_post_fusion.py`。
- 静态lib引用少的组合：`target_conditioned_local_search.py`与`local_attender.py`；仍需检查tools中的动态加载，不能仅凭无静态import删除。

上述18个独立模块都在`lib/models/layers/`。精确行号、源码hash、删除状态和关联调用点见[代码清单](code_cleanup_inventory.csv)，共36项：6个已删除import、29项待依赖处理、1项明确保留。

频域线仍按根规则停用；本轮没有扩展，也没有把历史频域接口残留机械删掉。M1、4090D复现配置、head、优化器和已有初始化行为均没有借清理进行修正。

## 4. 冗余报告：冗余的是行动计划角色

339份Markdown报告中未发现内容完全相同的副本。261份被CSV直接以路径提及，78份未被直接提及；后者中包含受保护M1原始报告，所以“未直接引用”不是删除依据。

以下7份报告的未来动作容易互相冲突，已标为历史计划；保留原文件和证据：

| 报告 | 不再进入默认上下文的原因 |
|---|---|
| `m1_early_gate_optimization_analysis_20261026.md` | 旧gate优化计划，部分归因依赖已失效逐序列provenance |
| `m1_msha_internal_input_interaction_audit_20261026.md` | 曾提出另一条QKV/输入路线，不是当前选定方案 |
| `m1_early_gate_value_interaction_recommendation_20261026.md` | receiver-centered提案已有后续测试与更正 |
| `m1_early_gate_value_crosslayer_literature_scout_20261026.md` | ECLVR为未训练设计，不能继续作为默认执行指令 |
| `m1_msha_2026_module_literature_scout_20261026.md` | 文献候选分支，不是已批准方法 |
| `m1_early_gate_sequence_scene_attribution_20261026.md` | FSCIR/FCCR/FCSV优先级多次变化，且gate-only proxy与full15 joint须区分 |
| `m1_dcvr_v1_v2_value_source_reallocation_20261026.md` | 历史FCSV-R行动方案，不代表当前待跑实验 |

另外18份文献/设计记录标为参考材料，15份清理记录保留维护历史，295份保留历史证据，4份作为当前入口。详见[逐报告清单](report_cleanup_inventory.csv)。所有项都显式记录`physical_delete=NO`。

不要删除：原始训练/测试记录、失败执行记录、配对不成立的更正、清理审计、资产主表和CSV。它们是识别过去错误的证据。若以后合并叙述，只合并文字入口，保留旧路径或提供可恢复映射。

## 5. Git与恢复

```text
master @ 09f3d57                    原分支保留
  └─ archive/research-state-20261026 @ c0f1aff
       └─ cleanup/m1-early-gate-20261026  本轮整理结果
```

参考源码从索引移出清单：[git_untracked_reference_files.csv](git_untracked_reference_files.csv)。源文件仍在`experiments/external_repos/`，原项目独立克隆`TBSI_original/`和所有历史snapshot也保留在磁盘；不是删除255个磁盘文件。

查看某一清理前代码片段：

```bash
git show c0f1aff:TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py
```

整个研究文本/源码还有本地恢复包`uncommitted_research_before.tar.gz`，其清单见`files_before.json`。数据和权重从未打进该包。不要用`git clean -fdx`清理本仓库，它会涉及忽略的实验产物。

既有`fix/all-bugfixes`、`融合内部调整`和`/tmp/tbsi_m1_09f3d57`工作树没有证据证明可废弃，本轮保留；`.git`原本约86MB，不是主要磁盘瓶颈，未通过清理历史来释放实验空间。

## 6. 检查边界

本轮仅做静态清单、AST语法解析、文件hash与Git差异核对；没有模型测试、训练、推理或性能复现。M1 YAML hash仍须等于`fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`。

机器可读文件：`code_cleanup_inventory.csv`、`report_cleanup_inventory.csv`、`config_cleanup_inventory.csv`、`inventory_summary.json`、`files_before.json`。维护资产ID：`PROJECT-CONTEXT-GIT-CLEANUP-20261026`，同步登记至中文主表与CSV。
