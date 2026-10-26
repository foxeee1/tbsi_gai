# M1 实验证据索引

更新时间：2026-10-12

这份索引是 M1 研究证据的单一入口。指标、配置哈希、checkpoint 哈希和实验状态以[实验资产数据库](experiment_asset_registry.csv)为准；本页负责说明每份证据回答什么问题、能支持到哪一级结论。

按日期梳理代码改动、验证结论、排除项和当前瓶颈，请先读[实验资产数据库的 M1 研究时间线](实验资产数据库.md#m1-研究时间线改动结论与排除项)。

## 1. 研究锚点

| 对象 | 文件/目录 | 用途 |
|---|---|---|
| 受保护 M1 | [配置](tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml) · [checkpoint](../output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar) | 所有 M1 候选的 parent；seed42、epoch15、full-244 参考指标 AUC 56.21、PR 70.21、NPR 66.20 |
| M1 参考报告 | [tbsi_e_mhsa_full_20260927.md](reports/tbsi_e_mhsa_full_20260927.md) | 训练、环境、固定权重和官方指标 |
| 资产主表 | [实验资产数据库.md](实验资产数据库.md) | 人工核对用权威记录 |
| 机器索引 | [experiment_asset_registry.csv](experiment_asset_registry.csv) | 脚本解析用 23 列登记表 |
| 清理记录 | [m1_stopped_local_candidates_checkpoint_cleanup_20261011.md](reports/m1_stopped_local_candidates_checkpoint_cleanup_20261011.md) | 已删除废弃 checkpoint，报告、配置、日志和结果仍保留 |

## 2. 最重要的科学结论

当前证据支持的 M1 失败链条是：

```text
TBSI/MHSA 后的搜索表征可读性下降
    -> CENTER 候选排序与 Hann 在线选峰不一致
    -> 选中 cell 后 offset/size 仍有误差
    -> 闭环状态更新放大少量中心偏差
    -> 目标出界与恢复不足
```

目前不能把首因归结为单一融合算子、局部建模缺失或 MCCA 实现错误。

最直接的证据是：

1. [搜索接口前后证据](reports/m1_search_stage_interface_evidence_20261010.md)：不支持“目标信息只是在融合之后才丢失”。
2. [候选召回与 CENTER 读出](reports/m1_candidate_recall_readout_20261010.md)：候选排序和 offset/size 读出同时受限。
3. [全量融合优先级](reports/m1_full244_fusion_mechanism_priority_20261010.md)：把融合、读出和闭环状态问题分开。
4. [MCCA 失败归因](reports/m1_mcca_failure_attribution_20261011.md)：计算图和梯度正常，但消息修正与最终定位行为没有稳定对应关系。

## 3. 推荐阅读顺序

### A. 先理解 M1 的真实问题

1. [全量测试失败地图](reports/m1_full_test_failure_map_20261005.md)
2. [全量场景挖掘](reports/m1_full244_scene_mining_20261009.md)
3. [MHSA 全量瓶颈分析](reports/m1_mhsa_full244_bottleneck_analysis_20261009.md)
4. [融合信息流诊断](reports/m1_fusion_flow_diagnostic_20261009.md)
5. [全量序列审计](reports/m1_full244_sequence_audit_20261009.md)
6. [全量融合机制优先级](reports/m1_full244_fusion_mechanism_priority_20261010.md)

### B. 再看决定方向的定位诊断

1. [搜索接口前后证据](reports/m1_search_stage_interface_evidence_20261010.md)
2. [D1/D2 候选召回与 CENTER 回归读出](reports/m1_candidate_recall_readout_20261010.md)
3. [新 train-side CENTER 诊断](reports/m1_center_readout_fresh_diagnostic_20261010.md)
4. [前三候选方向诊断](reports/m1_three_direction_diagnostic_20261010.md)
5. [96 序列双方向诊断](reports/m1_two_direction_broad_cohort_diagnostic_20261010.md)
6. [消息注入位置杠杆](reports/m1_message_position_leverage_20261012.md)

## 4. M1 信息流与场景诊断

| 证据 | 数据范围 | 结论级别 |
|---|---|---|
| [m1_fusion_flow_diagnostic](reports/m1_fusion_flow_diagnostic_20261009.md) | train-side trace，冻结权重 | 提出模板 MHSA、搜索融合、定位输出之间的机制假设；不提供性能结论 |
| [m1_fusion_flow_testset](reports/m1_fusion_flow_testset_20261009.md) | 30 序列固定权重诊断 | 描述性跨序列证据，不用于选择模块 |
| [m1_fusion_scene_attribute_diagnostic](reports/m1_fusion_scene_attribute_diagnostic_20261010.md) | 场景/属性分层 | 严重场景同时有中心选择和重叠失败；不证明局部模块是首因 |
| [m1_full244_sequence_audit](reports/m1_full244_sequence_audit_20261009.md) | full-244 归档预测 | 确认低分序列、失败事件和后续诊断边界 |
| [m1_full244_scene_mining](reports/m1_full244_scene_mining_20261009.md) | full-244 归档预测与属性 | 只读场景挖掘，不改模型、不挑测试子集 |
| [m1_temporal_trace_v1](reports/m1_temporal_trace_v1_20261006.md) | 时序 trace/replay 合同 | 用于确认闭环状态分析的数据合同 |

## 5. CENTER、候选召回和读出证据

| 证据 | 核心发现 | 可支持的方向 |
|---|---|---|
| [候选召回与读出](reports/m1_candidate_recall_readout_20261010.md) | 严重点上 GT cell 常在较大 top-K 内，但很少排到 top-1；GT-cell oracle 下 offset/size 仍有误差 | 候选质量、联合读出、状态恢复 |
| [CENTER fresh diagnostic](reports/m1_center_readout_fresh_diagnostic_20261010.md) | 新 train-side cohort 事件数量低于预设稳定性门槛，结论限于探索性机制证据 | 不足以单独支持新模块 |
| [Strip GT-cell 反事实](reports/m1_strip_gt_cell_counterfactual_20261010.md) | 选对 cell 可改善部分框，但目标证据与跨序列稳定性门槛不足 | 停止 score-only Strip ranking 路线 |
| [Strip 训练/推理合同审计](reports/m1_strip_score_training_inference_contract_audit_20261010.md) | Strip 分支接通并有训练信号，但没有证明它改变正确峰选择 | 排除“完全未接通”，保留接口不匹配解释 |
| [身份候选重排](reports/m1_mhsa_candidate_rerank_20261010.md) | MHSA 身份信号可观测，但 raw cosine/MLP 不能稳定改善候选排序 | 停止身份重排 |
| [身份机制验证](reports/m1_mhsa_template_identity_mechanism_20261010.md) | 模板身份线索存在，但不是可靠的在线位置决策分数 | MHSA 作为基础信息保留，不直接控制选峰 |

## 6. M1 候选模块实验

### 6.1 LCGA 与局部空间建模

| 文件 | 状态 | 结论 |
|---|---|---|
| [LCGA-style 忠实度与归因](reports/m1_lcga_attribution_fidelity_20261010.md) | 停止当前 style 版本 | 当前 style 缺少论文曲率路径，不能代表论文 LCGA；即时贡献不稳定 |
| [官方 LCGA 准入](reports/m1_lcga_faithful_gate_20261010.md) | 速度门止损 | 算子等价和数值门通过，但双路 search refiner 延迟代价过高 |
| [官方 LCGA 短训](reports/m1_lcga_faithful_short_20261010.md) | 完成，未晋级 | 3 epoch proxy 未形成跨序列定位改善，不进入 full training |
| [冻结 M1 LCGA](reports/m1_lcga_frozen_only_20261010.md) | 完成，未晋级 | 排除了“必须联合更新 head 才能看到收益”的简单解释 |
| [LCGA search refiner](reports/m1_lcga_search_refiner_20261010.md) | 预实验 | 记录融合前双路 search 插入及合同检查 |
| [LCGA 任务适配诊断](reports/m1_lcga_task_adaptation_diagnostic_20261010.md) | 只读诊断 | 当前主要风险是加入局部修正后扰动 M1 稳定 search 表征 |
| [LCGA/Strip 接口诊断](reports/m1_lcga_strip_interface_diagnostic_20261010.md) | 只读诊断 | 失败序列提示定位接口和闭环问题，不能直接归因局部空间不足 |
| [局部模块失败归因](reports/m1_local_module_failure_attribution_20261010.md) | 综合诊断 | “缺少局部建模”仍未被证明是 M1 首因 |

### 6.2 MCCA 模板跨模态消息适配

| 实验 | 范围/结果 | 状态 |
|---|---|---|
| [匹配 G0/G1](reports/m1_mcca_g0_g1_20261010.md) | 60 序列 proxy；早期 G1 AUC `+1.3995`，39/60 胜；不确定性区间跨零 | 仅 screening 正信号 |
| [fresh96 归因](reports/m1_mcca_g0_g1_20261010.md) | G1 相对 G0 AUC `-0.8453`，41/55；未复现 | 独立验证失败 |
| [full-244 配对](reports/m1_mcca_full244_paired_20261011.md) | G1-G0 AUC `+0.4488`，但 paired CI 跨零；G1-M1 不隔离续训效应 | 全量点估计正向但不确定 |
| [冻结 MCCA adapter](reports/m1_mcca_frozen_adapter_validation_20261011.md) | 相对冻结 M1 AUC `-1.0300` | 标准 tracking loss 不足以安全训练 adapter |
| [冻结 MCCA 归因](reports/m1_mcca_frozen_adapter_attribution_20261011.md) | 梯度、更新、零输出 parity 正常；目标-困难峰 margin 可被破坏 | 目标/接口失配得到支持 |
| [MCCA trace](reports/m1_mcca_frozen_adapter_attribution_20261011.md) | 发现少量响应变化可被闭环放大 | 闭环放大风险 |
| [排序监督](reports/m1_mcca_frozen_rank_20261012.md) | AUC `-0.6664`；rank loss 下降但闭环未改善 | 排序目标不足 |
| [可靠性门控](reports/m1_mcca_reliability_operator_20261012.md) | AUC `-1.3291`；零门控、梯度和参数更新通过 | 当前 gate 算子不支持 |
| [post-search MCCA](reports/m1_post_search_mcca_frozen_20261012.md) | 融合后、CENTER 前注入；AUC `-0.6812` | 注入位置假设不支持 |
| [MCCA 失败总归因](reports/m1_mcca_failure_attribution_20261011.md) | 汇总实现、优化、目标、位置和泛化解释 | 当前 MCCA 方向停止扩展 |
| [MCCA 两 seed确认](reports/m1_mcca_two_seed_confirmation_20261011.md) | 用户中断，未形成完整两 seed 结果 | 不能引用为多 seed 证据 |

MCCA 结果文件集中在：

- `output/experiments/M1_MCCA_G0_G1_20261010/`
- `output/experiments/M1_MCCA_G1_ATTRIBUTION_FRESH96_20261011_V2/`
- `output/experiments/M1_MCCA_FULL244_PAIRED_20261011/`
- `output/experiments/M1_MCCA_FROZEN_ADAPTER_3EP_20261011/`
- `output/experiments/M1_MCCA_FROZEN_RANK_3EP_20261011/`
- `output/experiments/M1_MCCA_FROZEN_RELIABILITY_3EP_20261012/`
- `output/experiments/M1_POST_SEARCH_MCCA_FROZEN_3EP_20261012/`

### 6.3 其他已做过的 M1 候选

这些实验是历史候选证据，适合放在相关工作或附录，不应与当前 MCCA/定位主线混写：

- [CFVI 代码属性分析](reports/cfvi_m0_m1_m2_code_attribute_analysis_20260930.md)
- [CFVI full matched M1](reports/cfvi_p1_full_matched_m1_20261001.md)
- [FMR full matched M1](reports/fmr_full_matched_m1_20261003.md)
- [HF-Relation loc9](reports/hf_relation_loc9_full_matched_m1_20261003.md)
- [TAFH full matched M1](reports/tafh_full_matched_m1_analysis_20261002.md)

它们的完整指标、配置、checkpoint 和状态必须回到 CSV 资产表核对，不从文件名推断“成功”或“失败”。

## 7. 证据等级和使用边界

| 标签 | 含义 | 论文使用方式 |
|---|---|---|
| `PROTECTED-BASELINE` | M1 固定 parent 和 full-244 参考指标 | 主基线 |
| `FULL244-DESCRIPTIVE` | 已查看的全量测试回放/配对结果 | 透明报告；不能作为未触碰测试集 |
| `TRAIN-SIDE-DIAGNOSTIC` | 预冻结 train-side 状态或事件诊断 | 机制证据，可用于提出/排除假设 |
| `PROXY-SCREENING` | 30/60/96 序列或短训结果 | 只能做筛选和归因，不能称稳定提升 |
| `CONTRACT-AUDIT` | shape、parity、梯度、状态、加载和推理合同 | 支持“是否接通”，不支持性能有效 |
| `STOPPED` | 明确停止的候选 | 作为失败边界和负结果证据 |
| `CANCELLED/FAILED-PREFLIGHT` | 未完成训练或在推理前失败 | 只能说明流程/数据问题，不能说明方法失败 |

## 8. 当前可以写进论文的结论

可以写：

- M1 的严重失败同时包含候选排序和 CENTER 回归读出问题。
- 融合前后接口诊断没有支持“信息只在 `tbsi_fuse_search` 后被破坏”。
- 多个模块计算图、梯度和参数更新正常，但闭环收益不稳定。
- MCCA 的标准监督、排序监督、可靠性门控和位置移动均未形成稳定改善。
- 现有证据更支持“表示修正与在线选峰/闭环状态之间的目标接口不一致”。

暂时不要写：

- “MCCA 原理无效”；
- “局部空间建模无效”；
- “MHSA 是 M1 失败的唯一原因”；
- “full-244 证明新模块有效”；
- “两 seed 或多 seed 已验证”。

## 9. 资产查找规则

1. 先查本页确定实验属于哪一类。
2. 再查 [CSV 资产表](experiment_asset_registry.csv)确认 ID、范围、指标、checkpoint 和状态。
3. 再打开对应报告了解假设、失败归因和下一步限制。
4. 最后才进入 `output/experiments/<ID>/` 查看原始 JSON、CSV、trace、日志和预测。

不要用目录名、文件名或聊天中的旧摘要替代资产表中的最终状态。
