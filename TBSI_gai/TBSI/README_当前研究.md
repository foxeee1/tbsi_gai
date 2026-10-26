# 当前研究入口

这是后续工作的默认入口。历史报告保留为证据，不共同充当当前计划。

## 当前范围

- 受保护研究参照：M1 `TBSI_E_MHSA_FULL_4090D_seed42`。4090D 等效复现组具有独立身份，按根规则及资产表分别引用。
- 方法研究 parent：Early-conditioned MSHA gate full15。保留其配置、checkpoint、原始日志与结果。
- 用户目标：沿 MSHA 模板交互/Value 内容建模完善论文；本轮任务是清理上下文与版本管理。
- 下一算子、训练合同和训练任务：**尚未选定，未启动**。DCVR-v2、ECLVR、STSER 和上一份报告的 A/B/C 都是历史设计或待选方案。

## 默认只读这些入口

1. 根目录 `AGENTS.md` 与本目录 `AGENTS.md`：实验边界和工作规则。
2. `experiments/实验资产数据库.md` 与同步 CSV：查询所涉及的实验身份。
3. `experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml`：M1 配置。
4. `experiments/tbsi_track/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023.yaml`：Early gate 配置；实际运行身份同时核对 retry 记录。
5. 涉及源码时直接阅读 `lib/models/layers/tbsi_layer.py`、`lib/models/tbsi_track/vit_tbsi_care.py` 和训练/推理调用入口。
6. 涉及最近重复性与短训时再读 `m1_early_gate_control_repeatability_20261026.md`、`m1_early_gate_value_recon_quick3_20261026.md`；具体更正见 `m1_early_gate_bottleneck_audit_20261026.md`。这些文件位于 `experiments/reports/`。

不要默认读取全部339份 Markdown 报告或全部335个工具脚本。所需历史证据通过资产ID定向找回。

## 目前可以说什么

| 事项 | 证据边界 |
|---|---|
| M1固定参考AUC56.21 | 受保护登记值；现存结果来源问题不能靠覆盖该值修复 |
| Early gate历史AUC56.56 | 独立运行点估计；匹配对照与旧逐序列provenance尚未闭合，不能直接写成已归因的稳定提升 |
| quick3短训 | 确实运行过；历史512×3、batch16/累积8只有12个更新边界，七序列不能代表完整性能 |
| 当前MSHA信息流 | 默认TBSI-E先更新模板，后续ViT再影响搜索；需要读实际feature flags确认是否存在额外支路 |
| 初始化及旧重构问题 | 当前源码存在统一初始化覆盖特殊输出层、旧重构作用于self/cross等事实；其解释历史掉点的比例尚未测定 |
| 多次模块失败 | 不能统一归因于MSHA、head、算子复杂度或训练量中的任何单一因素 |

## 清理与恢复

清理前本地Git提交：`c0f1aff98792d604110a1e006a7d2cef4394734d`，分支 `archive/research-state-20261026`。清理工作位于 `cleanup/m1-early-gate-20261026`。

代码片段、报告、配置和工具的逐项清单见 [`maintenance/cleanup_20261026/README.md`](../../maintenance/cleanup_20261026/README.md)。大型模块仍在原路径，待按依赖闭包处理；未依据失败指标直接删除。外部源码与历史快照在磁盘原位保留，从当前Git索引移出，清理记录注明恢复来源。
