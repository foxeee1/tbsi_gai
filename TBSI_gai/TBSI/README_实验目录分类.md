# 实验目录分类与命名约定

## 当前分类

历史实验目录保留原路径，避免破坏资产登记、报告引用和复现证据。按用途分类如下：

| 中文类别 | 目录范围 | 说明 |
|---|---|---|
| `01_锚点与复现` | `TBSI_E_MHSA_FULL_4090D_seed42`、`v1.0.1-*`、`v1.0.3-*` | 受保护 M1 与 4090D 等效复现组，禁止覆盖或改名 |
| `02_正式候选训练` | `M1_FUSION_*`、`M1_ADAPTIVE_*` | 当前独立候选训练和配对评测 |
| `03_机制诊断` | `M1_*DIAG*`、`M1_*TRACE*`、`M1_*READOUT*`、`M1_*SIGNAL*` | 固定状态、闭环 trace、信号准入和读出诊断 |
| `04_历史路线` | `M1_MCCA_*`、`M1_LCGA_*`、`DBF_*`、`tam_*` | 已停止路线的历史证据；可清理大型中间产物，但先保留报告和关键摘要 |
| `05_历史筛选` | `BTP_*`、`B3_*`、`B4_*`、`template_expert_*`、`S1_*`、`rpp-v2-*` | 旧代理筛选与历史对照 |
| `06_频域停用` | `CFVI_*`、`FMR_*`、`HF_RELATION_*`、`r2_*` | 按研究规则暂时停用，仅保留可审计历史证据 |
| `07_旧诊断与合成数据` | `single_modality_*`、`synthetic_*`、`innovation1_*`、`temperature_sweep`、`mhsa_*` | 早期机制探针和合成数据分析 |
| `08_缓存与回收站` | `.trash_*`、`.locks`、`tensorboard` | 临时缓存或已失效产物，确认无引用后可删除 |

## 新目录命名

新实验目录统一使用中文类别前缀和日期，例如：

`01_锚点与复现/实验_M1_融合诊断_20261013/`

`03_机制诊断/诊断_中心读出_20261013/`

`04_历史路线/归档_MCCA_G1_20261010/`

目录名必须同时包含用途、对象和日期；配置名、资产 ID、报告中的历史路径保持一致。受保护锚点和已有登记目录不批量重命名。

## 本次清理后的高置信候选

以下项目尚未删除，原因是报告或 manifest 仍直接引用它们：

- `M1_SEARCH_STAGE_TRAIN_DIAG_20261009/combined_final/selected_features.npz` 与 `analysis_final/selected_features.npz` 内容完全相同，各约 168M；保留一个即可，但删除前需把报告引用统一到规范路径。
- `M1_SEARCH_STAGE_TRAIN_DIAG_20261009/combined/selected_features.npz` 与 `combined_v3_48/selected_features.npz` 内容完全相同，各约 107M；同样需要先统一引用。
- `M1_FUSION_FLOW_FULL244_DIAG_20261009` 与 `M1_FUSION_FLOW_TESTSET_DIAG_20261009` 的部分 `frames.jsonl` 内容完全相同；应按报告用途决定保留 full244 还是 testset 副本。
- `M1_TWO_DIRECTION_DIAGNOSTIC_20261010/stage_capture/selected_features.npz` 约 462M，`M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz` 约 269M，属于大体积诊断中间量，需确认报告已抽取汇总后再删。

已清空 `output/.trash_*` 回收站目录，约释放 1M；没有删除上述仍有引用的诊断证据。
