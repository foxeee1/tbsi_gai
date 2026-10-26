# M1 S1 full-244 确认结果（2026-10-14）

## 运行合同

本次没有重新训练网络。S1 是在 train-side 固定状态上拟合的 bounded offset predictor，使用固定 `rho=0.5`、utility threshold `0.08` 和独立 offset/size policy；full-244 只验证该冻结策略能否从 proxy 推广。M1 与 S1 使用同一受保护 epoch15 checkpoint、同一 tracker 代码、同一官方 LasHeR testingset、FP32 和串行 tracker。M1 run `1030`、S1 run `1031` 各产生 244/244 个预测文件。

配置、checkpoint、代码、predictor 和 policy 的哈希均写在 `output/experiments/M1_S1_FULL244_CONFIRM_20261014/manifest.json`。M1 保护配置和 checkpoint 哈希仍为登记值，未被覆盖。

## 全量官方指标

| 指标 | M1 current parent | S1 frozen readout | S1 - M1 |
|---|---:|---:|---:|
| AUC / Success | 56.2077 | 56.4599 | **+0.2522** |
| Precision | 70.2073 | 70.5443 | **+0.3370** |
| Norm Precision | 66.1980 | 66.5330 | **+0.3351** |
| OP50 | 68.3339 | 68.7700 | **+0.4361** |
| OP75 | 47.1349 | 47.1488 | **+0.0139** |

逐序列 AUC 胜负平为 117/124/3；PR 为 89/99/56；NPR 为 87/107/50；OP50 为 94/95/55；OP75 为 84/117/43。总指标的方向是小幅正向，但 AUC、PR、NPR、OP50 均没有达到预设的约 1% 全量提升目标，OP75 基本不变。

Persistence audit 中，S1 的序列等权 IoU<0.5 比例为 0.3122，M1 为 0.3166；帧加权比例为 0.3695 对 0.3737。S1 出现持续失败的序列数为 135，M1 为 136；持续失败段总数为 434 对 430。最长失败段由 1494 帧降到 1277 帧，但总持续失败段数量略增，不能把结果概括为全面恢复。

## 分层序列结果

| Cohort | 序列数 | AUC Δ | PR Δ | NPR Δ | OP50 Δ | OP75 Δ | AUC 胜/负/平 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 已筛选 low-AUC62 | 62 | +3.7274 | +5.0960 | +4.7513 | +5.0720 | +3.0750 | 35/24/3 |
| normal20 sentinel | 20 | +0.0030 | +0.0205 | +0.0125 | +0.0059 | -0.0426 | 13/7/0 |
| 未参与筛选 remaining162 | 162 | **-1.0470** | **-1.4453** | **-1.3153** | **-1.2850** | **-1.1507** | 69/93/0 |
| 全部244 | 244 | +0.2522 | +0.3370 | +0.3351 | +0.4361 | +0.0139 | 117/124/3 |

因此 full-244 的正收益不是普遍融合修正能力，而是 low-AUC62 的恢复收益被 remaining162 的退化抵消后的结果。normal20 只能说明预选的正常 sentinel 没有明显变化，不能代表全部正常序列。

## 主要增益与退化序列

按 AUC 差，最大增益为：`ab_bikeoccluded` +57.63、`3bike1` +56.53、`leftfarboycomingpicktheball` +49.63、`right2ndflagformath` +27.79、`runningcameragirl` +23.11、`whiteboyrightcoccergoal` +19.07。最大退化为：`umbrellawillopen` -57.90、`bluegirlbiketurn` -32.63、`whiteridingbike` -26.73、`shinycarcoming` -15.20、`ab_whiteboywithbluebag` -14.33、`redmidboy` -12.42、`pinkwithblktopcup` -11.29、`rightbottlecomes` -9.65。

其中 `umbrellawillopen`、`bluegirlbiketurn` 和 `whiteridingbike` 是 full-244 中新的高风险退化序列；它们不在原 low-AUC62/normal20 proxy manifest 中，直接解释了 remaining162 的负均值。该结果支持“策略对场景分布敏感”的归因，不能支持全局有效的融合修正器。

S1 action audit 共 219,048 帧：offset 87,212（39.82%），no-op 131,836（60.18%）；size 和 joint 均未触发。也就是说，本次 full-244 实际只验证了 offset branch，不能据此评价 size correction 或联合动作。

## 与前面实验组对照

| 实验组 | 范围 | AUC 相对直接 parent | 其他主要结果 | 结论 |
|---|---|---:|---|---|
| Bounded predictor proxy | union82 | +2.23 | low-AUC +3.00；normal -0.14 | proxy promising，未验证 full |
| S1 proxy / independent readout | union82 | +2.83 | low-AUC +3.73；normal +0.03 | proxy 强正，但存在筛选偏差 |
| S1 full-244 本次 | full-244 | **+0.2522** | remaining162 AUC -1.047 | 未达到约1%，不晋级为全量方法 |
| Cdiag feature residual full | full-244 | -0.0274 | PR -0.0075；NPR +0.1000；OP50 +0.0899 | 已排除当前 feature-side residual 的整体涨点 |
| ABC Cdiag/Cspatial proxy | union82 | Cdiag +2.82；Cspatial +1.78 | normal 分别 +0.008、-0.118 | 仅支持 B 优先进入确认，不能替代 full 结果 |

S1 full-244 比 Cdiag full 的 AUC 高 `+0.2796` 个百分点，但距离预期的 1% 仍差约 `0.75` 个百分点；两者都没有证明当前方向能稳定带来目标幅度。S1 的主要优势是 readout-side 修正比 feature residual 更接近有效机制，主要弱点是未筛选场景退化。

## 归因与决定

- **已支持**：offset readout correction 在 low-AUC62 上有可重复的恢复信号，且在 full-244 上带来小幅净增。
- **已排除**：proxy 上的 +2%～+3% 不能作为全量涨点预期；entropy/reliability gate 没有解决 normal 风险；feature-side Cdiag residual 不是当前的有效主路线。
- **尚未支持**：S1 不是可直接晋级的全量方法。remaining162 的系统性负收益说明 admission 信号存在场景分布外推问题。

下一步只建议做一次单变量风险修正：固定 offset predictor 和修正幅度，针对 full-244 中的三类信息构造 deployment-visible admission gate（base peak margin、RGB/TIR disagreement、历史框一致性），只学习 `noop/offset` 放行，不重新训练 feature fusion、不加入 size/joint。训练和阈值必须只用 train-side，并在同一 remaining162 负收益清单上先做闭环 proxy；通过后再复测 full-244。若 gate 不能在保持 low-AUC 恢复的同时消除 remaining162 退化，应停止当前 S1 扩展。

## 证据

- Full-244 manifest、日志、预测覆盖和官方结果：`output/experiments/M1_S1_FULL244_CONFIRM_20261014/`
- 官方 eval artifact：`output/test/result_plots/M1_S1_FULL244_CONFIRM_20261014/eval_data.pkl`
- 逐序列官方对照：`output/experiments/M1_S1_FULL244_CONFIRM_20261014/per_sequence_official.csv`
- Persistence audit：`output/experiments/M1_S1_FULL244_CONFIRM_20261014/persistence_audit/official_metric_and_persistence.json`
