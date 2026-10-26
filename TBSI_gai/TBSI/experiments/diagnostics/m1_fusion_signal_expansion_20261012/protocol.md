# M1 帧内融合信号扩展筛查（2026-10-12）

## 目的

扩展原 `M1-FUSION-SIGNAL-ADMISSION-DIAG-20261012` 的同一信号筛查 cohort，在不训练模块的前提下检查特征能量、模板支持和 CENTER 熵信号在更多序列上的方向是否保持。

## 唯一改动

扩展采集 8 条满足 150-800 帧的 LasHeR trainingset 序列。采集后对照原始排除源审计发现，其中 3 条也出现在前序 MCCA attribution 选择集中；因此将 5 条符合排除规则的序列定义为主扩展集，另外 3 条仅作为重叠敏感性分析，不用于主结论。分别与旧 12 序列合并，重新做 leave-one-sequence-out probe 及同一信号分析。

## 固定条件

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15。
- 配置 SHA256：`fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`。
- Checkpoint SHA256：`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- 数据：LasHeR trainingset，`150-800` 帧；新增序列清单见 `sequence_list.txt`。
- 主扩展新序列：`2runfour`、`Blue_in_line_1202`、`The_girl_with_the_cup_1202`、`bikeout`、`carfarstart`，清单见 `primary_sequence_list.txt`。
- 重叠敏感性序列：`blkbikefromnorth`、`boyleave`、`man2startmoto`，三者已进入前序 MCCA attribution 选择集，清单见 `sensitivity_overlap_sequence_list.txt`。
- 与原 12 序列不重复；它们仍与旧 MCCA validation96 序列集合重叠，故扩展结果仍为探索性，非独立验证。
- 固定状态规则：`onset-3-iou-0.5`，`probe-reference-frames=2`；不施加模块干预。
- GT 仅用于离线固定状态选择、probe 拟合和评分，不进入运行时信号。

## 判读约束

序列是统计单位；不得把同一序列多个帧当作独立样本。保留原 12 序列的 `analysis_v3`，扩展后在新实验目录单独保存合并分析。AUC gap `<0.02` 视作弱/并列效用，不纳入方向准确率主统计。无论结果如何，都不能从该 train-side 小样本宣称 full-244 提升或一般化可靠性。
