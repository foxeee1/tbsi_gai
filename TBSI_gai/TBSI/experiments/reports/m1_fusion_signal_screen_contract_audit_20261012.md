# M1 帧内融合信号筛选合同审计（2026-10-12）

## 结论

**现有筛选不足以判定哪些信号能够指导 CENTER 的帧内融合。** 它可以记录局部能量、raw 模板 cosine 和特征冲突与既定代理量的关系；不能据此排除目标条件响应、照明退化或学习式效用信号，也不能归因于最终 search 融合前的位置。旧结果保留，但“没有信号通过准入”仅指这些手工信号在旧代理合同下未获准进入门控训练。

## 审计对象与发现

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15；本审计使用原 72 条 outcome-blind LasHeR train-side 序列的 144 个固定状态，其中 37 个 severe onset。冻结清单为 `experiments/diagnostics/m1_fusion_utility_signal_pilot_20261012/expansion72/manifest.json`，SHA256 `37eee640a1012e8bb4969cbff4c18952a72eca432a8780edab02764220640bab`。与已审计旧 validation/screening 队列无序列重叠；M1 训练暴露未知。
- **Probe 留出错误：**旧 `analyze_m1_fusion_signal_admission_20261012.py` 的 `fit_probe` 仅跳过当前状态索引，却把输出称为 `leave-one-sequence-out`。144 个状态中有 109 个在拟合数据中仍有同序列帧，37/37 个 severe 状态全部如此。旧 probe AUC 数值不能作为按序列留出证据。已先保存旧脚本快照，再修正为排除整条序列、按序列复用同一模型，输出到独立目录；原结果未覆盖。
- **控制区位置不匹配：**旧效用脚本取 `background` 掩码的前 `n_target` 个格子作等面积控制。144/144 个控制区虽等面积，但质心中位为 `(1.5, 0)`，离目标区质心中位约 `9.60` 个 16×16 网格 token（四分位 `9.51–9.74`）。Hann 加权和空间上下文都不同，故目标衰减与控制衰减的差值混入位置效应。
- **效用标签与任务不齐：**旧标签是冻结 CENTER 的目标最高分减背景最高分之变化。144 个状态中仅 43 个出现 RGB/TIR 净 GT 排名差异；37 个 severe 中仅 16 个。排名差异非零时，margin 效用差与排名方向同号为 `30/43`，severe 为 `11/16`。旧输出没有反事实预测框 IoU、offset/size 误差或困难负峰位置，所以不能把 margin 方向称为实际定位效用方向。它也只衰减 GT 目标区，未衡量抑制错误模态困难负峰可能带来的收益。
- **基线与样本选择：**旧“恒选多数”是在被评估的同一批效用标签上事后确定的组内多数方向，只是描述性难度参照，不是可部署的预注册基线。Severe onset 按 M1 失败轨迹选出，适合机制诊断，但不能代表随机帧分布。37 个 severe 仍不足以训练并确认复杂信号头的泛化。

Probe 留出修复只改变辅助 AUC 分析；先前固定 crop 衰减实验的 144 行原始数值没有重算或覆盖。那组结果仍可描述旧 margin 代理，但控制区与任务标签问题使其不能充当实际融合效用的准入判决。

## 修正后的 probe 复核

将同一 144 状态按序列排除后，probe AUC 差至少 `0.02` 的 severe 状态由旧口径 `11/37` 变为 `19/37`。在这 19 个状态上，raw 模板峰显著性差命中 `9/19`，评估集多数方向为 `10/19`；局部能量差命中 `5/19`。这只检验分支目标/背景可分性，仍不是非线性融合器与 CENTER 的反事实框定位。旧第一次修正输出带有硬编码的错误 cohort 限制文字，保留为 superseded；权威修正输出为 `output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/signals_analysis_sequence_holdout_v2/summary.json`，SHA256 `ee24cb1e27c6697edb9bf25001e1e37b3bbde5e24a5520f3d2cd29e82ce0ef74`。

代码证据：Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；修改前分析脚本快照 `experiments/snapshots/m1_fusion_signal_admission_pre_sequence_holdout_fix_20261012.tar.gz`，SHA256 `f5617c0015e90ed706f7c5a72ffbe0d7a2cafd317931fc086351511cea022432`；修正后脚本 SHA256 `60d838c4ab6093259809f37f43aa31886bb224ecb3cdf9127fb08149270e335c`。CPU 单进程重算；没有训练 M1、启动闭环测试或写入 checkpoint，GPU 时长 `0`。本审计的 AUC/PR/NPR/OP50/OP75/FPS 均为 `NA`。

## 下一版筛选合同

1. **先固定数据划分与标签。** 按序列在任何信号拟合、阈值选择和模块设计前划分开发与最终留出集。当前 72 序列已参与信号审计，只能继续用于开发；最终确认需要新的 outcome-blind 序列。比较必须沿用同一 crop、同一 M1 权重、精度与在线解码口径。
2. **让离线效用对应实际控制动作。** 在最终 search 融合输入处，对每路做与拟议空间路由幅度匹配的受控调节；目标区与困难负峰区分别干预，控制区匹配空间位置、面积和特征分布。记录在线 Hann 选格、目标对最强困难负峰的 margin、offset/size、实际预测框 IoU，以及正常帧损伤。只有标签方向对干预强度和控制位置稳定，才可用于信号训练。
3. **比较任务信号与简单对照。** 将照明/局部退化、能量差作为控制；另建只用当前可得模板和双模态 search 特征的轻量 provisional response，学习目标响应、困难负峰竞争与双路分歧。GT 只监督 train-side 响应头和离线评分，不能进入运行时信号。先确认响应图自身按序列留出的目标/困难负峰排序，再检验它能否预测上一步的反事实定位收益。
4. **帧内问题单独判定。** 上一帧一致性和下一帧 IoU 会混入状态反馈，另立时序实验；融合后的 CENTER entropy 只能用于后续状态或二次推理决策。当前不训练空间 gate/MCCA，也不把任何手工或学习型信号称为已验证。

用户提供的 IGF、目标 query、响应图及贡献估计等方法用于提出候选信号族；它们在各自任务中的收益不能替代本 M1 接口与数据口径下的准入证据。

修正分析命令及完整逐状态输出：

```bash
python3.8 tools/analyze_m1_fusion_signal_admission_20261012.py \
  --features output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz \
  --center-csv output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/per_arm.csv \
  --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/signals_analysis_sequence_holdout_v2 \
  --diagnostic-id M1-FUSION-SIGNAL-SCREEN-CONTRACT-AUDIT-20261012 \
  --cohort-provenance '72 outcome-blind LasHeR train-side sequences; zero overlap with audited MCCA validation96 and prior signal/admission cohorts per expansion72 manifest; M1 training exposure unknown; fixed states are exploratory.'
```

逐状态结果：`output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/signals_analysis_sequence_holdout_v2/per_state_signals.csv`。受保护 M1 config/checkpoint SHA256 仍为 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1` / `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
