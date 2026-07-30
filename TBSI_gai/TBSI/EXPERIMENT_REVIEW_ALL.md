# TBSI 全系列实验复盘与研究路线裁决

更新时间：2026-07-30

## 1. 执行摘要

本项目围绕 TBSI 先后探索了时序令牌、退化感知融合、差分投影、信号解耦、可靠性门控、模板条件桥接、频域冲突校准和反事实效用路由等多个系列。现有证据支持以下结论：

1. 原始全量基线仍是当前最优可学习模型：LasHeR test AUC `55.46`。
2. 所有已完成且可直接比较的全量可学习增强均低于该基线，下降范围为 `-0.41` 至 `-1.88 AUC`。
3. phase-1 oracle 达到 `56.43 AUC`，相对基线为 `+0.97`，证明基线附近存在逐帧校正空间；但 router-only 仅得到 `54.83`，说明该空间不能由当前低维统计路由器稳定预测。
4. mini A/B/C 上的提升不能作为晋级依据。多个方法在 mini 上平均提升约 `+0.6` 至 `+1.2 AUC`，进入全量后却统一反转。
5. 当前失败不是单一超参数问题，而是共同的决策错误：模块能够产生“不同的融合结果”，却无法可靠判断“何时修改基线会带来真实任务收益”。

研究路线裁决：

- 停止继续微调 `router-only`、标量门控、可靠性分数、频域阈值和同构残差桥。
- 不再把 Temporal Token 作为论文主线。
- 保留全量基线、oracle 候选生成和逐帧日志基础设施。
- 若继续，只推进一个结构上不同的方向：候选输出级的短时序验证与风险约束选择，而不是再次从 pooled feature 预测模态权重。

## 2. 证据等级与比较协议

### 2.1 可信度分级

| 等级 | 定义 | 用途 |
|---|---|---|
| A | 同一 LasHeR test 244 序列、完整输出、标准五指标 | 决定路线是否有效 |
| B | mini_lasher_test A/B/C，与各自基线成对比较 | 判断训练健康度和跨子集稳定性 |
| C | oracle、属性分析、bypass、合成干预 | 解释机制和估计上限 |
| D | smoke、未完成测试、仅训练损失、旧注释且缺少结果文件 | 仅作工程记录 |

全量基线的标准评测合同为：

| AUC | OP50 | OP75 | Precision | Norm Precision |
|---:|---:|---:|---:|---:|
| 55.46 | 67.28 | 46.73 | 68.98 | 65.23 |

基线权重：

`TBSI/output/experiments/v1.0.0-完美基线-全量/checkpoints/TBSITrack_ep0015.pth.tar`

### 2.2 为什么 mini 结果会误导

mini A/B/C 每组仅覆盖有限序列与属性组合，且不同子集的基线难度明显不同。一个模块只要偏向某类退化，就可能在 A 或 B 上获得 `+2 AUC`，同时在 C 上下降。全量评测则由大量正常、部分遮挡、尺度变化和复杂运动帧共同决定，正常样本上的轻微负迁移会淹没少数困难属性的收益。

因此，今后的晋级门槛应为：

- mini A/B/C 至少 `3/3` 正增益；
- 三组最小增益不低于 `+0.3 AUC`；
- 全量验证必须使用独立训练的全量模型；
- 最终只认可全量 `>= +0.3 AUC`，目标为 `>= +0.5 AUC`。

## 3. 全量实验总表

下表只列入已有完整标准评测结果的主要模型。

| 系列 | 模型 | AUC | 相对基线 | OP50 | OP75 | Precision | Norm Precision | 裁决 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | v1.0.0 完美基线 | **55.46** | 0.00 | 67.28 | 46.73 | 68.98 | 65.23 | 当前 incumbent |
| Signal Decouple | v1.6.0 | 54.00 | -1.46 | 65.53 | 44.92 | 67.52 | 63.67 | 停止 |
| Signal Decouple | v1.6.5 learn-scale | 53.58 | -1.88 | 64.99 | 44.74 | 67.05 | 63.16 | 停止 |
| Reliability | v2.1.0 reliability residual | 54.62 | -0.83 | 66.27 | 45.70 | 68.42 | 64.37 | 停止 |
| FCC | v4.1.0 | **55.05** | **-0.41** | 66.96 | 46.08 | 68.93 | 64.92 | 最接近基线，仍失败 |
| FCC | v4.1.6 middle | 54.12 | -1.33 | 65.74 | 45.43 | 67.51 | 63.63 | 停止 |
| FCC | v4.2.0 residual | 54.18 | -1.28 | 65.77 | 45.23 | 67.89 | 63.84 | 停止 |
| CUTR oracle | phase-1 oracle | **56.43** | **+0.97** | 68.50 | 48.38 | 70.22 | 66.47 | 仅为不可部署上限 |
| CUTR learnable | v5.0.0 router-only ep6 | 54.83 | -0.63 | 66.71 | 46.14 | 68.70 | 64.74 | 未达到 +0.5 目标 |

这里最关键的对照不是 `56.43 vs 54.83`，而是：

```text
oracle 可利用空间：       +0.97 AUC
learnable router 实际恢复：-0.63 AUC
相对目标 +0.50 的缺口：   -1.13 AUC
```

这表明问题不在于缺少候选动作，而在于动作效用对当前 router 输入不可辨识，或其监督目标与闭环跟踪收益不一致。

## 4. 各系列复盘

### 4.1 时序令牌系列

代表设计包括前置 Temporal Token、融合后置 token、帧对训练、不同 bottleneck、纯 cross-attention 和 EMA token。

已有专项结果：

| 版本 | AUC | 相对当时 DA 基线 55.83 |
|---|---:|---:|
| 前置 v2 4ep | 55.32 | -0.51 |
| 前置 v2 15ep | 54.65 | -1.18 |
| 后置 bn64 无帧对 | 54.58 | -1.25 |
| 后置 bn64 + 帧对 | 54.94 | -0.89 |
| 后置 bn128 + 帧对 | 55.40 | -0.43 |
| 纯交叉 v3 | 48.97 | -6.86 |
| EMA + 交叉 v4 | 未完成 | 不进入结论 |

专业判断：

- 帧对训练修复了训练/推理状态分布不一致，但没有创造足够的任务信号。
- token 在注意力交互中的占比很低，最优版本的主要作用更接近融合后特征自注意力，而不是有效时序记忆。
- 纯 token cross-attention 将高维空间结构压到少量原型子空间，出现明显信息瓶颈。
- TBSI 的固定首帧模板已经提供长期外观锚点；新增模块若没有独立运动监督，只能通过长梯度路径学习短时动态，信噪比不足。

结论：该系列已经完成结构性证伪，不建议再增加 token 数量、GRU、EMA 规则或 bottleneck 宽度。

### 4.2 DGS / 退化感知融合 / 差分投影

该系列从通道质量、MADC/CSR、自由路由、attention 路由、RGB-TIR 差分投影、保守缩放、模板对齐和自质量估计等方向反复尝试。

主要价值是暴露了两个问题：

- RGB-TIR 差异并不等于模态退化。颜色、姿态、热响应差异和真实故障会产生相似统计。
- 小路由器容易学习数据子集偏好，而不是样本级任务效用。

`v1.5.0-基线+dgs-v4-diffproj-全量v2` 留有训练和测试日志，但没有找到可信的完整标准分析表，因此不把它写入全量定量结论。这个系列可作为探索历史保留，不应再作为当前主线恢复。

### 4.3 Signal-Decouple / 条件传播

该系列包括直接信号解耦、zero-init、late-layer residual、全层弱残差、learnable scale、固定弱尺度、条件传播、soft-search reliability、template-search competition 和 reliability-guided interaction。

mini 上最有吸引力的结果包括：

| 方法 | mini A/B/C 平均 AUC 增益 | 正增益子集 |
|---|---:|---:|
| v1.6.0 signal-decouple | +1.17 | 3/3 |
| v1.6.5 learn-scale | +0.70 | 2/3 |
| v1.8.1 late reliability | +0.60 | 2/3 |
| v1.6.2 late residual | +0.49 | 2/3 |

但全量结果为：

- v1.6.0：`54.00`，相对基线 `-1.46`；
- v1.6.5：`53.58`，相对基线 `-1.88`。

属性分析显示部分困难属性确实上涨，例如某些低照度、热交叉或异常可见性类别；同时高频正常属性和常规运动场景回归。因为属性标签高度重叠，少数属性上涨不能抵消全量负迁移。

结论：信号解耦可以改变特征，但可靠性估计不足以承担逐帧干预决策。zero-init 和弱尺度只能降低破坏，不能把错误决策变成正确决策。

### 4.4 输出残差、可靠性残差与模板条件桥

该系列尝试把侵入位置后移，以保护 TBSI 主融合路径，包括 output residual gate、reliability-aware residual、late-layer、template prior、template-conditioned bridge、positive/clipped bridge、CFS reliability、competitive bridge 和 competitive residual。

代表性 mini 结果：

| 方法 | mini A/B/C 平均 AUC 增益 | 正增益子集 |
|---|---:|---:|
| v2.1.0 reliability residual | +0.98 | 3/3 |
| v3.1.0 CFS reliability bridge | +0.85 | 2/3 |
| v3.0.0 template-conditioned bridge | +0.83 | 2/3 |
| v3.2.0 competitive bridge | +0.33 | 2/3 |
| v2.0.0 output residual gate | -0.28 | 1/3 |

v2.1.0 是最具迷惑性的例子：mini 三组均上涨，范围约 `+0.55` 至 `+1.20`，但全量为 `54.62`，下降 `-0.83`。

这说明：

- 将模块后移确实减少了 backbone 污染；
- 但 mini 的三组正增益仍不足以保证全量泛化；
- 模板相似度更适合做“候选结果验证”，不适合直接映射成特征残差强度。

结论：停止继续设计同类 feature bridge；保留“模板作为验证信号”这一观察。

### 4.5 FreqGate / FCC 频域冲突校准

该系列覆盖频域门控、FCC、local/margin/learn-scale、low-mid/middle、threshold、selection、conflict activation、directional、layerwise、residual、frequency reliability、value gating 和 auxiliary loss。

代表性 mini 结果：

| 方法 | mini A/B/C 平均 AUC 增益 | 正增益子集 |
|---|---:|---:|
| v4.2.0 FCC residual | +0.96 | 2/3 |
| v4.1.6 FCC middle | +0.91 | 2/3 |
| v4.1.0 FCC | +0.86 | 3/3 |
| v4.1.10 low-mid-middle | +0.73 | 3/3 |
| v4.1.7 selection | +0.61 | 3/3 |
| v4.1.11 layerwise | +0.56 | 3/3 |
| v4.0.0 FreqGate | +0.02 | 1/3 |

全量却得到：

- v4.1.0：`55.05`，下降 `-0.41`；
- v4.1.6：`54.12`，下降 `-1.33`；
- v4.2.0：`54.18`，下降 `-1.28`。

FCC 是所有可学习全量方案中最接近基线的一支。这说明频域冲突统计包含一定有效信息，但它更像相关性描述，不是干预充分条件。middle 或 residual 版本增加校正强度后下降更大，进一步支持“误触发成本大于困难帧收益”的判断。

结论：FCC 可作为论文失败分析或 future work 的动机，不值得继续做阈值和层位搜索。

### 4.6 CDIO -> ICPO -> CUTR-Lite

CDIO 最初试图用低维时序状态、共同/差分创新和 Kalman 风格增益修正融合。评审后逐步修复为 ICPO/CUTR：

- 删除低维状态到高维特征的病态重建；
- 不再把 common/differential 残差解释为退化因果；
- 使用候选动作的真实 tracking utility 作为监督；
- 显式保留 keep-baseline 动作；
- 为路由器建立独立 KL 监督；
- 最终收缩为冻结 TBSI、只训练约 30 万参数的三动作 router。

phase-1 oracle：

| 数据集 | Baseline AUC | Oracle AUC | 增益 |
|---|---:|---:|---:|
| mini_lasher_test | 67.00 | 68.88 | +1.88 |
| LasHeR test | 55.46 | 56.43 | +0.97 |

全量 oracle 的动作比例：

| keep | RGB | TIR | temporal |
|---:|---:|---:|---:|
| 34.65% | 28.42% | 30.49% | 6.44% |

这说明模态方向候选比时序候选更重要，也说明逐帧最优动作切换频繁。随后 router-only ep6 在全量上得到 `54.83 AUC`，没有恢复 oracle gap。

失败原因不是“oracle 不存在”，而是：

1. oracle 标签由当前帧 GT IoU 产生，动作边界随细小框偏差剧烈变化，属于高噪声、低 margin 的决策目标。
2. pooled RGB/TIR/base 特征和 score 统计丢失了候选框的空间差异，无法辨识哪个动作会改善闭环轨迹。
3. frame-wise utility 忽略动作对下一帧搜索区域的影响；单帧最优不等于序列最优。
4. 冻结 backbone 保护了基线，却也限制了可分性。router 只能在固定表示上拟合不可分标签。
5. 期望动作把离散的 `RGB/TIR/keep` 候选平均成一个中间残差，可能不等价于任何一个真正有效候选。

结论：CUTR 的“任务效用而非抽象质量”仍是最有价值的思想，但当前 router-only 实现已被全量结果否定。

## 5. 跨系列共同失败机制

### 5.1 决策变量可观测，但效用不可辨识

几乎所有模块都能观测 RGB/TIR 差异、频域冲突、模板相似度或响应置信度，但这些量与“改动后是否涨 IoU”只有条件相关性。模型缺少候选输出、轨迹状态和下一帧影响，因此无法从观测统计推断干预效用。

### 5.2 正常样本的误触发成本更高

TBSI 基线已经较强。增强模块面对的是残余错误而不是普遍错误。困难帧收益只覆盖少量样本，而每个正常帧上的微小扰动都会累计到后续搜索区域，形成闭环误差。

### 5.3 训练目标与序列指标不一致

现有模块主要优化单帧 tracking loss。SOT 推理是闭环系统：本帧框决定下一帧裁剪区域。单帧 loss 的微小改善可能增加轨迹抖动、尺度漂移或未来丢失概率。

### 5.4 mini 选择偏差被反复放大

大量变体在同一 mini A/B/C 上迭代，实际上形成了对验证子集的隐式超参数拟合。即使每个实验没有直接使用 test GT 训练，反复依据子集结果选择结构也会产生 selection bias。

### 5.5 “更保守”只能趋近基线，不能超过基线

zero-init、弱 residual、clipping、late-layer 和 keep action 都降低了负迁移，但如果效用判断本身不准确，它们的上限只是把模型推回原始 TBSI，而不是产生稳定正增益。

## 6. 仍然成立的研究发现

虽然没有可发表的正向模型，现有实验不是没有价值。以下发现有较强证据：

1. TBSI 附近存在约 `+0.97 AUC` 的候选级 oracle 空间。
2. 该空间主要来自 RGB/TIR 候选选择，显式时序候选只占 `6.44%`。
3. 低照度、热交叉、异常可见性等属性上存在局部收益，但高频正常属性的回归决定全量结果。
4. 特征级差异/冲突适合生成候选，不适合独立决定是否执行候选。
5. 模板和短时序更适合作为候选验证信号，而不是直接生成高维增强特征。
6. 对强基线做外部增强时，核心问题是 selective intervention，不是表达能力不足。

## 7. 下一条唯一建议路线

### 7.1 候选输出级短时序验证

下一轮不再预测模态权重，而是先得到 `keep/RGB/TIR` 三个候选框与响应图，再在输出层做短时序风险选择：

```text
TBSI fused feature
    -> keep / RGB / TIR candidate heads
    -> candidate boxes + score maps
    -> temporal verifier
       - 与首帧模板的目标一致性
       - 与最近 2~3 帧速度/尺度模型的一致性
       - 响应峰值、熵、峰谷差和多峰风险
       - 候选间分歧与轨迹跳变风险
    -> keep-biased constrained selection
```

这条线与已失败 router 的本质区别是：

- router 在动作执行前，从压缩特征猜测效用；
- verifier 在候选执行后，直接比较候选的空间结果与短时序一致性；
- 选择目标从单帧 oracle imitation 改为序列风险最小化；
- 模板和时序只负责验证，不负责重建或注入高维特征。

其思想可从选择性预测、能量重排序、模型预测控制和风险敏感决策等领域迁移，而不是继续拼接多模态跟踪模块。

### 7.2 最小证伪实验

只使用现有 baseline checkpoint 和 oracle 候选，不训练新 backbone：

1. 离线读取三候选的 box、score map 和 GT。
2. 构造不使用当前帧 GT 的验证特征。
3. 先测一个解析评分器，再测一个极小 pairwise ranker。
4. 在完整 LasHeR test 上进行严格因果回放，候选选择只能使用当前及历史观测。
5. 同时报告 AUC、相对 oracle 恢复率、keep 比例、切换频率和轨迹跳变率。

止损线：

- 解析评分器不能达到基线 `+0.2 AUC`，不训练 ranker；
- ranker 不能达到 `+0.3 AUC`，停止整个外部校正方向；
- 只有达到 `+0.5 AUC`，才补 RGBT234、属性和效率实验。

## 8. 最终路线裁决

当前状态分类：

- `baseline_ready`：全量基线可信，可直接复用；
- `analysis_ready`：mini、全量、属性和 oracle 证据足以支撑失败分析；
- `main_result_missing`：尚无超过基线的可学习主结果；
- `paper_not_ready`：当前不能围绕 CUTR-Lite 正向方法写论文。

正式决定：

```text
Verdict: BRANCH
停止: Temporal Token、同类 reliability gate、FCC 阈值搜索、CUTR router-only
保留: baseline ep15、三候选生成、oracle 日志、标准全量评测
下一方向: candidate-level temporal verification
预算: 仅允许一次最小证伪 + 一次 learnable ranker
```

如果该结构上不同的方向仍失败，应停止围绕 TBSI 外部融合增强继续堆模块，转向数据/训练目标或更换基线，而不是开启下一轮门控变体。

## 9. 主要证据路径

- 全量基线配置与历史结果：`TBSI/experiments/tbsi_track/v1.0.0-完美基线-全量.yaml`
- 全量基线权重：`TBSI/output/experiments/v1.0.0-完美基线-全量/checkpoints/TBSITrack_ep0015.pth.tar`
- 时序专项报告：`TBSI/EXPERIMENT_REPORT.md`
- 全量方法对比：`TBSI/output/diagnostics/full_candidates_serial/v210_full/summary.json`
- FCC 全量：`TBSI/output/diagnostics/fcc_v410_full/analysis.console.log`
- Oracle 全量：`TBSI/output/diagnostics/cutr_phase1/cutr_oracle_phase1_lasher_full_baseline_v100/analysis_stdout.log`
- Oracle 动作统计：`TBSI/output/diagnostics/cutr_phase1/cutr_oracle_phase1_lasher_full_baseline_v100/oracle_action_summary.json`
- CUTR learnable 全量：`TBSI/output/runtime_logs/cutr_lite_routeronly_full_ep6_t8.analysis.log`
- 当前方法说明：`方案.md`
