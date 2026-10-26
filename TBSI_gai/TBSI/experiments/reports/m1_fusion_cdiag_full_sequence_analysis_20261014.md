# M1 Cdiag full-244 逐序列与信息流归因

## 分析边界

Parent 是固定 M1，candidate 是 full-train 后的 Cdiag。两者使用同一 244 条 LasHeR test 序列、同一 GT 和官方 evaluator。官方审计字段中 `M1` 表示 candidate，`B0` 表示固定 baseline；本报告统一按这个方向解释。

逐帧部分读取两套 tracker prediction txt 和同一 `init.txt`，排除初始化帧后比较：candidate 是否改变输出框、candidate-baseline 的 IoU 差、救援帧（IoU 差 > +0.05）和伤害帧（< -0.05）。这部分是可观测输出归因，不等同于隐藏层的严格因果归因。

## 全量逐序列结论

- 244 条序列严格比较：candidate 胜 115、负 126、平 3。
- 平均 sequence AUC delta：`-0.02735` 个百分点；中位数 `-0.02780`。
- 95% 以上序列的初始化帧之后输出发生变化（changed fraction `0.9426`），因此模块确实进入了推理信息流。
- 改变的帧中，救援帧 `16,979`，伤害帧 `17,013`，数量几乎相等；改变帧的平均 IoU delta 为 `-0.000061`。
- 这解释了 aggregate 指标的形态：少数严重失败序列有明显恢复，但大量中高质量序列出现小幅退化，整体抵消后 AUC 略降。

## 按 baseline 难度分层

| baseline AUC 桶 | 序列数 | 胜/负/平 | mean AUC delta | mean overlap delta | mean failure<0.5 delta | 救援/伤害帧 |
|---|---:|---:|---:|---:|---:|---:|
| low `<30` | 42 | 22/19/1 | +4.167 | +0.0442 | -0.0542 | 4635 / 3274 |
| mid `30-60` | 72 | 31/41/0 | -1.268 | -0.0127 | +0.0131 | 6545 / 6847 |
| high `>=60` | 130 | 62/66/2 | -0.696 | -0.0070 | +0.0083 | 5799 / 6892 |

Cdiag 的有效性主要集中在 baseline 已经严重失败的 low 桶；一旦 baseline 处于中高质量区间，伤害帧超过救援帧，且失败率恶化。这是条件性修复信号，不是通用融合增益。

## 最大增益序列

| 序列 | AUC delta | overlap delta | 救援帧 | 伤害帧 | persistent segment delta |
|---|---:|---:|---:|---:|---:|
| `ab_bikeoccluded` | +56.994 | +0.655 | 260 | 3 | 0 |
| `3rdfatboy` | +55.402 | +0.564 | 234 | 5 | -1 |
| `leftfarboycomingpicktheball` | +48.262 | +0.489 | 671 | 14 | 0 |
| `right5thflag` | +19.368 | +0.198 | 632 | 137 | -2 |
| `boy\`sheadingreycol` | +14.958 | +0.151 | 78 | 5 | +2 |
| `AQgirlwalkinrain` | +14.158 | +0.143 | 74 | 13 | 0 |

这些序列说明 Cdiag 有时能把已经失锁的轨迹重新拉回目标附近，但 `boy\`sheadingreycol` 同时增加了 persistent segment，说明“局部恢复”不等于稳定闭环恢复。

## 最大退化序列

| 序列 | AUC delta | overlap delta | 救援帧 | 伤害帧 | persistent segment delta |
|---|---:|---:|---:|---:|---:|
| `10runone` | -65.288 | -0.666 | 3 | 251 | +1 |
| `whiteridingbike` | -26.633 | -0.268 | 4 | 202 | -1 |
| `shinycarcoming` | -19.260 | -0.191 | 12 | 169 | 0 |
| `leftexcersicebookyellow` | -17.030 | -0.172 | 34 | 311 | +2 |
| `carcomingfromlight` | -15.236 | -0.153 | 6 | 130 | 0 |
| `rightbottlecomes` | -14.828 | -0.155 | 71 | 547 | -3 |

`10runone` 是最清晰的反例：candidate 几乎从第 2 帧开始持续改变输出，但 251 帧伤害、只有 3 帧救援，说明 residual 改变了闭环轨迹，却没有可靠的状态准入机制。

## 训练期信息流证据

- `gradient_flow.csv`：末期 total grad norm `0.0965`，4/4 residual 参数有梯度，zero-grad count 为 0；不存在明显梯度断路。
- `feature_changes.csv`：`gamma_rgb` 从约 `0.00056` 增至 `0.09468`，`gamma_tir` 从 `-0.00078` 变为 `-0.13207`；末期 RGB/TIR residual delta norm 约 `1230/1571`，说明跨模态修正不是 zero-gate。
- internal signal 日志显示 fused/template/search norm 随训练发生变化；这是模块确实改变了 forward 表示的证据，但日志没有序列 ID，不能把某个序列的涨跌直接归因到某个隐藏层通道。
- `loss_components.csv` 末期 total loss `0.7764`、IoU loss `0.8105`，训练目标可优化；但训练 loss 的下降没有对应到 full-test 的长期 recovery。

## 干净归因

可以排除：模块未更新、zero-gate、全局计算图断路、测试输出完全未改变。

当前证据支持：Cdiag residual 已进入 TBSI feature 信息流，并在少数严重失锁序列上产生局部恢复；但它没有判断何时应介入，也没有直接约束 center 候选、offset/size 解码或 crop 状态反馈。因此在中高质量序列上会持续施加小幅扰动，导致救援和伤害基本抵消，长期 persistent failure 不减少。

不能从本次日志声称：某个具体隐藏层或某个模态单独导致退化。要得到更强的算子级因果结论，需要下一次独立实验记录 per-sequence/per-frame gate、residual norm 与 center/readout 变化，并保持 no-op/abstain 对照。

## 下一步判定

停止无条件扩大 feature-side residual。若继续追求约 1% 提升，应做条件式 readout/state-recovery 路线：先在 held-out 序列上验证可部署 utility gate 能否区分救援帧与伤害帧，再训练直接影响 center/readout 和 crop recovery 的单变量模块。若 gate 仍不能区分两类帧，则当前信息流缺少可靠指导信号，继续堆叠融合算子没有证据基础。

## 完整产物

- 逐序列明细：`output/experiments/M1_FUSION_CDIAG_FULL_15EP_B64_W4_retry3_20261013/analysis_full_sequence/per_sequence_detailed.csv`
- 所有发生输出变化的帧：`.../analysis_full_sequence/changed_frames.csv`
- 聚合、分桶、top gains/losses：`.../analysis_full_sequence/summary.json`
- 生成脚本：`tools/analyze_m1_cdiag_full_20261014.py`
