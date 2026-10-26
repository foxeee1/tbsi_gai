# G0 信息源准入实验详细分析

日期：2026-10-04  
Parent：`TBSI-E-MHSA-FULL-20260927`  
Checkpoint：M1 epoch 15，SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`  
配置：`TBSI_E_MHSA_FULL_4090D_seed42`  
硬件/环境：1× RTX4090D 24 GiB；Python 3.8.20；PyTorch 1.9.0+cu111；CUDA 11.1；`TBSI_INFER_FP16=0`

## 1. 实验问题与合同

G0 检验的是：在 M1 实际闭环 search crop 不变的情况下，较早层 RGB/TIR search token 是否包含 location 9 输入特征之外、且与目标定位有关的条件信息。

本实验冻结 M1，不训练 SCSU，不修改预测框。使用固定 LasHeR train proxy 中 24 条序列、每条 2 个事件，共 48 个事件。前向只使用模板和 M1 产生的 search crop；GT 只在前向结束后投影到 search token 网格生成标签。

早期特征取 transformer block index 2，晚期特征取 location 9 TBSI 注入前的 search token。两者都是 16×16、768 维 token，因此本实验研究表征信息差异，不支持“恢复更高分辨率细节”的说法。

四个读出条件为：

| 条件 | 输入 |
|---|---|
| `L` | location 9 输入的 RGB/TIR search token |
| `L_plus_E` | `L` 加早期 RGB/TIR search token |
| `L_plus_Lcontrol` | `L` 加同维度的晚期 token control |
| `L_plus_Eshuffle` | `L` 加打乱事件来源的早期 RGB/TIR token |

每个条件使用相同的两层小型读出器结构、训练轮数和序列隔离规则。标准化参数只从 probe 训练序列拟合。

## 2. 执行完整性

### 2.1 预测 parity

导出旁路在 3 条序列、6 个固定事件上与关闭导出时逐事件比较预测框，最大绝对框差为 `0.0`。这说明特征导出没有改变 M1 闭环预测。

### 2.2 标签覆盖

| 项目 | 数量 | 比例 |
|---|---:|---:|
| 总事件 | 48 | 100% |
| 目标中心/目标 token 在 crop 内 | 43 | 89.58% |
| crop 外或无有效目标 token | 5 | 10.42% |
| 目标小于或等于 1 个 token | 5 | 10.42% |
| 进入 probe 的事件 | 43 | 89.58% |

5 个 crop 外或空标签事件被单独记录，没有计入可救援成功率。标签由真实 crop 到 16×16 网格的几何变换生成，背景优先使用目标框外的一圈邻近 token。

## 3. Probe 主结果

当前 sequence-isolated split 使用 18 条序列、31 个事件训练读出器，6 条序列、12 个事件验证。以下 AP 是验证事件的目标 token average precision。

| 条件 | 验证 mean AP | 验证 median AP | 相对 `L` mean ΔAP | 正 ΔAP 事件比例 |
|---|---:|---:|---:|---:|
| `L` | 0.79983 | 0.87489 | 0 | 0/12 |
| `L_plus_E` | 0.88199 | 0.88919 | **+0.08216** | 10/12 = 83.33% |
| `L_plus_Lcontrol` | 0.78883 | 0.88293 | -0.01100 | 5/12 = 41.67% |
| `L_plus_Eshuffle` | 0.81635 | 0.82681 | +0.01652 | 4/12 = 33.33% |

### 3.1 序列级结果

事件按序列聚合后，`L_plus_E` 的 mean ΔAP 仍为 `+0.08216`，6 条验证序列中 5 条为正：

| 序列 | `L_plus_E` ΔAP | `L_plus_Lcontrol` ΔAP | `L_plus_Eshuffle` ΔAP |
|---|---:|---:|---:|
| `2ndrunningboy` | +0.14696 | +0.05382 | +0.06975 |
| `biketurndark` | +0.11062 | +0.03176 | +0.09192 |
| `boyalone` | -0.00133 | +0.00655 | -0.03945 |
| `boybehindbus` | +0.07242 | -0.06823 | -0.01370 |
| `rightgirlwithumbrella` | +0.02891 | -0.08429 | -0.03555 |
| `tallboyNumber_9` | +0.13538 | -0.00563 | +0.02618 |

按序列进行 20,000 次 bootstrap，`L_plus_E` mean ΔAP 的 95% 区间约为 `[+0.0388, +0.1246]`。这是以 6 条验证序列为聚类单位的探索性区间，不能代替多序列、多 seed 验证。

### 3.2 困难事件

在 12 个验证事件中，M1 `L` AP 低于 0.60 的事件有 4 个：`G0-0001`、`G0-0009`、`G0-0011`、`G0-0019`。`L_plus_E` 的 ΔAP 分别为 `+0.2207`、`+0.2861`、`+0.2566`、`+0.1679`，4/4 为正，平均约 `+0.2328`。

这表明当前信号主要集中在 M1 读出较困难的事件，符合“早期表征可能保留晚期表征未充分利用的信息”的工作假设。但这些困难事件只覆盖 4 个事件，不能形成稳定困难属性结论。

### 3.3 排名指标的限制

`L_plus_E` 的 AP 普遍上升，但代码中的 `target_rank` 使用目标 mask 中第一个正 token 的位置作为单一代表，因此不是多 token 目标的完整排名指标。该指标在 `L_plus_E` 下只有 3/12 事件改善，和 AP 结论不一致；本报告以 AP 为主，并把 rank 结果视为辅助诊断，后续应改为目标 token 最佳 rank、目标 token 平均 rank 或 top-k recall 的预注册指标。

## 4. Control 与可解释性

### 4.1 `L_plus_Lcontrol`

`L_plus_Lcontrol` 的 mean ΔAP 为 `-0.01100`，中位数为 `-0.00818`。它没有复制 `L_plus_E` 的正增量，支持增量来自早期 token 内容，而不仅是简单增加输入维度。

不过当前实现的 late control 是在同一事件内对晚期 token 做固定 token roll，不是独立事件匹配的 late feature。它适合检查同维度和位置扰动，不能完全排除“早期输入的空间排列更适合读出器”的解释。扩大诊断应加入预注册的跨事件、同分布 late control，并固定其构造方式。

### 4.2 `L_plus_Eshuffle`

`L_plus_Eshuffle` 的 mean ΔAP 为 `+0.01652`，比真实 `L_plus_E` 低 `0.06564`，验证事件中只有 4/12 为正，序列级 bootstrap 95% 区间约为 `[-0.0223, +0.0589]`。真实早期特征与打乱早期特征之间存在明显的点估计差异，但当前样本不足以把 shuffle 差异称为稳定的对应关系效应。

打乱条件仍保留了一部分正增益，说明可能存在两类因素：早期特征本身提供了通用外观/模态信息，以及事件对应的早期空间结构提供了额外信息。扩大诊断必须把这两部分分开测量。

## 5. 统计和方法边界

1. 48 个事件中只有 12 个进入当前验证集；有效样本按序列聚类后只有 6 个序列。
2. 读出器是分别训练的 probe，不是同一个模型的嵌套检验；AP 绝对值受小样本、读出器训练和标准化影响，主要解释配对差值。
3. 当前 probe 只有 seed 42；没有第二个 probe seed，也没有预注册扩大诊断集上的重复拟合。
4. 当前 `L_plus_E` 的资源阈值 `mean ΔAP >= 0.03` 是工程准入规则，不是统计学显著性阈值。
5. GT 只用于 forward 后标签，parity 通过；但 crop 外事件无法评估同一 crop 内的救援能力，必须继续单独报告。
6. 该结果研究的是 token 级目标定位读出能力，不等价于 tracker AUC、PR、NPR 或训练后增益。

## 6. 决策

当前证据支持 `EXPAND_DIAGNOSTIC`：真实早期特征在冻结 M1 上表现出相对晚期特征的条件 AP 增量，且增量在 M1 较困难的 4 个验证事件上方向一致；late control 没有复现该增量，shuffle 的增量明显较小。

扩大诊断完成前，不应实现或训练 SCSU。下一阶段应固定扩大序列集合和第二 probe seed，使用序列级隔离，补充跨事件 late control、严格的早期特征 shuffle、困难事件分层，以及目标 token top-k recall/最佳 rank 等预注册指标。只有扩大诊断仍满足同向增量和 shuffle 衰减，才进入 C0/Cplain/Cscsu 的 P1 比较。

## 7. 证据

- 原始 runner：[runner.log](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_information_source_20261004/runner.log)
- 导出清单：[export_manifest.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_information_source_20261004/export/export_manifest.json)
- parity：[parity.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_information_source_20261004/export/parity.json)
- 标签汇总：[label_summary.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_information_source_20261004/labels/label_summary.json)
- probe 逐事件结果：[probe_metrics.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_information_source_20261004/probe/probe_metrics.json)
- 机器决策：[decision.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_information_source_20261004/decision.json)
