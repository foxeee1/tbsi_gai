# G0-Expand 外部序列验证分析

日期：2026-10-04  
Parent：`TBSI-E-MHSA-FULL-20260927`  
Checkpoint：M1 epoch 15，SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`  
配置：`TBSI_E_MHSA_FULL_4090D_seed42`  
硬件/环境：1× RTX4090D 24 GiB；Python 3.8.20；PyTorch 1.9.0+cu111；CUDA 11.1；FP32 tracker

## 1. 目的与冻结内容

G0-Expand 检验原 G0 的早期信息增量能否在未参与上一轮判断的新 train 序列上复现。原 G0 的 18 条 probe 训练序列和 6 条旧验证序列全部排除；从剩余 152 条 LasHeR train 序列中用 seed 42 固定抽取 24 条新序列，每条两个事件，共 48 个事件。

冻结内容包括 M1 epoch-15 checkpoint、block index 2 的早期 token、location 9 输入前晚期 token、M1 serial FP32 闭环、原始 search crop、序列清单、两个 probe seed `42/3407`、640 个固定优化 step 和六个条件。四类输入都使用 `[256, 3072]` 结构。

## 2. 导出完整性

| 项目 | 结果 |
|---|---:|
| 新序列 | 24 |
| 新事件 | 48 |
| 有效目标事件 | 47/48 = 97.92% |
| crop 外或空标签事件 | 1 |
| 小于或等于 1 token 事件 | 1 |
| Fresh parity 事件 | 6 |
| Fresh parity 最大框差 | 0.0 |

Fresh parity 在 3 条新序列、每条两个固定事件上比较 diagnostic 关闭/开启的 tracker 预测框，最大绝对差为 `0.0`。特征均为有限值，四类特征形状均为 `[256, 768]`。1 个 crop 外或空标签事件保留在事件文件中，没有进入 AP 计算。

## 3. 两 seed 主结果

表中 ΔAP 相对同一 seed、同一 fresh event 的 `L_zero`。`best_positive_rank` 是全部正 token 中排名最靠前者，修正了原 G0 使用第一个正 token 的问题。

### Seed 42

| 条件 | mean AP | mean ΔAP vs `L_zero` | mean best positive rank | 正 ΔAP 事件比例 |
|---|---:|---:|---:|---:|
| `L_zero` | 0.81603 | 0 | 4.09 | -- |
| `L_E` | 0.85876 | **+0.04273** | **2.26** | 80.85% |
| `L_Lcopy` | 0.82296 | +0.00692 | 3.94 | 74.47% |
| `L_E_spatial` | 0.84313 | +0.02709 | 2.68 | 70.21% |
| `L_E_event` | 0.85121 | +0.03517 | 2.83 | 68.09% |
| `XY_only` | 0.87409 | +0.05806 | 3.47 | 68.09% |

### Seed 3407

| 条件 | mean AP | mean ΔAP vs `L_zero` | mean best positive rank | 正 ΔAP 事件比例 |
|---|---:|---:|---:|---:|
| `L_zero` | 0.82268 | 0 | 3.09 | -- |
| `L_E` | 0.86854 | **+0.04586** | **2.17** | 93.62% |
| `L_Lcopy` | 0.82387 | +0.00119 | 3.02 | 59.57% |
| `L_E_spatial` | 0.84681 | +0.02413 | 2.40 | 68.09% |
| `L_E_event` | 0.85439 | +0.03171 | 3.32 | 74.47% |
| `XY_only` | 0.87330 | +0.05061 | 3.72 | 68.09% |

## 4. 信息源解释

`L_E` 在两个 seed 上均为正，均超过预注册的 `+0.03 AP` 资源阈值，并且两个 seed 都优于 `L_Lcopy`、`L_E_spatial` 和 `L_E_event`。`L_E` 的 best positive rank 也在两个 seed 上优于 `L_zero`，因此 AP 和排名方向一致。

这支持一个有限结论：早期 RGB/TIR search 表征包含 location 9 输入没有充分利用的条件定位信息，而且该信息具有一定的同事件空间对应性和事件特异性。

`L_E_event` 仍有正增量，说明早期表征的一般外观或模态信息也能帮助读出器；`L_E_spatial` 仍有正增量，说明空间错位并没有完全消除收益。因而不能把全部 `L_E` 增量解释为精确 token 对应信息。

## 5. 位置先验风险

`XY_only` 的增量分别为 `+0.05806` 和 `+0.05061`，高于 `L_E` 的 `+0.04273` 和 `+0.04586`。这说明当前 token 标签任务明显包含 crop 内的位置先验，`L_E` 的 AP 增量不能直接等同于视觉信息增量。

机器闸门仍允许进入 Cplain，是因为位置先验是单独报告的诊断对照，不是 `L_E` 的内容 control。但下一阶段 Cplain 必须加入位置先验消融或位置归一化分析，并报告 `L_E - XY_only` 的差异；不能把 `L_E` 的绝对 AP 提升写成纯粹的目标外观/跨模态信息。

## 6. 决策

最终机器决策为：

```text
CONDITIONAL_PASS_CPLAIN_ONLY
```

通过条件：

- fresh export parity 通过，最大框差 `0.0`；
- seed 42 和 3407 的 `L_E` mean ΔAP 均为正且不低于 `+0.03`；
- 两个 seed 上 `L_E` 均优于 `L_Lcopy`、`L_E_spatial` 和 `L_E_event`；
- AP 和 best positive rank 方向一致。

该决策只允许下一步做冻结 M1 上的 Cplain 闭环诊断。它不授权 SCSU 实现、SCSU 训练或 full confirmation。Cplain 必须与 M1 使用相同 crop、相同写入位置和 zero gate 起点，并增加 `gamma=0` 关门消融；只有 Cplain 本身出现可复现闭环收益，才有理由比较 Cscsu。

## 7. 运行限制

1. 新验证集有 24 条序列、47 个有效事件，仍属于一次扩大诊断，不是 tracker full-test 结果。
2. 每个条件独立训练读出器；AP 主要用于同事件配对差异，不代表训练后 tracking AUC。
3. 当前未报告 bootstrap 区间；后续 Cplain 前应以序列为聚类单位计算 paired bootstrap。
4. `L_E_event` donor 在同一 split 内选取，但当前实现按事件池轮换，尚未加入更细的非标签条件匹配。
5. 当前结果没有使用目标大小或 M1 AP 选择新序列；序列清单在导出前冻结。

## 8. 证据

- Manifest：[manifest.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_expand_20261004/manifest/manifest.json)
- Fresh export：[export_manifest.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_expand_20261004/export/export_manifest.json)
- Fresh parity：[fresh_parity.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_expand_20261004/export/fresh_parity.json)
- Probe summary：[probe_summary.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_expand_20261004/probe/probe_summary.json)
- Decision：[decision.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_expand_20261004/decision.json)
