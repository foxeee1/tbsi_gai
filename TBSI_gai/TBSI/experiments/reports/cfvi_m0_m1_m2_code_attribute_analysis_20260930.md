# CFVI M0/M1/M2 核心代码与属性对比

日期：2026-09-30  
范围：LasHeR RPP-v1 冻结 test proxy-60，seed 42，15 epoch，FP32 tracker，串行 `threads=0`。  
结论级别：单 seed、proxy-60 诊断，不是 full LasHeR 主结果。

## 1. 核心代码对比

| 项目 | M0-0 / M0-15 | M1 | M2 |
|---|---|---|---|
| 配置开关 | `CFVI_MODE="off"` | `CFVI_MODE=fixed` | `CFVI_MODE=dynamic` |
| CFVI 对象 | `self.cfvi = None` | `FixedCrossModalFreqInjection` | `DynamicCrossModalFreqInjection` |
| 频率分解 | 无 CFVI 路径 | 2-D Haar：LL/LH/HL/HH | 同 M1 |
| 源模态处理 | 不修改 V | 四个源频带等权拼接后 1x1 Conv 投影 | 源频带先按位置动态加权，再 1x1 Conv 投影 |
| 权重 | 无 | 固定 `[1,1,1,1]` | `1 + 0.5*tanh(gate_net(receiver, source))`，范围 `(0.5,1.5)` |
| 可学习缩放 | 无 | `lambda_rgb/lambda_tir`，初值 0.001 | 同 M1 |
| 动态 gate | 无 | 无 | `8*dim -> 64 -> 4`，最后一层零初始化 |
| 注入位置 | 原始 V | `v_rgb += local_rgb; v_tir += local_tir` | 同 M1 |
| Q/K | 不变 | 不变 | 不变 |

关键实现位置：

- CFVI 模块：[cfvi_module.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/cfvi_module.py:1)
- 实例化和 V 注入：[tbsi_layer.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py:50)
- 配置传递：[tbsi_track.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/tbsi_track/tbsi_track.py:388)
- M0 配置：[CFVI_M0_RPP_v1_seed42.yaml](/root/autodl-tmp/TBSI_gai/TBSI/experiments/tbsi_track/CFVI_M0_RPP_v1_seed42.yaml:10)

M0-0 与 M0-15 的区别只有 checkpoint 来源：M0-0 直接加载父 checkpoint，M0-15 使用相同 proxy train contract 重新训练 15 epoch。M0-15 因此可作为普通训练对照，但不能单独解释为 CFVI 的因果对照。

## 2. 总体结果

| 配置 | AUC | OP50 | OP75 | PR | NPR |
|---|---:|---:|---:|---:|---:|
| M0-0 父 checkpoint | 51.52 | 62.71 | 40.79 | 65.82 | 60.88 |
| M0-15 CFVI-off | 46.64 | 56.65 | 35.71 | 60.89 | 54.84 |
| M1 fixed | 46.03 | 55.88 | 34.94 | 60.18 | 54.17 |
| M2 dynamic | 45.77 | 55.29 | 35.27 | 60.08 | 53.56 |

相对 M0-15 的逐序列配对结果：

| 比较 | mean ΔAUC | median ΔAUC | AUC win rate | AUC large loss (≤-5) | AUC large gain (≥5) |
|---|---:|---:|---:|---:|---:|
| M1 - M0-15 | -0.61 | -0.30 | 45.0% | 15.0% | 10.0% |
| M2 - M0-15 | -0.87 | -0.14 | 46.7% | 13.3% | 11.7% |

这说明总体 AUC 下降主要由部分大幅退化序列贡献，不能概括为每个序列都退化。M2 的 PR/NPR win rate 接近或超过 50%，但整体均值仍为负，表现为异质性而非稳定收益。

## 3. 属性分组结果

属性来自冻结的 `Attributes&Order.zip`，属性为多标签；因此各行不能相加，且小组结果不具备独立样本意义。表中 AUC 单位为百分点，`M1/M2-M0-15` 也是百分点。

| 属性 | n | M0-0 AUC | M0-15 AUC | M1 AUC | M2 AUC | M1-M0-15 | M2-M0-15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NO | 11 | 64.39 | 62.51 | 65.85 | 61.84 | +3.34 | -0.67 |
| PO | 55 | 50.59 | 45.62 | 45.13 | 44.76 | -0.50 | -0.87 |
| TO | 29 | 46.39 | 40.72 | 39.29 | 39.62 | -1.43 | -1.10 |
| HO | 4 | 37.24 | 28.85 | 34.78 | 33.88 | +5.93 | +5.03 |
| MB | 30 | 45.33 | 44.95 | 42.94 | 43.60 | -2.01 | -1.35 |
| LI | 21 | 49.44 | 41.34 | 42.95 | 41.00 | +1.61 | -0.34 |
| HI | 12 | 51.25 | 42.97 | 40.09 | 42.23 | -2.88 | -0.74 |
| AIV | 4 | 43.13 | 42.72 | 39.01 | 40.57 | -3.71 | -2.16 |
| LR | 25 | 46.57 | 41.97 | 40.40 | 41.11 | -1.57 | -0.86 |
| DEF | 13 | 48.32 | 49.59 | 47.35 | 47.58 | -2.24 | -2.01 |
| BC | 27 | 51.00 | 46.39 | 44.97 | 45.45 | -1.42 | -0.94 |
| SA | 32 | 45.95 | 45.77 | 45.10 | 45.18 | -0.66 | -0.59 |
| CM | 42 | 53.93 | 48.12 | 47.79 | 47.06 | -0.34 | -1.06 |
| TC | 43 | 45.71 | 41.30 | 41.86 | 40.97 | +0.56 | -0.33 |
| FL | 8 | 51.79 | 45.85 | 47.49 | 42.86 | +1.63 | -2.99 |
| OV | 4 | 64.30 | 61.21 | 60.92 | 60.33 | -0.29 | -0.88 |
| FM | 52 | 50.52 | 44.55 | 43.84 | 43.73 | -0.71 | -0.83 |
| SV | 58 | 50.71 | 46.03 | 45.45 | 45.18 | -0.58 | -0.85 |
| ARV | 36 | 45.94 | 40.04 | 38.79 | 39.16 | -1.25 | -0.89 |

## 4. 属性解释边界

1. M1 在 `NO`、`LI`、`FL` 小组相对 M0-15 为正，但 `NO` 和 `FL` 只有 11 和 8 条序列；不能据此宣称对某属性稳定有效。
2. M1 在 `HI`、`DEF`、`MB` 退化最明显；M2 也在 `DEF`、`MB` 退化，说明动态 gate 没有解决这些困难条件下的失配。
3. M2 没有形成一致的属性优势：`HO` 是正向，但 n=4；`FL` 反而比 M1 明显更差；大样本 `PO/SV/FM` 仍为负。
4. M0-15 比 M0-0 的整体下降说明 proxy 训练本身造成了显著分布/优化变化。因而 M1/M2 相对 M0-0 的下降不能全部归因于 CFVI；正确的主要比较是 M1/M2 相对 M0-15。

## 5. 当前诊断与下一步

当前最可信的归因是 L4 coordination/distribution mismatch：CFVI 把跨模态频率信息加入 V，但训练只使用 176 条 train proxy 序列，且 M1/M2 单 seed 在困难属性上没有一致收益。M2 的输入条件 gate 增加了表示自由度，却没有改变总体负向方向。

建议下一步只做一个变量：固定 M0-15 的训练协议，在同一 proxy 和 seed 下记录每层 `lambda_rgb/lambda_tir`、M2 四频带权重的均值/方差，再决定是频带权重失效、注入强度失控，还是 proxy 分布问题。没有该诊断和第二 seed 前，不建议进入 full confirmation。

证据：

- `output/experiments/CFVI_M0_PARENT_RPP_v1_seed42/analysis/eval_data.pkl`
- `output/experiments/CFVI_M0_RPP_v1_seed42/analysis/eval_data.pkl`
- `output/experiments/CFVI_M1_RPP_v1_seed42/analysis/eval_data.pkl`
- `output/experiments/CFVI_M2_RPP_v1_seed42/analysis/eval_data.pkl`
- `experiments/proxy/lasher_rpp_v1/proxy_test_manifest.json`
