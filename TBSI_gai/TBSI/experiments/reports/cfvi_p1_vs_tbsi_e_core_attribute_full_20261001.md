# CFVI P1 与 TBSI-E：核心代码和 full-244 属性对比

## 结论

P1 是 TBSI-E 的单点 Value 注入扩展，不是另一套融合主干。它保留 TBSI-E 的 template-only joint MHSA，仅在 `TBSI location=9` 的 Value 上加入固定 Haar-CFVI 残差。full-244 结果总体持平：P1 AUC 56.20，TBSI-E M1 AUC 56.21，不能称为整体增益。

属性结果显示，P1 在 `NO`、`DEF`、`BC`、`SA`、`SV`、`ARV` 上的 AUC 相对 M1 为正，在 `LI`、`HI`、`AIV`、`OV` 上为负；但这些是同一测试集上的描述性分组，属性多标签，且小组如 AIV/OV 只有 7 条序列，不能据此声称属性专项有效。

## 核心代码修改

| 位置 | TBSI-E | P1 实际变化 |
|---|---|---|
| 融合主干 | `TBSI_FUSION: tbsi_e`，template-only joint MHSA | 不变 |
| Q/K | 标准 `qkv` 生成和 attention logits | 不变 |
| Value | 原始 `v` | `v_rgb += lambda_rgb * norm(proj(Haar(tir)))`；`v_tir += lambda_tir * norm(proj(Haar(rgb)))` |
| 频率处理 | 无 | 2x2 token block Haar，输出 `LL/LH/HL/HH`，四带拼接后 1x1 Conv、GELU、LayerNorm |
| 作用位置 | TBSI-E 的 3/6/9 模板融合层 | 仅 location 9，即 `backbone.tbsi_layers.2` |
| 模态范围 | RGB/TIR joint template MHSA | 双向跨模态 Value 注入；不修改 search token |
| 新参数 | 无 CFVI 参数 | 2,360,834 个参数：4D-to-D Conv 2,359,296，LayerNorm 1,536，两个 lambda |
| 训练协议 | SOT pretrain，15 epoch，batch 16，累积 8 | 相同；P1 仅把 `CFVI_MODE` 从 `off` 改为 `fixed`，并设置 `CFVI_LOC=[9]` |

对应代码：

- Haar 与投影：[cfvi_module.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/cfvi_module.py:13)
- 固定四频带注入：[cfvi_module.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/cfvi_module.py:106)
- CFVI 实例化与 Value 残差：[tbsi_layer.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py:53)
- `v` 注入位置：[tbsi_layer.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py:98)
- location 过滤：[vit_tbsi_care.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/tbsi_track/vit_tbsi_care.py:200)
- P1 配置：[CFVI_P1_FULL_MATCHED_M1_seed42.yaml](/root/autodl-tmp/TBSI_gai/TBSI/experiments/tbsi_track/CFVI_P1_FULL_MATCHED_M1_seed42.yaml:15)

代码中还增加了 route probe、attention diagnostics、template expert、bridge-pruner 等通用 feature gate；它们在 P1 配置中关闭，不属于本次 P1 的有效模型差异。`CFVI_MODE=off` 时保留 TBSI-E 路径，故不能把这些关闭的工具代码算作 P1 方法贡献。

## Full-244 总体对比

| 指标 | P1 | TBSI-E M1 | P1 - M1 |
|---|---:|---:|---:|
| AUC | 56.20 | 56.21 | -0.01 |
| OP50 | 68.38 | 68.33 | +0.05 |
| OP75 | 47.01 | 47.13 | -0.12 |
| Precision | 70.39 | 70.21 | +0.18 |
| Norm Precision | 66.43 | 66.20 | +0.23 |

## 属性分组对比

指标均按当前仓库 244 条测试序列计算。`d` 表示 P1 - M1，单位为百分点。属性来自冻结的 `TBSI_gai/data/lasher/Attributes&Order.zip`，每条序列可属于多个属性，因此各行不可相加。

| 属性 | n | P1 AUC | M1 AUC | dAUC | dOP50 | dOP75 | dPR | dNPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NO | 33 | 74.75 | 73.07 | +1.68 | +1.82 | +1.81 | +2.13 | +2.35 |
| PO | 214 | 53.71 | 53.73 | -0.01 | +0.09 | -0.13 | +0.21 | +0.20 |
| TO | 86 | 50.87 | 50.81 | +0.06 | +0.24 | -0.19 | +0.51 | +0.30 |
| HO | 16 | 53.68 | 53.60 | +0.08 | +0.51 | -0.15 | -0.16 | +0.54 |
| MB | 88 | 49.37 | 49.47 | -0.10 | +0.03 | +0.12 | +0.22 | -0.00 |
| LI | 51 | 48.42 | 50.06 | -1.64 | -2.05 | -0.82 | -1.73 | -1.51 |
| HI | 34 | 58.29 | 60.14 | -1.85 | -2.73 | -0.79 | -2.95 | -2.49 |
| AIV | 7 | 45.46 | 47.71 | -2.25 | -1.72 | -0.85 | -3.21 | -1.76 |
| LR | 68 | 45.35 | 45.75 | -0.40 | -0.13 | -0.74 | -0.11 | -0.15 |
| DEF | 46 | 58.88 | 56.28 | +2.59 | +3.57 | +2.31 | +3.16 | +3.65 |
| BC | 77 | 54.88 | 53.75 | +1.13 | +1.80 | +0.74 | +2.03 | +1.83 |
| SA | 106 | 49.76 | 49.04 | +0.72 | +1.09 | +0.77 | +1.24 | +1.06 |
| CM | 132 | 54.29 | 54.83 | -0.54 | -0.69 | -0.54 | -0.71 | -0.50 |
| TC | 147 | 49.96 | 49.80 | +0.15 | +0.49 | +0.16 | +0.36 | +0.57 |
| FL | 20 | 48.27 | 47.47 | +0.80 | +1.29 | -0.54 | +1.59 | +0.92 |
| OV | 7 | 58.22 | 60.64 | -2.41 | -2.69 | -2.20 | -3.00 | -2.40 |
| FM | 192 | 55.47 | 55.37 | +0.10 | +0.09 | -0.03 | +0.37 | +0.32 |
| SV | 221 | 56.19 | 55.76 | +0.43 | +0.61 | +0.26 | +0.71 | +0.74 |
| ARV | 108 | 52.38 | 51.78 | +0.60 | +0.82 | +0.57 | +0.73 | +0.98 |

## 属性解释

最明显的正向组是 `DEF` (+2.59 AUC, n=46)、`NO` (+1.68, n=33) 和 `BC` (+1.13, n=77)。最明显的负向组是 `OV` (-2.41, n=7)、`AIV` (-2.25, n=7)、`HI` (-1.85, n=34) 和 `LI` (-1.64, n=51)。大样本组 `PO` 基本持平，`FM`、`SV`、`ARV` 只有小幅正差。

这些结果只能作为 full-244 上的描述性异质性：没有按属性重新训练或挑选模型，也没有序列级 bootstrap 的属性置信区间；因此不能把 `DEF` 或 `NO` 写成已验证的适用场景。总体配对胜率仍为 47.95%，序列 bootstrap 95% CI 为 `[-0.012137, +0.012972]`。

## 证据路径

- P1 eval：[eval_data.pkl](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/P1/analysis/eval_data.pkl)
- M1 eval：[eval_data.pkl](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/M1_reference/analysis/eval_data.pkl)
- 配对结果：[paired_delta.csv](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/paired_delta.csv)
- P1 checkpoint：[TBSITrack_ep0015.pth.tar](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/CFVI_P1_FULL_MATCHED_M1_seed42/checkpoints/TBSITrack_ep0015.pth.tar)
