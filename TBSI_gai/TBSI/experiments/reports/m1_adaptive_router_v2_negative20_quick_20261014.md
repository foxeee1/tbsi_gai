# M1 disagreement-aware selective router：负收益 20 序列快速实验

## 实验目的

验证在 Cdiag full-244 退化序列上，加入 RGB/TIR disagreement-aware gate、降低 residual 上限并保持 no-op 初始化，能否减少错误介入并恢复 M1。

## 合同

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15，seed42。
- Candidate：`M1_ADAPTIVE_POST_ROUTER_V2_NEG20_QUICK_seed42`，独立配置和输出目录。
- 训练：LasHeR train 64 samples，1 epoch，batch2，worker4；只训练 adaptive router 的 12 个参数。
- 测试：固定 20 条 full-244 中 AUC 退化最大的序列；candidate 与 M1 使用同一 manifest、FP32、同一官方 evaluator。
- 结构：上下文加入 `abs(RGB-TIR)`；`MAX_RESIDUAL=0.05`；`GATE_BIAS=-4.5`；M1 主干冻结。

## 五指标

| 指标 | M1 baseline | V2 candidate | Delta |
|---|---:|---:|---:|
| AUC | 44.8731 | 41.8348 | -3.0382 |
| Precision | 57.31 | 53.07 | -4.24 |
| Norm Precision | 51.24 | 47.98 | -3.26 |
| OP50 | 52.8040 | 49.2992 | -3.5048 |
| OP75 | 27.9935 | 26.4413 | -1.5522 |

逐序列 AUC：candidate 胜 7、负 13、平 0。逐帧可观测输出变化中，救援帧 1,175，伤害帧 1,625；改变帧平均 IoU delta 为 `-0.03394`。candidate 的 persistent segment 总数比 M1 少 2，但最长失败段由 `314` 增至 `351`，整体失败率仍恶化，不能视为恢复收益。

## 逐序列分层

| baseline AUC 桶 | 数量 | 胜/负 | mean AUC delta |
|---|---:|---:|---:|
| `<30` | 3 | 0/3 | -1.652 |
| `30-60` | 13 | 5/8 | -4.144 |
| `>=60` | 4 | 2/2 | -0.486 |

最大改善：`boyride2path` `+4.011`、`boy\`headwithouthat` `+2.546`、`rightblkfatboyleftwhite` `+1.456`。最大退化仍集中在部分长序列，说明 gate 虽减少了部分干预，但没有获得可靠的“何时介入”判别能力。

## 归因

- 结构合同通过：M1 checkpoint 正常加载，仅新增 router 参数缺失；训练期间无主干解冻。
- 训练 loss 和 IoU 有限，说明 forward/backward 可运行。
- V2 的 disagreement 输入没有转化为可部署的可靠 gate：负收益子集上 changed fraction 仍约 `0.847`，救援/伤害比例约 `1175/1625`。
- 因此失败原因不再是单纯的 residual 幅度过大，而是训练目标没有提供“保持 M1”与“介入修复”的 utility/no-op 监督。普通 tracking loss 会驱动 router 改善当前训练样本，却不会约束长期闭环风险。

## 决策

`INCONCLUSIVE-NEGATIVE / V2 NOT ADMITTED`。disagreement-aware gate 这次快速训练没有修复负收益序列，不能进入 full training，也不继续堆叠 gate 结构。下一步若继续，只做 train-side、sequence-held-out 的 utility/no-op gate 可分性实验；需要先证明 gate 能区分有益 intervention、伤害 intervention 和 no-op，再进入新的联合 readout/state-recovery 模块。

## 证据

- 配置：`experiments/tbsi_track/M1_ADAPTIVE_POST_ROUTER_V2_NEG20_QUICK_seed42.yaml`
- 负收益清单：`experiments/manifests/m1_cdiag_negative20_20261014.txt`
- 训练输出：`output/experiments/M1_ADAPTIVE_POST_ROUTER_V2_NEG20_QUICK_20261014/`
- 官方审计：`output/experiments/M1_ADAPTIVE_POST_ROUTER_V2_NEG20_QUICK_20261014/official_audit/official_metric_and_persistence.json`
- 逐序列分析：`output/experiments/M1_ADAPTIVE_POST_ROUTER_V2_NEG20_QUICK_20261014/analysis/`
