# M1 STSER competition SOT 初始化 full15 结果

> 2026-10-25 归因更正：下文为历史初稿，性能数值仍有效，但“从SOT开始是相对M1的合同混杂”“负gamma/小写回证明失效”“有非零梯度即机制有效”等解释已更正。请以[详细归因与算子设计](m1_stser_sot_full15_causal_attribution_and_operator_design_20261025.md)及本文末尾更正为准。历史版本已存档，未修改原始日志和结果。

日期：2026-10-25（UTC）  
实验：`M1_STSER_SOT_FULL15_COMPETITION_20261025`  
Parent：`TBSITrack_SOT_Pretrained.pth.tar`；比较锚点为受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15

## 结论

本次实验完成了从 SOT 预训练权重开始的 LasHeR 全量 15 epoch 训练和 full-244 测试。相对受保护 M1，五项指标均为小幅下降：AUC/Success `55.98`（`-0.23`）、OP50 `68.05`（`-0.28`）、OP75 `46.89`（`-0.24`）、Precision `70.03`（`-0.18`）、Norm Precision `66.00`（`-0.20`）。因此本运行不支持“STSER competition 带来全量涨点”的论文主张，也没有达到预设的 `+0.5` 涨点目标。

该结果仍然给出有效机制证据：STSER 分支在训练图中可学习，router、experts、T2S attention 和 outer 参数都有非零梯度；路由分布发生了明显变化；但它对 search 特征的写回幅度很小，且 TIR 分支末期出现负 gamma。性能下降不能完全归因于 STSER，因为本运行相对 M1 是从 SOT 权重独立初始化并进行完整 15 epoch 训练，训练合同和参数轨迹与 M1 不同。结论分类为：`FULL-244 COMPLETE / AGGREGATE POINT-DROP / MECHANISM-SUPPORTED / OPERATOR-CAUSALITY-INCONCLUSIVE`。

## 训练与评测合同

- 初始化：`pretrained_models/TBSITrack_SOT_Pretrained.pth.tar`，SHA256 `1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b`；无 `INIT_CHECKPOINT`，未继承 M1 optimizer/scheduler state。
- 训练：LasHeR full train，`59,904` samples/epoch，15 epochs，3,744 batches/epoch，batch 16，gradient accumulation 8，有效 batch 128，seed 42，fresh AdamW，StepLR 在 epoch 10 衰减，4 workers，AMP，layer 0 冻结。
- 测试：LasHeR full-244，FP32，`threads=0`，template update off，template 128/factor 2，search 256/factor 4；结果文件为 244 个预测文件和 244 个时间文件。
- 环境：1x NVIDIA GeForce RTX 4090 D，Python 3.8.20，PyTorch 1.9.0+cu111，CUDA 11.1。
- 代码快照：Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；实际代码/配置哈希、命令和时间见 `experiments/snapshots/m1_stser_sot_full15_competition_20261025/provenance.txt`。
- 配置 SHA256：`3dbc8b7e34ba07b9cc4a6d07e4a5e4ec8c475319233188bc0073941af671044b`。
- checkpoint SHA256：`0cf5cf46781d2741bb5134cb69b99fbaf3013f7c273c4e6e9932595140189bd5`。
- 训练时间：2026-10-25 03:01:41--06:33:37 UTC；测试和自动分析完成于 08:38:22 UTC；总 GPU wall time 约 5.61 h。

## 与 M1 的同口径点估计

| 指标 | M1 protected | STSER SOT full15 | 差值（STSER-M1） |
|---|---:|---:|---:|
| AUC/Success | 56.21 | 55.98 | -0.23 |
| OP50 | 68.33 | 68.05 | -0.28 |
| OP75 | 47.13 | 46.89 | -0.24 |
| Precision | 70.21 | 70.03 | -0.18 |
| Norm Precision | 66.20 | 66.00 | -0.20 |

两组均记录为 LasHeR full-244、FP32、`threads=0`、template update off、4090D 环境，因此可以做固定 aggregate 的描述性比较。M1 的逐序列预测目录曾被后续运行改写，不能从现存文件生成有效的 M1-STSER 配对 CI、序列胜负或帧级恢复统计；本报告不把这些不可复核的配对结果当作证据。

## 方法与信息流

STSER competition 保留原有 T2S 和 shared/private/null 专家，只改变路由输入。对每个 search token，排除自身及 3x3 邻域，在同模态 search token 中取 top-4 非邻域竞争位置，形成 `query - competitor` 描述；该描述与 T2S 消息进入 router，router 决定消息进入 shared、private 或 null 专家；最后以有界 gamma 残差写回 search 特征，再进入 CENTER head。训练目标仍是原有 GIoU、L1 和 location loss，没有增加竞争排序监督。

这条链路在代码和日志中实际执行，关键证据为 `logs/stser_flow.csv`、`logs/stser_gradient_flow.csv` 和 `logs/feature_changes.csv`。它没有直接修改 CENTER 排序规则，也没有启用模板更新。

## 训练期机制观测

- `stser_gradient_flow.csv` 共 1,080 条记录。router 组 8/8 tensor、experts 18/18 tensor、T2S attention 6--8 tensor、outer 3/3 tensor 均出现非零梯度。router 平均梯度范数约 `3.88e-7`，experts 约 `1.44e-5`，T2S attention 约 `4.15e-5`，outer 约 `1.30e-4`。这证明分支可学习，但 router 梯度很小。
- 路由早期为混合状态；训练末期 RGB 概率约为 shared `0.99997`、private `1.3e-5`、null `1.3e-5`，TIR 概率约为 shared `2.0e-8`、private `0.999998`、null `2.1e-6`。模型最终选择“RGB 共享消息、TIR 私有消息”的近确定性策略，而不是持续使用三路互补融合。
- `feature_changes.csv` 显示训练末期 RGB 相对写回 `5.31e-5`，TIR 相对写回 `1.41e-3`；对应 gamma 为 `0.00255` 和 `-0.01816`。因此 TIR 写回方向为负且仍小于输入特征范数的千分之二，STSER 对后续 CENTER 输入的直接扰动有限。
- 全模型梯度日志每个采样点总范数均为 `0.1`，说明触发了全局梯度裁剪；每次有 2/265 参数无梯度，但 STSER 专项 tensor 仍有非零梯度。epoch15 训练末端 loss `0.86201`，GIoU `0.21423`，L1 `0.01858`，location `0.34063`，IoU `0.80016`，这些是训练 batch 统计，不是测试指标。

## 序列级分布

候选自身的 244 序列指标已保存到 `output/experiments/M1_STSER_SOT_FULL15_COMPETITION_20261025/analysis_lasher244/per_sequence_metrics.csv`。AUC 低于 20 的序列有 25 个，AUC 不低于 70 的序列有 84 个。最低 AUC 包括 `basketball849` 2.67、`1strowleftboyturning` 4.12、`ab_pingpongball2` 4.22、`rightcar-chongT` 4.52、`2girlup` 5.78；最高 AUC 包括 `darkcarturn` 91.04、`silvercarturn` 89.53、`boyruninsnow` 89.22、`whitecarturnleft` 88.55、`carturn117` 87.75。由于缺少可验证的 M1 同序列预测，这些只能用于候选内部的失败场景定位，不能称为相对 M1 的改善序列。

## 归因边界与下一步

最直接的事实是：竞争描述改变了路由，路由改变了模态消息分配，但末期特征写回很弱，full-244 没有形成正向指标传导。当前最有根据的机制解释是“search-search 自相似竞争”没有明确对应模板目标与干扰物的竞争，且路由最终塌缩为模态固定策略；它更像是在选择消息类型，而不是把目标/干扰差异写回 search token。

仍需保留的替代解释是训练合同：本次从 SOT 预训练权重直接训练，M1 是另一条训练轨迹；因此不能把 `-0.23` 全部归因于算子。若继续研究，应先做同一 SOT 初始化、同一 full-data 15 epoch schedule 的 plain T2S control；只有 control 接近当前结果时，才有资格把下一版目标条件竞争残差与当前 STSER 比较。下一算子应从模板目标支持分数选择 hard-negative 竞争池，并让 `target evidence - hard-negative evidence` 直接进入有界残差；第一版沿用原损失，避免竞争表征和新监督同时变化。

## 证据路径

- 配置：[M1_STSER_SOT_FULL15_COMPETITION_20261025.yaml](../tbsi_track/M1_STSER_SOT_FULL15_COMPETITION_20261025.yaml)
- provenance：`experiments/snapshots/m1_stser_sot_full15_competition_20261025/provenance.txt`
- checkpoint：`output/experiments/M1_STSER_SOT_FULL15_COMPETITION_20261025/checkpoints/TBSITrack_ep0015.pth.tar`
- 训练与诊断：`output/experiments/M1_STSER_SOT_FULL15_COMPETITION_20261025/logs/`
- full-244 tracking：`output/test/tracking_results/tbsi_track/M1_STSER_SOT_FULL15_COMPETITION_20261025/`
- 序列分析：`output/experiments/M1_STSER_SOT_FULL15_COMPETITION_20261025/analysis_lasher244/`

## 2026-10-25 追加归因更正

1. M1和本组均从同一SOT预训练开始，名义full15训练设定匹配；本组不是从M1 epoch15续训。剩余训练混杂是公共新参数初始化没有逐tensor配对保证及单seed差异，不能笼统称SOT起点不同。
2. quick3 control 的真实mode是`stser`，即无competition的STSER；并非`plain_t2s`或M1。因此quick3的+0.670不能直接作为相对M1的涨点。
3. 已有270个采样batch支持训练路由早期饱和：epoch2起RGB几乎固定shared，TIR后期几乎固定private。竞争输入只通过router进入输出，并未直接进入Value内容。这个结构缺口和路由变化是实证；它们解释全部-0.23的因果效力尚未确认。
4. 负gamma不是错误方向的证据；小的全局写回不能证明不影响目标token或闭环；实现是先限幅residual再乘gamma，gamma自身没有有界约束。
5. 全模型梯度记录在unscale和clipping后，不能从0.1恒定归因于裁剪；并非所有STSER tensor在所有采样点都有非零梯度。部分CSV字段是占位或表头错位，不能用作精确机制结论。
6. 名义7,020次更新机会，checkpoint有state的263个参数tensor均记录Adam step=7,016；4次差异的成因未确定，不能据此指认掉点。
7. 候选自身244序列的已有轨迹几何统计：1,063段连续≥5帧IoU<0.2的失败中，1,049段起点仍有≥90%GT框位于crop内。几何在框内不等于可见，也不能推断是相对M1新增失败，更不能直接认定MSHA或CENTER是首因。
8. 当前准确状态为`FULL-TRAIN/FULL-244 COMPLETE / AGGREGATE POINT-DROP / ROUTER-SATURATION-OBSERVED / TRACKING-CAUSE-UNRESOLVED`。下一步是独立router温度诊断及目标条件内容修正的设计/消融，不要求先补做另一条昂贵full15对照才能继续研究。本轮只离线分析，没有训练或跟踪测试。

补充报告：`experiments/reports/m1_stser_sot_full15_causal_attribution_and_operator_design_20261025.md`。逐epoch、已有轨迹统计及输入manifest：`output/experiments/M1_STSER_SOT_FULL15_COMPETITION_20261025/attribution_assets_only_20261025/`。
