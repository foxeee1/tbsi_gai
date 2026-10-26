# M1 STSER competition descriptor quick3

日期：2026-10-24  
研究 parent：`TBSI_E_MHSA_FULL_4090D_seed42` epoch15  
直接 control：`M1_STSER_QUICK3_20261024`（STSER v1）

## 研究问题

STSER v1 为每个 search token 独立计算 shared/private/null 路由。它没有使用其他 search 候选作为参照，因此尚不能称为候选竞争路由。本实验只增加一个竞争描述：对每个 search token 排除自身和 3x3 邻域，在同模态 search 特征中取 top-4 相似位置的加权池化，并将 `query - competitor` 送入原 router。T2S attention、专家、null route、gamma、CENTER 损失和训练进度均保持不变。

假设：当 crop 中存在目标与强干扰候选时，当前位置相对于非邻域候选的差异证据能改善消息写入，尤其改善高 IoU 阈值下的框定位。可证伪信号是候选在 OP75 上相对 STSER v1 稳定提高；反例是竞争路由只改变均值而不改善逐序列高阈值指标，或产生明显延迟成本。

## 合同与实现

- 新模式：`SEARCH_CONDITIONED_T2S_MODE=stser_competition`。
- 输入：`[B,256,768]` search token；输出形状不变。
- 竞争候选：同模态 cosine 相似度，排除自身与 3x3 邻域，top-k=4；索引只依赖输入，不读 GT。
- 新增路由量：`S_i - C_i`，其中 `C_i` 是竞争位置的温度加权特征池化。
- 初始化：外层 gamma 仍零初始化，保持 M1 初始化恒等行为；随机输入 forward/backward 和 full model 构建通过。
- 更新证据：候选 checkpoint 的 gamma 范数为 `0.001766`，`out_rgb` 权重范数为 `15.366`，competition router 首层权重范数为 `7.673`；新增路径确实被优化器更新。
- 训练：RPP-LasHeR 176 序列，10,112 samples/epoch，3 epochs，237 updates，seed42，batch16、accumulation8、worker4、AMP；训练第一次在保存阶段因磁盘满失败，失败日志保留，随后同一配置重跑成功。
- 测试：RPP-v2 锁定 60 序列，97,908 frames，FP32，serial `threads=0`，template update off。

## 配对结果

| 指标 | STSER v1 | Competition | 差值 | 95% sequence bootstrap CI | 胜/负/平 |
|---|---:|---:|---:|---:|---:|
| AUC / Success | 51.991 | 52.148 | +0.157 | [-0.538, +0.906] | 31/28/1 |
| Precision | 63.477 | 63.503 | +0.026 | [-0.874, +0.964] | 29/26/5 |
| Norm Precision | 60.468 | 60.692 | +0.224 | [-0.559, +1.127] | 24/28/8 |
| OP50 | 61.788 | 61.993 | +0.205 | [-0.624, +1.148] | 26/26/8 |
| OP75 | 43.720 | 44.302 | +0.583 | [+0.011, +1.264] | 36/17/7 |
| mean IoU | 0.52512 | 0.52684 | +0.00172 | [-0.00530, +0.00940] | 31/29/0 |

候选相对 plain T2S 的绝对值为 AUC `52.148`、Precision `63.503`、Norm Precision `60.692`、OP50 `61.993`、OP75 `44.302`；相对 plain 的 AUC 差值约 `+0.361`，仍未达到预设 `+0.5` 点准入线。记录的总推理速度为 STSER v1 `62.42 FPS`、Competition `58.06 FPS`，约下降 7.0%。

## 逐序列观察

AUC 最大增益：`midrunboywithwhite +11.883`、`carcominginlight +9.923`、`shinybikeboy2left +5.317`。AUC 最大退化：`leftmirrorside -7.038`、`ballshootatthebasket3times -6.107`、`carcomingfromlight -5.091`。这些序列名不能直接作为场景因果标签；它们只用于后续固定状态回放和候选互换审计。OP75 的正向均值主要来自相同的高增益序列，但仍需检查是否由少数长序列主导。

稳健性补充：按序列帧数加权后，AUC 差值为 `+0.026`、OP75 差值为 `+0.443`；AUC 去掉绝对变化大于 3 点的 10 条极端序列后均值为 `+0.024`。这说明宏平均 OP75 的正向方向存在，但整体收益仍受序列异质性影响。

## 结论

`PROXY-COMPLETE / CONTRACT-PASSED / LEARNABILITY-PASSED / OP75 INITIAL-SUPPORT / AUC CI-CROSSES-ZERO`。

竞争描述带来了可观察的高阈值定位正信号，说明相对候选证据比单位置模态一致性更有潜力；但总体 AUC 不稳定，延迟增加，不能直接启动 full training 或宣称稳定涨点。下一步应做同一固定 crop/state 的 target-vs-shifted competitor 反事实回放，检查 OP75 增益是否伴随 GT candidate 与 hard-negative 的 rank、CENTER heatmap 和消息范数改善。若竞争输入在这些指标上没有目标特异方向，停止扩大算子；若方向一致，再只调整一个变量（竞争 top-k 或受限写入强度）后做更长训练。

## 证据

- Config：`experiments/tbsi_track/M1_STSER_COMPETITION_QUICK3_20261024.yaml`，SHA256 `b76d310679633c5c67489cb0f8d1748ac339d8b1ad4c1a5c7c3a42ae7b77011e`。
- Module：`lib/models/layers/search_conditioned_t2s_router.py`，SHA256 `9e7a1599d818c57c27d5863505345e4fedda71d83f37dc8b40e1728d6452a3fc`。
- Checkpoint：`output/experiments/M1_STSER_COMPETITION_QUICK3_20261024/checkpoints/TBSITrack_ep0003.pth.tar`，SHA256 `61f36630a3a3ff0d7c032bde9a7046860c1c8676722d57702ae450f1c7658f25`。
- Summary：`output/experiments/M1_STSER_COMPETITION_QUICK3_20261024/paired_rpp60_summary.json`。
- Per sequence：`output/experiments/M1_STSER_COMPETITION_QUICK3_20261024/paired_rpp60_per_sequence.csv`。
- Logs/provenance：`output/experiments/M1_STSER_COMPETITION_QUICK3_20261024/logs/`、`experiments/snapshots/m1_stser_competition_quick3_20261024/`。
