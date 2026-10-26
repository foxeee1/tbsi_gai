# M1 融合前 MCCA 准入诊断协议（2026-10-12）

## 目的

判断是否有足够证据支持把 MCCA 类双路残差放到两路最终 search 特征与原始 `tbsi_fuse_search` 之间。当前只做准入诊断，不训练候选模块。

已知证据使该实验必须能够否定融合路线：72 条 train-side 序列的既有 search-stage probe 中，失配前驱/严重起点的 `fused` 特征相对 `concat` 探针 margin 更高约 11.54/12.18；32 个固定状态的消息位置诊断也没有找到稳定增益方向。因此，不能把“最终 search 特征弱”直接解释成“原融合器有害”。

## 预注册问题与假设

**主问题：**在真实闭环 crop 内的失配前驱和严重起点，是否反复出现一模态的目标证据优于其困难负峰、另一模态却更支持困难负峰，而且原始融合表示/定位读出低于较强单路？

- `H_FUSE`：上述跨模态冲突在独立 train-side 序列中重复出现，且原始融合后目标-困难负峰区分或定位低于较强单路。此时才支持继续研究融合前的条件化 MCCA。
- `H_UPSTREAM/READOUT`：两路单独已不可读，或融合输出不比较强单路差，或错误主要来自 CENTER 的选格/offset/size 读出。此时暂不支持 MCCA 融合改造，应回到 search 表征或定位读出。

此实验验证的是融合接口是否存在可利用缺口，不证明新增模块训练后必然提高完整跟踪指标。

## 实验单位与样本

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15；使用登记 checkpoint 与配置只读。
- 数据：LasHeR train-side。排除既有 search-stage、fusion-flow、candidate-recall、CENTER-readout、LCGA、MCCA、消息位置和相关固定状态诊断涉及的序列；排除规则及源清单哈希写入 manifest。
- Outcome-blind 选取最多 24 条 loader-compatible 序列，按目标面积、运动、RGB 亮度、TIR 局部对比平衡；若排除后不足 24 条，则纳入所有合格序列并把样本量限制写明。M1 训练序列暴露未知，故称“未参与既有诊断的 train-side cohort”，不称 M1 未见集。
- 每序列最多两个预先定义状态：一个正常参考帧；一个首个可见失配起点（连续 3 帧 IoU `<0.5`，且 GT 目标在当前 crop 覆盖率 `>=99.9%`）。没有满足条件的序列只贡献正常参考帧。GT 仅用于离线事件选择和评分。
- 基线先做完整闭环 trace，再在冻结的同一模板、状态、图像与 crop 上采集特征和输出；任何反事实都不得更新状态。报告 sequence-level 统计，帧不能当独立样本。

## 唯一诊断流程

1. 对 M1 每帧记录最终 `enc_rgb`、`enc_tir`、`tbsi_fuse_search` 输出、CENTER 原始 score、Hann score、选中格、offset/size 与解码框。核对固定状态 replay 的 M1 parity，checkpoint/config 哈希前后不变。
2. 用相同的序列分组交叉验证和预先固定的线性 probe 训练协议，分别评估 RGB、TIR、concat、fused 表征对 GT 目标支持区与困难负峰的区分。困难负峰定义为 GT 扩张区域之外的最高响应局部峰；probe 所有训练、标准化和阈值仅在训练折拟合。
3. 在原始 CENTER 输出上计算同一帧 Hann 后目标支持区与困难负峰的 margin，并报告实际 top-1、中心误差、offset/size 误差和 IoU。该项使用真实 M1 head，不用 GT 改写输入或挑选峰。
4. 对“较强单路可读、融合后变差”的帧做逐序列归因复核：检查该结论是否同时体现在 held-out probe margin 与真实 CENTER 定位读出；只出现在 probe 或只出现在 margin 的帧不计为融合失败证据。

不把 RGB-only/TIR-only 特征直接送进原始双路 Conv 当作主要证据，因为这会改变训练时的输入分布；不以 GT 区域模糊、清零或单模态强衰减得到的救回结果作为自然场景证据。

## 记录指标

- 主机制量：每个失配状态 RGB/TIR/concat/fused 的 held-out target-vs-hard-negative probe margin；融合相对较强单路的差。
- 原模型量：Hann 后 target-vs-hard-negative score margin、GT cell rank、top-1 是否落在目标支持区、预测中心误差、size L1、IoU。
- 分层：正常参考、失配前驱、严重起点；同时报告每序列配对值、冲突方向、序列胜率和按序列 bootstrap 区间。
- 工程量：trace 帧数、样本数、GPU 时长、固定状态 replay FPS（单独注明是否含 hook），模型/config/checkpoint 哈希、代码快照、数据路径、命令和完整分析输出。

## 决策门（项目筛选线，不是统计学定律）

只有同时满足下列条件，才允许实现一次最小融合前 MCCA 候选：

1. 至少 6 条不同序列出现同一方向的自然冲突；每条序列最多贡献 1 个失配状态进入主判定。
2. 冲突状态中，较强单路的 held-out probe margin 为正、另一模态为负；融合相对较强单路的 margin 下降，并且真实 CENTER 目标-困难负峰 margin 或中心/IoU 至少一项同向恶化。
3. 上述模式在至少三分之二冲突序列上复现，不能由单一序列或单一属性层驱动；正常帧没有显示普遍融合损伤。
4. 若提出“可靠性条件”，其输入必须是推理时可获得的信号；用留序列评估证明该信号能预测哪一路对目标-困难负峰更有利。GT、未来帧、测试集标签不得进入条件分支。

未通过任一项：结论写“暂不支持当前 MCCA 融合前候选”，不继续调 bottleneck、gate 或 loss。若两路都弱，归入上游表征问题；若 fused probe 不弱但 CENTER 错误，归入定位读出问题；若自然冲突存在但条件信号不可预测，当前门控方案尚未成立。

## 通过后的下一步（本协议不执行）

只实现一个候选：`enc_rgb, enc_tir` 进入原 `tbsi_fuse_search` 前，各自加一个小瓶颈双向残差 `phi(q,c,|q-c|)`，建议 bottleneck 32、输出投影零初始化；原 `tbsi_fuse_search`、CENTER、模板路径和 loss 不变。它属于融合前 search feature adapter，不是旧模板消息 MCCA，也不应称作显式 MHSA-message 控制。

进入短训前需通过 disabled/zero-output parity、shape/save-load、双路梯度、冻结 M1 参数与 BN eval/buffer 不漂移、小样本可拟合检查。比较只在新的固定 train-side dev 上做 adapter-only M1 对照；通过后再做同 parent、同 seed/步数的 G0/G1 matched continuation，并同时报告 `G1-G0` 与 `G1-M1`。本诊断通过仅代表允许进入该单变量实验，不代表论文主结果或创新性结论。

## 资源与解释边界

- 资源：单张 4090D、单诊断主进程；不启动 DataLoader 子进程，若实际脚本/loader 会创建 worker，启动前按顶层规则重新核算并发。开始执行前再次检查实际 GPU/CPU 资源占用。
- 不做训练、不写 checkpoint、不跑 test-244、不修改 M1 配置、权重或固定结果。
- 当前工作树代码可能与归档 M1 评测环境存在差异；复用 M1 权重的 replay 只作为当前工作树机制诊断，不是严格复现。
- 执行前已冻结本协议；实际清单、偏差和结果记录在下节。

## 执行结果（2026-10-12）

协议已执行到 MCCA 准入判定；没有训练候选模块。

- 首次 outcome-blind 名单 `m1_prefusion_mcca_admission_20261012` 在执行前审计时发现包含旧 MCCA validation/attribution 序列，未用于追踪或评分，标记为 `INVALIDATED-BEFORE-EXECUTION` 并保留清单。第二版把这些清单纳入排除源。
- 排除既有诊断 cohort、MCCA validation96 和 MCCA attribution 序列后只剩 15 条合格序列。按协议使用全部 15 条，没有放宽排除条件凑到 24。清单 SHA256 `0f655f6c66e5a5c092ded79c9d02000bcde24a665a7e70d24a8e1ca4b0f206a6`；序列清单 SHA256 `9d12f4...`（精确值见 manifest）。
- 固定 M1 闭环 trace 覆盖 15 条序列、6,167 跟踪帧，耗时 `179.15 s`；在同一状态采集 63 帧，覆盖 9 个 severe onset、9 个 pre-onset 和 39 个正常参考帧。重放与 trace 最大框 parity 误差 `4.96e-5 px`。配置/checkpoint 哈希与受保护值一致。
- 3-fold sequence-held-out probe 的 severe onset 结果：RGB/TIR/concat/fused target-vs-background margin 中位数分别为 `-9.51/+5.52/+3.69/-5.84`；目标区域 probe top rate 分别为 `3/9, 6/9, 5/9, 4/9`。probe margin 数值来自分别拟合的 head，只作描述；主要证据是每序列 top/readability 与同帧 CENTER 输出配对。
- 9 个 severe onset 中 5 条出现 RGB/TIR 单路 top 冲突；其中 3 条（`3girl1`、`blackruning`、`whitegirl2_0115`）是 TIR probe 指向目标、RGB 与 fused probe 未指向目标，且 CENTER GT rank 分别为 `4/4/8`、IoU 为 `0.026/0.287/0.000`。但其余 2 条冲突中，融合 probe 保留了目标 top；`4boysbesidesblkcar` 还出现相反的 RGB 有利冲突而 fused 仍保留目标 top。故属于少数可复现个案，不是全 cohort 普遍现象。
- severe onset 上 CENTER Hann target/background margin 中位数 `+0.218`、GT rank 中位数 `4`、IoU 中位数 `0.363`；强制 GT cell 的离线上界 IoU 中位数 `0.488`。这表明排序与回归读出仍混杂，不能把坏框全部归于融合表征。

**判定：`H_FUSE` 未达到预注册准入门。** 自然冲突 5/9，低于至少 6 条序列的项目门槛；其中 3 条同时表现出融合 probe top loss 与真实 CENTER 定位错误，但独立序列总数只有 9。当前证据支持“存在局部、值得进一步复核的 TIR 有用而融合后变弱案例”，不支持“原融合器普遍破坏目标证据”，也不支持直接启动 pre-fusion MCCA adapter-only 训练。对最终结果更稳妥的归类是 **尚未区分 / 不准入训练**，不能写成 MCCA 失败或融合创新已成立。

### 失败归因与后续边界

- 已排除：本次不是零输出、梯度或训练合同问题，因为本轮没有候选模块和训练。
- 暂不支持：通用双向 pre-fusion MCCA 能修复 M1 的主要定位失败；当前少量冲突方向不一致，且 probe 与 CENTER 排序/回归不总一致。
- 尚未区分：TIR 支持信号是否能由推理可用量可靠预测；3 个融合 top-loss 案例是否会在更大独立 cohort 重现；CENTER 选格或 offset/size 哪个限制最终框更多。
- 下一步不进入训练。若继续融合路线，先做一次独立小样本复核，只聚焦三个方向：冻结同一 crop 下的 TIR 目标/困难负峰响应、融合后实际 CENTER 是否失去该响应、是否存在可部署的无 GT 质量信号。通过后才实现零输出初始化的融合前 adapter，并加容量相近的 self-only 对照；若不能重复，则停止 MCCA 融合前分支。

## 实验身份与产物

- ID：`M1-PREFUSION-MCCA-ADMISSION-DIAG-V2-20261012`；Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`。追踪时使用了带本实验 ID 参数的 runner 内容；执行后发现该入口原文件是工作树中的未跟踪资产，已恢复其原 SHA256 `fee4c4bb5e68bda0390bd651e8ec1c09d56f02add81eee07d441da4d4092ebba`。等价的独立副本 `tools/run_m1_prefusion_mcca_admission_trace_20261012.py`（SHA256 `9e545581d40ef661f1197139c38c4a03dd8e21ff91c4f8b944b7fab6c9bde44d`）和完整相关源快照 `experiments/snapshots/m1_prefusion_mcca_admission_v2_sources_20261012.tar.gz`（SHA256 `4d8438be62e5e0ac94277794da55384255bcc1f69f2ee293f9bd4416d88b1128`）已保留用于复核。
- 受保护配置 SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，前后未变化。
- 数据清单：`experiments/diagnostics/m1_prefusion_mcca_admission_v2_20261012/manifest.json`；序列清单同目录 `sequence_list.txt`，SHA256 `b826b45018150bec8e91cb056cf0c02388b40f6cb2d5376799232dbb445eadc7`。
- 结果：`output/experiments/M1_PREFUSION_MCCA_ADMISSION_DIAG_V2_20261012/trace_run/`、`feature_capture/`、`probe/`、`analysis/`。逐状态合并表 `analysis/joined_per_state.csv`，完整汇总 `analysis/summary.json`。
- 资源：1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1；trace+feature capture 合计约 `0.052 GPU小时`；probe/聚合为 CPU 后处理。FPS、AUC/PR/NPR/OP50/OP75 不适用。
