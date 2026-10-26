# M1 STSER competition full15 结果与 M1 系列对比

日期：2026-10-25（Asia/Shanghai）  
实验：`M1_STSER_COMPETITION_FULL15_FROM_M1_20261024`  
Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15

> **归因校正（2026-10-25）**：STSER full-244 的记录 aggregate 相对 M1 固定 aggregate 五项均低，这是高度重合名义评测协议下的描述性点差；但不构成严格配对显著性或算子因果结论。旧 M1 预测目录已被后续运行改写，候选报告中的 parent 列、bootstrap 和胜负统计失效；STSER 还从 M1 epoch15 联合续训了15个 epoch，缺少同条件无 STSER 续训对照。

## 结论

全量训练和 full-244 测试已完成。competition full15 的官方 LasHeR test 指标为 AUC `55.93`、OP50 `67.88`、OP75 `46.64`、Precision `70.19`、Norm Precision `65.97`。与 M1 固定指标 `56.21/68.33/47.13/70.21/66.20` 相比，点差分别为 `-0.28/-0.45/-0.49/-0.02/-0.23` 个百分点。可确认登记点估计下降；无法从现有文件确认逐序列统计显著性，也不能把它单独归因给 STSER。

这是当前配置与单 seed 的结果，不证明所有竞争描述都无效。quick3 是 RPP-v2 60 序列、相对 plain T2S 的 proxy 比较；full15 是 LasHeR full-244、相对 M1 固定 aggregate 的比较，两者的对照和范围不同，因此 full15 不能直接反驳 quick3 的 `+0.583 OP75`。机制解释仍有限：训练记录没有采集到 STSER 专属梯度和逐层消息/竞争特征变化，不能据此解释为何性能下降。

## 实验合同与数据

- 从 M1 epoch15 checkpoint 初始化；seed 42；LasHeR full train，59,904 samples/epoch，15 epochs，每 epoch 3,744 batches，有效 batch 128。
- 现有 provenance 没有保存每 epoch 实际抽取的训练样本 manifest；因此仅匹配 seed 和样本总数，不足以证明与其他运行看到完全相同的 sample order。
- 优化器 AdamW，LR `1e-4`，backbone multiplier `0.1`，GIoU/L1/location loss 权重按 M1 合同，AMP 开启；训练使用 1× NVIDIA GeForce RTX 4090 D、4 个 DataLoader worker。
- LasHeR test full-244，219,292 frames，FP32、`threads=0`、template update off；全部 244 个结果文件存在。测试耗时约 2小时05分，测得 FPS 约 61.76。
- checkpoint SHA256：`bb81b1b32015fc478ce6a4077f0a9d3d37210165b1818baf67defb5332dc1a52`。
- config SHA256：`86126be62ded2a3d35facff6d1e6a8e298d9b42994045d10dae71263408da4d1`。
- 运行代码记录为 Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`，实际工作树含研究代码改动；结果仅对应 provenance 记录的该工作树。

## 与 M1 系列 full-244 结果对比

下表按 AUC、OP50、OP75、Precision、Norm Precision 排列。候选指标来自各自报告记录；不同初始化、训练量和优化合同不等同于受控横向消融。

| 实验 | 训练/初始化合同 | AUC | OP50 | OP75 | Precision | Norm Precision | 相对 M1 AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 protected | M1 epoch15 固定锚点 | 56.21 | 68.33 | 47.13 | 70.21 | 66.20 | -- |
| AMCM E2 10EP M1MATCH | 从 M1 初始化，full-data 10EP，M1-matched run | 56.09 | 68.18 | 47.06 | 69.99 | 66.15 | -0.12 |
| Cdiag full15 | 从 M1 初始化，full train 15EP，4 个 residual 参数训练 | 56.18 | 68.42 | 47.18 | 70.20 | 66.30 | -0.03 |
| STSER competition full15（本次） | 从 M1 初始化，full train 15EP，competition T2S router 联合训练 | 55.93 | 67.88 | 46.64 | 70.19 | 65.97 | -0.28 |
| N-gram + margin full15 | 从 SOT 预训练初始化，full train 15EP | 54.94 | 66.83 | 46.57 | 68.26 | 64.72 | -1.27 |
| AMCM E2 seed3407 | seed3407、10EP；second-seed check | 56.13 | 68.26 | 47.15 | 70.08 | 66.24 | -0.08 |

补充：早期 AMCM E2 seed42 报告曾记录 `56.68/69.03/47.60/70.83/66.88`，但后续 M1-matched full-data 10EP 记录为 `56.09`，seed3407 为 `56.13`。因此旧 seed42 正向结果属于合同/版本不同的历史结果，且没有被第二 seed 确认，不应作为当前最佳稳定组。

Early-conditioned MSHA gate full15 joint run 的原始 aggregate 曾报小幅正向，但 2026-10-24 provenance audit 发现现存 M1 prediction 文件与受保护锚点不一致，并判定其配对序列分析为 `ARCHIVED-ANALYSIS-ONLY / REPLAY-REQUIRED`。本报告不把该组列为可确认的直接性能排名依据。

## 序列分布与配对限制

本次候选 244 序列自身的 AUC 最低项包括 `basketball849` 3.16、`leftgirl'swhitebag` 3.23、`ab_pingpongball2` 3.29、`1strowleftboyturning` 3.52；最高项包括 `darkcarturn` 90.99、`silvercarturn` 89.91、`boyruninsnow` 89.40、`carturn117` 88.48。它们只描述候选绝对表现，不表示相对 M1 的改善或退化。

为生成诊断而将当前 M1 输出目录与候选输出重新配对后，M1 当前文件重算指标为 `56.77/69.07/47.69/70.93/66.91`，不等于受保护 M1 的登记指标 `56.21/68.33/47.13/70.21/66.20`。既有 provenance audit 已确认 M1 目录的序列文件被后续运行改变。因此本次诊断生成的 sequence-bootstrap CI、胜负数、frame recovery 和序列差值均不作为有效 M1 配对证据；配对表只保留作审计，不据此归因。

严格逐序列归因需要一套可验证的 M1 配对预测。本轮按用户要求不重跑 M1，因此不从当前混合预测目录计算配对 CI、序列胜负或帧级恢复归因；先用最重合的 M1 固定 aggregate 做点估计比较，并把下一候选的 control 与候选结果放在新建的同口径目录中。

协议判断：两组的记录在 LasHeR full-244、219,292 帧、FP32、`threads=0`、template update off、template 128/factor 2、search 256/factor 4、4090D 和软件版本上高度重合。M1 历史报告与 STSER provenance 记录同一 Git HEAD；但两者都包含未提交工作树，M1 报告未记录评测文件逐文件哈希，且 M1 当前结果目录不能复算固定分数。因此可以拿 M1 固定 aggregate 做最接近的点估计比较，不能宣称评测实现逐字节相同或完成了可复现配对比较。

| 评测合同项 | M1 锚点记录 | STSER full15 记录 | 可比性 |
|---|---|---|---|
| LasHeR 测试范围 | full-244 | full-244 / 219,292 frames | 相同名义范围 |
| crop | template 128/factor 2；search 256/factor 4 | 相同 | 相同 |
| 推理 | FP32；threads=0；template update off | 相同 | 相同 |
| 计算环境 | 1×RTX 4090D；Python 3.8.20 / PyTorch 1.9.0+cu111 / CUDA 11.1 | 相同 | 相同记录 |
| evaluator 代码身份 | 历史报告为 HEAD + working tree，无当时评测文件哈希 | 同 HEAD + 有研究工作树改动 | 不能证明逐字节相同 |

本轮尝试启动的 M1 同协议审计重放，按用户要求在 8/244 个序列后停止；这 8 个结果不参与任何指标。故此处仍采用最重合的既有 M1 aggregate，而不继续重放。

## 训练信息流、梯度和损失

预期信息流为：每个 search token 与排除自身及 3×3 邻域后的 top-4 同模态竞争位置比较，形成 `query - competitor`；该描述与 T2S message 和跨模态一致性一起进入 shared/private/null router；路由后的 T2S 消息经有界残差注入 search 表征，再进入 CENTER head。优化目标沿用 GIoU、L1 与 location/center loss，没有新增竞争排序监督。

- 训练日志确认 M1 初始化 checkpoint 被加载，新增 STSER 键在 M1 中缺失并以候选初始化。
- 训练诊断每 200 updates 记录全模型梯度范数；采样总范数约 `0.1`，达到全局梯度裁剪阈值。backbone/head 平均梯度均非零；记录为 2/265 参数无梯度，但日志没有输出参数名。
- epoch15 最后一个 batch：total loss `0.78973`，GIoU `0.19683`，L1 `0.01634`，location `0.31438`，训练 IoU `0.81459`。这是训练 batch 统计，不是独立验证性能。
- checkpoint 含 37 个新增 STSER competition state entries；RGB/TIR gamma 最终为约 `0.0000623/0.0009248`，说明输出残差尺度有非零学习。
- `internal_signals.csv` 记录的是三层 TBSI 输入/输出特征范数；`feature_changes.csv` 只有表头，没有行。现有 Diagnostics hook 不覆盖 `SearchConditionedT2SRouter`；`amcm_gradient_flow.csv` 也不是 STSER 梯度。因此虽然用户要求记录模块专属梯度和流特征变化，本次运行没有完整满足这一观测合同。现有证据只能证明优化过程继续、全模型梯度存在和 STSER gamma 非零，不能给出 router/expert 各参数梯度曲线、竞争 token 特征差、T2S 消息与注入前后特征变化的时间轨迹。

## 归因：事实与假设

**已有证据**

- 五项 full-244 aggregate 点估计均低于 M1，其中 OP75 低 `0.49`。因 parent 预测文件失配，没有有效逐序列 CI、胜负数或恢复/退化帧计数。
- STSER 最终 `gamma_rgb≈6.23e-5`、`gamma_tir≈9.25e-4`。`max_delta=0.10` 且残差按输入范数有界，因此末期注入幅度相对对应 search 特征范数的上界约为 RGB `6.23e-6`、TIR `9.25e-5`。这说明末期更新很弱，不证明早期也弱，也不证明它对框完全没有影响。
- 梯度日志只采样全模型，范数被裁到 `0.1`；没有 STSER 参数组梯度。`feature_changes.csv` 为空，无法证实 router 选择、竞争描述和注入前后特征的变化。
- competition descriptor 由同模态 search-search cosine 产生：排除 3×3 邻域后，每个位置取全局 top-4，再用 `query - competitor` 作为路由输入。竞争者不是按模板目标证据选出的。路由只改变 shared/private/null 的消息比例，最后仍经受限残差进入 CENTER head；训练没有增加目标—干扰物排序监督。

**当前最有根据、但仍待验证的解释**：这个算子定义的“竞争”与跟踪需要的“目标竞争”不一致。搜索区自相似可找到重复纹理或相似物，却不能判断谁是模板目标的高相似干扰物；而且被选出的全局竞争特征只用于决定如何路由 T2S 消息，并没有显式把目标证据与干扰证据做差后写回 search token。新分支末期残差尺度又很小，所以正向机制信号可能没有形成足够、方向正确的表征变化。

**同样必须保留的替代解释**：STSER 组是在 M1 epoch15 checkpoint 上继续 full-data 联合训练15 epoch；M1 对照是 epoch15 本身。因缺少同初始化、同 seed、同 schedule、无新模块的续训 control，性能差也可能主要来自继续训练造成的 backbone/head 漂移。现阶段不能断言“STSER 本身导致掉点”。

训练入口把 `INIT_CHECKPOINT` 作为模型权重初始化；这是新的 run id，没有继承 M1 的 optimizer/scheduler state，而是使用新建 AdamW 与该运行的 15 epoch schedule。因此候选测的是“载入 M1 权重后，以新优化器续训15 epoch并加入新分支”，不是单纯在同一优化轨迹上插入 STSER。此差异进一步说明为什么无模块续训 control 是必要的。

## 下一算子与实验闭环

针对的问题限定为：M1 已观察到 crop 内中心选择失配后可能发生反馈漂移、目标出界和恢复不足；这只是系统级失败链，不能反推某个融合算子是首因。STSER competition 的可检验子问题是：在同一 search crop 中，是否能通过模板目标证据区分目标支持位置与高响应干扰位置，并让这种差异进入融合后的 search token。

建议的下一算子是**目标条件 T2S 竞争残差**，保留原有 T2S、shared/private/null 专家和层位置，只替换竞争者定义及其进入残差的路径：

1. 从 search query 对 opposite-modality template 的现有注意力/相似度得到每个 search token 的目标支持分数 `s_i`；不读取 CENTER 输出作为目标证据。
2. 在空间上与当前位置分离的 token 中，按 `s_j` 选 top-k 高支持位置作为竞争池，以温度 softmax 或 log-sum-exp 得到 `c_i^-`。训练期用 GT box 排除目标正区域，推理期只用模板分数和固定空间排除规则。
3. 形成 `d_i = target_message_i - W(c_i^-)`，再由现有 shared/private/null 路由对 `d_i` 做有界残差写回 search token。这样竞争直接针对“模板也认为像目标”的远端干扰，而不是泛化的 search-search 外观相似。
4. 第一版沿用现有 GIoU/L1/location 目标，只改竞争表征；如果目标与 hard-negative 的排序 margin 仍不动，第二阶段才单独加入小权重 ranking loss，避免算子与监督同时变化后无法归因。

实验顺序建议：

1. 先做最近因果对照：以固定 M1 epoch15 为初始化，跑 full-data 15 epoch 的无新增模块 continuation control，optimizer、LR schedule、AMP、seed 和 worker 合同匹配 STSER full15。它不重训或覆盖 M1 锚点，用来估计单纯续训漂移；由于旧 STSER run 未保存实际训练 sample manifest，这个对照首先是方向性诊断，不能单凭它形成精确配对因果量。
2. 若 control 的表现仍接近 M1，再启动目标条件竞争候选。为形成可写论文的因果对照，control 与该候选需从同一 M1 初始化并共用显式保存的 full-data sample manifest、优化器/LR schedule 和测试合同；只把 competition descriptor 换成上面的 target-conditioned descriptor。无需重跑 M1 测试，比较两组新结果即可。
3. 记录 STSER 分组的 gamma/梯度、router 三路概率和熵、shared/private/null 消息范数、正目标与 hard-negative 选择率、注入前后 search-token cosine/范数，以及目标—干扰排序 margin。现有全模型裁剪梯度与空 `feature_changes.csv` 不足以归因。
4. 判定：若 continuation control 也出现接近的掉点，先改续训 schedule/parent 漂移；若 control 保持而目标条件候选提升 margin、产生可测的特征更新并改善 full-244，支持该算子链；若 margin 提升但定位指标不动，瓶颈在融合特征向定位输出的传导；若 margin 不动，停止当前 competition 算子族。

本轮不重跑 M1 测试。旧 M1 目录不用于序列配对；下一轮新 control 与候选必须各自写独立目录并保存244序列清单及结果哈希，逐序列分析仅在同一新口径内进行。

## 判定与下一步

判定：`AGGREGATE POINT-DROP VS M1 / PAIRED SEQUENCE ATTRIBUTION INVALID / OPERATOR CAUSALITY CONFOUNDED BY CONTINUATION / FLOW OBSERVABILITY INCOMPLETE`。本轮不重跑 M1。下一步优先用同初始化的无模块 full-data continuation control 量出续训漂移，再决定是否训练目标条件竞争算子；暂停单纯扩大 top-k、温度、router 宽度或专家数。

## 证据

- 运行配置：`experiments/tbsi_track/M1_STSER_COMPETITION_FULL15_FROM_M1_20261024.yaml`
- provenance：`experiments/snapshots/m1_stser_competition_full15_20261024/provenance.txt`
- 训练/测试/官方分析：`output/experiments/M1_STSER_COMPETITION_FULL15_FROM_M1_20261024/logs/`
- checkpoint：`output/experiments/M1_STSER_COMPETITION_FULL15_FROM_M1_20261024/checkpoints/TBSITrack_ep0015.pth.tar`
- full-244 tracking：`output/test/tracking_results/tbsi_track/M1_STSER_COMPETITION_FULL15_FROM_M1_20261024/`
- candidate per-sequence absolute metrics：`output/experiments/M1_STSER_COMPETITION_FULL15_FROM_M1_20261024/analysis_vs_m1/per_sequence.csv`
- current-parent paired audit（因 parent 文件不匹配而不用于归因）：`output/experiments/M1_STSER_COMPETITION_FULL15_FROM_M1_20261024/analysis_vs_m1/summary.json`
