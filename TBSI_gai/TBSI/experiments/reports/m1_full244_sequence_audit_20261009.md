# M1 LasHeR 全量逐序列审计与下一步诊断设计（2026-10-09）

## 对象与可比边界

- 受保护 parent 是 `TBSI_E_MHSA_FULL_4090D_seed42`（M1），seed 42、epoch 15、FP32；配置 SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`，权重 SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。训练为 LasHeR train、15 epoch、batch 16×梯度累积 8、AMP；YAML 的验证集写作 `LasHeR_test`，因此后续模块选择必须转到预先冻结的 train-side 数据。参考训练环境为 1×RTX 4090 D、Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1。
- 模型的 RGB/TIR 经共享参数的 patch embedding、位置编码和 ViT blocks 分路编码；每路模板 64、搜索 256 tokens。在 blocks 3/6/9 后，`tbsi_e` 的联合 MHSA 更新两路**模板**，搜索 tokens 保留分路；模板与搜索随后在各自 ViT block 中交互。backbone 输出排列为 `[RGB template, RGB search, TIR template, TIR search]`，共 640 tokens。head 前将两路搜索拼接，经已训练 `tbsi_fuse_search`（默认 3×3 Conv-BN-ReLU）得到 `[B,768,16,16]` 搜索特征，再进入 CENTER head。以上是 M1 的两个不同融合接口；后骨干 DBF 不能称作 blocks 3/6/9 的同位 MHSA 替换。
- 本次只读复用原有 full-244 预测、逐序列官方指标和 LasHeR 19 个序列属性标签。旧失败地图曾错用 B0 目录，已在 2026-10-05 的官方口径审计中纠正；本次脚本校验 M1 244 条成员、预测/计时长度及 AUC/OP50/OP75 parity。B0 为历史不同训练合同的参考，逐序列差值只供描述，不能当作 MHSA 或 DBF 的因果效应。

## 全量逐序列结果

官方序列等权成功曲线给出 AUC **56.2077**、OP50 **68.3339**、OP75 **47.1349**；原 M1 报告的 PR/NPR 为 **70.21/66.20**，本次未重新计算。244 条有效序列共有 219,292 帧预测。`_time.txt` 有 219,292 个正值，按文件记时求和所得 **76.02 FPS**，仅是该归档计时定义下的推算值。

| 现象 | 数值与口径 |
|---|---:|
| IoU<0.5，序列等权 / 帧加权比例 | 31.66% / 37.37% |
| 至少一个连续 ≥30 帧 IoU<0.5 的序列 | 136/244 |
| 持续低重叠段 / 段内帧数 | 430 段 / 64,656 帧 |
| 持续段占全部 IoU<0.5 帧 | 78.91% |
| 持续段延伸至序列末尾的序列 | 75/244 |
| 受影响序列首次持续段的相对起点中位数 | 视频长度的 21.79% |

最后一项按**每条受影响序列**取首次事件，再对 136 条取中位数；`679.5` 是**全部 430 个事件**起始帧号的中位数，不能将二者混称，也不能笼统说多数失败发生在视频后期。`75` 表示至少一个持续段没有后续 `IoU>=0.5` 的评测帧，不是目标必然一直可见且无法恢复。官方 overlap 对无效 GT 帧记为 `-1`；本组共有 533 个无效 GT 帧、分布于 23 条序列，持续段数字因此是评测口径的低重叠统计，不应直接解释为 430 次视觉漂移。

逐序列审计表覆盖全部 244 条，并保留每条 AUC、属性、首个/最长/累计持续段、中心与尺度误差及计时。例如 `1strowleftboyturning` AUC 4.04、单段 700 帧、起于第 32 帧；`1stcol4thboy` AUC 8.28、13 段累计 2,824 帧；`1strowrightgirl3540` AUC 21.46、最长单段 1,494 帧。这些是失败形态例子，不作为候选模块的测试集选样。部分近零 GT 框会使归一化中心误差极大，不能用未经稳健处理的该列排序或解释机制。

属性层面，TC 147 条平均 AUC 49.80，非 TC 97 条 65.91；TC 组 99/147 条有持续段。LR 68 条平均 45.75，SA 106 条 49.04，LI 51 条 50.06。这些属性高度重叠：TC 中 137 条同时 PO、90 条同时 SA、52 条同时 LR。它们描述困难样本分布，不表明 TC、低照度或融合本身是失败原因。M1 对历史 B0 的 TC 组平均 AUC 差还是 +1.75，因此不能声称 M1 在 TC 上相对 B0 退化。

## 诊断结论与融合假设边界

**唯一得到全量结果支持的主要弱项**：M1 在相当多的序列上进入长时间低重叠状态，恢复能力存在明显问题。当前只能称为观测到的失败形态；搜索目标可能已离开 crop、被遮挡或不可见，亦可能是定位响应、模板状态或模态融合出了问题。归档 full-244 只有融合后的框，没有 M1 同状态 RGB-only/TIR-only 预测、逐帧 crop-to-image 变换或融合层状态，故“是否由模态冲突或 MHSA/IGF 对应接口引起”**暂时无法判断**。

已有 M1 前 64 帧联合 MHSA 统计中，跨模态 attention mass、模态不一致与收益/退化没有稳定关联；这不支持直接宣称模板 MHSA 过融合。另一个已有单 seed、3 epoch、60 条测试代理短训中，post-backbone DBF Differential/Vanilla/Legacy AUC 为 49.07/50.58/52.47；Differential 对 Legacy 为 -3.40。该 DBF 替换绕过已训练的 M1 3×3-BN-ReLU，且新模块优化器 LR 为 `1e-5`、legacy 融合为 `1e-4`，不能把负分数单独归因差分机制，也不能将其作为再试另一个注入位置的理由。另有 search-only G1 在 train-side 30 条序列得到 Legacy 71.15、Vanilla 69.86 AUC，但两者训练 worker/RNG 不匹配、Differential 尚未补齐，仍是探索性负信号。论文 IGF 使用当前 RGB 图像引导照明权重等路径，现有本地 DBF 不等价于 IGF。

## 仅建议的一步：训练侧固定状态因果诊断

1. 在 LasHeR **train** 侧冻结序列/事件清单及 SHA256：从现有 M1 train-side 闭环轨迹按预写规则选持续低重叠事件及匹配正常窗口；不根据 full-244 测试中的具体序列、属性分数或 DBF 测试代理重新挑模块。先用已有三序列 M1 trace 检查采集字段，再把诊断上限设为约 16 条序列、每条至多一个事件和一个对照窗口。记录真实 RGB/TIR 图像质量、GT 可见性、实际 crop 与 GT/crop 覆盖、前一帧 tracker 状态和置信响应；持续段中无效 GT 帧另列。
2. 在**同一冻结状态与同一 crop**上执行原始 M1 单步前向，并分别捕获 blocks 3/6/9 模板 MHSA 输入/输出、两路搜索特征、`tbsi_fuse_search` 输入/输出及 head 响应。预先固定温和的单模态扰动与无扰动控制，量化单模态信息可用性、融合前后目标/干扰响应和单步定位变化；遮挡、crop-out、非可见目标单列。模态屏蔽或特征干预改变输入分布，只能提供机制线索，不能当作重新训练后的性能结果。全程聚合标量和少量帧图，不导出全量特征。
3. 决策门槛：只有在多个独立 train 序列中同时看到**目标在 crop 内、至少一模态有可用目标信息、融合后目标响应或定位反而恶化**，且问题可定位到一个接口，才定义单一 DBF 假设。若定位到模板 MHSA，候选只替换/残差增强该处的模板交互并保留 token 合同；若定位到搜索融合，则优先保留已训练 M1 readout 的单一可关闭残差。两种位置不能在同一实验同时改。若这些条件不成立，停止 DBF 缝合，转向 crop/模板/定位原因。此诊断不直接形成模块效果结论；后续短训需相同 parent、seed、数据、worker/RNG、进度、优化器 LR、显式 checkpoint 和独立 run 目录，先过 shape/parity/梯度/保存加载，再以 train-side 冻结开发集筛选，最后才安排独立确认。

**成本与状态**：本轮新增训练 0、推理 0、GPU 时长 0；只新增只读分析脚本与报告。下一步固定状态诊断预计以最多约 0.5–1 GPU 小时为停止预算，先在已有三序列 trace 上计时校准；不启动长训练或 full-244 重测。当前磁盘仅约 5.4 GiB 可用，应仅保留小型事件清单、标量和必要帧图。现有 M1 权重/config、预测和旧 DBF 负结果均不修改。

证据：`output/experiments/M1_FULL244_SEQUENCE_AUDIT_20261009/{summary.json,per_sequence.csv,attribute_summary.csv}`；`tools/analyze_m1_full244_sequences_20261009.py`；`experiments/reports/m1_full_test_failure_map_20261005.md`；`experiments/reports/mhsa_failure_analysis_v1_stage_b_20260928.md`；`experiments/reports/dbf_igf_contract_audit_20261009.md`。
