# M1 MCCA 两 seed 匹配确认实验（2026-10-11）

## 预注册问题与结论状态

问题：冻结当前双向 MCCA 消息适配结构后，在完整 M1-style 15 epoch 训练合同下，相对同 seed、同数据、同日程的 G0 对照，收益是否能在两个 seed 上复现？

当前状态：`INTERRUPTED / USER-STOPPED`。用户要求停止当前任务后，launcher 与 G0 seed42 训练进程已终止；本轮后续三臂取消。

## 实验合同

- Parent：受保护的 `TBSI_E_MHSA_FULL_4090D_seed42` 模型结构与 M1 训练配方；直接效应比较为同 seed 内 G1−G0，不把 seed 42 的旧 M1 指标当成匹配 G0。
- G0：`MCCA_MODE='off'`；G1：`MCCA_MODE='bidirectional'`；其余配置逐项相同。
- 两 seed 分别为 42、3407；训练次序 G0-42 → G1-42 → G0-3407 → G1-3407，全程串行。
- 初始化：当前仓库可用的 `pretrained_models/TBSITrack_SOT.pth.tar`，SHA256 `1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b`。这是与 M1-style 训练入口一致的 SOT 初始化文件；M1 历史日志记录的文件名为 `TBSITrack_SOT_Pretrained.pth.tar`，该旧名目前不在磁盘，因此不宣称初始化文件名/文件 provenance 与历史运行严格一致。G0/G1 使用完全相同的当前文件。
- 数据：LasHeR 官方训练 split 的 977 条有效序列中，冻结 96 条作为闭环 train-side validation，余下 881 条用于训练。训练与验证清单 SHA256 分别为 `ba9a0aa45665af0e82f3d8813223aa5d50b63f9b3fcfdbc3bce5209317cec475`、`87c3b09c5c688f43e370d35ea027247ad387cb69717fa159fc72a71f33eb05e0`；无序列交集。新 96 条按固定 SHA256 排序从未列入现有 proxy/诊断清单的合格序列中选取，没有使用任何模型指标；selection manifest SHA256 `5c7aa417c026a6f575b5b75f058aa5f33cb75910aa0cf8f219876109b426d9f5`。
- 每臂 15 epoch、59,904 samples/epoch、batch 16、gradient accumulation 8、468 optimizer steps/epoch（7,020 steps/run）、AdamW、LR `1e-4`、第 10 epoch step decay、backbone multiplier `0.1`、GIoU/L1 `2.0/5.0`、AMP true。GPU 主机进程 + worker 1 共 2 个进程；prefetch 2、非持久 worker。
- 固定环境：1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1；FP32 tracker、`threads=0`。
- validation：四臂都对同一份 96 序列清单做完整闭环测试；主要比较 AUC、PR、NPR、OP50、OP75、逐序列 G1−G0 差值、胜负与 bootstrap 区间，并逐 seed 单独报告。两个 seed 只提供初步重复性证据，不包装成充分多 seed 结论。FPS独立记录，不作本轮晋级/调参目标。
- 训练期间 trainer 内部 sampled validation 使用训练清单，仅作为 loss/过程诊断，不作为独立性能验证；独立性能验证仅指闭环 96 序列。

## 可比性边界

这不是受保护 M1 的严格复跑。M1 原训练使用全部 977 条 LasHeR train 序列、worker 4、历史记录的初始化文件名；本轮将新冻结 96 条从训练样本中留出，并使用 worker 1 遵守资源上限，初始化使用当前可核对的 SOT checkpoint。每个 seed 的 G0/G1 内部为匹配对照；与已登记 M1 单次指标的直接差值不是 MCCA 的因果效应。

M1 预训练 checkpoint 曾接触 LasHeR training split，因此这 96 条对候选 G0/G1 训练是序列留出，对 pretrained parent 不是严格意义上的未见数据。此前已看过 test-244；此轮完全不使用 test-244。

## 资产与运行位置

- 机器清单与配置：`experiments/proxy/mcca_confirm_two_seed_20261011/`
- 源码/配置/清单快照：`output/experiments/M1_MCCA_CONFIRM_TWOSEED_20261011/provenance/source_snapshot.tar.gz`，SHA256 `dae1bd1f695e669fe845f6117c85b22a1511742a18b073d7ce365f03c5b89437`
- Launcher：`tools/run_m1_mcca_confirm_two_seed_20261011.sh`
- Run state、日志与指标：`output/experiments/M1_MCCA_CONFIRM_TWOSEED_20261011/`
- Checkpoints：`/autodl-fs/data/M1_MCCA_CONFIRM_TWOSEED_20261011/output/experiments/`
- Train-side predictions：`output/test/tracking_results/tbsi_track/M1_MCCA_CONFIRM_*_<runid>/`
- M1 YAML 和 epoch-15 checkpoint 启动前 SHA256 与受保护登记完全一致；不修改 M1 配置、权重、结果或输出目录。

## 判定规则

- 支持继续：两个 seed 的 G1−G0 AUC 均为正，其他定位指标没有一致性退化，且逐序列改善并非只由极少数序列贡献；仍需明确这只是两 seed、train-side validation 的初步确认。
- 不支持：两 seed 方向相反，或两个 seed 都没有正向定位信号；停止当前 MCCA，不用 test-244 事后挑选。
- 不确定：一个 seed 正、一个不正，或差异区间很宽；如实归档为不确定，不加第三 seed（按用户约束），不将它升级为稳定收益。

## 进度

停止记录：G0 seed42 已运行至 epoch 1 的 750/3744 batch 后停止；未完成 epoch，未产生 checkpoint，也未进行闭环验证。G1 seed42、G0 seed3407、G1 seed3407 均未启动。GPUHours 未可靠采集，登记 `NA`。日志出现一次 `Non-finite norm` 警告，周期性诊断在 step 200/400/600 记录 2/228 个零梯度参数；仅作为中断前观察保留，不构成性能或收敛结论。日志和诊断文件保留在 `/autodl-fs/data/M1_MCCA_CONFIRM_TWOSEED_20261011/output/experiments/M1_MCCA_CONFIRM_G0_seed42/logs/`，无 partial checkpoint。
