# M1 固定状态动作级融合回放（2026-10-12）

## 结论

**判定：三个单信号和联合信号均未通过固定状态真实框效用准入；停止当前 ±15% 信号引导算子，不进入 full-244 闭环。** 主要效应不是可测的融合修复：在唯一按序列留出的 24 条序列（44 个固定状态）上，S1/S2/S3 的序列宏平均 decoded IoU 相对 no-op 分别为 `-0.0039/-0.0086/-0.0039`；S1+S2 为 `-0.0055`，S1+S2+S3 为 `-0.0106`。全部与 no-op 的 95% 配对序列 bootstrap 区间包含零或为负；没有一个信号臂超过固定 TIR 控制。多信号投票未表现出联合收益。

这是对**预选固定状态的离线动作效用**结论，不是闭环 tracking AUC 结论，也不能证明融合对 M1 的总体失败无关。结果更具体地表明：当前“按信号选择 RGB/TIR 互反缩放 ±15%”没有在独立序列上改善同 crop、同 state、同 CENTER head 的真实框；其峰命中和框回归误差整体恶化，因而停止该算子与当前信号合同。

## 实验合同

- Parent：`TBSI_E_MHSA_FULL_4090D_seed42` epoch 15。受保护 config/checkpoint SHA256 分别为 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e` 与 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，实验前后未变。
- 数据：LasHeR train-side 72 条序列，原诊断每序列取两个预先选定固定状态，共 144 个状态。信号校准的冻结 split 是 48 train / 24 heldout；由于本 replay 使用原 cohort，主结论只计算 24 条 heldout 序列的 44 个状态。训练 48 条的结果只用于展示拟合内外差异，不能作为效能证据。
- 固定条件：同一模板特征、保存的最终 RGB/TIR search feature、trace 中同一 `tracker_state_before`、同一 GT、同一 crop transform、同一 M1 fusion convolution、CENTER score/offset/size head 和 Hann window。每个动作仅改最终 search 拼接前两路特征幅度。无 MHSA、fusion convolution、CENTER 参数或 checkpoint 改动；不运行闭环状态更新。
- 动作：no-op、固定 RGB +15%、固定 TIR +15%、固定 seed 3407 的随机方向、S1/S2/S3、S1+S2、S1+S2+S3、多信号低置信度 no-op（`abs(mean signal prediction) < 0.005`）。后者阈值在本次 replay 前固定，仅作诊断控制，不作为已验证部署阈值。
- 信号工件：`train72_screen_v1/signal_calibration.json` SHA256 `c621708e7262916859e1749c893a189f2a8b763260b01cfd601a5ca1d649dc3b`；S1 response heads SHA256 `c3006ad39d2b627919264755fb8551dc128e78979c5a1fe34b2c79a6e9584079`。信号标定仅基于预先固定的 48 train sequences，24 heldout sequences 没有用于拟合。
- 主指标：每状态经原 Hann peak 和原 offset/size 共同解码后的真实 bbox IoU；先对每条序列的状态取均值，再对序列做宏平均。95% bootstrap 区间以序列为重采样单位，20,000 次，seed 3407。对 no-op、固定 TIR、random 的配对对比均列出；没有做多臂多重比较校正，因此所有次要对比为探索性描述。
- 环境：1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8；FP32；单实验进程。GPU 小时未由运行器采集，登记为 NA。Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`，工作树源快照见下方。

## 留出序列主结果

`delta / CI` 是相对同一序列 no-op 的 paired sequence macro-IoU 差及 95% bootstrap CI。对固定 TIR 和 random 的差值另列用于核验信号是否超出常数/随机动作。此表不是 tracking AUC 表。

| 条件 | 序列宏平均 IoU | 相对 no-op delta [95% CI] | 对 no-op 胜/负 | 相对固定 TIR delta [95% CI] | 相对 random delta [95% CI] |
|---|---:|---:|---:|---:|---:|
| no-op | 0.7551 | 基准 | 基准 | +0.0019 [-0.0074, +0.0100] | +0.0113 [+0.0040, +0.0213] |
| 固定 RGB +15% | 0.7540 | -0.0011 [-0.0126, +0.0069] | 13/11 | +0.0009 [-0.0169, +0.0161] | +0.0102 [+0.0028, +0.0184] |
| 固定 TIR +15% | 0.7532 | -0.0019 [-0.0101, +0.0072] | 11/13 | 基准 | +0.0094 [-0.0014, +0.0240] |
| 随机方向 | 0.7438 | -0.0113 [-0.0212, -0.0041] | 9/15 | -0.0094 [-0.0238, +0.0014] | 基准 |
| S1 | 0.7513 | -0.0039 [-0.0111, +0.0049] | 7/17 | -0.0019 [-0.0059, +0.0010] | +0.0074 [-0.0024, +0.0219] |
| S2 | 0.7465 | -0.0086 [-0.0200, +0.0003] | 10/14 | -0.0067 [-0.0224, +0.0046] | +0.0027 [-0.0028, +0.0086] |
| S3 | 0.7512 | -0.0039 [-0.0160, +0.0054] | 13/11 | -0.0020 [-0.0185, +0.0115] | +0.0074 [-0.0001, +0.0160] |
| S1+S2 | 0.7496 | -0.0055 [-0.0102, -0.0015] | 2/10（12 平） | -0.0035 [-0.0114, +0.0026] | +0.0058 [-0.0010, +0.0158] |
| S1+S2+S3 | 0.7445 | -0.0106 [-0.0221, -0.0020] | 8/16 | -0.0087 [-0.0240, +0.0019] | +0.0007 [-0.0039, +0.0059] |
| 三信号低置信度 no-op | 0.7452 | -0.0099 [-0.0210, -0.0017] | 7/15（2 平） | -0.0080 [-0.0229, +0.0019] | +0.0014 [-0.0026, +0.0061] |

拟合内与留出方向相反：在校准所用 48 条 train 序列上，S1/S2/S3 相对 no-op 分别 `+0.0065/+0.0060/+0.0089`，S1+S2+S3 为 `+0.0092`；在独立 24 条 heldout 序列上，这些差值变为 `-0.0039/-0.0086/-0.0039/-0.0106`。这是当前结果里最明确的机制信号：按当前标签/信号合同拟合出的效用方向存在明显拟合内乐观，未能迁移到留出序列。它支持“当前信号泛化失败”，但不证明任何信号学习都无法泛化。

固定状态末端框的 no-op IoU 与原 `per_arm.csv` baseline 最大绝对差为 `3.74e-5`，均值差 `6.22e-7`。这验证 feature replay 和框解码口径对齐；这不是归档 M1 全序列逐帧 parity。

## 归因边界

### 已排除

- **不是状态错配造成的动作比较差异。** feature、trace、图像按 sequence/frame 唯一 join；no-op 对原 baseline 几乎数值一致。
- **固定幅度不是可忽略的中性扰动。** 预设随机方向在留出状态上比 no-op 低 `0.0113` IoU，95% CI 不跨零；动作方向确实可以改变真实解码框，而不是只改变 margin。
- **当前信号没有显示独立序列动作效用。** S1/S2/S3 未超过 no-op、固定 TIR或随机方向的配对不确定性；S2、联合臂的负差尤其明显。不能把 earlier closed-loop 62 条低 AUC 子集的增益归因给这些信号。
- **当前动作没有修正 Hann 选峰。** 留出状态 Hann peak 位于 GT cell 的比例：no-op `6/44`；固定 RGB/TIR `5/44`；S1 `5/44`；S2/S3/S1+S2+S3 `4/44`。动作后的平均 peak-to-GT 距离也未改善。

### 本实验仍不能区分

- **选择与回归谁是 M1 全局首因。** 留出固定状态下 no-op 的 CENTER GT-cell rank 均值为 `5.64`，但强制 GT cell 的离线 oracle IoU 只比实际 baseline 高 `0.0025`。这说明该状态队列里单独纠正选格的可挽回空间很小；同时预测框 center error 和 size error会变差（no-op `7.84 px / 0.130`，S1+S2+S3 `8.21 px / 0.142`）。它支持“本动作扰动了选峰及框读出，未产生净收益”，但不足以证明 M1 的所有困难轨迹主要由 offset/size 或其他某一项导致。
- **幅度与方向机制没有被拆开。** 这里只有 ±15% 固定幅度，以及一个随机方向实现；还没有 `±7.5%/±30%` 剂量响应，也没有多随机种子平均。因此不能断言所有幅度或随机策略都会退化。
- **固定状态不等于闭环效用。** 这些是依据既有 trace 选出的正常/失配状态；没有让每个动作从该状态继续跟踪，因此无法量化轨迹分叉、目标出界或恢复能力。
- **留出样本很小且是诊断选样。** 24 序列、44 状态（含 10 severe、10 matched-good），不代表 244 条测试序列。10 条序列的 matched-good 描述切片中没有任何调制臂构成已验证的正常帧安全保证。
- **随机控制只有一个 realization。** 该 control 可证明任意方向扰动会有影响，但不能作为随机策略的均值估计。
- **当前代码不是归档 M1 逐帧 parity。** 结论仅相对于相同当前源码和固定 M1 epoch-15 checkpoint 的 feature replay；不能写成相对于受保护历史 full-244 指标的直接因果结论。

## 失败运行与复现

早期 replay harness 有三次短失败，均未产生候选结果：CENTER head 返回四项而工具最初只解包三项；独立信号头留在 CPU；组合信号没有像 tracker `forward()` 一样传播模板 mask / S3 crop 特征。逐项修正后，最终 V2 replay 完成。失败签名保留在会话与工作树历史；资产状态记录为 harness 修复后完成，不将失败版本当作结果。

复现命令（工作目录 `TBSI_gai/TBSI`）：

```bash
TBSI_INFER_FP16=0 python3.8 tools/replay_m1_fixed_state_actions_20261012.py \
  --features output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz \
  --trace-dir output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run/trace \
  --trace-dir output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank24/trace \
  --trace-dir output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank48/trace \
  --output-dir output/experiments/M1_FIXED_STATE_ACTION_REPLAY_V2_20261012
python3.8 tools/analyze_m1_fixed_state_actions_20261012.py \
  --actions output/experiments/M1_FIXED_STATE_ACTION_REPLAY_V2_20261012/per_state_actions.csv \
  --screen output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/train72_screen_v1/per_state.csv \
  --output-dir output/experiments/M1_FIXED_STATE_ACTION_REPLAY_V2_20261012/analysis \
  --bootstrap 20000
```

Source snapshot: `experiments/snapshots/m1_fixed_state_action_replay_20261012_sources.tar.gz`, SHA256 `27b1af4eb1e811049174d7a554fb7fc1c0d39e283c58dd9debca5c9b2ffd27a5`。Replay script SHA256 `5c61399db8ea833e90470c03606bfa899c8da1cab086a2528d9e201c66e20398`；analysis script SHA256 `3108a329a03d730fb7965480b097c7f66a189856d0ad63f77a188e9eed8d1fd2`；features SHA256 `214a24efd960a3529608fde36158fb8e611a492b6817e2c3b5d1babdb495804e`；sequence list `output/experiments/fixed_state_sequence_list.txt` SHA256 `6e940f05edf73dc2a2aee4b839b08e7cc277ccde2a54cc5e7a3b419f21b5a066`；heldout paired results SHA256 `41b0700e37400b3d6dbb9ecd96d18490eb86fdc96d24c19a21f188d110517d23`。

Harness failures and their fixes are kept at `output/experiments/M1_FIXED_STATE_ACTION_REPLAY_V2_20261012/harness_failures.md`.

## 路线决定

**停止**当前“信号驱动 ±15% search 双路互反缩放”路线，不追加更复杂的联合投票，也不进入 full-244。**保留的发现**是在线 action utility 必须直接对齐原 CENTER 的最终框；常量与 learned signal 必须在未参与校准的序列上比较。若继续 M1 定位归因，下一轮应使用新的独立 train-side 序列，固定 state 并拆分：no-op、GT-cell 选峰 oracle、真实 Hann peak + 原 offset/size、GT-cell + 原 offset/size、GT-cell + 重新估计/GT size 对照。该诊断可直接量化选峰和回归各自的可挽回上限；在它完成前，不再设计新融合信号。
