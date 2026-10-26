# M1-S1 闭环状态分叉风险信号诊断（2026-10-15）

## 问题与设计

上一轮 selective state commit 使用固定状态下的单步效用作为风险标签，不能证明其能预测多帧状态反馈伤害。本实验在同一帧序列上并行推进 M1 与 S1 两条完整闭环轨迹，检验当前帧处理前可观测的状态分歧是否能排序未来定位差值。

- Parent：受保护 M1 epoch15 checkpoint；S1 bounded readout predictor 与 independent action policy 冻结。
- 数据：LasHeR trainingset 冻结72序列名单中按名称 SHA256 排序选出的12条完整序列，整序列为分析单位；没有使用 testingset。
- 分支：同一帧、同一初始化、FP32；M1 分支关闭 readout predictor，S1 分支使用冻结 predictor/policy 并以 S1 框更新内部状态。因而从轨迹开始后，readout 动作和后续状态都会不同。
- 风险分数：当前处理前的归一化中心差、log 尺寸差、联合 L2 分歧；不使用未来帧或 GT。
- 标签：未来10/30个 tracked frame 的 `mean(IoU_M1-state - IoU_S1-state)`；正数表示 S1 状态反馈更差。
- 不拟合预测器、不调阈值。主要结果为 pooled AUROC/Spearman、分数最高四分位的有害率及逐序列方向一致性。

## 结果

12/12 序列完成，共3,675个配对帧；10帧标签覆盖3,555帧，30帧标签覆盖3,315帧。两条轨迹都按同一图像序列推进，rollout 用时176.49秒。环境为1×NVIDIA GeForce RTX 4090 D、Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1、FP32、单进程、0 worker。

| 预测分数 | Horizon | 未来 S1 伤害比例 | Spearman | AUROC | 最高四分位伤害率 | 相对总体富集 |
|---|---:|---:|---:|---:|---:|---:|
| 中心分歧 | 10 | 0.332 | -0.260 | 0.501 | 0.172 | 0.518× |
| 尺寸分歧 | 10 | 0.332 | -0.250 | 0.496 | 0.207 | 0.623× |
| 联合分歧 | 10 | 0.332 | -0.256 | 0.501 | 0.189 | 0.569× |
| 中心分歧 | 30 | 0.387 | -0.266 | 0.481 | 0.211 | 0.545× |
| 尺寸分歧 | 30 | 0.387 | -0.255 | 0.476 | 0.229 | 0.592× |
| 联合分歧 | 30 | 0.387 | -0.258 | 0.481 | 0.234 | 0.604× |

联合分歧的逐序列 Spearman 方向明显混杂：10帧只有6/12为正，30帧为7/11为正。示例中，`rightestblkboy2` 的相关为 +0.308/+0.349，而 `boyunderthecolumn` 为 -0.334/-0.267，`basketballshooting2` 为 -0.286/-0.367。说明分歧大不等同于状态反馈危险，可能同时代表 S1 有益修正或有害漂移。帧级相关不作显著性推断，帧数不是独立样本量。

## 归因与结论

结论分类：**暂不支持**“单一几何状态分歧分数可作为 M1/S1 分支差异风险门控”。这不是对所有闭环风险特征的否定；本实验只排除了中心/尺寸状态间距作为无条件单调的分支差异信号。由于两条完整轨迹的 readout 动作和后续状态都会不同，本实验**不能把未来差异归因于某次状态提交**，也不能作为单次状态回退器的因果训练标签。前一轮 same-state utility 风险器和本轮简单分歧分数都没有显示稳定的排序依据，继续拟合分类器或阈值缺少证据。

下一步应转向事件级机制实验：在相同 S1 前缀和同一帧上分叉，只改变该帧提交 S1 框或 M1 框；之后两支均恢复 S1 readout policy，直接测未来定位与恢复差。若单次提交效应在 train-side 事件上异质且可由可见信号区分，再设计 selective gate；若效应本身接近零或方向不稳定，转测固定短暂状态保持策略。完成该机制检验前，不做 full test 或在官方 test 上调策略。

## 复现与资产

- 命令：`/root/autodl-tmp/conda_envs/tbsi/bin/python -u TBSI_gai/TBSI/tools/run_m1_s1_closed_loop_risk_diagnostic_20261015.py --sequences 12`
- 独立运行目录：`output/experiments/M1_S1_CLOSED_LOOP_RISK_DIAGNOSTIC_20261015/`
- 文件：`manifest.json`、`sequence_manifest.txt`、`paired_rollout.csv`、`paired_rollout.jsonl`、`metrics.json`。
- M1 YAML/checkpoint SHA256 与受保护值一致；S1 predictor/policy SHA256 与既有冻结值一致。
- 首次启动因相对工作目录错误未能打开脚本，未产生进程或数据；使用上列根目录命令启动成功。
- 不报告 AUC/PR/NPR/OP50/OP75：该训练集诊断不是正式跟踪基准评估。
