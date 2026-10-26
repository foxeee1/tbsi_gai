# Single-Modality Degradation v1：四格诊断报告

日期：2026-09-26  
状态：`DIAGNOSTIC`；不启动融合归因探针，不启动训练  
父问题：受控单模态局部退化是否通过 TBSI 模板桥接造成另一模态定位损失  

## 1. 实验边界

本实验只使用 LasHeR train 的固定 24 条开发序列、72 个 GT 事件；不使用测试序列、不读取已取消的 169 行盲审资产、不生成 P 标签。每个事件从相同 tracker 状态和 RNG 快照分叉，评估 `00/10/01/11` 四条件，记录干预当帧 `offset=0` 和后续 `offset=1..5` 的 IoU。

三个退化套件独立分析，不混合均值：RGB/TIR 高斯模糊（sigma=3.0）、RGB/TIR 低照度（强度乘 0.35）、RGB/TIR 对比度压缩（factor=0.40）。模型、权重、模板更新、注意力和输入尺寸均未修改。

交互项同时给出两种符号：

`iou_interaction = IoU11 - IoU10 - IoU01 + IoU00`；

`error_interaction = e11 - e10 - e01 + e00 = -iou_interaction`，其中 `e=1-IoU`。因此正的 `error_interaction` 才表示双模态额外错误。置信区间按序列聚类 bootstrap（24 条序列，5,000 次，seed=42）；事件中相邻帧不作为独立样本。

## 2. 完整性检查

- 原始结果：5,184 行 = 72 事件 × 3 套件 × 4 条件 × 6 offsets。
- 三个套件均包含 72 事件、24 条序列、offset 0--5；`skipped_events.json` 为空。
- 离线分析输出：`output/experiments/single_modality_degradation_v1/batch/analysis_v1/`。
- 原始 CSV SHA256 以及逐事件、逐序列明细保存在 `analysis.json`、`event_summary.csv` 和 `sequence_summary.csv`。
- canonical YAML SHA256 仍为 `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`，未修改。

## 3. 结果摘要

下表是事件均值；`post-only` 为先在每个事件内平均 offset 1--5，再汇总。括号内为序列聚类 bootstrap 95% CI。`error I` 是错误率交互项。

| 套件 | 范围 | 单 RGB 损失 `00-10` | 单 TIR 损失 `00-01` | 双模态损失 `00-11` | error I |
|---|---:|---:|---:|---:|---:|
| blur | offset 0 | 0.11050 | 0.03363 | 0.20108 | 0.05695 [0.02229, 0.08990] |
| blur | offset 1--5 | -0.00361 | 0.00048 | 0.00468 | 0.00781 [0.00054, 0.01713] |
| lowlight | offset 0 | 0.00047 | 0.02134 | 0.02666 | 0.00485 [-0.00186, 0.01389] |
| lowlight | offset 1--5 | -0.00157 | -0.00031 | -0.00115 | 0.00073 [-0.00361, 0.00507] |
| contrast | offset 0 | 0.00795 | 0.01770 | 0.01706 | -0.00859 [-0.03086, 0.00478] |
| contrast | offset 1--5 | -0.00123 | -0.00220 | -0.00235 | 0.00108 [-0.00151, 0.00342] |

稳健性摘要：

- blur 的当帧双模态额外错误最明显，但 `error I` 去掉最大 3 个事件后为 `0.04647`，去掉最大三分之一事件后为 `0.00947`；其后续帧交互在去掉最大三分之一事件后为 `-0.00616`。
- lowlight 的当帧交互区间跨零，后续帧区间跨零；去掉最大三分之一事件后的 post-only `error I=-0.00601`。
- contrast 的当帧 IoU 交互均值为负（对应错误率交互为 `-0.00859`），后续区间跨零；去掉最大三分之一事件后的 post-only `error I=-0.00572`。

## 4. 路由判断

这轮结果支持一个有限的诊断事实：强 RGB 模糊在干预当帧可以造成明显定位损失，且双模态条件在原始事件均值上比两个单模态条件呈现额外错误。但该交互不满足本轮预设的稳定性要求：效应主要集中在当帧和少数事件，去掉最大三分之一事件后不再保持正值；低照度和对比度也没有给出稳定的后续交互。

因此不能把结果表述为“桥接传播机制已被证明”，也不能据此训练可靠性门控或新的融合模块。第二轮定点消息缩放及随机区域对照暂缓，训练明确保持 `STOP_NO_TRAINING`。若后续继续该方向，应先预注册一套不依赖极端事件的退化事件分层与定点/随机归因对照；本报告本身不改变 canonical baseline，也不为新方法提供正结果主张。

## 5. 复现信息

- Python：`/root/autodl-tmp/conda_envs/tbsi/bin/python`，Python 3.8.20；PyTorch 1.9.0+cu111。
- GPU：1× NVIDIA GeForce RTX 4090 D；FP32 tracker inference，`TBSI_INFER_FP16=0`。
- 配置：受保护 baseline 配置名仅作为只读推理 contract；epoch 15 checkpoint。
- checkpoint：`output/experiments/v1.0.0-单卡3090-48G-等效全局128基线/checkpoints/TBSITrack_ep0015.pth.tar`。
- Git HEAD：`09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；工作树含已有用户修改，需结合完整工作树审计解释。
- 原始结果：`output/experiments/single_modality_degradation_v1/batch/`。
- 分析入口：`tools/analyze_single_modality_degradation_v1.py`。
