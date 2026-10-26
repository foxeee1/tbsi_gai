# PVL 两个注入位置五轮筛选计划（2026-10-20）

## 问题与可证伪假设

PVL 的概率交叉注意力能否在不改变 M1 主干和监督定义的条件下改善 RGB/TIR 帧内目标匹配？模板 loc6 适配应更影响模板跨模态消息；搜索融合 Neck 适配应更直接改变进入 `tbsi_fuse_search` 的 16×16 搜索表征。任一位置若在同初始化、同训练量和同测试名单下未超过 parent control，或收益只来自少数序列，则不支持将其升级成长训候选。

## 三组对照

1. `M1_PVL_CONTROL_PRETRAIN_5EP_20261020`：M1 架构，PVL 关闭。
2. `M1_PVL_TEMPLATE_LOC6_MATCHED_5EP_20261020`：仅模板 TBSI loc6 启用 PVL。
3. `M1_PVL_SEARCH_NECK_MATCHED_5EP_20261020`：仅搜索融合 Neck 启用 PVL。

三组均使用 `TBSITrack_SOT.pth.tar` 初始化、seed 42、LasHeR train 512 样本/epoch、batch 2、FP32、worker 4、5 epochs、同一优化器/损失/增强/调度。此为 screening 训练进度，不是 M1 正式训练复现。

## 测试名单和解释边界

固定 manifest：`experiments/manifests/m1_pvl_position_screen_20261020.txt`。包含简单 3 条、中等 3 条、原 M1 低 AUC 3 条和额外困难 4 条。名单来自已观察过的 LasHeR test；它用于候选筛选，不是独立验证。最终报告同时给三组配对的整体与分层指标、逐序列差异、五指标（AUC/PR/NPR/OP50/OP75）、FPS、参数/FLOPs、训练信号和 M1 epoch15 上下文值。M1 epoch15 只作历史背景，不进入三组因果对照。

## 代码与计算图变更

- 新增独立 `PVLAdapter`，每路输入/输出保持 `[B,N,768]`；训练期通过 adapter 私有 RNG 采样 Value，eval 使用 Value 均值；零初始化残差尺度保证初始化旁路。
- PVL 参数初始化与 ViT 通用初始化隔离，已用相同 seed 构造三组并验证 267 个共享 state tensor 逐元素一致。
- 模板位置：TBSI loc6 的双模态模板 token，在 joint template MHSA 前处理。
- Neck 位置：搜索 RGB/TIR token 进入原 `tbsi_fuse_search` 前处理，输出仍为 `[B,768,16,16]`。
- 所有新行为由独立 `MODEL.PVL` 配置启用；M1 YAML 不改。

## 通过/停止条件

- 代码阶段：禁用模式的模型构造/状态加载与原路径一致；两位置随机输入 forward、零门控 parity、有限值、反向梯度、参数量和 FLOPs 检查通过。
- 训练阶段：三组均从相同公开预训练权重完成 5 epochs；每组记录准确初始化文件、权重哈希、配置哈希、快照、日志、checkpoint 和完整命令。失败/中断也登记资产。
- 结论阶段：只有候选在相同固定 manifest 上的增益不由一两条序列主导、且复杂度/稳定性可接受，才支持下一步独立验证。否则保留为机制探索或停止该层位。

## 资源和文件

任务串行执行，一个训练主进程加四个 DataLoader worker；测试单独运行。每个 run 使用独立 config 与输出目录。M1 受保护 config/checkpoint 哈希在代码改动前已核验，并保存当前 dirty source snapshot。

## 最终筛选决策（2026-10-20）

完成同初始化匹配筛选后，两处 PVL 均低于 control：模板 loc6 AUC `48.15`（delta `-4.18`），搜索 Neck `44.07`（delta `-8.26`）；两组均为 13 序列 5 胜、8 负。结论为**不晋级长训**。Neck 在历史 M1 低 AUC 桶平均 AUC delta `-38.33` 且理论计算增量更高，优先停止；loc6 也没有足够证据保留为训练候选。

由于候选由公开 SOT 初始化训练，历史 M1 低 AUC 桶只是回看选出的测试子集，不能将结果解释为对 M1 epoch15 的直接退化或机制归因。完整逐序列分析、复杂度与限制见 `experiments/reports/m1_pvl_two_position_matched_screen_20261020.md`。若重开 PVL，应先建立 train-side 固定帧目标/硬背景参与度与 rank 诊断，并记录 PVL 专属激活/梯度；当前结果不支持扩大旁路模块搜索。
