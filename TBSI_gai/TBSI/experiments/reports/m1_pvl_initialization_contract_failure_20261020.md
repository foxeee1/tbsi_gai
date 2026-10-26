# PVL loc6 初始实验归因合同失败（2026-10-20）

## 记录

首个 loc6 试运行 `M1_PVL_TEMPLATE_LOC6_PRETRAIN_5EP_20261020` 完成 5 epoch 和 13 序列测试，指标为 AUC/OP50/OP75/PR/NPR `41.17/47.75/36.68/49.32/48.64`；同轮控制为 `52.33/62.09/45.57/63.86/62.07`。这组差异不作为 PVL 效果结论。

## 失败归因

新增 PVL adapter 在主干初始化过程中创建，并参与 ViT 的通用权重初始化。当时 PVL 随机参数初始化消耗全局 RNG，使公开预训练文件未覆盖的 TBSI-E 随机层可能与控制组初值不同。因而这次比较不满足单变量初始化合同。该实现问题足以使本轮结果失去干净归因资格；不能据此断言 PVL 导致退化。

## 修复与验证

- PVL 构造使用隔离的 RNG 上下文与固定初始化种子。
- ViT 通用初始化跳过 PVL 子树，不改变 M1 主干模块初始化顺序。
- 训练期 Value 重参数采样使用 adapter 私有 generator，不推进共享模型 dropout 随机数流；eval 使用 Value 均值，推理确定。
- 用 seed 42 从公开 `TBSITrack_SOT.pth.tar` 分别构造控制、模板 loc6 和搜索 Neck 模型。三个模型共有的 267 个 state tensor 全部逐元素相同，候选不存在额外 absent shared key。
- 旧试验的 config、checkpoint、日志、预测和 eval pickle 保留在原目录；不覆盖或删除。它只登记为 `INVALID-INITIALIZATION-CONTRACT`。

## 下一步

使用新配置名 `M1_PVL_TEMPLATE_LOC6_MATCHED_5EP_20261020` 和 `M1_PVL_SEARCH_NECK_MATCHED_5EP_20261020`，与已完成且 disabled 路径逐元素兼容的控制组比较。若两位置仍失败，再根据新增模块梯度、残差幅度和逐序列变化判断表示/监督适配问题，不扩展模块搜索。
