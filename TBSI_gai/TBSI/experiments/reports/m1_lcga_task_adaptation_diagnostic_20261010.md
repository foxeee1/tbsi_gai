# LCGA 固定状态失配诊断与任务适配边界（2026-10-10）

## 结论

继续扩大忠实 LCGA 训练/测试没有根据。两条最大退化序列的定点回放提示，优先解决的不是更换空间算子，而是**避免加入 LCGA 后联合续训把 M1 的上游 search 表征推离稳定状态**，同时确认曲率分支真的影响定位。当前忠实版的外层残差注入在这些帧上的相对 L2 约 `3e-4`；候选与 control 的融合层输入差异约 `1.7%-2.1%`，后者主要不是当前 checkpoint 的即时 LCGA 注入。

本诊断只在已经用于短训评价的 LasHeR 训练侧 dev30 中，事后选取两条最大退化序列各一个损失相关分叉帧及其前一帧。它是**解释已有失败的探索性诊断**，不能用这两条序列选择下一版超参数，也不能作为新方法泛化证据。未运行训练、test-244 或新的 dev 评测。

## 固定状态协议

- Control：`M1_LCGA_FAITHFUL_CONTROL_seed42` epoch 3，SHA256 `d24faa0fa97c9f7f83c12ac6641c9091c6e0cd699a3bb87a174f41401e5a1920`；候选：`M1_LCGA_FAITHFUL_PROXY_seed42` epoch 3，SHA256 `f628599219d954e4b9f4b57263b04c03baa5f2f0e49c95a88f39018ddbae97ff`。保留两份 checkpoint 和原逐帧预测。M1 受保护资产不参与修改。
- `leftwhiteblack` 取 3209/3210，后者归档 IoU 从 control/candidate `0.89/0.88` 跳至 `0.84/0.00`；`girlbikeinlight` 取 199/200，归档 IoU 从 `0.65/0.65` 至 `0.71/0.32`。二者是从共同可跟踪状态出现的损失相关分叉，不是整个序列的最早非零框差。
- 两模型从同一首帧建立模板，用 control 的上一帧归档框固定 tracker 状态与 search crop；FP32 单帧回放。另对候选执行零外层残差门控。实际 RGB/TIR 输入和模板张量逐值相同；GT 均完整位于固定 crop。记录原始响应、最终 search-to-fusion 输入、融合输出、CENTER 峰、GT 区域与背景峰值差。张量保留为压缩 NPZ。
- 离线将两套融合特征交叉送进两套 CENTER head，再将两套融合前 search 特征交叉送进两套融合层和 control head。原配对的 head 响应与 GPU 回放最大绝对差低于 `3e-5`。交叉组合不是已训练 tracker，解释仅限这几个固定帧的接口敏感性。

## 关键证据

| 事件帧 | 固定 crop 下 control/候选中心误差 | 两臂 CENTER 峰到 GT 中心 | 候选外层门控置零 | 接口交换 |
|---|---:|---:|---|---|
| `leftwhiteblack` 3210 | `0.84 / 71.38 px` | `16.19 / 85.97 px` | 候选仍错，`71.38 px` | 候选融合特征 + control head 仍峰 `[8,11]`；control 特征 + 候选 head 峰 `[8,8]`。候选融合前 search + control 融合/head 仍 `[8,11]`；control search + 候选融合/control head 为 `[8,8]`。 |
| `girlbikeinlight` 200 | `4.90 / 4.81 px` | 两臂均 `9.96 px` | 峰不变 | 所有特征/融合/head 交叉组合峰均 `[8,8]`。 |

`leftwhiteblack` 3210 的 response target-background margin：control `+0.00365`，候选 `-0.00447`；候选外层门控置零仍约 `-0.00448`。将候选的*未注入 LCGA 的* search 特征交给 control 融合层与 control head，margin `-0.00231`、错误峰不变；反向用 control search 经候选融合层和 control head，margin `+0.00063`、目标峰不变。这个固定帧将失配定位到**融合前 search 表征差异**，而非候选 CENTER head 或忠实 LCGA 的即时残差。候选与 control 是联合续训后不同权重；不能把该上游差异唯一归因于 LCGA 的训练作用，也不能说明受保护 M1 原始 MHSA 已损坏。

`girlbikeinlight` 200 使用 control 上一帧归档框时两臂都正确，且所有交叉组合都保持目标峰。这说明归档闭环差异**没有在该固定输入下复现**。两臂上一帧归档框中心仅差 1 px；但是用候选自己的*整数归档框*重放，仍与候选归档当前框最大差 `12.59 px`。归档框丢失亚像素和可能的内部状态，故目前不能断言“1 px 状态差造成失败”。不继续围绕这条已见过结果的序列调参。

## 下一版边界

保留 LCGA 家族，但将下一实验登记为**任务适配新版本**，不能再叫“官方忠实 LCGA”或复用本轮 checkpoint。优先假设是：固定 M1 backbone、原搜索融合和 CENTER head 的权重及 BN 统计，仅训练 LCGA 局部残差，可以避免上游 search 漂移；这是检验局部细化是否能改善既有读出的一项训练合同变化。第二个独立机制门是让曲率支路在外层仍保持 identity 的条件下真正获得梯度（例如只改变内部曲率系数初始化），并用同权重曲率关断测实际响应变化。**这两步应分开验证，不一次改冻层、alpha、窗口和插入位置。**

在训练前先用旧 train-side 数据检查：冻结路径的 forward/BN 严格不变、外层零门控 parity、曲率卷积梯度与响应变化、两顺序完整 tracker FPS。原 10% 速度停止线已由用户为上一轮显式放宽，但下一版要重新预先写明预算；不能事后依据结果移动门槛。若通过，再冻结一个未使用过的 train-side dev 清单，只做一次 matched 短训。旧 dev30 与已评价的新 dev30 仅供诊断，不再用来选版本。此处没有创建新配置、没有开始下一轮训练。

## 证据与命令

Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；诊断脚本 SHA256 `8432d77c652beda0cfe10e8652ee0e3925624a1c8d95e85ffb48c3c7e2283420`、交叉读出脚本 SHA256 `bc0923d81c255bb3b0588fdaa3eeadf8593ae17194d26bffcdeb478c7b9f0299`。`first_split.json` SHA256 `8c8d99468514c6891417c594dbe1de879b6b568589971ed959fe6b299257ee0c`；`first_split_tensors.npz` SHA256 `8a6aa401b02761ed2877da3ca6200a9f3d9d6b016f05bdfbd5584e7dce55c337`；`head_swap.json` SHA256 `a6d5da8a6c1d7f0332a6df1ec5f3a3ad689d0b03c9f889f0fbea6488f9409980`。运行环境为 1x RTX 4090 D、Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1；GPUHours 未单独测量，记 NA。

```bash
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python tools/diagnose_lcga_first_split_20261010.py
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python tools/analyze_lcga_first_split_head_swap_20261010.py
```
