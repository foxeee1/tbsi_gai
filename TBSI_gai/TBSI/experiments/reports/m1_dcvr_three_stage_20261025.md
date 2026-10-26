# M1 DCVR 三阶段实验记录（2026-10-25）

## 目的与合同

DCVR（Degradation-aware Cooperative Value Routing）只改 TBSI-E 原生 template MSHA 的 Value token 路由：Q/K、attention logits、CENTER head、模板更新和 search-to-template 路径保持不变。关闭环境变量 `TBSI_DCVR_MODE` 时为 M1 原生路径；开启时按早期 RGB/TIR token 与 search 的匹配分布计算熵、top-1/top-2 竞争和跨模态一致性，并对两个模态的 Value token 做零均值、前景预算归一化的 token 重分配。

三阶段预注册为：

1. 固定状态同 crop/state 的 M1-off 与 DCVR-on 信息流回放；
2. 同一 M1 checkpoint 上的 RGB/TIR blur、lowlight、contrast 单模态/双模态干预；
3. 同一 SOT 初始化、seed、optimizer、3 epoch 合同的 matched quick train/test。

数据、代码和输出均独立于受保护 M1。阶段 1/2 输出放在 `/autodl-fs/data/tbsi_dcvr/`，避免占用仓库空间。

## 阶段 1：固定状态机制验证

- parent：`TBSI_E_MHSA_FULL_4090D_seed42` epoch15，FP32；原有固定状态事件清单 35 帧按脚本现有合同聚合为 7 个序列 severe-onset 事件。
- M1-off：`phase1_m1_off_retry3`；DCVR-on：`phase1_m1_on_retry3`，`alpha=0.25`。
- 两臂使用相同 crop、state_before、GT 和 tracker 初始化；各自产生 224 条 stage-level 记录。
- DCVR 证据均非退化常数：平均跨模态 agreement `0.6648`，RGB/TIR 匹配熵 `0.8830/0.8621`，竞争 `0.5546/0.5589`，source-scale std `0.0893/0.0755`。
- fixed-state IoU 差（on-off）为 `-0.00062`；`center_gt_rank` 仅 `shinycarcoming` 从 179 变 181，其余 6 个事件不变。

结论：`DIAGNOSTIC-COMPLETE / COMPUTATION-PASSED / CENTER-ORDERING-NOT-SUPPORTED`。算子确实改变了 Value 路径，但当前路由没有稳定改变 CENTER 排序，不能把阶段 1 写成机制成功。

## 阶段 2：受控退化干预

- 同一 M1 checkpoint、同一 8 个预注册事件、horizon=5；每个 suite 包含 clean、RGB-only、TIR-only、双模态四条件，共 576 行/臂。
- 退化强度：Gaussian blur sigma=3；lowlight factor=0.35；contrast factor=0.40。
- DCVR-on 相对 matched off 的平均 IoU 差：

| suite | RGB-only | TIR-only | 双模态 |
|---|---:|---:|---:|
| blur | +0.00067 | -0.00019 | -0.00079 |
| contrast | -0.00114 | +0.00066 | +0.00068 |
| lowlight | +0.00039 | +0.00263 | +0.00169 |

clean 条件平均差为 `+0.00554`，说明当前 DCVR 也会改变正常路径。结果支持“效应依赖场景/退化类型”的观察，但没有支持稳定的退化补偿方向。

结论：`DIAGNOSTIC-COMPLETE / DEGRADATION-SPECIFIC-SIGNAL / ROBUSTNESS-NOT-SUPPORTED`。阶段 2 不能单独支撑论文收益，后续必须使用更大事件集或训练后模型验证。

## 阶段 3：matched quick train/test

- control：`M1_DCVR_QUICK3_CONTROL_20261025`；candidate：`M1_DCVR_QUICK3_CANDIDATE_RETRY1_20261025`。
- 两者均从 `TBSITrack_SOT_Pretrained.pth.tar` 从零开始，seed42、LasHeR train 512 samples/epoch、3 epoch、AdamW、batch16、accumulation8、worker4；candidate 唯一开启 `TBSI_DCVR_MODE=on, alpha=0.25`。
- 同一 7 序列 test manifest，FP32、threads0、template update off。

| sequence | control AO | candidate AO | delta |
|---|---:|---:|---:|
| 3rdfatboy | 0.7853 | 0.7818 | -0.0035 |
| 4thboywithwhite | 0.0969 | 0.0984 | +0.0015 |
| ab_bikeoccluded | 0.3532 | 0.6819 | +0.3287 |
| boydownplatform | 0.7704 | 0.7708 | +0.0005 |
| leftfarboycomingpicktheball | 0.5437 | 0.5559 | +0.0122 |
| shinycarcoming | 0.3910 | 0.3862 | -0.0048 |
| womanback2car | 0.7664 | 0.7626 | -0.0038 |
| **mean** | **0.52954** | **0.57680** | **+0.04726** |

训练日志显示两臂 loss/IoU 曲线接近；candidate 训练阶段的无 mask 输入已显式记录为 uniform foreground fallback，推理阶段仍使用真实 mask。收益主要由 `ab_bikeoccluded` 单序列驱动，不能称为稳定 0.5% 全量提升。

结论：`PROXY-COMPLETE / CONTRACT-PASSED-AFTER-INPUT-FIX / EXPLORATORY-POSITIVE / SINGLE-SEQUENCE-DOMINATED`。阶段 3 值得进入更大 matched manifest 的 quick 回测，但不应直接进入 full-244 训练。

## 合同问题与修正

第一次 candidate 训练因训练入口没有传 `template_fg_masks` 中止；该错误已登记为输入合同问题，DCVR 现在在训练无 mask 时使用显式均匀预算并写入 `fg_mask_fallback_uniform`，没有修改 M1 默认路径。固定状态脚本新增 `--split`，默认仍为 trainingset，本轮显式使用 testingset。

## 清理

已删除以下废弃路线的 checkpoint 文件和 tracking result 文件：STSER SOT/content quick、STSER competition quick/full 的权重与结果。保留其 `experiments/reports/`、logs、摘要、快照和资产登记。M1、4090D 复现组、Early gate full15 和本轮 DCVR 权重均保留。

## 下一步

先在预注册的 60 序列 matched manifest 上复跑阶段 3 control/candidate，并保留逐序列配对置信区间；若 `ab_bikeoccluded` 优势消失或收益集中度仍过高，停止当前 DCVR 算子扩展。只有在大 manifest 上出现跨序列正向且阶段 1 的 Value→CENTER 传导证据改善后，才考虑 full-15/full-244。

主要证据：

- `/autodl-fs/data/tbsi_dcvr/phase1_m1_off_retry3/`、`phase1_m1_on_retry3/`
- `/autodl-fs/data/tbsi_dcvr/phase2_degradation_m1_off_smoke/`、`phase2_degradation_dcvr_smoke_retry1/`
- `output/experiments/M1_DCVR_QUICK3_CONTROL_20261025/`
- `output/experiments/M1_DCVR_QUICK3_CANDIDATE_RETRY1_20261025/`
