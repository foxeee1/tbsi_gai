# M1 Early Template Value Reconstruction Fixed-State Screening

## 运行合同

- Parent: `M1_SCREENING_FIXED_STATE_FLOW_CALIBRATION_V8_20261022`，其 checkpoint 为受保护 M1 之外的同一 fixed-state calibration parent。
- Candidate: `M1_EARLY_TEMPLATE_VALUE_RECON_SCREEN_20261023` 的 1-epoch checkpoint，仅训练新增 `EarlyTemplateValueReconstruction` 参数。
- 范围：LasHeR trainingset calibration cohort，23 个序列，46 个固定状态（每序列 `pre_onset` 与 `severe_onset`）。
- 目的：验证 cross-modal Value 聚合前的证据条件化 template Value 重构，是否改变目标/背景结构、CENTER 排序和固定回放定位。
- 解释边界：这是机制筛选，不是 full tracking 性能结论。

## 合同预检

`py_compile`、zero-init disabled parity、forward shape、保存加载和单 batch 梯度检查通过。零初始化时输出最大绝对误差为 `0.0`；`up` 参数有非零梯度，`down` 首轮梯度为零符合零初始化 up 的预期。最小可拟合性训练完成 8 个 batch，loss 全程有限。

## 与 M1 的探索性结果（不可作因果归因）

早期汇总曾把 candidate 与 `M1_SCREENING_FIXED_STATE_FLOW_CALIBRATION_V8_20261022` 按 `event_id` 配对，并得到下表。但两个分支使用了不同 checkpoint：parent 是受保护 M1 epoch15，candidate 从 Early-Conditioned Gate full-15 checkpoint 初始化后又训练 1 epoch。因此这些差值混合了 parent 模型差异、前序 full-15 Gate 训练及新增重构参数，**不能用于判定重构算子的因果效果**，仅保留为探索性现象：

| 指标 | 平均变化 candidate-parent | 胜/负/平 |
|---|---:|---:|
| fixed replay IoU | `+0.04346` | 23/20/3 |
| CENTER GT hard-background margin | `+0.00436` | 27/19/0 |
| CENTER GT rank（越低越好） | `-0.23913` | 7 个状态改善、1 个退化、38 个不变 |
| post-fusion query hard-background margin | `+0.00494` | 25/21/0 |
| post-fusion query strongest-background margin | `+0.00104` | 23/23/0 |
| post-fusion query GT rank（越低越好） | `+0.26087` | 7 个改善、13 个退化、26 个不变 |

分层结果显示明显的阶段异质性：

| 阶段 | replay IoU 平均变化 | CENTER margin 平均变化 | CENTER rank 平均变化 |
|---|---:|---:|---:|
| pre_onset | `-0.03968` | `-0.00096` | `+0.04348` |
| severe_onset | `+0.12659` | `+0.00968` | `-0.52174` |

severe-onset 的正向变化主要由少数序列贡献（`rightof2boys`、`theleftestrunningboy`、`Aboydownbike`、`bike150`、`whiteboy242`），同时 `Amidredgirl`、`outer2leftmirrorback`、`blkboywithglasses` 等序列退化。因此不能把总体均值直接解释为稳定收益。

## 机制判读

1. 输出 Value 的 target norm 平均从 `17.8668` 增至 `18.5795`（约 `+4.0%`），background norm 从 `24.6759` 增至 `25.0696`（约 `+1.6%`）。target-background cosine 从 `0.6586` 增至 `0.6681`，绝对变化约 `+0.0096`；这说明存在有限的方向/关系变化，但幅度变化占主导。
2. CENTER rank 在 46 个状态中 38 个不变，post-fusion GT rank 也有 26 个不变。因而该算子没有稳定穿透 CENTER 的离散选峰瓶颈。
3. fixed replay IoU 的改善与 CENTER margin 变化相关（Pearson `r≈0.606`），与 post-fusion hard-background margin 的相关性较弱（`r≈0.170`）。这更接近局部几何/幅度扰动改变回归结果，而非一致的目标身份重排。
4. pre-onset 下降而 severe-onset 上升，说明当前 1-epoch 新分支尚未学到跨阶段一致的重构规则；也可能存在短训和 parent 初始化的混杂，不能归因于算子在完整训练下必然失败。

## 结论

**结论分类：`INCONCLUSIVE / MECHANISM-SIGNAL-BUT-NOT-ADMISSION`。**

合同、梯度和固定状态执行均通过；severe-onset 有局部正信号，但 CENTER 排序大多不变，且 pre-onset 反向。当前证据不足以启动 15 epoch 全量训练，也不足以宣称模板 Value 重构解决了 M1 的定位失败链条。

## Strength 0/1 配对对照

使用同一 1-epoch candidate checkpoint、同一事件清单和同一 tracker，只将 `EARLY_TEMPLATE_VALUE_MAX_DELTA` 从 `0.10` 改为 `0.0`。结果如下：

| 指标 | strength-1 减 strength-0 | 胜/负/平 |
|---|---:|---:|
| fixed replay IoU | `-0.000248` | 18/24/4 |
| CENTER GT rank | `0.000000` | 0/0/46 |
| CENTER GT hard-background margin | `+0.000101` | 24/22/0 |
| post-fusion query GT rank | `0.000000` | 0/0/46 |
| post-fusion query hard-background margin | `+0.000051` | 27/19/0 |
| fusion output target norm | `-0.004037` | 15/31/0 |

pre-onset IoU 变化为 `-0.000015`，severe-onset 为 `-0.000482`。因此第一轮 candidate 与 strength-0 的差异几乎不改变离散排序，不能解释上一节 severe-onset 的正向 fixed-state 差异。当前算子在这个 checkpoint 和 `max_delta=0.10` 合同下没有可观测的定位传导效应。

## 最终筛选结论

**结论分类更新为：`REFUTED-AS-LOCATING-MECHANISM / STOP-CANDIDATE`。** 合同和梯度路径是有效的，但 strength 0/1 的因果对照排除了“该重构残差稳定改善 CENTER/回放定位”的解释。停止该候选的 3-epoch 或 15-epoch 训练，不叠加模板更新或第二个融合模块。
