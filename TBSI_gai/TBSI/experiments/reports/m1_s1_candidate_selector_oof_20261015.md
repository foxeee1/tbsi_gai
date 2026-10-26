# M1 候选级选择器 OOF 诊断（2026-10-15）

## 结果

在 72 条 train-side 序列、144 个固定状态上，冻结 fused/RGB-only/TIR-only 三个候选，只用候选框几何特征做 sequence leave-one-out ridge selector。GT 只用于训练折标签和留出评估，推理选择不读取 GT。

| 方法 | 平均 IoU | 中位数 IoU | 相对 M1 fused | 序列聚类 95% CI |
|---|---:|---:|---:|---:|
| M1 fused | 0.64382 | 0.69920 | 0 | [0, 0] |
| OOF geometry selector | 0.62091 | 0.67810 | **-0.02291** | **[-0.03508, -0.00279]** |
| Best-of-three GT oracle | 0.68778 | 0.71826 | +0.04396 | [+0.02014, +0.05297] |

selector 的选择次数为 fused/RGB/TIR = `129/12/3`；与 oracle 选择一致的比例为 `53.47%`。相对 fused，只有 `5/144` 状态改善，`9/144` 状态退化。oracle 的互补候选信息存在，但当前几何特征不能稳定识别它。

## 归因

结论标签：**已排除当前几何 selector 作为可用路由器**；**已支持候选集合继续作为诊断工具**；**暂不支持接入闭环或 full-244**。

失败原因更接近表示/置信度缺失，而不是候选不存在：单步候选框的几何关系无法区分“单模态偶然偏好”与“真正包含目标的候选”。当前 selector 没有使用候选响应图、峰值置信度、熵、跨模态响应分歧和历史 state 一致性，因此不能判断候选是否可靠。

同时，selector 只在固定 crop/state 上评估。它没有解决 crop 内中心选择错误、目标出界和后续状态反馈问题，不得外推为 tracking AUC 提升。

## 下一步

停止当前几何 selector 的扩展，不接入 tracker。下一项最小实验只增加候选级 response evidence：保存三候选各自的 peak score、top-2 margin、entropy、peak cell、候选框中心/尺寸分歧和相对 M1 state 的位移，再做同样的 sequence-LOSO selector。状态提交继续冻结为 M1 state。若加入 response evidence 后留出收益仍接近零，则候选路由的瓶颈在 crop/center-cell 生成或候选 head 本身，应停止后融合修正路线；若出现稳定正收益，再做 output-only 闭环。

## 产物

- OOF 指标：[metrics.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_S1_CANDIDATE_SELECTOR_OOF_20261015/metrics.json)
- 逐状态：[per_state_selector.csv](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_S1_CANDIDATE_SELECTOR_OOF_20261015/per_state_selector.csv)
- 逐序列：[per_sequence_selector.csv](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_S1_CANDIDATE_SELECTOR_OOF_20261015/per_sequence_selector.csv)
- 带候选框标量坐标的候选审计：[retry3](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015_retry3/metrics.json)
