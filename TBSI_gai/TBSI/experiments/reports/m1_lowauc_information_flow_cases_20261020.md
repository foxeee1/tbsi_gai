# M1 低 AUC 代表序列信息流诊断（2026-10-20）

## 问题与结论

目标是在有改善迹象的低 AUC 序列中，定位目标/背景区分首次可观察到的减弱，区分表征变化、CENTER 读出和闭环反馈，并判断旁路后融合与主回路各自能处理的问题。

本次使用冻结的 M1 epoch15 checkpoint，在 `ab_bikeoccluded`、`3rdfatboy`、`leftfarboycomingpicktheball` 三条 Cdiag full-244 大幅增益序列，以及 Cdiag 显著退化的 `rightbottlecomes` 上，分别取 replay trace 中首个 crop-contained、连续 30 帧 IoU<0.5 的起点和前一帧。每帧从 trace 的同一 `state_before` 重建 crop；记录 ViT block 输出、TBSI loc3/6/9、最终 backbone norm、`tbsi_fuse_search` 前后几何和 CENTER score map。block/TBSI margin 是搜索 token 对模板前景原型与固定 hard-background 位置的 cosine 差，不是 CENTER logit；它用于找表征信号变化，不能单独解释最终预测。无训练、无参数干预。

**判定：尚未区分一个统一的算子首因。** 这四例呈现至少两种失效形态：早层 RGB/TIR 目标证据从一开始就弱或发生分化；以及 CENTER 的位置/框解码失配。MHSA 中的早层变化是最早被测到的表征差异，但并未通过 block 级干预证明为首因。`tbsi_fuse_search` 后目标区域与背景位置的 feature L2 通常增大，不能据此称为目标选择性修正；已有 search-stage 资产也未显示它是普遍的破坏环节。状态反馈会把局部读出误差放大成后续 crop 漂移和出界，但本次固定帧分析不重新估计该放大幅度。

## 资产与口径

- Parent：受保护实验 `TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15。配置 SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，均与资产记录一致。
- 数据：`/root/autodl-tmp/TBSI_gai/data/lasher/testingset/` 下四条序列；追溯性 case study，不能推及总体或作为验证集结果。
- 固定状态来源：`M1_FUSION_FLOW_FULL244_DIAG_20261009/closed_loop_trace/trace`。该 trace 是当前工作树 replay，不等同 archive 逐帧 parity；四条序列的两帧 fixed replay IoU 均与此 trace IoU 一致。
- 执行：FP32、uncertainty mode `off`、无训练；1× NVIDIA GeForce RTX 4090 D；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8。运行环境与 M1 锚点软件环境不同，故为冻结 checkpoint 的诊断 replay，不称为严格 M1 环境复现。
- 代码快照：Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；工作树有 dirty changes，因此 `summary.json` 同时记录执行路径上 12 个代码文件的 SHA256；runner SHA256 `80254c3972e3f41cbb59bb8d4ec5c5a0903ce2c8da2650ced1a1037c804935e5`。
- 完整逐事件/逐层数据：`output/experiments/M1_LOW_AUC_INFORMATION_FLOW_CASES_20261020/{per_event.csv,stage_metrics.csv,selected_events.json,summary.json,run.log}`。

## 逐事件结果

| 序列 | M1 AUC → Cdiag AUC | 前帧 → onset IoU | CENTER top-1 到 GT 中心距离（格） | onset 的最早可见特征现象 |
|---|---:|---:|---:|---|
| `ab_bikeoccluded` | 8.93 → 65.92 | .601 → .365 | .89 → .05 | block00 的 RGB margin 为 -.100，TIR 为 +.212；到 block02 RGB/TIR 已分别回正至 +.186/+ .355。早层 RGB 弱信号被后续 block 修复，CENTER top-1 在 onset 落回 GT 中心；但预测 IoU 仍仅 .365，优先怀疑框尺度/offset 或 GT 区域定义，而非错峰。 |
| `3rdfatboy` | 18.53 → 73.93 | .764 → .258 | .44 → 1.00 | block00 RGB/TIR margin 从前帧 .138/.170 降至 .076/.058；随后继续衰减，block08 RGB 仅 +.009，最终 norm RGB -.009。TBSI loc3/6 局部略有恢复但没有扭转；onset CENTER top-1 偏离 GT 中心约一格。属于早期模态证据走弱并传至读出的例子。 |
| `leftfarboycomingpicktheball` | 19.49 → 67.75 | .573 → .339 | 1.29 → .94 | onset RGB 在 block00/02 仍强（+.171/+.165），TIR margin 从 block00 +.071 到 block02 -.016，block05 仍近零；loc3 后稍回到 +.001，loc6/9 再近零或为负，最终 norm TIR -.038。最早分化在早层 TIR，融合层仅局部补回，没有持续修复。CENTER top-1 距中心不到一格但 IoU 下降，解码/尺度仍需检查。 |
| `rightbottlecomes` | 29.16 → 14.33 | .576 → .372 | 1.58 → 1.68 | RGB 从 block00 起 margin 约 +.002，之后始终接近零；TIR block00 仅 +.008，后续上升到最终 norm +.113。定位 top-1 前后都在 GT 中心外约 1.6 格。可见 TIR 语义证据增强没有自动校正中心选择；Cdiag 在此类输入上明显有害。 |

这些 AUC 数值来自已完成的 Cdiag full-244 报告，只用于说明案例为何入选；本次不是 Cdiag 的复现实验或因果比较。

## 信息流解释

1. **初始 patch/token 到 MHSA blocks。** 本诊断最早可观测点是 block00 的输出，不含 patch embedding 前后的原始图像可分性。因此只能说“到 block00 已见弱证据”，不能判定偏差由数据、patch embedding、位置编码还是 MHSA attention 首先生成。`ab_bikeoccluded` 表明早层弱模态可被 block02 修复；`3rdfatboy` 和 `leftfarboycomingpicktheball` 则显示早层下滑/模态分化会继续传到后层。
2. **TBSI loc3/6/9。** 在 `3rdfatboy` 与 `leftfarboycomingpicktheball`，可见局部 margin 回升，但到后层没有稳定保持。`rightbottlecomes` 的 TIR margin 在后层提高，却没有对应中心回归。故 TBSI/MSHA 既不是全然失活，也没有证据表明它是统一修正器或统一破坏源。
3. **`tbsi_fuse_search`。** 四例中融合输出的 target-mean 与固定 hard-background feature L2 多数高于单模态输入，但 RGB/TIR 到融合输出的 cosine 一致性也更高。L2 变大只说明几何距离改变，不说明方向把目标排在背景之前；本次没有做融合旁路/置零对照，不能把该操作归类为因果性修正或放大。此前 search-stage probe（72 train-side sequences、104 held-out events）显示失配前驱的 margin 在融合输入/输出均为负，绝对值中位数由 15.913 降至 4.377；严重起点由 18.220 降至 6.038，不支持普遍的二次恶化。
4. **CENTER。** 原先用 GT 区域最大响应计算 rank 会掩盖中心偏移，因此本版补充了 GT 几何中心格、top-1 到中心距离与中心/峰值响应。四例说明 rank/峰值 margin 单独不够：`3rdfatboy` 是明显错位；`ab_bikeoccluded` onset 的 top-1 正在 GT 中心但 IoU 低；另两例 top-1 距中心不足两格却 IoU 仍低。下一层应读取 CENTER 的 size/offset 在 GT center cell 与峰值 cell 的解码，并拆分中心误差和框尺度误差。
5. **状态闭环。** 既有 M1 轨迹资产显示，失配段多数起点目标仍在 crop 内，而后续相当一部分事件出界，首次出界距起点中位数 19.5 帧；因此错误读出之后的 state 更新/crop 重采样是可确认的误差放大链条。CENTER 局部 oracle 的严重 onset IoU 上限高于实际 top-1，也表明“局部有可达解”但不等于推理信号能选中它。本次固定帧不测试反馈干预。

## 后融合与主回路的适用边界

| 观察到的问题 | 后融合旁路可解决的范围 | 主回路应承担的范围 | 当前证据边界 |
|---|---|---|---|
| 融合前有正确目标位置/模态证据，经过融合后排序才丢失 | 可在融合后修正已存在的 feature margin；必须对准 GT 与 hard-background 的 signed margin，并保证进入 CENTER 后仍保留 | 如果丢失在 TBSI 内或模态表征本身，后融合看不到已丢失的可辨信息 | 现有 search-stage 资产未支持融合是普遍破坏点；多个 residual/NLformer 候选有非零注入但背景主导、rank 无改善，不能靠放大强度解决缺少目标选择性的问题 |
| RGB/TIR 早层证据弱或相互矛盾 | 只有当另一模态或模板仍保留可定位目标的信号时，后融合才可重新加权/传递；需有可靠可观测路由信号 | 改进目标条件化的 token 交互与模态选择，让所需信息在 loc3/6/9 后仍可被读出；但须先证明干预改变后续 CENTER 排名 | 单模态 oracle 平均有 headroom，但简单 OOF selector 5/144 改善、9/144 退化；目前“有信息”与“可部署选择”之间有缺口 |
| 特征仍区分目标，但 CENTER 选峰/框解码错 | 只改 CENTER 前 feature 可能扰动峰值，不能保证尺寸/offset 或标签坐标对齐变好 | 优先检查 CENTER score、Hann window、GT cell、offset/size 解码和训练损失对齐；错误峰局部可达时再测读出机制 | 严重 onset 的 CENTER 局部 oracle 有明显 headroom；top-k ranker 在 held-out 低于 no-op，故现有可见信号不足以部署自动选峰 |
| 错峰后持续漂移、目标出界 | 每帧后融合只能影响当前 crop 内特征，不能单独挽回已经出界或错误状态导致的无目标 crop | 状态更新应加入失败拒绝、恢复/重定位或可校准的置信度，单独验证状态转移与恢复 | 失配到出界链条有既有轨迹证据；尚未证明哪一种恢复机制能稳定改善闭环 |

## 下一步判别实验与结论状态

结论状态：**已排除**“所有低 AUC 案例都由 `tbsi_fuse_search` 普遍抹掉已有目标 margin”；**暂不支持**“MHSA 某个单层是已证实首因”；**支持下一步单变量检验** CENTER 解码与早层表征的分组诊断。

优先在独立 train-side 序列、同一 crop/初始化模板/状态、失败 onset 与 matched-good 控制上，做两个可逆、分开报告的测试：

1. 固定最终 feature，只离线交换 GT-center 与原始 peak cell 的 CENTER size/offset 读出，验证四个案例中低 IoU 是否主要来自位置还是尺度；同时用非目标 cell 对照，不能把 GT oracle 当部署策略。
2. 固定 CENTER/head，分别对 block00→block02 的早层流、TBSI loc3/6/9 输出做零化/替换或局部 token swap，测目标中心/最强背景 signed margin 是否改变并传至 CENTER。若 earliest intervention 仍不改变 CENTER，则停止把该层当修复靶点。

仍应停止无差别扩展旁路模块。现阶段旁路的能力边界是“重排尚存的局部信息”；主回路的能力边界是“改变信息生成/交互，但需要监督让变化对准目标排序”。两者都不能仅凭特征距离、模块残差能量或小样本闭环增益宣称解决目标/背景区分。
