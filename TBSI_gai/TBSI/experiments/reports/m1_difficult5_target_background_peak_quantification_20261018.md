# difficult5 目标峰与背景峰量化

## 定义

对每一帧的 16x16 CENTER map 先应用推理时 Hann 窗。目标峰是与 GT 框有空间重叠的 cell 中最高分，背景峰是所有非 GT cell 中的最高分：

\[
s_t=\max_{u\in\Omega_{gt}} s(u),\quad
s_b=\max_{u\notin\Omega_{gt}} s(u),\quad
\Delta=s_t-s_b.
\]

`rank` 是目标峰在整张 Hann map 中的降序名次。失败帧定义为 `Δ <= 0`。数值来自当前候选 checkpoint 的 fixed-state trace，五条序列均使用同一 M1 state trajectory 作为输入。

## 全序列分布

| 序列 | 有效帧 | 失败帧 | 目标峰均值 | 背景峰均值 | margin 均值 | 目标 rank 均值 | 背景峰距均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `manoncall` | 520 | 43 | 0.5512 | 0.0965 | 0.4547 | 1.28 | 2.47 |
| `boyss` | 327 | 80 | 0.4503 | 0.2954 | 0.1549 | 3.54 | 2.65 |
| `girldownstairfromlight` | 1154 | 157 | 0.5004 | 0.1560 | 0.3444 | 5.62 | 2.56 |
| `bike2left` | 256 | 15 | 0.6979 | 0.0427 | 0.6552 | 1.30 | 3.61 |
| `girlinrain` | 311 | 0 | 0.8246 | 0.0078 | 0.8168 | 1.00 | 3.16 |

## 失败帧分布

| 序列 | 失败帧 | 目标峰均值 | 背景峰均值 | margin 均值 | rank 均值 | 背景峰距均值 |
|---|---:|---:|---:|---:|---:|---:|
| `manoncall` | 43 | 0.2293 | 0.2904 | -0.0612 | 4.42 | 2.44 |
| `boyss` | 80 | 0.2533 | 0.4654 | -0.2122 | 7.54 | 3.05 |
| `girldownstairfromlight` | 157 | 0.1108 | 0.3384 | -0.2276 | 33.52 | 4.22 |
| `bike2left` | 15 | 0.0607 | 0.3385 | -0.2779 | 51.40 | 7.79 |
| `girlinrain` | 0 | NA | NA | NA | NA | NA |

## 共性与差异

共同点是目标背景 margin 接近或小于零，且背景峰并非随机低分噪声，而是具有足够高响应的竞争峰。`girlinrain` 没有该现象，目标峰均值 0.825、背景峰均值 0.008，说明当前困难主要是峰竞争而非所有序列的模态融合失效。

失败峰有两个明显子型：

1. **近距离竞争型**：`manoncall`。目标 rank 仍约 4，目标和背景分数接近，背景峰距约 2.4 token。需要利用局部峰形、模态一致性和时间连续性做细分。
2. **目标抑制/远背景型**：`girldownstairfromlight` 与 `bike2left`。目标峰很低，rank 分别约 33.5 和 51.4；背景峰距增大到 4.2 和 7.8 token。单纯提高目标峰或使用中心先验会误伤这类样本。

`boyss` 同时具有较高背景峰和中等 rank，属于过渡型。它的恢复不能简单解释成目标峰翻转。

## 推荐建模方式

把问题建模为 **候选峰效用排序**，而不是直接回归一个 RGB/TIR 融合门：

1. 从 Hann map 取 top-K 候选峰，并为每个候选保留 `score`、局部 PSR、top1-top2 margin、峰宽/二阶曲率、到 crop 中心距离。
2. 在候选位置提取 RGB/TIR 各自 score、模板相似度、模态差异、跨模态对应一致性和 reliability probability。
3. 加入时间特征：相对上一状态的位移、尺度变化、上一帧峰的连续性、候选是否落在上一框预测邻域。
4. 用一个小型共享 MLP 对每个候选输出 utility，再用 listwise softmax 或 pairwise hinge 排序。训练标签使用候选框与 GT 的 IoU/中心距离，困难背景峰作为 hard negative。

建议效用形式：

\[
u_i = f_\theta([s_i,\mathrm{PSR}_i,\Delta_i,d_i,
s_i^r,s_i^t,|s_i^r-s_i^t|,c_i^{rt},
\delta_i^{temp},\rho_i^{rgb},\rho_i^{tir}]),
\]
\[
L=\max(0,m-u_{gt}+u_{bg}^{hard})+\lambda L_{smooth}.
\]

其中 `bg_hard` 应从当前 top-K 背景中动态选择，`L_smooth` 约束相邻帧 utility 不发生无依据的剧烈跳变。推理时由 utility 选择峰，再交给原 CENTER box decoder 和 state update，避免改变已有状态合同。

## 下一步最小验证

固定当前 checkpoint，不训练 backbone 和 CENTER head；先做离线候选排序可拟合性检查：

- 每帧取 top-16 候选，按 GT IoU 生成标签；
- 训练/验证按序列划分，防止同一序列相邻帧泄漏；
- 对比 raw score、`score + center prior`、小型 utility MLP 三种排序；
- 主要指标为失败帧 top-1 修正率、GT rank、top-5 recall、margin 和闭环首次分叉帧，而不是只看总体 AUC；
- 只有离线排序在两类失败子型都有效，才进入 tracker 闭环测试。

证据目录：`output/experiments/M1_AMCM_DR_HANN_FAILURE_PAIRWISE_PROXY_20261018/fixed_state_signal_audit_difficult5/trace_amcm/`。
