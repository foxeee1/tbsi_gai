# Single-Modality Blur v2：新序列复核

日期：2026-09-26  
状态：`DIAGNOSTIC`；用于复核当帧 RGB 模糊现象，不是未见训练验证集

## 协议

从旧 SJI/单模态 v1 使用过的 24 条序列之外，固定抽取 24 条 LasHeR train 序列，每条按 `round(linspace(1, n_frames-1-horizon, 3))` 取 3 个事件，共 72 个事件。序列和帧在运行前写入 `output/experiments/single_modality_blur_v2/manifest/manifest.json`，不依据 tracker IoU 选样本。epoch-15 checkpoint 可能已经见过这些 train 序列，因此不能称为 unseen validation。

只比较 `00` 原始输入和 `10` RGB 全图高斯模糊（sigma=3.0）；TIR 不变。每个事件从相同 tracker 状态和 RNG 快照分叉，记录 offset 0--5。尺寸层预先固定为目标面积占图像面积 `<1%`、`1%--5%`、`>=5%`。

## 结果

`drop = IoU00 - IoU10`：

| 范围 | 均值 | 中位数 | 序列 bootstrap 95% CI | 去掉最大 3 事件 |
|---|---:|---:|---:|---:|
| offset 0 | 0.11261 | 0.07137 | [0.08128, 0.14797] | 0.08851 |
| offset 1 | 0.02193 | 0.00640 | [0.00561, 0.04087] | 0.02183 |
| offset 2 | 0.01270 | 0.00041 | [0.00150, 0.02863] | 0.01270 |
| offset 3 | 0.00087 | 0.00014 | [-0.00524, 0.00683] | 0.00087 |
| offset 4 | 0.00522 | 0.00122 | [0.00097, 0.01010] | 0.00522 |
| offset 5 | 0.00527 | 0.00015 | [0.00029, 0.01079] | 0.00527 |

offset 0 去掉最大三分之一事件后均值仍为 `0.03790`。尺寸层 offset 0 均值为 small `0.13914`、medium `0.08165`、large `0.04187`；对应序列区间均在零以上，但 large 只有 9 个事件、7 条序列，应视为较弱分层证据。

## 结论边界

新固定序列复核支持继续研究“强 RGB 模糊下的当帧融合鲁棒性”：当帧损失跨多条序列存在，且不由前三个事件单独支撑。它不支持持续模板污染的表述；后续帧损失显著减弱，offset 3 的区间跨零。

本实验只说明受控全图高斯模糊下的 GT 评测现象，不能直接称为真实安防摄像头故障，也不能声称模型未见过这些序列。

## 证据与复现

- Manifest：`output/experiments/single_modality_blur_v2/manifest/`。
- 原始结果：`output/experiments/single_modality_blur_v2/batch/blur_event_results.csv`，864 行，72 事件，0 跳过。
- 分析：`output/experiments/single_modality_blur_v2/batch/analysis_v1/analysis.json`。
- 脚本：`tools/build_single_modality_blur_v2_manifest.py`、`tools/single_modality_blur_v2.py`、`tools/analyze_single_modality_blur_v2.py`。
- 运行环境：1× RTX 4090 D；Python 3.8.20；PyTorch 1.9.0+cu111；FP32 tracker inference；`TBSI_INFER_FP16=0`。
