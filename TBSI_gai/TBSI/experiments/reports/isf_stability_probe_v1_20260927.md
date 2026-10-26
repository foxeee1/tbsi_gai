# ISF Stability Probe v1

日期：2026-09-27  
状态：`DIAGNOSTIC`

## 目的

在固定 `single_modality_blur_v2` 的 72 个事件上，从相同 tracker state 和 RNG 快照分叉，比较第一层 `ca_s2t_v2f` 前 64 个 template message 的局部模态稳定性与当帧 IoU 下降是否相关。不训练、不改权重、不改 canonical YAML。

## 协议

- 条件：`00`、`R_target_blur`、`T_target_blur`、`R_background_blur`；局部 Gaussian blur sigma=3.0。
- 数据：24 条 LasHeR train 序列、72 事件、每事件 4 条 forward，共 288 条结果。
- checkpoint：epoch 15 canonical checkpoint；FP32，`TBSI_INFER_FP16=0`。
- message：`tbsi_layers[0].ca_s2t_v2f.attn_reshape[:, :64, :]`，预期 shape `[1, 320, 768]`。

## 结果

| 条件 | mean msg_cos_drop | mean iou_drop | Pearson r | Spearman rho | 95% CI(iou_drop) |
|---|---:|---:|---:|---:|---:|
| RGB target blur | 0.03433 | 0.05555 | 0.09897 | 0.07089 | [0.02925, 0.08647] |
| TIR target blur | 0.00017 | 0.02093 | 0.18524 | 0.13848 | [0.00225, 0.04893] |
| RGB background blur | 0.01139 | -0.00171 | -0.01066 | -0.17532 | [-0.00538, 0.00057] |

Hook audit 通过：72 事件均为 4 条件、输出 shape `[1, 320, 768]`，每个事件各条件 hook 调用数一致。目标区域模糊确实带来可测消息变化和 IoU 下降，但消息变化与 IoU 下降的相关性很弱；背景对照不显示正向关系。因此本轮不支持“局部模态干预下 message stability 可作为定位损失机制指标”的主张，不启动新融合模块训练。

## 证据与复现

- 原始结果：`output/experiments/isf_stability_probe_v1/batch/event_results.csv`
- hook 审计：`output/experiments/isf_stability_probe_v1/batch/hook_audit.json`
- 元数据：`output/experiments/isf_stability_probe_v1/batch/metadata.json`
- 分析：`output/experiments/isf_stability_probe_v1/batch/analysis/analysis.json`
- 脚本：`tools/isf_stability_probe_v1.py`、`tools/analyze_isf_stability_probe_v1.py`
- manifest SHA256：`45c13f5721a181a1e88a109a42acba83b589cbc9680b4a00551d095295264683`
- canonical YAML SHA256：`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
- 环境：1× RTX 4090 D 24GB；Python 3.8.20；PyTorch 1.9.0+cu111；FP32。
