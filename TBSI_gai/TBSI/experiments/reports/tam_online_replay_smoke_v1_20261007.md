# TAM online image replay smoke v1

日期：2026-10-07  
父实验：`TAM-TRACE-SMOKE-20261007`  
实验级别：auxiliary/dev；training-contract smoke，不是跟踪评测

## 问题与合同

验证冻结 M1 是否能从真实 LasHeR train 图像按已接受的 tracker crop 回放，在线生成当前帧和上一帧 raw head token，并仅优化因果 TAM 旁路。GT 只用于 actor 的事后训练目标；crop、历史顺序和序列归属全部来自冻结 M1 trace。没有启动 full-data training、full-244 或任何 AUC 比较。

固定范围为 `abreastinnerboy`、`biketurndark`、`right2ndfarboytakinglight2left` 三条已有 trace 序列，共 1374 行、1371 个历史链接；manifest 由 trace 的实际 crop 矩形连接到真实图像路径，未用 GT 重裁。M1 checkpoint 为 epoch 15，SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。

## 执行

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
TBSI_INFER_FP16=0 PYTHONUNBUFFERED=1 ./tools/run_causal_tam_image_replay_smoke.sh
```

实现文件：

- `tools/generate_m1_crop_replay_manifest.py`
- `tools/run_causal_tam_image_replay_smoke.py`
- `tools/run_causal_tam_image_replay_smoke.sh`

运行环境：1x NVIDIA GeForce RTX 4090 D；Python 3.8.20；PyTorch 1.9.0+cu111；CUDA 11.1；seed 42；`TBSI_INFER_FP16=0`。代码快照：`09f3d57` + working tree；canonical baseline YAML SHA256：`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`，运行前后未改动。

## 结果

最终固定 6 个连续历史对通过：

- `TAM_IMAGE_REPLAY_SMOKE_PASS`
- 所有 6 个 loss、raw token、TAM 输出和梯度均 finite
- `gamma` 从 0 开始，末步为 `0.0001809388`
- 首步 gate 打开后残差比例为 0；后续最大记录比例为 `5.55e-05`，低于 0.10 上限
- 非 TAM 参数及 buffer state digest 保持不变
- 当前/历史 token 均满足 `[1,256,768]`
- 未计算 AUC、PR、NPR、OP50、OP75、FPS；这些指标为 `NA`

## 失败与修复记录

第一次 image smoke 因脚本把返回的 raw feature 直接以错误维度交给 actor，停止在第一步，错误为 `current must have shape [B, grid_size^2, dim]`。修复为对 feature map 显式 flatten，并保持 batch 维后重跑。此前 manifest 初次生成还因每个 trace row 重复读取 checkpoint 计算 SHA256 被中止；随后改为每次运行只计算一次。两次失败输出和日志保留，不进入模型结论。

## 判断

本实验只支持“真实图像在线 raw-token 回放和 TAM-only 训练接口可执行”。它不支持 TAM 提升跟踪，也不授权 full-244。下一步若继续，只能先做同一 image-backed actor 的小规模固定训练审计和 gamma=0/冻结状态关门核对；不得把本 smoke 的 loss 或 gamma 变化解释成任务收益。

证据：`output/experiments/tam_online_replay_smoke_v1_20261007/m1_crop_replay_manifest.json`、`smoke_summary.json`、`run.log`。

## 后续 parity 审计（2026-10-07）

从每条序列均匀选取 5 个有历史当前帧，共 15 个样本，使用同一 checkpoint、同一 FP32 配置和 trace crop 做 online raw-token 重算。结果为：shape 全部一致 `[1,256,768]`，数值均 finite，crop 元数据一致；但 checksum 一致数为 `0/15`，mean abs diff 的样本均值为 `0.11990465`，最大 abs diff 为 `5.41395235`。

因此当前状态更新为 `TAM_IMAGE_REPLAY_PARITY_STOP`：真实图像路径和接口可执行，但尚未证明其逐项复现 accepted M1 raw trace。未启动 100/500-step image actor、RPP-v1 proxy 或 full-244。下一步只允许继续核对原 tracker 的图像读取、模板输入、帧索引和 preprocessing；parity 关闭前不产生训练稳定性或性能结论。

证据：`output/experiments/tam_online_replay_smoke_v1_20261007/parity_audit.json`。

## Parity 修复与 100 步训练合同审计

先前的 parity STOP 已定位并修复了三项回放差异：红外图像必须保持 tracker 的三通道 BGR→RGB 读取；trace `frame_id` 直接对应 zero-based dataset image index；模板必须使用 `init.txt` 初始化框和 dataset index 0 图像。修复后每条序列均匀选取 5 个样本，共 `15/15` raw token checksum 一致，shape/crop 一致，`max_abs_diff=0`。

随后在同一 manifest 上执行 100 个真实图像历史对的 TAM-only actor audit。结果：gamma `-9.99999e-05→0.0118128`；gate 打开后 Q/K/V/out 梯度均非零；coverage 范围 `0.703125–1.0`；最大残差比例 `0.1000000089`，仅为 float32 舍入超过 0.10 的约 `9e-9`，在 `1e-6` 合同容差内；所有 loss/梯度/输出 finite；非 TAM state 未变；optimizer state 含 9 个 TAM 参数张量；checkpoint 保存/恢复通过。

这一步支持“真实图像→精确 M1 raw token→TAM→head loss→TAM update”的稳定训练合同，不支持跟踪性能或 full-244 结论。证据：`output/experiments/tam_online_replay_train_audit_v1_20261007/smoke_summary.json`、`tam_image_replay_smoke_checkpoint.pth`。

## 全训练 metadata replay 尝试

按冻结的 `ffm_train_sequences.txt`（152 条 train sequence）启动串行 M1 metadata-only replay，使用 `TBSI_TRACE_SAVE_RAW=0`，只保存实际 tracker crop/state 的 JSONL。运行在完成 5/152 条、12170 帧和 12165 个历史链接后人工中止；完成序列为 `rightbackcup`、`boywithwhitebackpack`、`nearestleftblack`、`blackdresswithwhitefar`、`left4thgirlwithwhitepants`。首条长序列实测约 42.5 FPS，但按当前入口估算剩余运行时间过长，未将部分输出当作全训练 manifest。

状态：`M1_METADATA_REPLAY_PARTIAL_RESOURCE_STOP`。部分结果保留在 `output/experiments/m1_train_metadata_replay_v1_20261007/`；尚未生成完整 sampler audit，不授权 RPP-v1 proxy training。后续需要优化为单模型/跨序列复用或批处理入口后，再重新生成完整 152 序列 manifest。
