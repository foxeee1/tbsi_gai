# CFVI P1 频率机制重定义审计

## 审计范围

本次只做代码、现有 checkpoint、训练诊断和模板数据管线审计，没有训练新模型，也没有用短训结果做候选淘汰。对象是 `CFVI_P1_FULL_MATCHED_M1_seed42` full matched run。

## 1. 实际频率算子

P1 的输入是 TBSI-E location 9 的模板 token：每个模态为 `[B, 64, 768]`。实际路径为：

```text
[B,64,768]
  -> [B,768,8,8]
  -> Haar: LL/LH/HL/HH, each [B,768,4,4]
  -> bilinear upsample, each [B,768,8,8]
  -> concat [B,3072,8,8]
  -> 1x1 Conv 3072->768 + GELU
  -> flatten + LayerNorm
  -> [B,64,768]
  -> lambda scaling
  -> add to the opposite modality's V
```

Haar 的四个固定组合在 [cfvi_module.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/cfvi_module.py:13)；投影和上采样在 [cfvi_module.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/cfvi_module.py:68)；固定四带注入在 [cfvi_module.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/cfvi_module.py:106)。

这不是严格的 Haar 可逆变换等价：频带先从 8x8 降到 4x4，再双线性插值回 8x8，已经改变了空间采样。但它也没有实现真正的频带选择约束：四个带在上采样后直接拼接，任意 1x1 Conv 可以自由混合四带，随后再经过 GELU 和 LayerNorm。因此当前实现更准确的描述是“固定局部频率重排/降采样后的跨模态投影”，而不是已经证明的频带专属交换算子。

## 2. V、attention 和接线位置

在 [tbsi_layer.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py:88) 中，RGB/TIR 模板先拼成 `[B,128,768]`，生成 `q,k,v`；12 个 head 下 `v` 为 `[B,12,128,64]`。CFVI 输出重排为同样的 head 形状，并执行：

```text
v_rgb <- v_rgb + local_tir
v_tir <- v_tir + local_rgb
attention <- softmax(q @ k^T)
output <- attention @ v
```

所以 P1：

- 不改 Q/K 和 attention logits；
- 不处理 search token；
- 不使用跨模态位置匹配或相关性矩阵；
- 不区分目标区域和背景区域；
- 只在 TBSI location 9 实例化，位置选择由 [vit_tbsi_care.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/tbsi_track/vit_tbsi_care.py:181) 控制；
- 新增参数为单个 location 的 2,360,834 个参数。

P1 配置明确为 `CFVI_MODE: fixed`、`CFVI_LOC: [9]`、四带 mask 全 1：[CFVI_P1_FULL_MATCHED_M1_seed42.yaml](/root/autodl-tmp/TBSI_gai/TBSI/experiments/tbsi_track/CFVI_P1_FULL_MATCHED_M1_seed42.yaml:15)。

## 3. 模板裁剪、对齐和 mask

训练和测试都使用模板尺寸 128、patch size 16，因此得到 8x8 模板 token。

- 测试端在 [tbsi_track.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/test/tracker/tbsi_track.py:64) 用目标框和 `template_factor=2.0` 对 6 通道 RGB/TIR 共同 crop，再 resize 到 128，之后拆成 RGB/TIR 三通道。
- 训练端在 [processing_utils.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/train/data/processing_utils.py:12) 按目标框中心裁正方形区域，尺寸为 `ceil(sqrt(w*h)*factor)`，再 resize 到 128。
- 训练端的 6 通道样本先经过 joint transform，再拆分为 visible/infrared：[sampler.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/train/data/sampler.py:150)。这保证了数据增强几何变换共享，但不等于特征在每个高频位置都严格对应。
- 代码计算了 8x8 的 template foreground weights，但只有 template expert 模式才消费它；P1 的 expert mode 为 `off`，CFVI 没有接收该 mask：[tbsi_track.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/test/tracker/tbsi_track.py:71) 和 [tbsi_layer.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py:214)。

已导出 8 对同帧 RGB/TIR 模板 crop，使用当前数据集 init box、factor 2.0、128x128：

- [template_pairs_montage.jpg](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/template_pairs/template_pairs_montage.jpg)
- [template_pairs.csv](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/template_pairs/template_pairs.csv)

样本包含 `1blackteacher`、`ab_bikeoccluded`、`3rdfatboy`、`carcominginlight`、`whiteridingbike`、`AQgirlwalkinrain`、`3bike1` 和 `womanback2car`。这些 crop 说明输入框和几何裁剪是配对的，但不能由肉眼样本证明 token 高频的跨模态对应关系。

## 4. Full-run 诊断

诊断文件有 275,833 条记录。原始 JSONL 没有 epoch/step 字段；本审计依据固定训练合同 `59904 / 16 = 3744` 个 batch/epoch，重建前 56,160 条为 15 个训练 epoch，余下 219,673 条单独标记为训练后 forward，不能混入训练曲线。

| 训练阶段 | lambda RGB checkpoint | lambda TIR checkpoint | rho RGB mean | rho TIR mean |
|---|---:|---:|---:|---:|
| epoch 5 | 0.000936 | -0.001024 | 0.001674 | 0.002503 |
| epoch 10 | 0.003857 | -0.005420 | 0.004694 | 0.009932 |
| epoch 15 | 0.006690 | -0.009920 | 0.010363 | 0.015334 |

训练 epoch-1 的 rho 均值仅为 RGB `0.000872`、TIR `0.000783`。因此完整训练确实逐渐使用了分支，但使用幅度仍约为 1%--1.5%，明显小于此前不匹配短训诊断中的约 19%。这两套训练合同不同，不能直接作因果差值；可以确认的事实只是：本轮分支没有被完全剪掉，但没有成为强注入路径。

源频带能量均值也有稳定的 LL 主导：

- RGB 接收 TIR：epoch 1 为 `LL/LH/HL/HH = 0.8695/0.0518/0.0587/0.0200`，epoch 15 为 `0.8812/0.0460/0.0547/0.0181`；
- TIR 接收 RGB：epoch 1 为 `0.8382/0.0621/0.0721/0.0276`，epoch 15 为 `0.8502/0.0565/0.0682/0.0252`。

这里的 energy ratio 是 Haar 输出的源能量，不是定位贡献，也不是投影后注入贡献。`out_norm`、Conv 权重和 lambda 后，不能从 LL 占比推出 LL 的 AUC 贡献。

现有 `gradient_flow.csv` 有每 epoch 的整体 backbone/head 梯度汇总，但没有 CFVI Conv、LayerNorm、lambda 的逐参数梯度。因此“lambda 变小/变号的梯度原因”仍未被直接测量。

审计脚本和输出：

- [audit_cfvi_p1_frequency_mechanism.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/audit_cfvi_p1_frequency_mechanism.py)
- [cfvi_frequency_audit.md](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/mechanism_audit/cfvi_frequency_audit.md)
- [cfvi_epoch_summary.csv](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/mechanism_audit/cfvi_epoch_summary.csv)
- [cfvi_epoch_summary.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/mechanism_audit/cfvi_epoch_summary.json)

## 5. 当前机制判断

目前证据更支持以下可证伪判断，而不是直接选择 attention 或 Value：

1. P1 的频率输入是模板特征 token 的局部 2x2 频率成分，不是原图频率。
2. P1 没有目标/背景选择，也没有跨模态局部对应约束；高频转移可能同时包含目标边界、背景纹理和配准误差。
3. 四带在投影前没有独立的结构化处理，频带贡献无法从当前 `source_band_energy_ratio` 单独识别。
4. Full training 中 lambda/rho 的变化说明分支被优化到弱影响区，但现有日志不足以判定是梯度学习失败、投影归一化抵消，还是跨模态转移本身在损失上不划算。

因此当前不能合理地直接声称“应把频率放进 attention”，也不能直接实现动态频带门控。若继续，唯一优先的代码审计补充应是一次不改变模型输出的 instrumentation：记录 CFVI Conv/LayerNorm/lambda 的梯度范数、投影前后各带能量、目标 mask 内外的 rho，以及 RGB/TIR 对应 token 的频带相关性；在此之前不安排新训练。

## 6. 复现命令

```bash
cd /root/autodl-tmp
/root/autodl-tmp/conda_envs/tbsi/bin/python \
  TBSI_gai/TBSI/tools/audit_cfvi_p1_frequency_mechanism.py \
  --diagnostics TBSI_gai/TBSI/output/experiments/CFVI_P1_FULL_MATCHED_M1_seed42/logs/cfvi_diagnostics.jsonl \
  --gradient-flow TBSI_gai/TBSI/output/experiments/CFVI_P1_FULL_MATCHED_M1_seed42/logs/gradient_flow.csv \
  --output-dir TBSI_gai/TBSI/output/experiments/cfvi_p1_full_matched_m1_20261001/mechanism_audit \
  --checkpoint TBSI_gai/TBSI/output/experiments/CFVI_P1_FULL_MATCHED_M1_seed42/checkpoints/TBSITrack_ep0005.pth.tar \
  --checkpoint TBSI_gai/TBSI/output/experiments/CFVI_P1_FULL_MATCHED_M1_seed42/checkpoints/TBSITrack_ep0010.pth.tar \
  --checkpoint TBSI_gai/TBSI/output/experiments/CFVI_P1_FULL_MATCHED_M1_seed42/checkpoints/TBSITrack_ep0015.pth.tar
```
