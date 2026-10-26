# M1 IGF pre-fusion proxy：全 244 序列报告

## 目的与复现边界

本实验依据论文：*RGB-T Tracking With Template-Bridged Search Interaction and Target-Preserved Template Updating*, IEEE TPAMI, 47(1), 2025, 634--647。论文未提供官方实现，因此本实验是按论文机制语义完成的 **IGF-inspired reproduction approximation**，不是官方代码逐行复现。

研究问题是：在 `tbsi_fuse_search` 之前，能否用模板目标信息引导 RGB/TIR search 特征融合，改善 M1 的中心选择和后续跟踪。

## 实现合同

```text
RGB/TIR template tokens (B,64,768 each)
                 |
        IGF cross-attention guidance
                 |
RGB/TIR search tokens (B,256,768 each) + bounded residual
                 |
         tbsi_fuse_search (Conv-BN-ReLU)
                 |
             CENTER head
                 |
         box prediction / state update
```

`IGFPreFusion` 使用 search token 作为 query，将 RGB/TIR template token 拼接为 key/value，并分别把引导残差注入 RGB、TIR search。`gamma` 零初始化，残差为 `tanh(gamma) * 0.10`；模板、search 和输出形状保持原合同。CENTER head、decoder、tracker state 和 template update 没有改动。M1 默认关闭该 feature gate，保护配置和 checkpoint 未修改。

## 运行条件

- 实验：`M1_IGF_PREFUSION_PROXY_20261017`
- parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15
- 配置：`experiments/tbsi_track/M1_IGF_PREFUSION_PROXY_20261017.yaml`
- 训练：LasHeR_train，256 samples/epoch，3 epochs，batch 2，FP32，worker 1；仅训练 `igf_prefusion.*`
- 初始化：M1 epoch15 checkpoint
- 测试：LasHeR official testingset，全 244/244 序列，FP32，官方 evaluator
- 代码：Git `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` 加工作树 IGF 变更
- 配置 SHA256：`4fa830b1423424e5a2fe6a11c3319f9e32f86910379e5709cb938d4c6db12cc3`
- IGF 源文件 SHA256：`cc755ef37e580d4047b8ca390178d6e42a3964e51616d9069bf623a1f576955d`
- checkpoint SHA256：`4508f457fe3d583918e9462713452f9d7465b4b77f8b6a347454ad24cb5ea4f3`
- 训练后 `igf_prefusion.gamma=0.0022794066`

## 全量结果

| 方法 | AUC/Success | Precision | Norm Precision | OP50 | OP75 |
|---|---:|---:|---:|---:|---:|
| M1 | 56.2077 | 70.21 | 66.20 | 68.3339 | 47.1349 |
| S1 | 56.4599 | 70.54 | 66.53 | 68.7700 | 47.1488 |
| IGF proxy | 56.2896 | 70.27 | 66.45 | 68.3696 | 47.4382 |

相对 M1：AUC `+0.0819`，OP50 `+0.0357`，OP75 `+0.3033`，Precision 约 `+0.06`，Norm Precision 约 `+0.25`。相对 S1：AUC `-0.1703`，OP50 `-0.4004`，OP75 `+0.2894`。

逐序列 AUC 曲线胜负为：提升 113 条、退化 131 条。最大增益为 `whitegirlinlight +0.4503`，最大退化为 `ab_bikeoccluded -0.5799`。因此 OP75 的局部改善没有转化为 AUC/OP50 的一致改善。

## 结论与下一步

结论为 `PROXY-FULL244 / SMALL POSITIVE VS M1 / BELOW S1`。本结果支持“模板引导 search pre-fusion 在当前接口上可训练且有轻微局部收益”，但不支持“IGF 已解决退化问题”或“优于 S1”。`gamma` 仍接近零，且逐序列收益方向不稳定，剩余瓶颈仍可能位于 crop 内中心选择、框回归与状态反馈闭环，而非单一融合算子。

该实验是单 seed、3 epoch proxy，不是多 seed 稳健结论。下一步应先做 disabled/zero-gate parity 与固定 crop 的残差、中心命中和失败帧诊断，再决定是否增加 IGF 训练进度或改变引导信号；不应直接继续堆叠 head 模块。

证据目录：`output/experiments/M1_IGF_PREFUSION_PROXY_20261017/`；全量预测：`output/test/tracking_results/tbsi_track/M1_IGF_PREFUSION_PROXY_20261017/`。
