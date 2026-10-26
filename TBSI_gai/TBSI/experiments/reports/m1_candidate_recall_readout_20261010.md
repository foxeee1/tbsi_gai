# M1 D1/D2 候选召回与 CENTER 回归读出诊断

## 结论

在这批固定状态 train-side 失配事件中，M1 的错误不是单一的“选错格”或“框回归差”。严重起点上目标格通常仍存在于较大的候选集合内，但 Hann 排序很少把它排到首位；即使离线强制选择 GT 格，offset/size 解码仍不够准确。因而当前证据支持**候选排序失配与 CENTER 回归读出误差并存**，不能把收益押在单纯重排或单纯加局部模块上。

## 实验身份与协议

- ID：`M1-CANDIDATE-RECALL-READOUT-20261010`；parent 为受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15。
- 数据：LasHeR `trainingset`，60 条 outcome-blind、loader-compatible 序列；按目标面积、相对运动、RGB 亮度、TIR 局部对比代理量分层平衡。序列清单 SHA256：`fb0ad869806bb86d1ef2e52f4f8a049964a24f3345b0047186f991b6fb2e540d`；manifest SHA256：`985dc0539142101aeafaa09e57950053e3bbe7746e69b4fe15475c609ce4583a`。
- 每条序列最多取一个正常匹配帧、一个失配前驱、一个严重起点，共 123 个事件：`matched_good=60`、`pre_onset=31`、`severe_onset=32`，覆盖 60 条独立序列。严重起点由连续 3 帧 IoU `<0.5` 且 GT 全部留在搜索 crop 内定义。
- D1：对真实在线 Hann 加权 score 排序，测 GT 精确 cell 与 GT 支持区域在 top-K 的召回，`K=1,3,5,8,16,32,256`；另测每个 top-K 中事后 oracle 框质量。
- D2：固定同一帧的 CENTER size/offset maps，分别解码 top-1、GT cell、GT 周围 3×3 及 top-K oracle cell。GT/oracle 仅为离线反事实，不进入在线推理。
- 计算量：单张 RTX 4090 D，trace 20,930 帧，推理耗时 805.0 秒（约 0.224 GPU 小时）；分析约 8 秒。配置 SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- 运行环境实际为 Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8；受保护 M1 的登记环境为 PyTorch 1.9.0+cu111、CUDA 11.1。使用了登记的 M1 权重/配置，但这是当前工作树上的 train-side 诊断 replay，不是归档逐帧复现或正式 benchmark。GT-cell 分析解码相对当前 replay top-1 的最大框差仅 `0.000051 px`；该 parity 不等于与归档 M1 预测逐帧一致。
- 无训练、无测试集推理；AUC/PR/NPR/OP50/OP75/FPS 均为 `NA`。带 JSON trace 写盘的吞吐不作为 tracker FPS。

## D1：候选召回

| 状态 | 事件数 | GT 精确格 top-1 | GT 精确格 top-8 | GT 精确格 top-16 | GT 支持区 top-8 | top-8 oracle IoU 中位数 | top-16 oracle IoU 中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 正常匹配 | 60 | 15.0% | 91.7% | 98.3% | 100% | 0.841 | 0.848 |
| 失配前驱 | 31 | 32.3% | 80.6% | 100% | 100% | 0.728 | 0.764 |
| 严重起点 | 32 | 6.3% (2/32) | 59.4% | 96.9% (31/32) | 96.9% (31/32) | 0.534 | 0.566 |

整体序列 macro 的精确 GT-cell top-K 召回为：top-1 `17.1%`（95% CI `7.8–21.1%`）、top-8 `80.5%`（`76.1–91.4%`）、top-16 `98.4%`（`93.3–100%`）。GT 支持区 top-8 召回为 `99.2%`。精确格召回偏低部分来自“单格”定义；支持区召回更能表示目标邻域是否有响应。

**解释：**严重起点大多不是“整张搜索图完全没有目标候选”。`31/32` 的精确 GT cell 到 top-16 才被覆盖，且 `31/32` 的 GT 支持区域已在 top-8。但实际 top-1 命中 GT 精确格仅 `2/32`，说明困难主要包括目标峰相对干扰峰排序靠后。另一方面，top-16 oracle IoU 中位仅 `0.566`，证明扩大候选并不能自动恢复高质量框。

## D2：选格后的框读出

| 状态 | top-1 IoU 中位数 | GT-cell IoU 中位数 | GT-cell 中心误差中位数 | GT-cell offset L1 | GT-cell size L1 | GT-cell 绝对宽高误差 |
|---|---:|---:|---:|---:|---:|---:|
| 正常匹配 | 0.795 | 0.775 | 0.239 token | 0.151 token | 0.300 token | 4.35 px |
| 失配前驱 | 0.670 | 0.631 | 0.464 token | 0.278 token | 0.628 token | 6.90 px |
| 严重起点 | 0.290 | 0.448 | 1.046 token | 0.620 token | 0.944 token | 11.21 px |

严重起点中，GT-cell oracle 相对实际 top-1 的 IoU 中位提升 `+0.151`，中心误差中位减少约 `0.679 token`；但 GT-cell 的 size L1 仍为 `0.944 token`（约 11.21 px 的平均绝对宽高误差），并未随选格自动修复。top-8 内事后挑选最高 IoU 框的中位 IoU 为 `0.534`，top-16 为 `0.566`；top-16 oracle 框的中心误差、offset L1、size L1 中位数仍分别为 `0.555`、`0.335`、`0.772 token`。

GT 周围 3×3 只作为局部 oracle：严重起点其最高 IoU 中位数 `0.559`、最小中心误差中位数 `0.500 token`。这指出目标邻域有可恢复框，但现成回归读出和 score 选择共同限制了质量。oracle 数字是上界，不能作为实际方法收益。

## 归因与边界

1. **候选排序确实是问题之一。**严重起点 top-1 精确格命中 `2/32`，而 top-16 覆盖 `31/32`；M1 的 CENTER score/Hann 选峰没有把已有候选稳定排到第一。
2. **单纯候选重排不够。**即使 top-16 中事后选 oracle，IoU 中位只有 `0.566`；强制 GT-cell 后 size L1 仍高。这些状态下的 offset/size 不能视为可靠。
3. **也不能说“局部空间建模已被证明无效/必要”。**本实验只定位 M1 固定状态 CENTER 候选及回归瓶颈，没有干预融合特征、没有训练新 loss，也没有测试真实闭环修复效果。
4. **下一步方法选择：**若继续验证，应把单一主假设收窄为“score 与 offset/size 的联合定位一致性”，并在新 train-side holdout 上做固定状态小实验；不要再单独训练身份分数重排器。融合模块只有在证明融合前存在可用目标证据、且经融合后丢失时才重新进入候选。

## 证据与复现

- 正式汇总：`output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/analysis_final/summary.json`
- 逐事件 / 逐序列：同目录 `per_event.csv`、`per_sequence.csv`
- trace：`output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/trace/`；run state `output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/run_state.json`
- cohort：`experiments/diagnostics/m1_candidate_recall_readout_v1/`
- 源码快照：`output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/provenance/source_snapshot.tar.gz`，SHA256 `4b5e0d83761dd28f67779cc25a13a0a0352753a32b742d752c3e9d9fcf6495da`
- 当前 Git `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；分析脚本 SHA256 `212e0d3a96d686f0db01daf65c59077d484f4295ae1d11e5cbe0bde4dfe7051e`。修正宏平均 CI 和补齐回归汇总后的权威产物位于 `analysis_final/`；早先 `analysis/` 与 `analysis_corrected/` 是保留的中间产物，不应引用为最终汇总。
