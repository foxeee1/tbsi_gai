# G0 FFM post-TBSI 短程闭环实验报告

## 结论

本实验完成了预注册的唯一 FFM 位置：block index 5 输出、第二次 TBSI（index 6）之前；只写两路 search token，M1 冻结，FP32，seed 42，Cdiag/Cspatial 各 1 epoch。P0 与 gamma=0 关门合同通过，但方向闸门不通过：Cspatial 没有优于 Cdiag，且困难 rescue 事件窗口表现低于 C0。因此当前 post-TBSI FFM 位置 `NO-GO`，不启动第二 seed、K/V 乱序、换位置、改 loss 或 full-244。

## 合同与环境

- parent：`G0-POST-TBSI-HARD-20261005`
- train：固定 `ffm_train_sequences.txt`，152 条序列；dev：固定 `ffm_dev_rescue_sequences.txt`，24 条序列；不相交
- 评估：24 条 rescue 序列全轨迹；训练 1 epoch、batch 16、原 tracking loss、FP32
- C0：M1 epoch-15；Cdiag：逐位置跨模态 Linear；Cspatial：双向 CrossAttention，hidden 192、4 heads
- 硬件：1x RTX4090D 24GB；Python 3.8.20；PyTorch 1.9.0+cu111；CUDA 11.1；`TBSI_INFER_FP16=0`
- canonical baseline SHA256：`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`，本轮未修改

## P0 与训练有效性

`G0-FFM-P0-20261005` 通过：输入输出 `[B,256,768]`、有限值、zero-gate identity、首步 gate 梯度、gate 更新后 Q/K/V 梯度、非零 gate 输出变化均满足；冻结 M1 非旁路 state 未变化；完整 backbone 的 gamma0 与关闭模块最大差为 `0.0`。Cdiag adapter 参数 1,179,650，Cspatial adapter 参数 1,182,338。

训练均跑满 672 个 batch：Cdiag 平均训练 IoU `0.78935`，Cspatial 平均训练 IoU `0.78824`。这些是训练监控量，不替代闭环测试指标。

## 全轨迹结果

| 条件 | mean AUC | median AUC | 10% trimmed AUC | mean IoU | 10% trimmed IoU |
|---|---:|---:|---:|---:|---:|
| C0 | 0.539459 | 0.529440 | 0.548645 | 0.538617 | 0.548034 |
| Cdiag | 0.557286 | 0.557327 | 0.566521 | 0.556808 | 0.566274 |
| Cspatial | 0.546648 | 0.565634 | 0.558396 | 0.545948 | 0.558011 |
| Cspatial-gamma0 | 0.539459 | 0.529440 | 0.548645 | 0.538617 | 0.548034 |

相对 C0：Cdiag `ΔAUC=+0.017827`，14 胜/10 负，trimmed `ΔAUC=+0.011087`；Cspatial `ΔAUC=+0.007189`，11 胜/13 负，trimmed `ΔAUC=+0.002723`。Cspatial 相对 Cdiag 的 mean AUC 差为 `-0.010638`，不支持“空间交叉注意力比简单对应更有价值”。Cspatial 仍有 3 条序列 AUC 下降至少 0.05，最大下降 `-0.157605`。

## 困难事件窗口

按 G0-Hard 已登记的 rescue 事件，在固定 dev 清单中得到 24 个事件；窗口为事件帧 `[t,t+30]`，GT 仅用于前向后的评分。

本轮没有保存可复用的 C0 帧前 tracker state，因此“同一 state、同一 crop 的单步反事实”未执行，不能把本报告的窗口结果冒充该层证据；它被记录为 `NOT_EVALUATED`。本次 No-go 仅依据已完成的全轨迹与固定窗口结果。

| 条件 | 事件窗口 mean IoU | IoU>=0.5 恢复率 |
|---|---:|---:|
| C0 | 0.207197 | 20/24 = 0.833333 |
| Cdiag | 0.219907 | 19/24 = 0.791667 |
| Cspatial | 0.199477 | 16/24 = 0.666667 |
| Cspatial-gamma0 | 0.207197 | 20/24 = 0.833333 |

Cspatial 相对 C0 的窗口 IoU 为 `-0.007721`，恢复率下降 4 个事件；Cspatial 相对 Cdiag 的窗口 IoU 为 `-0.020430`。这直接否定本轮继续投入的必要条件。当前事件的中心距离和预测/GT 面积比已记录在 `window_analysis.json`；本轮不把未预注册的中心/尺度阈值再解释成新机制证据。

## 关门与证据边界

gamma0 与 C0 24/24 序列长度一致，逐框最大绝对差 `0.0`。所以退化不是关门路径破坏 M1，而是 residual 开启后的闭环作用。P0 通过只证明实现合同和可训练性，不能证明任务收益；Cdiag 的正均值也不能授权 Cspatial，因为 Cspatial 的困难窗口、胜负和相对 Cdiag 均不满足继续门槛。

## 证据路径

- P0：`output/experiments/g0_ffm_20261005/p0_result.json`
- 训练/启动日志：`output/experiments/g0_ffm_20261005/runner.log`
- 全轨迹分析：`output/experiments/g0_ffm_20261005/analysis.json`
- 困难窗口分析：`output/experiments/g0_ffm_20261005/window_analysis.json`
- Cdiag checkpoint：`output/experiments/G0_FFM_CDIAG_seed42/checkpoints/TBSITrack_ep0001.pth.tar`
- Cspatial checkpoint：`output/experiments/G0_FFM_CSPATIAL_seed42/checkpoints/TBSITrack_ep0001.pth.tar`
- 固定 manifest：`experiments/proxy/lasher_rpp_v1/g0_ffm_20261005/manifest.json`

## 资产定位

本实验是 G0-Hard 之后的一次受限探索性闭环验证，不是 full confirmation，也不是论文主结果。结论为：困难组冻结特征存在可读取 retrieval 信号，但当前 post-TBSI FFM 旁路没有把它转化为可靠跟踪收益；该位置封存。
