# TAM 固定 crop 单步反事实（2026-10-07）

## 目的与范围

检验一个最小、可证伪命题：经真实历史/current crop 几何对齐的上一帧 M1 raw head token，经现有小 TAM residual 后，是否改善当前帧原始 Center head 的预测框。

本实验复用已完成的 TAM-only 100-step image replay actor-audit checkpoint，不新增训练。该 checkpoint 仅在 `abreastinnerboy` 上训练；评估使用未参与该 adapter 训练的两条 LasHeR train trace 序列，每条按有效历史帧等距固定抽取 16 帧，共 32 个同状态、同 crop 单步事件。输入特征和 crop 来自已验证的 M1 temporal trace；GT 仅用于前向完成后的框评分。没有运行闭环跟踪、完整 proxy 或 full-244。

## 对照

- `M1/C0`：直接将保存的当前帧 raw head feature 输入原始 Center head。
- `TAM-aligned`：同一当前特征和同一 head，加入上一帧 raw feature；使用 trace 中实际 current/history crop 坐标及真实因果帧号。
- `TAM-gamma0`：同一模块把 gate 置零，检查逐框 parity。
- `TAM-unaligned`：同一历史 tensor，但错误地将 history crop 设为 current crop，作为几何对齐诊断对照。

两条序列为 `biketurndark`、`right2ndfarboytakinglight2left`。adapter seed 为 42；M1 checkpoint epoch 15。固定 crop 单步 IoU 使用 crop-normalized head box 与通过 M1 crop affine 映射后的 GT box计算。

## 结果

| 指标 | 结果 |
|---|---:|
| 事件数 | 32（16/序列） |
| M1 mean IoU | 0.62109 |
| TAM aligned mean IoU | 0.59462 |
| 配对 mean ΔIoU | -0.02647 |
| 配对 median ΔIoU | -0.00062 |
| 逐事件胜 / 平 / 负 | 11 / 4 / 17 |
| 10% paired-delta trimmed mean | -0.00236 |
| 去掉最大退化事件后的 mean ΔIoU | -0.00232 |
| 序列级 mean ΔIoU | `biketurndark`: -0.00078；`right2ndfarboytakinglight2left`: -0.05216 |
| mean feature residual ratio | 0.05505；最大 0.07169 |
| 平均历史有效覆盖率 | 0.87354 |
| `gamma=0` 最大预测框差 | 0.0 |
| TAM unaligned mean ΔIoU | -0.02658 |

最严重事件为 `right2ndfarboytakinglight2left` frame 622：M1 IoU `0.77493`，TAM aligned 与 unaligned IoU 均为 `0`；框中心从 crop-normalized x=`0.5071` 移到 `0.7751`。该事件残差比为 `0.05988`，不是接近 0.10 写入上限。aligned 与 unaligned 在该事件的预测相同。除该事件外，平均 ΔIoU 为 `-0.00232`，截尾均值为 `-0.00236`，因此总体并非由它掩盖的正向收益。

## 判断

**当前 checkpoint/样本的最小假设未获支持。** `gamma=0` parity 和非 TAM 状态只读检查通过，真实 crop overlap 足够（平均覆盖率约 87%），但开门后没有改善当前帧 box；逐事件中退化多于获益。正确 crop 对齐与错误 crop 对照的均值几乎相同，且没有显示正确对齐的增量。

这不能证明 TAM 在其他初始化、更多训练或其他事件上必然无效；但它足以否决把当前 100-step 单序列 checkpoint 作为性能证据或启动 full-244 的依据。尤其 frame 622 显示小于上限的残差仍可能造成单帧严重 head 跳变，不能把 zero-gate parity 当成开门稳定性的保证。当前不继续增加训练步数或调 gate。

## Provenance 与证据

- 代码快照：`09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` + working tree。
- M1 epoch-15 SHA256：`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- TAM actor-audit checkpoint SHA256：`0ddf9a05d5405214b43eccf7efb6a320444047beb3fd1a9b891d9609cc137d58`；step 100，训练序列 `abreastinnerboy`，gamma `0.0118128`。
- 环境：1x NVIDIA GeForce RTX 4090 D；Python 3.8.20；PyTorch 1.9.0+cu111；CUDA 11.1；FP32。
- 工具：`tools/eval_tam_fixed_crop_counterfactual.py`。
- 机器结果：`output/experiments/tam_fixed_crop_counterfactual_v1_20261007/result.json`。
- GPU 时长：NA（未单独计时）。AUC、PR、NPR、OP50、OP75、FPS：NA（非 tracker benchmark）。
- canonical baseline YAML SHA256 仍为 `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`，未修改。

## 先前 60 序列探针状态更正

`TAM-RPP-ENGINEERING-PROBE-20261007` 已无活跃进程。它在本次目标切换前只运行到 C0 的部分序列（日志出现 14 条序列起始，结果目录有 8 条完整预测文件）；没有完成 60 序列 C0 分析，没有启动 TAM100 tracker 测试。该运行记为 `INTERRUPTED-SCOPE-CHANGE`，其 partial predictions 保留但不计算 benchmark 指标，也不作为本单步实验的证据。RPP-v1 历史准入仍未通过。
