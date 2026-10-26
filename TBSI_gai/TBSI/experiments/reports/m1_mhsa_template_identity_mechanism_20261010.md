# M1 MHSA 模板身份线索机制验证（2026-10-10）

## 问题与范围

验证候选机制：TBSI-E/MHSA 后的模板特征是否携带目标身份线索，并在搜索困难时将目标位置排在背景困难负峰之前；若有信号，再判断它是否影响 M1 实际 CENTER 选峰。此处不训练、不改 M1/checkpoint、不评估 test-244，也不声称新模块收益。

## 固定实验合同

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15。运行前后 checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`。
- 数据：复用预冻结的 LasHeR train-side 96 序列 cohort（清单 SHA256 `b27b6a0426cc8e5c05a4cfe1ddf86b392768fe5c7b386f72c74622b03e5bbf3b`）。选择 42 个 matched-good、40 个 pre-onset、44 个 severe-onset 固定状态帧，每序列至多一个各类事件。
- 固定状态：保留原始初始化模板、trace 中的 state/crop；对同一搜索图像分别跑正确模板和 3 个确定性错模板。错模板优先匹配目标面积与运动代理四分位，作为身份替换压力对照，不代表可部署方法。
- 表征：从同一次 TBSI forward 的最终 `[B,640,C]` 特征中，按布局读取 RGB template/search（`0:64`/`64:320`）及 TIR template/search（`320:384`/`384:640`）。利用初始化目标前景 mask 加权池化模板 token，对两路 search 网格做 cosine map；另记录同一次前向真实 CENTER 的 Hann score 目标 rank、目标/困难背景峰 margin、框 IoU。
- 运行：1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8；FP32；无训练、无 DataLoader worker。GPU 前向诊断耗时约 14.6 秒（不含模型加载）；环境与 M1 登记环境不同。
- 当前工作树固定状态重放最大框差 `0.0000687 px`，小于 `0.01 px`；保护权重/config 未变化。该重放仍非归档 M1 逐帧等价。

## 结果

在错模板控制的逐帧中位数上比较正确模板（self）与错模板（wrong）：

| 固定状态 | 序列/帧 | CENTER 目标 top-1 self/wrong | Hann CENTER target-vs-hard-background margin 差值 self-wrong（中位数；正值条数） | 框 IoU 差值 self-wrong（中位数；正/负/平） | RGB/TIR 模板 cosine margin 差值中位数 |
|---|---:|---:|---:|---:|---:|
| matched-good | 42/42 | 81.0% / 71.4% | +0.176（38/42） | +0.116（36/6/0） | +0.027 / +0.020 |
| pre-onset | 40/40 | 70.0% / 45.0% | +0.110（34/40） | +0.163（34/6/0） | +0.021 / +0.019 |
| severe-onset | 44/44 | 29.5% / 36.4% | +0.018（25/44） | 0.000（21/16/7；均值 -0.033） | +0.020 / +0.019 |

严重起点处，self 相较错模板的 CENTER margin 平均差为 `+0.041`、中位数 `+0.018`，但 target top-1 少 3/44 个事件，IoU 平均反而低 `0.033`。这说明余下目标区域会获得稍强的分数间隔，不等价于实际选峰或闭环框得到可靠恢复。严重起点两路模板 cosine 的 target-vs-hard-negative margin 绝对值在 self 与 wrong 下仍为负；中位 self-wrong 差仅 RGB `+0.020`、TIR `+0.019`，没有显示强而稳定的身份峰值分离。

严重起点按预冻结代理分层的 self-vs-wrong IoU 正负交错：运动 Q1–Q4 的正/负/平为 `3/3/1`、`2/6/1`、`10/2/1`、`6/5/4`；目标面积 Q1–Q4 为 `6/5/3`、`6/2/2`、`5/4/2`、`4/5/0`。RGB 亮度和 TIR 对比组也不单调，每组仅 7–15 个 onset，不用于选择子群或宣称属性效应。

## 判定

**机制得到部分支持，但不支持直接进入训练。** 正常帧和失配前驱中，正确模板相对错模板显著改善 M1 自身 CENTER margin 和框；说明当前模型确实利用初始化模板信息。可是严重失配起点上，自模板相对错模板的选峰率没有优势、框收益异质且总体偏负。更重要的是，错模板重放是大幅替换初始化外观的压力对照；self 优于错模板并不能证明“新增 MHSA 条件融合路由”会优于原 M1。

因此当前主张应收窄：MHSA 后模板表征具有一定可用身份信息，但在目标已进入严重失配起点时，这些线索不足以稳定压过实际困难负峰。若要继续融合路由，需要先提出不依赖错误模板的直接路由反事实（同一正确模板下开/关模板身份到融合决策的路径），并证明在严重起点提高 CENTER target top-1，而不扰动 matched-good；本轮数据不足以授权训练或扩大到测试集。

## 复现与证据

- 脚本：`tools/diagnose_m1_template_identity_mechanism_20261010.py`，SHA256 `e722138d982bc047ae67029e2bd59439de6a5678d7bffb19247cef56257c5e2e`。
- 命令：`python tools/diagnose_m1_template_identity_mechanism_20261010.py --wrong-templates-per-query 3`。
- 完整逐臂结果：`output/experiments/M1_MHSA_TEMPLATE_IDENTITY_MECHANISM_20261010/per_arm.csv`；配对结果：`.../paired_self_vs_wrong.csv`；汇总：`.../summary.json`。
- smoke（3 序列、单错模板）在 `/tmp/m1_template_identity_smoke_20261010/`，重放 parity 通过。
- 限制：相似度 map 是诊断读出，不是原生 CENTER 分数；错模板会改变外观条件；当前工作树与归档预测不逐帧一致；M1 训练期见过 LasHeR train，cohort 只相对既有诊断新鲜；属性均为代理量。
