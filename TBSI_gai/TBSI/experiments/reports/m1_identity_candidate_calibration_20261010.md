# M1 MHSA 身份候选校准预实验（2026-10-10）

## 问题与结论

在不训练 M1 主网络、不改变 CENTER 回归头的前提下，检验一个小型模板身份匹配器能否把正确候选格从 M1 Hann top-8 中排到前面，并且不伤 matched-good / pre-onset 帧。它只回答固定状态的选格问题，不回答闭环 AUC、full test 或在线融合收益。

**结论：停止当前候选身份校准路线，不进入主网络融合训练。** raw cosine 在新队列上没有 severe-onset 收益；calibration 学得的匹配 MLP 与 C3 学习组合都未增加 severe target-top1，validation 也找不到满足预设安全条件的切换门。final 组中，学习排序把少数正常/前驱帧换向目标支持区，但 IoU 中位数下降；严重起点 target-top1 从 M1 的 4/11 降至 3/11。不能据此否定模板身份信息，而是说明当前候选级表征与监督仍未形成足够安全、可迁移的在线排序分数。

## 协议冻结

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15。Config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`；checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，运行前后相同。
- 新 train-side cohort 仅从已有排除规则后的 204 条 eligible 序列中 outcome-blind 平衡选择，序列级分为 calibration 60、validation 18、final holdout 18。V2 manifest SHA256 `dacea7648f3750a76f1da66a7d0844386f165383abffb01dade8f268217c445b`；三个名单 SHA256 分别为 calibration `918d46f613855b5ecf48a25029bb6651467406227ad3665abd918267abb35909`、validation `bf238dae12b25699eb08f02519022d234897aca20e70726ca88c5c95b812e5a2`、final `51394c1f3b78bf0ffdac2c8bdf4d142f260318b412a51e639cc909db0a525b7e`。三组序列互斥。
- V1 validation 名单中的 `orange` 有 304 个不满足本地 LasHeR loader 既有文件名解析合同的红外帧。V1 trace 在序列构造时失败、未完成任何跟踪；原 manifest/失败输出保留。V2 按同一 outcome-blind 元数据平衡规则以 `lover_1202` 替换，其他名单不变。
- 每序列最多取一个 matched-good、pre-onset、severe-onset 事件；无 onset 的序列提供一个 good-only 帧并记作 matched-good。总计 calibration 131 事件（128 含 top-8 正/负候选）、validation 34 事件、final 38 事件，其中 final matched-good 17、pre-onset 10、severe-onset 11。
- 对每个固定状态重放原 M1 模板和 crop，记录 CENTER Hann top-8 候选的融合 search token、TIR/RGB 模板前景池化向量、Hann 分数和 GT-support 标签。重放框与 trace 误差逐帧须小于 `0.01 px`，否则中止。
- C0 是 M1 Hann argmax；C1 是固定 raw cosine、top-8 内 z-score 后以固定 `lambda=0.5` 与 CENTER 分数相加；C2 是 39,329 参数 MLP，以 calibration 上候选 target-support vs hard-negative 的 pairwise softplus ranking loss 拟合；C3 是 calibration 学得的 `a*S+b*q+c*S*q`。M1 backbone、融合器、CENTER、offset/size、checkpoint 均未训练或改动。
- validation 仅选择 safe-gate：最多少 1 个 matched-good target-top1、pre-onset mean IoU 不降、换峰距离不超过 3 token。没有一组候选门槛满足全部约束，因此 C2/C3 safe arms 关闭，按设计回退 M1。保存模型和 gates 后才读取 final holdout。

## 结果

下表为 final holdout，target-top1 表示所选格落在 GT 支持区；误差单位为 16×16 网格 token。

| 状态 | 方法 | 事件 | target-top1 | 中心误差中位数 | IoU 中位数 | top-8 含目标 |
|---|---|---:|---:|---:|---:|---:|
| matched-good | C0 M1 | 17 | 82.4% | 0.205 | 0.856 | 100% |
| matched-good | C1 raw cosine | 17 | 82.4% | 0.205 | 0.856 | 100% |
| matched-good | C2 learned matcher | 17 | 88.2% | 0.304 | 0.845 | 100% |
| matched-good | C3 learned combination | 17 | 88.2% | 0.219 | 0.850 | 100% |
| pre-onset | C0 M1 | 10 | 50.0% | 0.487 | 0.681 | 100% |
| pre-onset | C1 raw cosine | 10 | 60.0% | 0.487 | 0.649 | 100% |
| pre-onset | C2 learned matcher | 10 | 60.0% | 0.579 | 0.624 | 100% |
| pre-onset | C3 learned combination | 10 | 60.0% | 0.579 | 0.624 | 100% |
| severe-onset | C0 M1 | 11 | 36.4% | 1.575 | 0.256 | 81.8% |
| severe-onset | C1 raw cosine | 11 | 27.3% | 1.882 | 0.192 | 81.8% |
| severe-onset | C2 learned matcher | 11 | 27.3% | 1.598 | 0.259 | 81.8% |
| severe-onset | C3 learned combination | 11 | 27.3% | 1.598 | 0.259 | 81.8% |

Severe-onset 的 11 个事件中，C2/C3 各有 2 个序列 IoU 上升、3 个下降，其余无变化；中位 IoU 变化约 `+0.0033`，中心误差中位没有改善。更关键的是，top-8 中有目标支持格的仅 9/11；剩余 2 个事件对任何 top-8 重排都不可恢复。matched-good 上 C2/C3 虽由 14/17 升到 15/17 target-top1，但 C2 IoU 中位 `−0.0113`、C3 `−0.0062`；pre-onset IoU 中位分别 `−0.0571`，不是可接受的免费增益。

C3 学得系数约为 `a=0.216, b=2.973, c=-0.016`，排序主要由 learned identity q 驱动；C2/C3 final 排名和指标相同。训练 pairwise loss 很低（MLP `1.13e-4`），但跨序列 final 未改善，说明拟合/梯度正常不等于排序可迁移。验证 safe-gate 不可行，故 C2/C3 safe 结果与 C0 相同。

## 归因与决策

1. **不是候选排序分支未接入**：匹配器在 calibration 可拟合，final 多个帧也改变排序/选格；但这不产生 severe-onset 的跨序列净收益。
2. **主要失败是可迁移校准不足与目标支持信息不足**：原始融合 token 与模板原型形成的匹配分数，在新序列上不能稳定压过 CENTER 的困难负峰；有 2/11 severe 事件 top-8 根本不含目标支持格。
3. **常规帧副作用真实存在**：top1 命中率增加不等价于定位改进；选中格的 offset/size 读出可使 IoU 变差。
4. **研究路线决策**：停止此小型候选排序头与当前 raw-cosine 路线；不继续调阈值、不扩展到完整融合训练。MHSA 身份信息“能改变候选顺序”仍成立，但“校准后能安全改善中心定位”未获支持。下一步优先回到 CENTER 的 score 与 offset/size 联合读出问题，或寻找 top-K 召回之前的候选生成瓶颈；不能再把本结果归因成未验证的融合模块收益。

## 复现与证据

- Cohort 构建：`tools/build_m1_identity_calibration_cohort_20261010.py`；V2 loader 合同修订：`tools/revise_m1_identity_calibration_cohort_20261010.py`。
- 校准/评估：`tools/run_m1_identity_calibration_20261010.py`。
- 代码 Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`（工作树另有既存修改；本次入口以脚本 SHA256 固定）。
- 脚本 SHA256：build `d6ab0b81b6241c2e199302685cf9bd626cc0d72e9fab84a3f998ee9a1606a1bc`；revise `d51453065873d4266cf36036273e53e09c8a8634a586ca032a9e47a2075f22`；runner `b288b08acab2452655dcb1266e414464318347fd9e76c8b5242c4b7ee8a7ccf9`。
- Trace 命令分别使用 calibration/validation/final 名单调用 `tools/run_m1_search_stage_train_trace_20261009.py`，输出位于 `output/experiments/M1_IDENTITY_CALIBRATION_20261010/trace_calibration/` 与 `output/experiments/M1_IDENTITY_CALIBRATION_V2_20261010/{trace_validation,trace_final_holdout}/`。
- 固定状态命令：`python tools/run_m1_identity_calibration_20261010.py --cohort-dir experiments/diagnostics/m1_identity_calibration_v2 --trace-calibration output/experiments/M1_IDENTITY_CALIBRATION_20261010/trace_calibration --trace-validation output/experiments/M1_IDENTITY_CALIBRATION_V2_20261010/trace_validation --trace-final output/experiments/M1_IDENTITY_CALIBRATION_V2_20261010/trace_final_holdout --output-dir output/experiments/M1_IDENTITY_CALIBRATION_V2_20261010/results`。
- 结果：`output/experiments/M1_IDENTITY_CALIBRATION_V2_20261010/results/summary.json`、`final_per_event.csv`、三组压缩候选特征、`calibrated_ranker.pth`、`validation_gates.json`。V1 loader 失败记录在 `output/experiments/M1_IDENTITY_CALIBRATION_20261010/trace_validation/manifest.json`。
- 环境：1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8、FP32。约 0.40 GPU-hour（calibration/validation/final trace 与固定状态重放运行时间合计）；无需并发训练。无 AUC/PR/NPR/OP50/OP75/FPS，均不适用。
- 限制：18 条 final 仅有 11 个 severe-onset，属探索性 screen；train-side cohort 仍来自 M1 训练分布；当前工作树重放非归档 M1 逐帧等价；不能推断闭环收益、test-244 表现或普遍性。
