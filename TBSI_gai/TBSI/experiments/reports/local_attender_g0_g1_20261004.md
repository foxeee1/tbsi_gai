# Local Attender 目标条件 search 融合：G0/G1 审计

日期：2026-10-04  
Parent：`TBSI-E-MHSA-FULL-20260927`  
代码：`09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` + working tree  
范围：固定 RPP-v1 train proxy 中 24 条序列、每条 2 个事件，共 48 个闭环事件；冻结 TBSI-E M1 epoch-15 checkpoint；FP32 tracker。

## G0 数据与位置审计

前向只使用首帧模板和当前帧 RGB/TIR search 特征；GT 框只在 forward 完成后用于目标/背景 token 分层。最终 M1 token contract 为每模态 `64 template + 256 search`，通道 `768`，search 为 `16x16`。

| 指标 | 结果 | 预注册判断 |
|---|---:|---|
| 事件总数 | 48 | 固定合同 |
| 目标 token 覆盖事件 | 43/48 = 89.58% | >80% |
| 目标局部 rescue rate（RGB->TIR） | 52.13% | >50% |
| 目标局部 rescue - 背景局部 rescue | +0.01112 | >0.01 |
| 目标模板相似度 margin（RGB） | -0.20294 | 应 >0 |
| 目标模板相似度 margin（TIR） | -0.16966 | 应 >0 |

局部五点邻域在目标 token 上有轻微较高的 rescue 信号，但模板—search 相似图无法区分目标与背景；两个模态的目标相似度均低于背景。该结果不支持“目标相似图能够提供可靠目标条件”的前提，因此 G0 `STOP`，不进入 G2。

5 个事件的当前 search crop 不覆盖 GT，已保留在 `event_rows.csv` 并从目标分层均值中显式排除；没有用插值或估计值填补。

## G1 Local Attender P0

使用用户提供的 Local Attender 局部加权聚合逻辑，改为同分辨率 `16x16` search、固定五点偏移、`conv_res=False`。隔离 adapter 为 `TargetConditionedLocalSearchAdapter`，C1/C2 共用参数结构，差异只在 similarity map 是否置零。

CPU P0 结果：

- 输入：template `[2,64,768]`，search `[2,256,768]`；
- shape 和 finite forward：通过；
- gamma=0 与原 search 输出最大误差：`0.0`；
- 非零 gamma 最大输出差：`0.088761`；
- 两步 optimizer 检查：gamma 梯度均非零且有限；
- 五点局部 search shuffle：产生可测输出变化；
- G1 状态：`DIAGNOSTIC-PASS`，但仅说明算子契约成立，不说明跟踪有效性。

G1 CUDA 首次调用未在 30 秒内产出 artifact，未把该尝试当作通过；CPU P0 已完成并保存，未启动训练。

## 决策

停止 Local Attender G2。没有创建 C0/C1/C2 配置、checkpoint 或 tracking 目录；不进行短程继续训练，也不进入 full-244。原因是 G0 的目标条件相似图在真实闭环事件上反而表现为背景更高，无法把后续增益归因于目标条件局部选择。

## 证据

- G0 原始事件：[event_rows.csv](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/local_attender_g0_retry1_20261004/event_rows.csv)
- G0 汇总：[summary_repaired.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/local_attender_g0_retry1_20261004/summary_repaired.json)
- G1 P0：[local_attender_g1_p0_cpu_retry1_20261004.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/local_attender_g1_p0_cpu_retry1_20261004.json)
- Local Attender：[local_attender.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/local_attender.py)
- Adapter：[target_conditioned_local_search.py](/root/autodl-tmp/TBSI_gai/TBSI/lib/models/layers/target_conditioned_local_search.py)
- G0 runner：[audit_local_attender_g0.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/audit_local_attender_g0.py)
- G1 runner：[p0_local_attender.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/p0_local_attender.py)
- G0 汇总脚本：[summarize_local_attender_g0.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/summarize_local_attender_g0.py)
