# G0-Expand / Cplain 闭环诊断分析

日期：2026-10-05  
Parent：`G0-EXPAND-CONDITIONAL-PASS-20261004`  
最终机器决策：`CPLAIN_STOP_NO_GO`

## 1. 决策问题

G0-Expand 只允许了一次低成本的 Cplain 闭环诊断：冻结 M1 参数，在 search 路径的 location 9 输入前加入 block 2 的 RGB/TIR early-search residual，并训练旁路投影和两个标量 gate。要回答的问题是：probe 中观察到的早期特征增量，能否在完整 tracker 闭环中稳定改变跟踪决策并改善结果。

本轮不测试 SCSU，不做 full-244，也不把 probe AP 当作 tracking AUC。闭环闸门预先要求：Cplain 相对 C0 有可解释的序列级收益、没有明显退化，并且训练后强制 `gamma=0` 能回到 C0。

## 2. 固定合同与实现

| 项目 | 固定内容 |
|---|---|
| M1 parent | `TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef` |
| 训练/评估序列 | 训练 152 条固定序列；评估 24 条固定 G0-Expand 新序列；sequence ID 完全不相交 |
| 评估事件 | 24 条序列、完整闭环序列结果；不是 probe token 任务 |
| C0 | 原 M1 checkpoint，旁路关闭 |
| Cplain | block 2 early RGB/TIR search token，各自 Linear 投影和 scalar gate，location 9 search-only residual；M1 参数冻结 |
| gamma=0 | 使用同一个训练后 checkpoint，强制两个 gate 为零 |
| 推理 | FP32，`TBSI_INFER_FP16=0`，相同 crop、tracker、评估代码 |
| 训练 | 1 个短 epoch，seed 42，AMP 关闭，旁路-only optimizer |

实现文件：`lib/models/layers/early_search_residual.py`、`tools/run_g0_cplain.sh`、`tools/g0_cplain_analyze.py`。配置和序列清单均为独立文件，未修改 canonical baseline YAML。

## 3. 运行完整性与第一次作废运行

第一次 Cplain 训练虽然设置了冻结参数，但未冻结 M1 子模块的训练状态，导致共享 BatchNorm `num_batches_tracked` 等 buffer 发生变化。该 checkpoint 与 M1 共有 39 个状态项不同，不能用于 Cplain 或 gamma=0 结论。

该运行没有混入最终统计，完整证据保留在：

`output/.trash_g0_cplain_invalid_20261004/`

随后修复 `CPLAIN_ONLY` 状态逻辑：无梯度 M1 子模块保持 `eval()`，只有 adapter 保持 train mode。修复后重新检查 checkpoint：M1 共享参数和 buffer `different_total=0`，仅旁路参数发生变化。最终统计只使用 retry2 目录。

## 4. 闭环结果

| 条件 | 序列数 | Mean AUC | Mean IoU |
|---|---:|---:|---:|
| C0 | 24 | 0.718779 | 0.720613 |
| Cplain | 24 | 0.719125 | 0.720971 |
| gamma=0 | 24 | 0.718779 | 0.720613 |

Cplain 相对 C0 的配对结果：

- Mean ΔAUC：`+0.000346`
- Mean ΔIoU：`+0.000357`
- AUC：`8 win / 16 loss / 0 tie`
- C0 AUC < 0.5 的困难序列：2 条；其中 2 条改善
- AUC 下降至少 0.05：4 条
- IoU 下降至少 0.05：4 条

配对差异由少数序列主导。例如 `midblkboyplayingphone` AUC `+0.281261`，但 `boybackpack` 为 `-0.126869`，`redgirlafterwhitecar` 为 `-0.080056`，`rightbhindbike-gai` 为 `-0.062432`，`1strowleft3rdgirl` 为 `-0.053839`。因此均值的微小正值不能解释为稳定的闭环收益。

## 5. 关门与可学习性检查

### gamma=0 parity

gamma=0 与 C0 比较了全部 24 条序列：序列长度全部一致，最大逐框绝对框差为 `0.0`。该结果通过了完整 tracker 级 parity，说明旁路没有在关闭时改变 M1 的前向行为，且配置/checkpoint 覆盖路径可追踪。

### gate 与 residual

训练后：

- `gamma_rgb = -0.0124245`
- `gamma_tir = +0.0139027`
- 在冻结 G0-Expand 导出 crop 上估计的 residual/late norm：RGB `0.00507`，TIR `0.00732`

gate 非零和 residual 存在只证明旁路被训练并写入了非零信号，不能替代 tracker 闭环闸门。当前信号很小且符号相反，结合逐序列负向结果，不能支持继续堆叠模块。

## 6. 与 G0-Expand probe 的关系

第一关允许进入 Cplain，但并没有证明纯视觉早期信息已经被隔离出来：G0-Expand 的 `XY_only` 相对 `L_zero` 增量为 seed 42/3407 的 `+0.05806/+0.05061`，高于 `L_E` 的 `+0.04273/+0.04586`。因此 probe 结果最多支持“早晚表征旁路值得一次闭环验证”。

Cplain 的闭环结果没有把这个弱条件证据提升为可用跟踪改进：均值改善接近零，胜负明显偏负，并出现严重退化。闭环证据优先于 probe AP，故不能准入 SCSU。

## 7. 机器决策

```text
CPLAIN_STOP_NO_GO
```

理由：

1. 实现合同通过：M1 冻结状态修复后共享状态不变，gamma=0 完整 tracker parity 通过。
2. 方法效果不通过：Cplain 的平均 AUC/IoU 增益只有 `+0.000346/+0.000357`。
3. 稳定性不通过：AUC 逐序列 `8/16`，并有 4 条 AUC 和 4 条 IoU 严重退化序列。
4. 当前不具备足够证据证明早期旁路能稳定改善跟踪决策。

后续动作：停止 Cplain 扩大、停止 SCSU/P1、停止 full-244；保留代码、配置、checkpoint、作废运行和本报告作为负向/边界证据。该结论只针对本次固定 Cplain 合同，不替代 canonical M1 baseline，也不构成 full-test 主结果。

## 8. 证据路径

- 第一关报告：`experiments/reports/g0_expand_analysis_20261004.md`
- 闭环汇总：`output/experiments/g0_cplain_20261004/closed_loop_summary_retry2.json`
- C0：`output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_g0_cplain_c0/`
- Cplain：`output/test/tracking_results/tbsi_track/G0_CPLAIN_DIAGNOSTIC_seed42_g0_cplain_trained_retry2/`
- gamma=0：`output/test/tracking_results/tbsi_track/G0_CPLAIN_GAMMA0_seed42_g0_cplain_gamma0_retry2/`
- 首次作废运行：`output/.trash_g0_cplain_invalid_20261004/`
- 配对分析代码：`tools/g0_cplain_analyze.py`
- 启动脚本：`tools/run_g0_cplain.sh`
