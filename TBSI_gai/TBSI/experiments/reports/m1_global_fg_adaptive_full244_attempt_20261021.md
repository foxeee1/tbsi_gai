# M1 Adaptive Full-244 测试报告

> 2026-10-21 provenance 更正：下文 43.65 AUC 是旧保存轨迹的历史 evaluator 输出，**不是有效的匹配 FP32 方法比较**。同一 checkpoint 的 FP16 首更新在代表序列出现 AMCM entropy NaN 与 `0,0,10,10`，FP32 首更新正常；旧断点结果的精度口径无法统一核验。详见 `m1_amcm_adaptive_precision_contract_audit_20261021.md`。原始结果和叙述保留用于审计；独立 FP32 全量复核结果另行登记。

## 目的

在同一 LasHeR testingset full-244 口径下比较 M1、foreground-only、mix50 和 adaptive，确认 25-event proxy 结论是否能推广到全测试集。

## 入口核验

- 数据集注册名：`lasher_test`
- full 测试入口：`tracking/test.py`
- M1 受保护结果未重跑、未覆盖
- 候选 checkpoint 通过 `TBSI_CHECKPOINT_OVERRIDE` 显式指定
- 首次字符串 run_id 试验在数据加载前失败；第二次错误的数字 run_id 试验因 checkpoint epoch 合同不匹配而中止；这两次均不产生有效指标

## 运行状态

adaptive checkpoint 已按 M1 的 FP32 串行入口完成 `lasher_test` 全 244 条序列。评测命令使用显式 `TBSI_CHECKPOINT_OVERRIDE`、`--threads 0`、`--num_gpus 1`、`--runid 3`；断点续测复用同一输出目录，已完成序列自动跳过。

结果目录 `output/test/tracking_results/tbsi_track/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021_3/` 包含 244 个 bbox 文件和 244 个 time 文件；日志为 `output/experiments/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021/full244_test_3.log`。

官方评测结果（244/244）：AUC/Success `43.65`，Precision `54.14`，Norm Precision `51.33`，OP50 `52.71`，OP75 `36.85`。

## 判定

**ARTIFACT-COMPLETE / PRECISION-CONTRACT-INVALID**（原判定 `FULL244-COMPLETE / NEGATIVE-VS-M1` 已撤回）。

旧保存轨迹的 evaluator 指标低于受保护 M1 参考（AUC `56.21`、Precision `70.21`、Norm Precision `66.20`、OP50 `68.33`、OP75 `47.13`），但已发现半精度 NaN 合同问题，不能用这组数据判断 adaptive entropy fallback 的真实增益。独立 FP32 全量复核才可形成结论。

## 历史中断记录

首次 `threads=2` 运行和中途停止的状态保留在本报告历史中；随后使用同一目录按 M1 串行口径断点完成，最终指标以上述官方结果为准。
