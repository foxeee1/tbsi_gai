# 阶段一：TSMS 最小可证伪实验

## 目标

验证训练期的 Target-guided Sparse Modulation 是否能让 SMSA 的稀疏支持更多覆盖目标区域，同时不破坏跟踪优化。TSMS 只使用 search 的训练标注，推理阶段完全关闭，因此不会把 GT 泄漏到测试模型。

## 实验组

| 组别 | 模型 | 状态 | 用途 |
|---|---|---|---|
| A0 | 原始 TBSI | 已有 canonical 结果 | 54.82 参考 |
| A1 | TBSI + SMSA | 已有 full 结果 | 55.40 直接 parent |
| A2 | TBSI + SMSA + TSMS coverage | 本轮实现 | 只改变训练损失 |

A2 配置为 `experiments/tbsi_track/v1.3.0-tsms-rpp-15ep.yaml`，独立输出目录为 `output/experiments/v1.3.0-tsms-rpp-15ep/`。CTVM、TCMR、background penalty、dynamic alpha 和 router 均不启用。

## A2 的唯一新增机制

对 L3/L6/L9 的 RGB2F/TIR2F SMSA attention，按 head 和 query 平均得到 `Abar ∈ R^256`。用 search GT box 在 16×16 search token 网格上生成带 1 token dilation、0.5 token soft edge 的 mask `M_gt`，只增加：

```text
GTMass = sum(Abar * M_gt)
Lcov = mean(max(0, rho - GTMass))
L = Ltrack + 0.05 * Lcov, rho = 0.20
```

不增加背景抑制项，先避免把“目标覆盖不足”和“背景惩罚过强”混为一个结论。attention 的 forward 数值不变，训练 actor 只从已有 attention 图构造辅助损失。

## 固定数据和执行口径

- train/test 使用已冻结的 sequence-level stratified RPP manifest；不重新抽样、不根据结果改成员。
- 使用完整 15 epoch proxy schedule，不使用通用 3 epoch proxy 替代正式协议。
- backbone、分辨率、batch、梯度累积、optimizer、AMP 和 augmentation 保持 A1 parent 不变。
- 测试使用固定分层 test proxy、`threads=0`、`TBSI_INFER_FP16=0`。
- RPP gate 已知未通过（proxy 排序不能预测 full AUC），所以 A2 proxy 结果只能用于机制/稳定性筛选；不得直接宣称 A2 full gain。
- 真正的 tracking 结论必须由 A1 vs A2 的 full confirmation 给出。

## 必记指标

训练日志按 `TSMS/L{3,6,9}/{RGB2F,TIR2F}/` 记录：

`GTMass`、`GTPrecision`、`GTRecall`、`BGMass`、`effective_support`、`entropy`、`zero_ratio`。

同时记录 `Loss/tsms_cov`、`TSMS/Lcov_raw`、tracking loss、IoU、SMSA 原有 entropy/support/norm error，以及最终 AUC、Success、OP50、OP75、Precision、Norm Precision。

## 判定门

1. **实现门**：A2 能完成 forward/backward；`TSMS/available=6`；无 NaN/Inf；A1 disabled parity 不受影响。
2. **机制门**：相对 A1，GTMass/GTRecall 有稳定上升，effective support 不塌缩，BGMass 不出现系统性异常上升，`Loss/tsms_cov` 不长期爆炸。
3. **跟踪门**：proxy 只作筛选。若 A2 proxy 明显不稳定或明显退化，停止 full；若接近 A1，保留进入 full confirmation，不凭 proxy 的微小 AUC 差异淘汰。
4. **最终门**：full confirmation 中 A2 必须相对 A1 在主指标和多 seed/序列分布上得到支持，才把 TSMS 纳入创新点一；否则记录为机制有效但 tracking utility 不足的负结果。

## 统一入口

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
bash scripts/run_tsms_phase1.sh          # 只校验并打印计划，不启动训练
bash scripts/run_tsms_phase1.sh --run    # 磁盘/进程/manifest 检查通过后才启动 A2
```

脚本不会修改 canonical baseline，不会覆盖 A0/A1 目录；A2 的训练、测试、分析 stdout 和 manifest 均写入独立目录。
