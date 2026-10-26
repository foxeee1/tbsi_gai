# 实验资产数据库说明

人工查看和增量维护的主表是 `experiments/实验资产数据库.md`，使用中文分组展示，避免超宽表难以核对。
机器可读同步表是 `experiments/experiment_asset_registry.csv`；每次更新 Markdown 主表时必须同步更新 CSV。
`experiments/registry.csv` 仅作为兼容旧记录保留；新条目必须先进入中文主表和详细 CSV。

## 增量更新规则

每个完成、失败、中断、诊断或 screening 运行都必须在结束后立即登记一行。
后续测试、分析或 provenance 修复更新原 ID，不创建含义不清的新行，也不覆盖历史记录。

必填字段为：

- 身份：`ID`、日期、parent、commit 或完整快照、config、seed；
- 协议：数据集、评测范围、checkpoint、结果范围；
- 指标：AUC、PR、NPR、OP50、OP75、FPS；
- 资源：GPU 型号、硬件/软件配置、GPU 时长；
- 解释：验证假设、实验结论、状态和证据路径。

未记录项目使用 `NA`。必须使用明确状态，例如 `VALID`、`PROVISIONAL`、`DIAGNOSTIC`、
`TRAINING-ONLY`、`RUNNING` 或 `HISTORICAL-UNVERIFIED`；不得从其他运行推断缺失指标。

对于解析 `eval_data.pkl` 的记录，指标提取口径为：success curve 均值作为 AUC，
20 像素处 precision curve 作为 PR，20 像素处 normalized precision curve 作为 NPR，
success curve 在 0.50/0.75 overlap 阈值处的值作为 OP50/OP75。此类记录必须保留 artifact 路径和评测范围。
