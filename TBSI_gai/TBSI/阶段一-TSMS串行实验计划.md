# 阶段一 TSMS 串行实验计划

## 控制变量

parent 固定为 A1 `TBSI + SMSA`。A2 只增加训练期 TSMS coverage loss；backbone、输入分辨率、采样器、batch、梯度累积、optimizer、AMP、学习率和 Entmax alpha 全部保持一致。推理代码不启用 TSMS，也不使用 GT。

## 串行顺序

1. **3 epoch full-data preflight**：使用 `v1.3.1-tsms-full-3ep.yaml`，完整 LasHeR train sampler，只检查训练是否正常。
2. **preflight gate**：检查训练日志是否存在、无 NaN/Inf/Traceback、能输出 `TSMS/available`，并确认 TSMS loss/统计不是空值。3 epoch 不判断涨点。
3. **15 epoch full train**：preflight 通过后，从相同预训练权重重新开始 `v1.3.1-tsms-full-15ep`，不从 3 epoch checkpoint 续训，避免改变训练起点。
4. **full test**：只使用 full checkpoint，在 `lasher_test` 的 244 条序列上 `threads=0`、`TBSI_INFER_FP16=0` 测试并分析。

## 结果判定

- 3 epoch 失败：停止，不启动 full；定位 TSMS 实现问题。
- 3 epoch 通过但 full 训练异常：停止并记录为训练稳定性失败。
- full 完成后：比较 A0、A1、A2 的 AUC/Success、OP50、OP75、Precision、Norm Precision，以及逐序列 ΔAUC、属性分布和 TSMS 的 L3/L6/L9、RGB2F/TIR2F 统计。
- 只有 A2 相对 A1 的 full 结果和机制统计同时支持时，才把 TSMS 作为创新点一的有效组成；否则保留为“机制约束有效但 tracking utility 不足”的负结果。

## 统一入口

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
bash scripts/run_tsms_phase1_serial.sh       # 只做安全检查和打印，不启动
bash scripts/run_tsms_phase1_serial.sh --run # 串行执行全部阶段
```

脚本会在 3 epoch 通过后删除仅由 preflight 产生的 checkpoint，但保留 preflight 日志；这一步只释放本次新生成的临时权重，不触碰 A0/A1 或 canonical baseline。
