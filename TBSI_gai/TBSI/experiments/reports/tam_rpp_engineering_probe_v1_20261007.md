# TAM RPP Engineering Probe v1

状态：`INTERRUPTED-SCOPE-CHANGE`   
日期：2026-10-07  
父级：`TAM-IMAGE-REPLAY-TRAIN-AUDIT-20261007`  
级别：exploratory engineering probe；不是通过 RPP-v1 历史准入后的 screening

## 问题与假设

在固定 RPP-v1 60-sequence test manifest 上，将已有 100-step image-backed TAM adapter 接入完整在线 tracker；与同一 M1 epoch-15 checkpoint 做 FP32 配对测试。零假设是该短训 adapter 不改变或不能改善冻结 M1 闭环；该测试只用于观察接线、缓存时序和明显漂移，不据此筛选/准入 TAM。

## 固定合同

- C0：冻结 M1 epoch-15，独立配置 `TAM_RPP_ENGINEERING_C0_seed42`。
- TAM100：相同 M1 + 100-step actor audit adapter，独立配置 `TAM_RPP_ENGINEERING_TAM100_seed42`。
- Dataset：冻结 `proxy_test_sequences.txt` 的 60 条 LasHeR test 序列；与 proxy train 的 176 序列交集为 0。
- Inference：FP32 (`TBSI_INFER_FP16=0`)，固定 6 threads；同一 evaluator。
- TAM100 权重来自 100-step image replay actor audit；未在本轮新训练。
- RPP-v1 历史 anchor admission 尚未证实，因此任何指标都只作探索性工程观察，不作 proxy screening/selection 或论文结果。
- 不运行 full-244；不改冻结 manifests、canonical baseline YAML 或输出目录。

## 命令

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
./tools/run_tam_rpp_engineering_probe.sh
```

## 结果与中断

本运行没有完成。检查时无活跃 TBSI 进程；日志仅记录到 C0 部分序列，出现 14 条序列起始，C0 结果目录有 8 条完整预测文件。60 序列 C0 分析未完成，TAM100 tracker 测试未启动，故 AUC/PR/NPR/OP50/OP75/FPS 汇总均为 NA。部分结果保留，不得视为配对测试或 proxy screening。

因用户将目标收敛到固定 crop 单步可证伪诊断，本探针记为 `INTERRUPTED-SCOPE-CHANGE`，不在本轮恢复。见 `experiments/reports/tam_fixed_crop_counterfactual_v1_20261007.md`。

启动失败记录：首次启动被一个包含正则字样的 `watch nvidia-smi` shell 误触发进程检查；修复后 checkpoint merge 首次因缺少仓库 `lib` 导入路径失败，之后修复并生成 merged checkpoint。随后仅有部分 C0 运行，本轮范围切换后结束；没有 TAM100 结果。所有日志和部分预测保留。
