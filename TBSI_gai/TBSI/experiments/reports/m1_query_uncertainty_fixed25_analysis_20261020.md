# M1 Query-conditioned uncertainty 固定帧逐序列归因

## 实验合同

每个事件使用同一首帧模板、同一 `tracker_state_before`、同一 crop 和同一 GT；分别运行 M1 `off` 与 `query_soft (tau=0.5,T=0.15)`。因此这里测的是当前帧的直接定位因果变化，不包含候选修改导致的后续状态漂移。M1 权重为受保护 epoch15，FP32。

## 总体结果

| 范围 | event IoU delta | margin delta | GT rank delta | 胜/负 |
|---|---:|---:|---:|---:|
| 25 events | -0.01584 | -0.00096 | +0.04 | 10/10 |
| 20 failure | -0.01976 | -0.00098 | +0.05 | 8/7 |
| 5 control | -0.00012 | -0.00086 | 0.00 | 2/3 |

唯一发生 top-1 归属翻转的是 `rightcar-chongT`：GT rank `1→2`，peak margin `+0.0030→-0.0026`，fixed-state IoU `0.499→0.101`，IoU 下降 `-0.3981`。这说明最终定位对极小 margin 改变高度敏感。

## 逐序列结果

完整字段见 `output/experiments/M1_QUERY_UNCERTAINTY_FIXED25_ANALYSIS_20261020/per_event.csv`。关键列为 `query_iou_delta`、`margin_delta`、`gt_rank_delta`、`top1_flip`、`feature_target_bg_l2_delta`、`feature_target_bg_cosine_delta`。

| sequence | IoU delta | margin delta | rank M1→Q | top1 flip |
|---|---:|---:|---:|---|
| girl`sblkbag | +0.0015 | +0.0008 | 1→1 | no |
| 1strowleftboyturning | +0.0001 | -0.0007 | 1→1 | no |
| ab_pingpongball2 | -0.0002 | +0.0004 | 1→1 | no |
| boy`sheadingreycol | -0.0004 | -0.0007 | 1→1 | no |
| rightcar-chongT | **-0.3981** | -0.0056 | **1→2** | **yes** |
| biketurnright | -0.0002 | -0.0007 | 2→2 | no |
| basketball849 | -0.0020 | -0.0032 | 1→1 | no |
| leftgirl'swhitebag | -0.0013 | -0.0006 | 2→2 | no |
| leftboy2jointhe4 | +0.0000 | -0.0088 | 2→2 | no |
| 2girlup | +0.0003 | +0.0002 | 2→2 | no |
| leftpingpongball | +0.0000 | +0.0006 | 89→89 | no |
| leftunderbasket | +0.0001 | -0.0003 | 2→2 | no |
| ab_bikeoccluded | +0.0047 | -0.0003 | 1→1 | no |
| lastleftgirl | +0.0000 | -0.0018 | 2→2 | no |
| bottlebetweenboy`sfeet | +0.0002 | -0.0007 | 1→1 | no |
| 1stcol4thboy | +0.0000 | -0.0007 | 8→8 | no |
| rightgirltakingcup | +0.0008 | -0.0020 | 3→3 | no |
| middrillmaster | +0.0007 | +0.0021 | 2→2 | no |
| boyunder2baskets | -0.0012 | -0.0016 | 2→2 | no |
| whitesuvturn | +0.0000 | +0.0039 | 2→2 | no |
| darkcarturn | -0.0002 | -0.0004 | 1→1 | no |
| silvercarturn | -0.0003 | -0.0011 | 1→1 | no |
| boyruninsnow | +0.0001 | -0.0004 | 1→1 | no |
| whitecarturnleft | +0.0003 | -0.0032 | 1→1 | no |
| girlafterglassdoor | -0.0005 | +0.0008 | 1→1 | no |

## 哪些信号影响了定位

1. **直接决定峰归属的是 peak margin，而不是整体 feature norm。** `feature_target_bg_l2` 平均只变化 `-0.01398`，target/background cosine 平均变化约 `1.2e-5`；融合输出范数平均变化 `-0.1208`，相对输出约 `-0.04%`。
2. **query participation 没有稳定提高 GT margin。** 平均 margin 下降 `0.00096`，多数事件仍保持原 GT rank；它改变了特征数值，但没有改变 CENTER 的离散峰排序。
3. **真正的灾难来自边界事件。** `rightcar-chongT` 的 M1 margin 只有 `+0.0030`，query 调制后变为 `-0.0026`，一次 top-1 翻转就造成 `-0.398` IoU。其余 24 个事件没有 top-1 翻转，IoU 变化均不超过 `0.005`。
4. 之前闭环实验中出现的 `whitesuvturn`、`leftunderbasket` 大幅变化，在固定状态分析中消失，说明它们主要是后续 tracker 状态反馈造成的，不是当前帧 query modulation 直接改善了目标峰。

## 结论

当前模块没有提供稳定的目标/背景表征分离。它对定位的可观测影响主要是对接近决策边界的峰做极小扰动，且方向不可控；因此不能进入训练。M1 的主要瓶颈仍是“模板条件没有可靠地决定 search 中哪个局部证据是目标”，而不是简单的 token participation 强度不足。

证据：

- [逐事件 CSV](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_QUERY_UNCERTAINTY_FIXED25_ANALYSIS_20261020/per_event.csv)
- [汇总 JSON](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_QUERY_UNCERTAINTY_FIXED25_ANALYSIS_20261020/summary.json)
- [分析脚本](/root/autodl-tmp/TBSI_gai/TBSI/tools/run_m1_query_uncertainty_fixed25_analysis_20261020.py)
