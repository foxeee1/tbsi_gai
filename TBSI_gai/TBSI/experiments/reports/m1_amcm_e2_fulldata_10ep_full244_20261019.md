# E2-FULLDATA-10EP M1-Matched Full-244 Analysis

## 条件

- 实验：`M1_AMCM_E2_FULLDATA_10EP_M1MATCH_20261018`
- Parent：M1 `TBSI_E_MHSA_FULL_4090D_seed42`
- 训练：LasHeR train，59904 samples/epoch，10 epochs，batch16，累积8，有效 batch128，AMP，worker4
- 测试：LasHeR test full-244，FP32，checkpoint `TBSITrack_ep0010.pth.tar`
- AMCM：post-fusion、`legacy` mode、仅训练 `amcm_reliability.*`

## 总体结果

| Run | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 56.21 | 68.33 | 47.13 | 70.21 | 66.20 |
| S1 | 56.46 | NA | NA | NA | NA |
| E2 proxy/短训 | 56.68 | NA | NA | NA | NA |
| E2 full-data 10EP | 56.09 | 68.18 | 47.06 | 69.99 | 66.15 |

相对 M1，E2 full-data 为 AUC -0.12、OP50 -0.15、OP75 -0.07、Precision -0.22、Norm Precision -0.05。相对原 E2 proxy/短训，AUC 下降 0.59；因此 proxy 中的正增益没有在 M1-matched full-data 训练中复现。

## 逐序列结果

逐序列 AO 的简单平均为 56.093。相对 M1，244 条序列中 52 条提升超过 0.05，190 条下降超过 0.05，2 条近似持平；逐序列平均差为 -1.03，中位数差为 -1.07。该方向统计与官方全局 AUC 不完全相同，因为官方 AUC 按帧和曲线聚合。

相对 M1 的主要增益：`blktribikecome` (+57.78)、`leftfarboycomingpicktheball` (+48.86)、`large` (+12.09)、`whiteboyrightcoccergoal` (+11.55)、`leftboyoutofthetroop` (+8.02)、`whiteskirtgirlcomingfromgoal` (+6.09)。

相对 M1 的主要退化：`boyleftblkrunning2crowd` (-29.67)、`bluegirlbiketurn` (-23.02)、`bluebuscoming` (-21.31)、`leftexcersicebookyellow` (-17.99)、`ab_whiteboywithbluebag` (-15.90)、`carcomingfromlight` (-15.82)、`redmidboy` (-14.41)、`pinkwithblktopcup` (-13.31)、`shinybikeboy2left` (-11.58)、`rightblkboy4386` (-8.84)。

完整逐序列表见 [per_sequence_comparison.csv](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_AMCM_E2_FULLDATA_10EP_M1MATCH_20261018/analysis/per_sequence_comparison.csv)。

## 特征和梯度证据

- epoch1：RGB/TIR 输入范数约 630.6/581.0，修正范数约 15.3/5.1。
- epoch5：修正范数约 34.5/22.0。
- epoch10：修正范数约 41.8/29.0，约为输入范数的 6.6%/5.0%。
- AMCM 平均梯度范数约从 epoch1 的 0.0153 增至 epoch10 的 0.0223，`zero_grad_count=0`。

这证明 AMCM 不是恒等映射，也不是没有梯度；它学习了输入条件的有界修正。但当前 legacy post-fusion 形式没有跨模态消息传递、历史状态或上一框条件，最终主要是对 TBSI fused feature 做平均门控缩放。大量序列退化说明“有动态修正”不等于“修正方向与 CENTER 定位目标一致”。

## 结论

结论：**暂不支持** E2 full-data 在当前 AMCM 合同下优于 M1。已排除“模块完全没有梯度/完全恒等”的解释；尚未区分的是后融合位置、监督目标与中心定位目标不一致、以及目标条件不足三种原因。下一步应优先做单变量的 directed residual 或 target-conditioned localization 对照，并保留同一 full-244 口径，暂不继续增加 AMCM 宽度或多尺度复杂度。
