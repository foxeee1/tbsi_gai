# M1-AMCM-NLFORMER-CONSENSUS-RESIDUAL-ROUTE-20261020

## Question

在 AMCM 内只把 RGB/TIR 模板匹配图改为双模态一致性路由，能否让 NLformer residual 只修正共同支持的目标区域，并改善 M1 已知困难帧中的目标/背景区分？

## Fixed conditions

- Parent: M1 protected epoch-15 model; M1 checkpoint 未用于训练初始化。
- Candidate initialization: `TBSITrack_SOT_Pretrained.pth.tar`。
- Changed variable: AMCM mode `nlformer_consensus_residual_gate` only。
- Training: LasHeR train, 512 samples/epoch, 3 epochs, FP32, batch 2, `NUM_WORKER=4`。
- External path unchanged: backbone, TBSI, `tbsi_fuse_search`, CENTER head, loss and tracker state。
- Diagnostic: same 10 fixed M1 failure events and same hard-background selection as the preceding AMCM route experiments。

## Mechanism

Let `S_rgb` and `S_tir` be normalized template-to-search matching maps. The route is

```text
C(x) = min(S_rgb(x), S_tir(x))
R(x) = exp(C(x) / 0.07) / mean_x[exp(C(x) / 0.07)]
Y_m(x,c) = X_m(x,c) + alpha * R(x) * Delta_m(x,c),  m in {rgb,tir}
```

`alpha=0` is identity. No auxiliary target/background supervision is enabled, so the route is trained only through the existing box losses.

## Results

| quantity | M1 arm | consensus route | interpretation |
|---|---:|---:|---|
| fixed-state mean IoU | 0.30133 | 0.25323 | delta `-0.04810` |
| GT rank mean | 2.8 | 4.2 | worse |
| GT rank improved | - | 1/10 | not repeatable |
| top-1 changed to GT | - | 0/10 | no recovery |
| target/background margin mean | -0.00096 | -0.06993 | worse |
| RGB residual target/background | 0 | 0.26827 | all 10 events background-dominant |
| TIR residual target/background | 0 | 0.26828 | same failure in both modalities |
| RGB/TIR residual spatial CV | 0 | about 1.04 | residual is structured, but in the wrong places |

Per-event values are in `output/experiments/M1_AMCM_NLFORMER_CONSENSUS_RESIDUAL_GATE_FIXED10_DIAG_20261020/per_event.csv` and `paired_per_sequence.csv`.

## Sequence-level reading

- Positive IoU deltas: `boyss` (+0.1568), `girldownstairfromlight` (+0.0827), `rightbluewhite` (+0.0528), `foamatgirl\`srighthand` (+0.0304), `leftopenexersicebook` (+0.0002)。
- Negative deltas: `blkboydown` (-0.2731), `umbrella` (-0.2369), `ab_bolstershaking` (-0.2053), `7rightorangegirl` (-0.0638), `rightdarksingleman` (-0.0249)。
- The largest failures (`blkboydown`, `umbrella`, `ab_bolstershaking`) have GT rank 6/17/2 after routing and strongly negative margins. The route does not recover an off-center target; it amplifies a consensus location that can also be background.

## Information-flow diagnosis

The route receives spatial template similarity, so the input contains location information. The residual is not uniform: spatial CV is about 1.04. However, the matching agreement is not a target certificate; a background patch visible in both modalities also receives a high consensus value. Because no loss directly ranks GT cells above hard background cells, the existing box loss does not force `R(x)` to be target-selective.

The post-module feature measurements show very small changes: mean relative norm change is approximately `+0.003%` for RGB target, `-0.019%` for RGB background, `-0.095%` for TIR target and `-0.030%` for TIR background. Thus the residual is largely attenuated or re-normalized before the measured fused representation reaches the score map. The resulting fused feature distances can change, but not in a consistent target-favoring direction.

## Decision

`FIXED10-COMPLETE / NOT-PROMOTED`.

This candidate is **not** admitted to the 50-sequence test. The experiment distinguishes two hypotheses:

1. **Residual uniformity:** rejected. The route creates strong spatial variation.
2. **Spatial variation is target-selective and survives fusion:** not supported. It is background-dominant in all ten events and is mostly attenuated downstream.

The remaining bottleneck is not the absence of a more complex consensus operator. It is the missing target-vs-hard-background training contract plus weak effective injection into the score-producing fusion path. The next allowed experiment, if continuing this line, must change one AMCM-internal variable that can test one of these two causes: either an AMCM-only contrastive/ranking signal for `R(x)` or an AMCM-only injection form whose post-fusion magnitude is measurable. Do not increase route strength or start full training for this version.
