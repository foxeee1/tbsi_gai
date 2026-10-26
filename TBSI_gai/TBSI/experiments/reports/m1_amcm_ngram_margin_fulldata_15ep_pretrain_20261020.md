# M1 N-gram Local Fusion Full-data Confirmation

## Protocol

- Parent: protected M1, `TBSI_E_MHSA_FULL_4090D_seed42`, fixed epoch15 reference.
- Candidate: N-gram Local Fusion V1 inside AMCM with `AMCM_REPR_LOSS` (weight 0.5, margin 0.05, top-k 8).
- Initialization: only `TBSITrack_SOT_Pretrained.pth.tar`; `INIT_CHECKPOINT` is empty. The M1 checkpoint was not loaded.
- Training: LasHeR train, 59,904 samples/epoch, 15 epochs, batch 16, accumulation 8, AMP, `NUM_WORKER=4`, seed 42.
- Test: LasHeR test, all 244 sequences, FP32, sequential断点补测. Existing 74 sequence outputs were skipped; the remaining 170 were evaluated.

## Overall Results

| Method | AUC/Success | Precision | Norm Precision | OP50 | OP75 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 56.21 | 70.21 | 66.20 | 68.33 | 47.13 | NA |
| N-gram + margin | 54.94 | 68.26 | 64.72 | 66.83 | 46.57 | 56.03 |
| Delta | -1.27 | -1.94 | -1.48 | -1.50 | -0.56 | NA |

The candidate is below M1 on every primary metric. This is a full-data, single-seed negative result, not evidence that every possible N-gram formulation is invalid.

## Sequence Analysis

Per-sequence paired metrics are stored in `m1_ngram_margin_full_vs_m1_per_sequence_20261020.json`. Largest AUC regressions were `10runone` (-62.06), `boyatdoorturnright` (-54.06), `blkboydown` (-38.73), `bluegirlbiketurn` (-36.98), and `midof3girls` (-35.21). Largest gains were `blktribikecome` (+60.20), `3rdfatboy` (+58.12), `whiteboyrightcoccergoal` (+25.35), `girlrightthewautress` (+24.99), and `leftuphand` (+20.51).

The wide positive/negative spread shows that the margin objective activates the module but does not produce a consistently target-selective correction. It trades errors between sequences rather than repairing the M1 crop-local peak failure as a stable behavior.

## Training Signals

`amcm_repr` decreased from about 0.062 at the start to about 0.028 at epoch15, and AMCM gradients remained finite. Therefore the negative result is not explained by a dead module or missing optimizer path. Earlier fixed-frame audits still found GT rank improvement 0/10 and loss-gradient direction inconsistent with target localization. The full-data result is consistent with that mechanism diagnosis.

## Decision

`NO-GO / NOT PROMOTED`. Do not use this run as the new parent for another loss or route sweep. Preserve it as the complete negative confirmation of the current N-gram-plus-margin contract. The next method should first change the AMCM target-selection signal or supervision contract, with a fixed-frame spatial selectivity audit before another full training run.

Evidence:

- Config: `experiments/tbsi_track/M1_AMCM_NGRAM_MARGIN_FULLDATA_15EP_PRETRAIN_20261020.yaml`
- Checkpoint: `output/experiments/M1_AMCM_NGRAM_MARGIN_FULLDATA_15EP_PRETRAIN_20261020/checkpoints/TBSITrack_ep0015.pth.tar`
- Training/test logs: `output/experiments/M1_AMCM_NGRAM_MARGIN_FULLDATA_15EP_PRETRAIN_20261020/logs/`
- Full results: `output/test/tracking_results/tbsi_track/M1_AMCM_NGRAM_MARGIN_FULLDATA_15EP_PRETRAIN_20261020/`

## Per-sequence diagnosis

The paired JSON is the complete per-sequence record. It contains 244 rows with
candidate/M1 AUC, Precision, Norm Precision, and their paired differences:
`m1_ngram_margin_full_vs_m1_per_sequence_20261020.json`. The following bins use
the AUC difference, so they are an exact partition of the valid rows rather than
a sample:

| AUC difference | Count | Mean difference | Interpretation |
|---|---:|---:|---|
| <= -10 | 26 | -22.46 | large regression |
| (-10, -2] | 50 | -4.52 | moderate regression |
| (-2, 2) | 114 | -0.22 | practically unchanged |
| [2, 10) | 38 | +4.80 | moderate gain |
| >= 10 | 16 | +21.43 | large gain |

This is the complete sequence membership of the non-neutral bins:

- Large regression: `10runone`, `7rightorangegirl`, `ab_bolstershaking`,
  `ab_rightlowerredcup_quezhen`, `ab_whiteboywithbluebag`, `basketballathand`,
  `bikeboyintodark`, `blkboydown`, `bluegirlbiketurn`, `boyatdoorturnright`,
  `boyss`, `darktreesboy`, `leftbottle2hang`, `leftfarboycomingpicktheball`,
  `lefthyalinepaper2rgb`, `lowerfoamboard`, `midof3girls`, `minibusgoes2left`,
  `rightblkboy4386`, `rightblkfatboyleftwhite`, `rightbottlecomes`,
  `rightdarksingleman`, `umbrella`, `umbrellabyboy`, `whitecarturnright`,
  `whiteridingbike`.
- Moderate regression: `1strowrightgirl3540`, `2runseven`, `ab_bikeoccluded`,
  `ab_blkskirtgirl`, `ab_girlcrossroad`, `ballshootatthebasket3times`,
  `besom3`, `bikeboyturntimes`, `bikefromlight`, `bikegoindark`, `bikeinrain`,
  `blkboy\`shead`, `blkgirlumbrella`, `blueboy`, `boy2buildings`,
  `boy2treesfindbike`, `boyaftertree`, `boyinplatform`, `boyoncall`,
  `boyplayphone`, `boywalkinginsnow2`, `broom`, `carcomingfromlight`,
  `carcominginlight`, `couple`, `firstexercisebook`, `girllongskirt`,
  `girlof2leaders`, `leftboyoutofthetroop`, `leftexcersicebookyellow`,
  `lefthyalinepaperfrontpants`, `leftmirror`, `leftrushingboy`,
  `littelbabycryingforahug`, `mandownstair`, `manoncall`, `midblkgirl`,
  `midboyNo_9`, `pinkwithblktopcup`, `raincarturn`, `rainycarcome_ab`,
  `redroadlatboy`, `rightblkboystand`, `rightboy504`, `rightcomingstrongboy`,
  `rightwaiter1_quezhen`, `turning1strowleft2ndboy`, `whitecarcomeinrain`,
  `whiteofboys`, `whiteskirtgirlcomingfromgoal`.
- Moderate gain: `1strowrightdrillmaster`, `3rdgrouplastboy`, `baggirl`,
  `belowdarkgirl`, `bike2trees`, `blkboyback`, `blkboylefttheNo_21`,
  `blkcaratfrontbluebus`, `blkhairgirltakingblkbag`, `boy2basketballground`,
  "boy`headwithouthat", `boy\`sheadingreycol`, `boytakingbasketballfollowing`,
  `boyunder2baskets`, `darkouterwhiteboy`, `foamatgirl\`srighthand`,
  `girldownstairfromlight`, `hyalinepaperfrontface`, `large`,
  `leftgirl'swhitebag`, `leftmirrorside`, `leftopenexersicebook`,
  `leftunderbasket`, `mangetsoff`, `mansimiliar`, `midrunboywithwhite`,
  `pingpingpad3`, `redtricycle`, `right2ndflagformath`, `right5thflag`,
  `shinybikeboy2left`, `shinycarcoming2`, `umbrellawillopen`, `whitefargirl`,
  `whiterunningboy`, `whitesuvturn`, `yellowgirl118`, `yellowskirt`.
- Large gain: `3rdfatboy`, `4thboywithwhite`, `blkboyhead`, `blktribikecome`,
  `bluebuscoming`, `boywaitgirl`, `girlrightthewautress`, `leftboy2jointhe4`,
  `leftuphand`, `midgreyboyrunningcoming`, `redmidboy`, `rightbluewhite`,
  `rightcar-chongT`, `swan_0109`, `whiteboyrightcoccergoal`,
  `whitecarturn683`.

The extreme paired rows show that the change is not a small calibration shift:

| Sequence | M1 AUC | Candidate AUC | Delta AUC | Delta Precision | Delta Norm Precision |
|---|---:|---:|---:|---:|---:|
| `10runone` | 79.48 | 17.42 | -62.06 | -78.29 | -76.32 |
| `boyatdoorturnright` | 61.15 | 7.09 | -54.06 | -61.13 | -62.35 |
| `blkboydown` | 49.77 | 11.03 | -38.73 | -57.32 | -48.78 |
| `bluegirlbiketurn` | 59.55 | 22.57 | -36.98 | -49.10 | -41.89 |
| `midof3girls` | 49.55 | 14.34 | -35.21 | -40.55 | -37.36 |
| `blktribikecome` | 23.88 | 84.07 | +60.19 | +71.13 | +70.42 |
| `3rdfatboy` | 18.53 | 76.65 | +58.12 | +71.00 | +69.79 |
| `whiteboyrightcoccergoal` | 39.09 | 64.44 | +25.35 | +30.97 | +30.68 |
| `girlrightthewautress` | 24.71 | 49.69 | +24.98 | +33.44 | +30.19 |
| `leftuphand` | 51.95 | 72.45 | +20.51 | +25.09 | +26.05 |

The 114 near-neutral sequences indicate that the adapter is often not changing
the tracker decision enough to matter. The 26 large regressions and 16 large
gains are both broad enough to rule out a universal target-preserving behavior.
However, this table alone cannot identify whether a particular row failed from
an incorrect crop peak, target suppression, or later state drift: the current
per-sequence artifact has no frame-level boxes, response maps, or residual
maps. Those claims require a separate fixed-frame instrumentation run.

## Core equations and causal reading

For each modality, the module first creates local context

```text
L_m = GELU(M_m([C_3(X_m), C_5(X_m), C_7(X_m)])),  m in {rgb,tir}.
```

The template evidence is a soft token match, not a spatially aligned target
label:

```text
s_m(i,j) = <norm(X_m(i)), norm(T_m(j))>
a_m(i,j) = softmax_j(s_m(i,j)/0.07)
q_m(i)   = sum_j a_m(i,j) s_m(i,j).
```

The residual predictor receives

```text
E(i) = [L_rgb(i), L_tir(i), |L_rgb(i)-L_tir(i)|, q_rgb(i), q_tir(i)].
```

It predicts a bounded, mean-centered message and routes it with the opposite
modality's centered similarity:

```text
u_m = 0.10 * tanh(F_m(E));       u_m <- u_m - mean_i(u_m)
r_rgb = sigmoid((q_tir - mean(q_tir))/0.07) * (0.5 + p_tir)
r_tir = sigmoid((q_rgb - mean(q_rgb))/0.07) * (0.5 + p_rgb)
X'_m(i) = X_m(i) + r_m(i) u_m(i).
```

`X'_rgb` and `X'_tir` are passed into `tbsi_fuse_search`; this is a genuine
pre-fusion feature residual, not post-fusion score scaling. Nevertheless, the
route is still derived from an unlabeled template-similarity peak. A high
background similarity is therefore eligible to receive the same correction as
the target. The spatial mean subtraction removes only a constant component; it
does not make the residual target selective.

The auxiliary objective uses a single global template prototype and a coarse
GT-cell mask:

```text
f_i = norm(0.5*(norm(X'_rgb(i)) + norm(X'_tir(i))))
p   = norm(0.5*(mean(T_rgb) + mean(T_tir)))
s_i = <f_i,p>
L_repr = 0.50 * ReLU(0.05 - max_GT(s_i) + max_BG(s_i)).
```

This loss can reduce the strongest background score relative to the strongest
GT cell, but it does not supervise the residual direction, the full target
region, the response-map shape, or the subsequent TBSI attention. Thus the
observed finite gradients and decreasing loss establish that the branch learns,
but they do not establish that it learns the desired crop-local correction.

## Conclusion from the paired evidence

The full test supports `NO-GO / NOT PROMOTED` for this exact contract. The
negative result is best attributed to a target-selection/coordination mismatch
(the module is trainable and connected, but its similarity route and prototype
loss do not distinguish target from hard background reliably). It does not yet
prove that all N-gram local operators are unusable. A valid next experiment is a
single fixed-frame audit that records target/background residual energy, GT rank,
peak margin, and the same quantities immediately after `tbsi_fuse_search`.
Only a repeatable spatially selective signature on the known failure frames
justifies another full training run.
