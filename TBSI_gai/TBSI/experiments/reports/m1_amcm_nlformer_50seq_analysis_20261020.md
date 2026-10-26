# M1 AMCM NLformer 50-sequence Analysis

## Evaluation scope

The frozen 50-sequence cohort is the prior E2 manifest
`experiments/proxy/lasher_e2_50seq_20261017.txt`, reused without changing
membership. Candidate predictions were generated with the 3-epoch NLformer
AMCM checkpoint and published SOT-pretrained initialization. M1 predictions
come from the protected full M1 result directory.

The reported AUC here is a transparent proxy computed from per-frame IoU as
the mean success rate over thresholds 0.00--1.00. It is not the official
LasHeR evaluator's full-test AUC and must not be presented as a full benchmark
metric.

## Aggregate result

| Scope | M1 mean IoU | NLformer mean IoU | Delta | M1 proxy AUC | NLformer proxy AUC | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Frozen 50 sequences | 0.54668 | 0.53370 | -0.01298 | 0.54747 | 0.53468 | -0.01279 |

Sequence-level wins/losses: `21 / 29`. The result is exploratory because the
candidate was initialized from the general SOT checkpoint while M1 was already
trained for 15 epochs.

## Baseline-stratified result

Using the M1 proxy AUC only as a diagnostic split:

| Group | Count | Mean delta proxy AUC | Wins | Losses |
|---|---:|---:|---:|---:|
| M1 proxy AUC < 0.50 | 19 | +0.02687 | 8 | 10 |
| M1 proxy AUC >= 0.50 | 31 | -0.03710 | 13 | 18 |

This is not a reliable improvement on the failure cohort: the low-baseline
group has a small positive mean but still has more losses than wins. The
negative high-baseline group shows a stability cost on sequences where M1 was
already tracking well.

## Largest changes

Largest gains by mean IoU were `3rdfatboy` (+0.5955), `4men` (+0.3447),
`4thboywithwhite` (+0.1365), and `AQgirlwalkinrain` (+0.1224). The two first
gains are recovery-like jumps from very poor M1 tracks and should be checked
for whether the module selects a new peak early in the sequence.

Largest losses were `10runone` (-0.6349), `ab_bolstershaking` (-0.2424),
`7rightorangegirl` (-0.2129), `ballshootatthebasket3times` (-0.1627), and
`ab_whiteboywithbluebag` (-0.1395). These are not isolated fixed-frame
effects; the full-sequence averages show sustained drift or early failure.

## Mechanism interpretation

The module is active: the training log reaches RGB/TIR residual-to-input RMS
around 3.3%/3.5% on average, and all three NLformer branches have nonzero
gradients. The 50-sequence result therefore does not indicate a dead graph.

It also does not establish target-selective correction. The mixed sign pattern
and the high-baseline degradation are consistent with the local/covariance
operator changing the selected peak without a sufficiently target-conditioned
constraint. In particular, the current template adapter is only a zero-init
template summary bias; no GT/background residual, rank, or margin signal was
recorded during this run.

## Decision

`NOT-PROMOTED / DIAGNOSTIC-ONLY`.

Do not increase training length or run full-244 yet. The next valid experiment
is a fixed-frame mechanism audit on the same 50-cohort representatives,
prioritizing the largest gains and losses. It must measure local, covariance,
and complementary branch changes at GT and strongest-background locations, GT
rank and margin before/after `tbsi_fuse_search`, and whether the first peak
change precedes state drift. A matched parent training run is required before
any performance claim.

Evidence:

- `output/experiments/M1_AMCM_NLFORMER_QUICK15_3EP_20261020/analysis_50seq/per_sequence.csv`
- `output/experiments/M1_AMCM_NLFORMER_QUICK15_3EP_20261020/analysis_50seq/summary.json`
- `output/test/tracking_results/tbsi_track/M1_AMCM_NLFORMER_QUICK15_3EP_20261020_50SEQ/`
