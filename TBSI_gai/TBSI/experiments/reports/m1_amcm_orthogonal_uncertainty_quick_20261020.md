# M1 AMCM Orthogonal-Uncertainty Quick Closed Loop

## Scope

This run changed only the AMCM-internal mode to `orthogonal_uncertainty` and
injected its two corrected streams at the existing pre-fusion AMCM boundary.
Backbone, TBSI, `tbsi_fuse_search`, CENTER head, tracker state, and external
losses were unchanged. The run used the fixed ten-event cohort from
`M1_AMCM_TMATCH_FIXED10_DIAG_20261019` and prefix closed-loop tracking to each
event frame.

This is a proxy diagnostic, not a full-244 result. The candidate was warm
started from the protected M1 epoch-15 checkpoint and trained for 3 epochs with
512 samples/epoch, batch 2, FP32 and `NUM_WORKER=4`; therefore it is not a
from-scratch or matched full-training comparison.

## Module equation

For search features $x_r,x_t$, the implementation forms

\[
s=(x_r+x_t)/2,\quad d_m=x_m-s-(\langle x_m-s,\hat{s}\rangle\hat{s}),
\]

where $d_m$ is the component orthogonal to the shared feature. A five-channel
router input is `[mean(|xr-xt|), cos(xr,xt), ||xr||, ||xt||,
| ||xr||-||xt|| |]`. The router predicts a two-way softmax $p_r,p_t$ and two
bounded residuals from `[s,d_r,d_t]`:

\[
x'_r=x_r+p_t\,0.1\tanh(f_r([s,d_r,d_t])),\qquad
x'_t=x_t+p_r\,0.1\tanh(f_t([s,d_r,d_t])).
\]

The residual projections and router output are zero initialized, preserving
identity at initialization. However, this mode does not consume template
tokens or target matching scores; its reliability is search-only.

## Event-level result

| Sequence | Frame | M1 IoU | Candidate IoU | Delta |
|---|---:|---:|---:|---:|
| 7rightorangegirl | 343 | 0.2559 | 0.0000 | -0.2559 |
| ab_bolstershaking | 2952 | 0.3501 | 0.3477 | -0.0023 |
| blkboydown | 49 | 0.2895 | 0.2860 | -0.0035 |
| boyss | 218 | 0.3421 | 0.3194 | -0.0227 |
| foamatgirl`srighthand | 67 | 0.3215 | 0.0000 | -0.3215 |
| girldownstairfromlight | 708 | 0.2927 | 0.3020 | +0.0093 |
| leftopenexersicebook | 120 | 0.3029 | 0.3020 | -0.0009 |
| rightbluewhite | 236 | 0.2403 | 0.2310 | -0.0093 |
| rightdarksingleman | 883 | 0.3827 | 0.4500 | +0.0674 |
| umbrella | 197 | 0.0000 | 0.0000 | 0.0000 |

Mean delta is `-0.05395` (median `-0.00290`), with 2 wins, 7 losses and 1
tie. The two large losses are complete event failures; the gains are isolated
and do not establish a repeatable correction.

## Mechanism evidence

- Logged residual norm / input norm averaged about `0.00357` (RGB) and
  `0.00049` (TIR). The residual is therefore close to an amplitude-level
  perturbation relative to the incoming features.
- The route is initialized at a spatially uniform 0.5/0.5 split. No template
  token appears in the orthogonal router input, so it cannot know whether a
  disagreement belongs to the target or to a background peak.
- The training log shows nonzero AMCM gradients after warm-up, so this is not a
  dead graph. It does not prove the gradient points toward target/background
  separation; the event result instead shows unstable direction across the
  fixed cohort.
- The implementation has no per-event target/background residual map or GT
  rank logging for this mode. Consequently the evidence chain stops before
  proving a target-selective spatial correction.

## Decision

`NOT-SUPPORTED` under the current contract. The failure is primarily a
representation/conditioning mismatch, not an argument against orthogonal
decomposition itself: the module estimates uncertainty from search statistics,
does not use the available template evidence, and applies a very small bounded
additive residual. It should not be promoted or scaled to full data. A future
single-variable AMCM-only revision would need to add target-conditioned
matching to the router and log target/background residual, rank and margin
before any further training.

Evidence: `output/experiments/M1_AMCM_ORTHOGONAL_UNCERTAINTY_QUICK15_3EP_20261020/`.
