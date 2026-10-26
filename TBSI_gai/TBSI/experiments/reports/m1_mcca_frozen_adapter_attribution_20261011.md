# M1 MCCA Frozen-Adapter Negative-Gain Attribution (2026-10-11)

## Decision summary

The frozen-backbone experiment rejects the exact contract "standard tracking loss can safely train the existing MCCA message adapters" on the registered 96-sequence validation cohort. It does **not** show a connectivity failure and does not prove that every supervised MCCA design is invalid.

The most defensible attribution is **L4: coordination/objective and interface mismatch**, with a secondary **L3: position mismatch**. There is a possible L1 optimization-efficiency issue (gradient clipping and short training), but it is not sufficient to explain the result and must not be used as the primary explanation without new measurements.

## Evidence already established

Parent is protected M1 epoch 15; candidate trains only six bidirectional MCCA adapters for three epochs. All other weights and buffers remain bitwise unchanged.

| Check | Observation | Meaning |
|---|---:|---|
| Zero-output parity | max difference `0.0` | the adapter has a valid identity initialization |
| Adapter gradients | all six output projections finite/nonzero | the path is connected |
| Zero-gradient parameters | `0/24` at sampled optimizer steps | no persistent dead adapter path |
| Frozen M1 changes | `0` parameters and buffers | no accidental backbone/head drift |
| Validation AUC delta | `-1.0300` | frozen adapter contract did not improve closed loop |
| Validation PR/NPR delta | `-1.4211/-1.2296` | degradation is not AUC-only |
| Sequence deltas | 56 wins / 40 losses, but large losses dominate | small changes are mostly harmless; a few failures are catastrophic |
| Largest losses | `man2startmoto -50.71`, `boyleave -48.17`, `firstleftrunning -14.04` | closed-loop amplification is plausible |
| Largest gains | `redbackpackgirl +18.75`, `blkbikefromnorth +12.49`, `mirrorfront +10.10` | the adapter is scene-sensitive, not inert |

The paired bootstrap intervals cross zero, so the point estimate is exploratory. The single-pass FPS observation (`73.25 -> 66.84`) is a cost signal, not a repeated latency conclusion.

## Why nonzero gradients do not imply the right learning signal

The training actor optimizes

\[
 L = 2L_{GIoU} + 5L_{L1} + L_{focal}.
\]

The CENTER head computes the selected cell as

\[
 k = \arg\max_{u,v} S(u,v),\qquad
 b = [u_k+\Delta x_k, v_k+\Delta y_k, w_k,h_k].
\]

The `argmax`/`torch.max(...).indices` decision is discrete. Consequently:

1. `L_focal` can shape the **raw** score map toward the Gaussian training target.
2. `L_{GIoU}` and `L_{L1}` can update offset/size at the currently selected cell and can backpropagate through the feature path, but they do not provide a differentiable gradient saying "select neighboring cell j instead of k".
3. Online inference applies a Hann window before selecting the cell. That window and the closed-loop crop/template update are absent from the training objective.

Thus the adapter can receive healthy gradients while still learning a correction that does not improve the actual online decision. This is a concrete objective mismatch, not a generic claim that an auxiliary loss is always needed.

## Why the MCCA interface is vulnerable

At each template MHSA site the adapter receives

\[
 z = [q,\;m,\;|q-m|], \qquad
 \Delta m = W_2\,\mathrm{GELU}(W_1 z),
\]

and adds `Delta m` to the projected cross-modal message before the residual stream continues. The adapter does not observe:

- search-time target-vs-hard-negative score margins;
- the Hann-weighted decision;
- RGB/TIR quality or spatial misregistration;
- motion/crop uncertainty;
- whether the current template state is already contaminated.

It therefore learns a global template-message correction from local sampled crops. The same correction can help one scene and alter a fragile early peak in another. Since tracking is closed loop, an early one-cell change can modify the next crop and template, producing a much larger sequence-level loss than its feature perturbation would suggest.

This explains the observed shape (many small wins/losses and a few very large regressions) better than a simple "module has no effect" explanation.

## L3 position mismatch

The diagnosed failure is primarily expressed at search-time:

`search representation -> raw score map -> Hann-weighted peak -> offset/size readout -> closed-loop state`.

MCCA changes an upstream template message. It can influence search features only indirectly through later transformer blocks, fusion, and the head. This is a long causal distance from the decision that fails. A template correction may improve identity information while simultaneously perturbing local spatial evidence or the regression branches. The current experiment did not measure enough intermediate signals to decide whether relocation to the search/fusion interface is required, but the location is a credible secondary cause.

## L1 optimization concerns: real but unproven

The adapter uses a zero-initialized output projection. This preserves identity, but at the first update the internal bottleneck receives little or no useful gradient until the output projection moves. That is a short-training efficiency cost, not a permanent gradient break; later sampled steps show finite gradients for all 24 tensors.

The training contract also clips gradients at `0.1`. The diagnostic log reports total norms at exactly `0.099999` in later samples, which is consistent with clipping saturation because logging occurs after clipping. It is not possible to infer the unclipped norm from that file. A future run must log pre-clip and post-clip norms before changing LR or initialization.

These issues may limit a three-epoch adapter-only run, but they cannot explain away the negative validation result without a matched longer/alternative optimization experiment. They are not permission to tune on the current 96-sequence cohort.

## What this result does and does not prove

It proves:

- the adapter is connected, updates, and preserves the frozen M1 state;
- the unchanged tracking loss can train it, but does not yield a reliable held-out closed-loop gain;
- the current interface/objective combination is unsafe because its scene-dependent corrections are not tied to the online peak decision.

It does not prove:

- that MCCA cannot help with a search-aware or reliability-aware objective;
- that all negative sequences are caused by one common visual attribute;
- that a different insertion point or training schedule would improve M1;
- that the one-pass FPS difference is a stable deployment number.

## Minimal next diagnostic before adding supervision

Do one read-only paired trace on the already registered positive and negative sequences, without training or changing checkpoints. For the first divergence frame and a fixed number of subsequent frames, record:

1. each layer/direction `||Delta m|| / ||m||` and `||Delta m|| / ||q||`;
2. target-region and hardest-negative-region residual norms;
3. raw score peak, Hann peak, GT cell, and target-hard-negative margin;
4. whether the selected cell is in the top-8/top-16 GT recall set;
5. offset and size errors after forcing the same selected cell;
6. the first frame at which M1 and MCCA crops/templates diverge.

Interpretation:

- Large residual with a margin moving in the wrong direction on negative sequences: coordination/reliability failure; add a quality-gated or search-aware correction only in a new contract.
- Tiny residual everywhere: optimization/scale issue; inspect pre-clip norms and training duration before architectural changes.
- Similar score maps but different offset/size: MCCA is acting on the wrong branch for this failure; do not add identity/ranking loss to the template message blindly.
- Early peak flip followed by crop divergence: closed-loop amplification; any new objective must explicitly protect hard-negative ranking and normal-frame invariance.

## Current recommendation

Do not claim "MCCA failed because it needs an auxiliary loss" yet. The supported statement is:

> The current MCCA message adapter is trainable but its standard crop-level tracking objective is not aligned with the Hann-weighted, discrete CENTER decision and closed-loop state transition. This coordination/interface mismatch produces scene-dependent corrections and rare catastrophic regressions, so the adapter-only contract is not promotable.

Run the read-only trace above once. Only if it shows a reproducible target-vs-hard-negative failure should a new, preregistered supervised experiment be designed; otherwise relocate or stop this MCCA version.

## Read-only paired trace result

The registered trace was completed on the three largest validation regressions (`man2startmoto`, `boyleave`, `firstleftrunning`) and three largest gains (`redbackpackgirl`, `blkbikefromnorth`, `mirrorfront`). Both fixed checkpoints were run in closed loop; 8,906 frames were recorded using the tracker's integer crop rule. This is a mechanism diagnostic selected from the already completed val96 result, not a new benchmark. Final output is `output/experiments/M1_MCCA_FROZEN_ADAPTER_ATTRIBUTION_TRACE_20261011_V3/`.

| Sequence | M1 Hann target/background margin | MCCA margin | M1 Hann GT rank | MCCA rank | M1 center error | MCCA center error |
|---|---:|---:|---:|---:|---:|---:|
| `man2startmoto` | `+0.448` | `-0.278` | 5 | 172 | 0.41 | 12.01 |
| `boyleave` | `+0.806` | `+0.797` | 5 | 5 | 0.14 | 0.15 |
| `firstleftrunning` | `-0.393` | `-0.356` | 19 | 20 | 4.83 | 4.81 |
| `redbackpackgirl` | `-0.277` | `-0.299` | 131 | 123 | 8.10 | 7.71 |
| `blkbikefromnorth` | `+0.498` | `+0.497` | 2 | 2 | 0.18 | 0.17 |
| `mirrorfront` | `+0.444` | `+0.439` | 4 | 4 | 0.20 | 0.19 |

The MCCA correction is therefore demonstrably active and condition-dependent. In `man2startmoto`, the first state difference larger than 2 pixels appears at frame 20 while both models still have high IoU (`0.963` M1 and `0.971` MCCA); the later large sequence loss is consistent with closed-loop amplification after an initially small decision change. In the gain sequences, the correction is neutral or slightly helpful, while in `redbackpackgirl` both systems already have a weak target margin and MCCA only modestly reduces the error.

Across all six sequences, the median Hann margin changed from `+0.267` to `+0.118`, the median GT rank from 7 to 10, and the median Hann center error from `0.430` to `0.633` tokens. Layer/direction residual ratios were not negligible: medians ranged from `0.256` to `0.974` of the incoming message norm. This rules out “the adapter learned almost nothing” as the explanation for the observed negative gain.

The trace supports the objective/interface diagnosis: the adapter can make a large template-message change without a reliability signal that protects the target-hard-negative margin. It does not yet prove that a new auxiliary loss will solve the problem; it identifies the measurable quantity that such a loss would need to control.
