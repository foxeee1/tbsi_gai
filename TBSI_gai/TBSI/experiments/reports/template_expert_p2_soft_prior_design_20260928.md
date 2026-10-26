# Template Expert P2: Soft Target Prior Design

Date: 2026-09-28
Parent result: `TEMPLATE-EXPERT-{C,G,T}-P1-20260928`
Parent evidence: `experiments/reports/template_expert_p1_rpp_v1_serial_20260928.md`

## Decision

Do not send the current target-weighted HMoE to full-244 and do not try to rescue it with more epochs, more experts, extra losses, entmax, gates, or template updating. The next bounded question is whether T failed because its target prior is too hard, specifically because it suppresses token routing and token write-back at the same time.

## Code finding

The P0 audit found an additional geometry issue before training: the existing
box-to-token helper returned intersection area as a fraction of the whole
template image, rather than coverage of each token patch. For a 50% x 50%
box, the mean value was about 0.0039 instead of 0.25, and a fully covered
patch could never reach weight 1. P2 therefore includes an explicit mask
normalization fix (`intersection_area / patch_area`) before the prior
comparison. P1 metrics remain unchanged and are not retroactively relabeled.

The current `MaskedTemplateHMoE.forward` does both:

1. Dispatch: `dispatch_logits += log(token_weight)` after repeating the template weights over slots.
2. Write-back: `return moe_out * token_weight.unsqueeze(-1)`.

Thus a background token can be downweighted before expert aggregation and then have its expert residual set to zero on output. This is stronger than target prioritization, but the P1 result alone does not establish that it caused the metric loss. P2 tests this possibility after correcting the token-weight geometry. The T checkpoint remains valid evidence for the original implementation, not for the corrected one.

## Stage A: no-training mechanism audit

Use the existing T checkpoint, the same fixed RPP-v1 manifest, and a small fixed sample of sequences or the first template forward from all 60 sequences. Instrument `MaskedTemplateHMoE` without changing outputs.

Record per RGB/TIR template:

- `fg_ratio = mean(token_weight)`; mean, median, min, max, fraction below 0.05, fraction above 0.80.
- `gamma`, `moe_out_norm`, `base_norm`.
- `expert_ratio_pre = ||gamma * moe_out|| / (||x|| + 1e-6)`.
- `expert_ratio_post = ||gamma * moe_out * token_weight|| / (||x|| + 1e-6)`.
- pre-mask expert output norm on foreground and background tokens; post-mask norms on both groups.
- dispatch entropy and foreground/background dispatch mass.

The instrumentation must be detached, sampled at a fixed cadence, and must not alter the forward path. Report the two residual ratios separately. A very small `expert_ratio_pre` means the metric result cannot be attributed primarily to hard write-back; a large gap between pre and post ratios is direct evidence of output suppression.

Acceptance: finite statistics, no output change relative to the uninstrumented T checkpoint, and a saved JSON/CSV artifact. This stage is diagnostic only and does not support an accuracy claim.

## Stage B: minimal proxy training frontier

Keep fixed: M1 parent epoch-15 checkpoint, seed 42, 3 training epochs, optimizer, batch, accumulation, AMP, data order, location 9, 4 experts, 4 slots, hidden size 4, corrected token-mask geometry, RPP-v1 train/test manifests, FP32 serial testing, and all backbone/head settings.

Run only these two new candidates, with C and the completed T result as existing references:

| ID | Dispatch prior | Output write-back | Purpose |
|---|---|---|---|
| `T-DISPONLY-P2_seed42` | `m_i` via `log(m_i)` | no mask multiplication | Isolate whether write-back suppression caused T's loss |
| `T-SOFT-P2_seed42` | `w_i = 1 + 0.5 m_i` via `log(w_i)` | no mask multiplication | Test target priority while retaining context |

Do not sweep lambda. `lambda=0.5` is predeclared. Do not initialize the new candidates from T; initialize both from the same M1 epoch-15 checkpoint so the comparison does not inherit T's hard-routing parameters. Each new module still starts with zero residual `gamma` and is trained under the same settings as P1.

The minimal result table must include C, G, T, T-DISPONLY and T-SOFT. Report AUC, OP50, OP75, Precision, Norm Precision, mean sequence FPS, parameter count, and per-sequence deltas versus C. RPP-v1 remains screening evidence, not full benchmark evidence.

## Predeclared routing rules

- If Stage A shows large pre/post residual suppression and T-DISPONLY recovers most of T's lost AUC without adding a new loss, the primary failure is hard output write-back. Continue only with T-SOFT as the method candidate.
- If T-SOFT is within 0.2 AUC points of or above C and does not lose more than 5% mean FPS relative to C, promote it to the required multi-seed proxy check. This is a screening gate, not a paper claim.
- If T-DISPONLY remains clearly below C and Stage A shows a non-negligible residual, target dispatch itself is implicated; stop this mask-based route rather than adding capacity.
- If both candidates are within noise of C or worse, stop the target-prior HMoE branch. Retain G/T as negative or null evidence and return to a structurally different fusion idea.
- No full-244, no extra seeds, and no further architecture changes are authorized by this P2 design until one candidate passes the proxy gate.

## Resource and comparability boundary

Stage A is a low-cost diagnostic. Stage B requires two additional 3-epoch trainings and two 60-sequence serial tests; it is expected to be expensive on the current single RTX4090D. The RPP-v1 frozen manifests and output roots must remain unchanged. Canonical baseline YAML and canonical output directories remain read-only.

The existing P1 results remain:

- C: AUC 52.02, mean FPS 72.45.
- G: AUC 52.01, mean FPS 69.95.
- T hard: AUC 50.96, mean FPS 69.55.

All P2 outcomes, including failed or inconclusive slices, must be added to both experiment asset tables before any route decision.

## Current route

`launch_analysis_campaign` for Stage A, then `launch_experiment` for Stage B only if the diagnostic is numerically valid. The current route is not full confirmation and does not yet support a claim that soft target conditioning improves tracking.
# P2 proxy results (2026-09-28)

The two candidates were trained for 3 epochs from the same TBSI-E MHSA
epoch-15 parent checkpoint, seed 42, and evaluated serially on the frozen
RPP-v1 60-sequence test list with FP32 inference. The canonical baseline YAML
was unchanged (SHA256
`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`).

| Candidate | AUC | OP50 | OP75 | Precision | Norm Precision | Mean FPS | Result |
|---|---:|---:|---:|---:|---:|---|
| P1-C parent control | 52.02 | 63.60 | 40.72 | 66.82 | 61.32 | NA | reference |
| T-DISPONLY-P2 | 52.06 | 63.23 | 41.66 | 65.93 | 61.02 | 67.98 | +0.04 AUC vs C |
| T-SOFT-P2 | 51.17 | 61.91 | 40.60 | 64.80 | 59.90 | 64.92 | -0.85 AUC vs C |

Both proxy evaluations completed 60/60 sequences with 120 text outputs per
run. Final checkpoint SHA256 values are
`609d6eeb0a58e1c9012a7a7f6f690d81204be93b27ad1c13b0d0918f610abd70`
(DISPONLY) and
`a91e5913af1b4f39c4345c5869decc744217d72e522c5f2cca6e26080920e000`
(SOFT). The trained scalar residuals were `0.001181156956590712` and
`0.0007471884600818157`, respectively.

The P2 decision is **stop/封存**: hard dispatch does not provide a meaningful
proxy improvement over the matched P1-C control, while the soft prior
degrades all reported tracking metrics. Do not spend a second seed or full
244-sequence confirmation on either candidate. The first DISPONLY test
attempt is retained as an infrastructure failure because the evaluator used
its legacy checkpoint path; the retry used an explicit checkpoint override and
is the only valid metric run.
