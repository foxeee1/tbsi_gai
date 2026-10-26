# M1 AMCM-lite reliability proxy screen (2026-10-17)

## Scope

This is an engineering adaptation named AMCM-lite, not a reproduction of an official AMCM implementation. It tests one hypothesis: target-conditioned, spatial RGB/TIR reliability modulation before the existing `tbsi_fuse_search` can improve difficult M1 sequences while leaving the CENTER head, decoder, tracker state, crop, and template update unchanged.

## Contract

- Input: RGB/TIR search feature maps (`B,C,16,16`) and RGB/TIR template tokens.
- Context: depthwise 3x3, 5x5, and 7x7 search context branches.
- Quality signals: RGB/TIR context, absolute cross-modal difference, and template similarity maps.
- Output: bounded spatial residuals, `rgb * (1 + delta_rgb)` and `tir * (1 + delta_tir)`, with `|delta| <= 0.10`.
- Placement: immediately before the original `tbsi_fuse_search` projection in `forward_fusion_only`.
- Training: M1 epoch15 checkpoint initialization; only `amcm_reliability.*` parameters train; frozen parent remains in eval mode so BatchNorm buffers do not drift.
- Inference: no persistent AMCM state; tracker state and template update are unchanged.

## Minimal training

Config: `experiments/tbsi_track/M1_AMCM_LITE_RELIABILITY_PROXY_20261017.yaml`.

LasHeR train, 256 samples/epoch, 3 epochs, batch size 2, FP32, one DataLoader worker, 4090D. Validation IoU was 0.68938, 0.71309, and 0.72380 at epochs 1--3. The final checkpoint SHA256 is `a5786f1f683d9cd7c2617b4e6b7cb9b8462d21dc1cff827e95f7caec26784c40`.

The first run was retained under `output/experiments/M1_AMCM_LITE_RELIABILITY_PROXY_20261017_BNDRIFT/` as a contract failure because frozen BatchNorm buffers changed. It is not used for the method conclusion.

## Difficult5 result

Fixed manifest: `m1_target_conditioned_center_proxy_difficult5_20261016.txt`, FP32, serial tracker, 5/5 sequences. The M1 comparison uses the protected epoch15 checkpoint and the same evaluator.

| Method | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 66.71 | 78.00 | 54.34 | 89.36 | 78.00 |
| AMCM-lite | 64.70 | 75.52 | 53.75 | 85.15 | 76.79 |
| Delta | -2.01 | -2.48 | -0.59 | -4.21 | -1.21 |

Per-sequence AUC deltas (AMCM-lite minus M1): `manoncall +0.50`, `boyss -9.38`, `girldownstairfromlight -1.31`, `bike2left +0.02`, `girlinrain +0.09`.

## Decision

`SCREEN-COMPLETE / NOT-SUPPORTED`. Under this short training contract, AMCM-lite does not improve the difficult5 parent. The result does not establish that all reliability estimation is invalid, but it rejects this spatial residual design and its current supervision as a promotion candidate. The concentrated `boyss` failure is consistent with the already observed crop/center reachability and closed-loop state bottleneck; it is not evidence that the CENTER head or `tbsi_fuse_search` alone is the root cause.

## Clean attribution audit

The real TBSITrack graph was audited with fixed synthetic token inputs and the actual CENTER head path. The audit artifact is `output/experiments/M1_AMCM_LITE_RELIABILITY_PROXY_20261017/diagnostics/amcm_contract_20261017.json`.

| Check | Observation | Attribution |
|---|---:|---|
| Zero-gate identity | max absolute difference `9.54e-7` | Forward contract passes |
| CENTER task gradient | AMCM total gradient norm `0.5531`; all 8 parameter groups nonzero | Information and gradient flow pass |
| One-step update | AMCM update norm `0.0851` | Optimizer reaches the module |
| Frozen parent | max drift `0` | No BN/state contamination in the corrected run |
| Fixed-batch fit | loss `8.8002 -> 7.7107` in 20 steps | The CENTER objective can drive the adapter |
| Learned feature effect | mean absolute fused-feature change `0.00181`, max `0.02134` | The trained adapter is active but its residual is small |

This separates the claims: implementation validity, learning/update validity, and tracking improvement. The first two are supported; the third is not supported on difficult5. The remaining attribution is therefore `L4 coordination/distribution or closed-loop mismatch`, with `L2 information bottleneck` still possible. It is not evidence for `L1 learning failure`.

## Fixed-state signal attribution

To test whether the learned gate contains usable frame-level reliability information, M1 and AMCM-lite were replayed on the same 123 training events from the existing candidate-recall audit. Each arm used the identical `tracker_state_before`, crop, template, and frame. No candidate or state from either arm was fed back. The AMCM gate maps were logged without changing the prediction.

| Quantity | Result |
|---|---:|
| Events / sequences | 123 / 60 |
| M1 fixed-state mean IoU | 0.622826 |
| AMCM fixed-state mean IoU | 0.622842 |
| Mean IoU delta | +0.000015 |
| Positive / negative events | 66 / 50 |
| Pearson(delta IoU, RGB gate magnitude) | -0.073 |
| Pearson(delta IoU, TIR gate magnitude) | -0.075 |
| Pearson(delta IoU, gate disagreement) | -0.074 |

Gate statistics were nearly constant across events: RGB signed mean `-0.001736` (std `0.000200`), TIR signed mean `+0.001209` (std `0.000131`), and RGB/TIR disagreement mean `0.002945` (std `0.000331`). The severe-onset subgroup had a small descriptive delta (`+0.000037`) but no usable gate-to-benefit relationship.

Attribution: the module passes implementation and optimization contracts, but its learned output has collapsed to a weak, nearly global modality bias. It is not functioning as a target-conditioned per-frame reliability estimator under the current CENTER supervision. This is an `L2/L4` failure (signal identifiability and objective/decision mismatch), not an `L1` gradient-flow failure. AMCM-lite is stopped; no full-244 promotion or architecture expansion is justified.

Do not run full-244 or expand AMCM-lite. The next research direction should use the already observed CENTER candidate headroom and test a directly supervised candidate utility signal under fixed state, with a no-op fallback; it should not add another unsupervised reliability gate.

Evidence: final checkpoint and logs under `output/experiments/M1_AMCM_LITE_RELIABILITY_PROXY_20261017/`; tracking results under `output/test/tracking_results/tbsi_track/M1_AMCM_LITE_RELIABILITY_PROXY_20261017/`.
