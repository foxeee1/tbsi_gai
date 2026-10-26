# M1 Modality Candidate Availability Audit

## Research question

At the same frozen M1 crop and tracker state, does either modality's candidate box localize the target better than the fused candidate, and does the union of candidates have enough headroom to justify candidate-level fusion routing?

## Hypothesis and null

- Alternative: RGB-only or TIR-only candidates recover useful target localization missing from the fused candidate; the per-state best-candidate oracle improves held-out sequence IoU.
- Null: single-modality candidates add no meaningful oracle headroom, so candidate routing is not the next useful fusion intervention.

## Run contract

- Parent: protected M1 epoch15, same 72-sequence / 144-state fixed-state cohort as the true-action and direction-feature audits.
- Sole factor: candidate feature source: cached M1 fused feature, shared M1 fusion projection with TIR input zeroed, or shared M1 fusion projection with RGB input zeroed.
- Keep M1 fusion projection, CENTER head, Hann window, decoder, crop, tracker state, GT, and sample membership fixed. No predictor/gate/state changes and no training.
- Metrics: decoded IoU by arm; paired per-state deltas to fused; sequence-macro bootstrap CI; frequency each single-modality candidate beats fused; best-of-three oracle headroom.
- Interpretation boundary: best-of-three uses GT to select a candidate and is only an upper bound. It is not a deployable performance estimate.
- Decision rule: proceed to an OOF candidate selector only if oracle best-of-three gains at least +0.010 mean IoU over fused and its sequence-cluster 95% CI lower bound is above zero. Otherwise stop candidate routing for this cohort.
- Data: LasHeR train-side fixed 144 states across 72 sequences; no official test, no closed-loop trajectory, no training.
- Resources: one tracker process, zero workers; FP32 on one RTX 4090 D.

## Minimal code map

- Add a standalone audit runner using saved modality/fused feature tensors and the frozen M1 fusion projection/head.
- Emit a manifest, aggregate metrics, per-state candidate rows, and per-sequence comparison under a new output directory.
- Do not edit model/tracker code or the protected config/checkpoint.

## Evidence target

- Minimum: exact member/key alignment, no-op fused IoU parity, finite arm metrics, and candidate recomposition parity.
- Solid: all 144 fixed states, per-sequence deltas, sequence-cluster confidence intervals, complete provenance.
- Maximum: official test of a frozen learned selector; out of scope unless the oracle audit supports it.
