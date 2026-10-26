# M1-S1 center/crop reachability audit (2026-10-16)

## Question

Can post-fusion offset/size correction solve the severe failures, or is the target center already missed by the CENTER response ranking inside the fixed crop?

## Protocol

This is a train-side fixed-state replay of protected M1 epoch15. It uses the existing 60-sequence, 123-state candidate-recall audit. For each state, every 16x16 cell is decoded with the original size and offset maps. The audit records the M1 top-1 cell, GT cell rank, top-k recall, GT-cell counterfactual IoU, and all-map oracle IoU. GT is used only for offline counterfactual measurement.

## Results

| cohort | states | top-1 IoU | GT-cell IoU | GT-cell minus top-1 | GT cell top-1 recall | GT cell top-8 recall | all-map oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 123 | 0.6228 | 0.6701 | +0.0473 | 17.1% | 80.5% | 0.7647 |
| matched-good | 60 | 0.7932 | 0.7785 | -0.0146 | 15.0% | 91.7% | 0.8491 |
| pre-onset | 31 | 0.6861 | 0.6682 | -0.0179 | 32.3% | 80.6% | 0.7647 |
| severe-onset | 32 | 0.2422 | 0.4687 | **+0.2265** | **6.25%** | **59.4%** | 0.6064 |

The severe-onset 3x3 best-cell mean IoU is 0.5745, so most of the recoverable center gain is spatially local but is not selected by the current score peak. The all-map oracle remains well below a perfect box because size/offset errors remain after center replacement.

## Sequence-level signal

The largest severe-onset center-oracle gaps are `blkgirlfromcolumn1_quezhen` (+0.782), `rightof2boys` (+0.765), `rightfirstboy-ly` (+0.672), `left4thblkboyback` (+0.602), and `ab_shorthairgirlbike` (+0.503). These are diagnostic examples, not a test-set ranking.

## Attribution

**Supports next center/crop intervention.** Severe failures are not explained by a small offset/size residual alone: the selected center is wrong and the target cell is often outside the top-k response ranking. This is evidence for a position/selection bottleneck (L3), while the residual size/offset gap means center selection will not solve every failure.

The result does **not** prove that a learned selector will achieve +22.6 points, because GT-cell replacement is an offline oracle. It also does not explain normal-sequence behavior: matched-good and pre-onset GT-cell replacement is slightly worse on average, so any future center intervention must be failure-conditioned and retain a no-op fallback.

## Next experiment

Use the same fixed crop/state and one new variable: a local center-cell reranker or top-k center candidate branch conditioned on failure evidence, with M1 top-1 as the default. Measure severe-onset recovery and matched/pre-onset non-degradation separately. Do not add a more complex offset/size predictor until this center intervention is tested.

## Evidence

- Derived output: `output/experiments/M1_S1_CENTER_CROP_REACHABILITY_20261016_retry1/summary.json`
- Per-state output: `output/experiments/M1_S1_CENTER_CROP_REACHABILITY_20261016_retry1/per_event.csv`
- Per-sequence output: `output/experiments/M1_S1_CENTER_CROP_REACHABILITY_20261016_retry1/per_sequence.csv`
- Source audit: `output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/analysis/`
- Script: `tools/analyze_m1_s1_center_crop_reachability_20261016.py`

## Limitations

Train-side fixed-state evidence only; no full-244 test, no closed-loop state update, no learned online selector, and no multi-seed conclusion.
