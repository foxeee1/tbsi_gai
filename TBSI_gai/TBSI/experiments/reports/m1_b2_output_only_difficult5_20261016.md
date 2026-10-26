# B2 output-only state attribution

## Question

Does B2's full-test regression mainly come from committing the corrected box to tracker state, or is the corrected output itself harmful?

## Setup

- Same B2 epoch-3 proxy checkpoint and same five-sequence difficult set used by the previous B2 screen.
- B2 residual output is unchanged.
- A baseline CENTER head is decoded from the pre-complementary M1 feature on the same frame.
- The corrected B2 box is returned, while the pre-complementary M1 box is committed as the next tracker state.
- This is controlled by `TBSI_RETURN_B2_BASELINE=1` and `TBSI_FEATURE_STATE_MODE=output_only`; default behavior remains unchanged.

## Results

| Sequence | M1 AUC | S1 AUC | B2 coupled | B2 output-only | output-only - M1 |
|---|---:|---:|---:|---:|---:|
| bike2left | 79.62 | 79.49 | 78.65 | 78.88 | -0.74 |
| boyss | 54.10 | 46.13 | 45.93 | 45.37 | -8.72 |
| girldownstairfromlight | 46.50 | 46.47 | 47.58 | 47.42 | +0.92 |
| girlinrain | 84.40 | 84.49 | 84.37 | 84.22 | -0.18 |
| manoncall | 68.94 | 67.77 | 68.85 | 68.38 | -0.57 |
| **mean** | **66.71** | **64.87** | **65.08** | **64.85** | **-1.86** |

Complete metrics are in [`difficult5_output_only.csv`](../../output/experiments/M1_TARGET_COMPLEMENTARY_ADAPTIVE_PROXY_20261016/full244_analysis/difficult5_output_only.csv). Runid 247 contains the output-only predictions. Runid 246 was an attempted full-244 output-only run; it was stopped after 8/244 sequences because the baseline-head double decode increased runtime to several hours. Its partial outputs are retained as an incomplete run.

## Conclusion

`STATE-FEEDBACK-DOMINANT` is **not supported for B2**. Keeping M1 state does not recover the missing gain: output-only is slightly worse than coupled on this fixed set and remains below M1. The B2 residual/gate output contract is therefore the current bottleneck. Stop adding state commit logic or another fallback layer to B2. Any next fusion experiment must first demonstrate a positive fixed-state output effect on the same difficult set; otherwise the post-fusion B2 route should be abandoned.
