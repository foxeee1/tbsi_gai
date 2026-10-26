# S1 state reachability diagnostic

## Question

Can deployment-visible state displacement and action confidence predict future harm from committing S1 state?

## Protocol

- Reused the existing train-side paired M1-state/S1-state rollout, 12 sequence folds and 3675 paired frames.
- Labels: future H10/H30 IoU difference; harmful means the S1-state rollout is worse than the M1-state rollout.
- Features: center/size/joint action scores, signed center and log-size state gaps, and absolute gap magnitudes.
- Fitted a regularized logistic model with sequence leave-one-out evaluation. No test thresholds or tracker behavior were changed.

## Results

| Future horizon | Sequence-LOSO macro AUROC | Folds |
|---|---:|---:|
| H10 | 0.528 | 12 |
| H30 | 0.405 | 12 |

The short horizon result is close to random. The long horizon result is below random and unstable across held-out sequences.

## Conclusion

`DIAGNOSTIC-COMPLETE / REACHABILITY-SIGNAL-NOT-SUPPORTED`. The current state gaps and action confidence do not provide a usable state-reachability gate. This does not prove that all reachability features are useless because the sample contains only 12 train-side sequences, but it does rule out immediately wiring this feature set into S1 inference.

The remaining plausible route is train-side state-aware learning with explicit multi-step crop supervision, or a direct center/candidate correction target. Another hand-designed admission threshold is not justified by this evidence.

Evidence: `output/experiments/M1_S1_STATE_REACHABILITY_DIAGNOSTIC_20261016/metrics.json` and the source paired rollout `output/experiments/M1_S1_CLOSED_LOOP_RISK_DIAGNOSTIC_20261015/paired_rollout.csv`.
