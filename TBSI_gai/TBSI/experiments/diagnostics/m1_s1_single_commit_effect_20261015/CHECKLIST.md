- [x] Correctly scope the previous whole-trajectory comparison as non-causal for a single commit.
- [x] Freeze an independent sequence selection rule and one-frame intervention contract.
- [x] Verify protected assets, source list, resources, and output paths.
- [x] Run S1 action discovery on the fixed train-side sequences.
- [x] Replay event prefixes and execute paired one-frame state commits.
- [x] Check event-output parity, state divergence, and complete future labels.
- [x] Summarize event and sequence effects with cluster-aware uncertainty.
- [x] Update report, manifest, both asset registries, and decision status.

## Result

- Retry2 and retry3 use the same 21 offset events in 11 sequences; retry3 adds same-frame S1-vs-M1 raw IoU. Future effect values match exactly across retries.
- Immediate readout mean/median IoU delta is `+0.000662/0`, sequence-cluster 95% CI `[-0.005541,+0.007267]`; positive/negative/zero counts are `7/7/7`.
- State commit minus fallback is H10 `+0.005118` (95% CI `[-0.001200,+0.010340]`) and H30 `+0.009954` (95% CI `[-0.006332,+0.031023]`); both inconclusive.
- Decision: do not fit a gate from this cohort or repeat rejected peak-offset/top-2 direction features. Next audit fused/RGB-only/TIR-only candidate availability at the same fixed states; evaluate any selector leave-one-sequence-out, and evaluate closed-loop state policy separately.
- Full report: `experiments/reports/m1_s1_single_commit_effect_20261015.md`.
