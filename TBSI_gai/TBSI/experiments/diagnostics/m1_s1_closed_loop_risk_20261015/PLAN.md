# M1/S1 Closed-Loop Risk Signal Diagnostic

## Selected idea

Advance frozen S1-state and M1-state trackers over the same train-side sequences and frames. Test whether state divergence observable before a frame predicts the next fixed-horizon localization harm of S1-state relative to M1-state.

## Run contract

- Parent: protected M1 epoch 15; frozen S1 predictor and independent action policy.
- Research question: does current state divergence rank future S1-vs-M1 localization harm in closed-loop rollouts?
- Null: state divergence has no useful sequence-held-out association with future harm.
- One changed variable: internal state feedback path (S1 coupled state vs M1 state); weights, frames, initialization, precision and readout policy are fixed.
- Data: deterministic first 12 sequences from the existing 72-sequence LasHeR trainingset manifest; whole sequences are the sampling and holdout unit. No test split and no threshold fitting.
- Labels: for each frame, mean IoU(M1-state) minus mean IoU(S1-state) over the next 10 and 30 tracked frames. Positive values mean S1-state is worse.
- Frozen observable scores: normalized center/state displacement, log-size displacement, and their combined Euclidean score, measured from the two branch states before processing the current frame.
- Evaluation: per-frame Spearman correlation and binary AUROC for future harm (delta > 0), sequence-held-out rank summaries, and top-quartile harm enrichment. Results are exploratory and train-side only.
- Stop: stop before rollout if protected hashes, manifests, parent artifact hashes, data coverage or resource checks fail; stop if tracker contracts cannot be initialized in FP32.

## Code map

- New standalone runner under `tools/`; no model/tracker code changes.
- Save immutable 12-sequence manifest, per-frame paired rollout CSV, metrics JSON, runtime log, and environment/provenance manifest under a new output directory.

## Evidence target

- Minimum: complete paired rollouts, finite labels, exact branch initialization and complete sequence coverage.
- Solid: sequence-held-out association and top-quartile enrichment reported with per-sequence values.
- Maximum: out of scope; no predictor fit or full cohort extrapolation in this pass.

## Decision rule

If held-out association is weak or inconsistent across sequences, do not train a selective state gate; investigate state restoration/latency as a direct mechanism. If association is positive and replicated across held-out sequences, expand to the full frozen train-side sequence manifest and fit a risk model there before any test evaluation.

## Outcome

Completed on 12 whole train-side sequences (3,675 paired frames). The joint state-divergence score had AUROC 0.501/0.481 and Spearman -0.256/-0.258 for 10/30-frame harm labels; high-score quartiles were less harmful than average and per-sequence directions were mixed. Classify as `not supported` for this score. Do not expand to predictor fitting or full test; next test a fixed 1-frame delayed state commit as a direct mechanism intervention.
