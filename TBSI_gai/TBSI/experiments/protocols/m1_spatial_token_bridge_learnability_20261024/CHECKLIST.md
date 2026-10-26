# Candidate 2 Learnability Checklist

## Done

- [x] Recover existing Candidate 2 assets and its no-go full-training result.
- [x] Confirm the new diagnostic changes only bridge-only training duration.
- [x] Validate config, protected M1 hashes, source bundle parity, and resource availability.
- [x] Write preregistration and training contract.

## Done

- [x] Run bridge-only 10-epoch training; retain logs and epoch 1/5/10 checkpoints.
- [x] Run explicit off/target/shifted fixed-state comparisons at epochs 1, 5, 10.
- [x] Compare residual norms, target/shifted margins, CENTER/query rank, and replay IoU by event and sequence.
- [x] Update report and both asset registries; classify the current interface as weakly coupled under this diagnostic.

## Stop Conditions

- [x] Bridge gradients and parameter updates were present; no non-finite training signal was observed.
- [x] Do not launch a full run: the preregistered mechanism gate did not pass.
