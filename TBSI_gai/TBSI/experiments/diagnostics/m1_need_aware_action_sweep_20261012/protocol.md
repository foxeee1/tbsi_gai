# M1 Need-Aware Action and Signal Screening (2026-10-12)

## Research question

Can deployment-visible, input-derived signals identify fixed M1 states where a small RGB/TIR input adjustment improves the true decoded CENTER box over doing nothing? This is signal screening only; no fusion module is trained.

## Hypotheses

- Null: on sequence-held-out states, candidate signals do not improve selected-action decoded IoU over no-op or fixed-direction controls, or their benefit is offset by harmful intervention on oracle no-op states.
- Alternative: a small, fixed, low-capacity signal model predicts positive RGB/TIR action utility on held-out sequences, preserves oracle no-op states, and improves sequence-macro decoded IoU over no-op and fixed-dose controls.

## Parent and fixed contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`.
- Checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Data: outcome-blind, previously unselected LasHeR training sequences with 40-149 frames; 32 sequences selected by deterministic quartile balancing over target-area, relative-motion, RGB target luminance and TIR target-ring contrast proxies. Exact list and source hashes are in `manifest.json`.
- Fixed state: same trace `tracker_state_before`, image crop, template, saved RGB/TIR search features, M1 fusion convolution, CENTER score/offset/size head, Hann window and original decoder for every action.
- Actions: no-op; reciprocal RGB-favoring and TIR-favoring input scaling at rho 0.05, 0.10 and 0.15; one deterministic random-direction rho 0.10 control. No MHSA or checkpoint changes.
- Label: for each action, `u_a = decoded_bbox_IoU(action) - decoded_bbox_IoU(no-op)`.
- Signals: crop RGB/TIR luminance and contrast; modality template-support response peak and peak margin; response entropy; cross-modal map correlation and peak distance; M1 CENTER entropy and top-1/top-2 margin; local branch energy/coherence at M1's predicted peak; current state object area and normalized inter-state motion. No GT-derived quantity is an input signal.
- Analysis: sequence leave-one-out ridge regression, fixed alpha=10 after train-fold standardization; compare selected action against no-op, each fixed action and random control. Sequence is the statistical unit; report sequence-macro paired bootstrap intervals (10,000 resamples, seed 3407). Feature ranking is descriptive and uses the same cross-validation predictions.
- Main outcomes: selected-action utility, no-op specificity when oracle best utility <=0.005, harmful intervention rate when selected utility <-0.01, mean regret to the best tested action including no-op, risk/coverage using train-fold residual RMSE as an explicitly uncalibrated uncertainty proxy, and per-scene-label results.

## Stop and claim boundaries

- No signal/operator advancement if held-out selected action does not exceed no-op and the fixed RGB/TIR controls, or if the signal chooses harmful actions on oracle no-op states at an unacceptable rate.
- This is an exploratory train-side screening result. M1's exact training-sequence exposure cannot be reconstructed from available artifacts; it is independent of recorded prior signal-diagnostic cohorts, not independent of M1 training.
- Fixed-state utility is not closed-loop AUC. Any positive result only supports a later short-rollout test, not a tracking-performance claim.
- If the cohort yields fewer than 24 usable sequences or fewer than 30 fixed states, stop and report the sample shortfall without changing the selection rule.

## Run outputs

- `manifest.json`, `sequence_list.txt`, `trace_run/`
- `feature_capture/selected_features.npz`, `feature_capture/per_arm.csv`
- `action_replay/per_state_actions.csv`, `action_replay/per_state_signals.csv`
- `analysis/summary.json`, `analysis/per_sequence.csv`, `analysis/signal_screen.csv`
- Full command, environment, source snapshot and outcome in `experiments/reports/m1_need_aware_signal_action_sweep_20261012.md` and both asset registries.
