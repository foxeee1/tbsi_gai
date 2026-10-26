# M1 MSHA query-role fixed-state preflight (2026-10-22)

## Decision

**MECHANISM DIAGNOSTIC COMPLETE / SPATIAL QUERY EFFECT SUPPORTED / TARGET-RANKING BENEFIT INCONCLUSIVE / NO TRAINING.** The first capture failure was traced to the runner's post-event M1 state-advance forward overwriting the last `SHIFTED_QUERY` arm's cached features. After detaching collectors before that forward, the preregistered 28-event replay passed the branch, tensor, state/crop, spatial-selectivity, and CENTER-input capture contracts. The structural query-position effect is established on this exposed calibration cohort, but its target-ranking and localization benefit is not established.

## Parent and cohort

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- YAML SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Cohort: existing `M1_CENTER_READOUT_FRESH_DIAG_20261010` train-side cohort, 24 sequences selected by fixed hash rank with previously used sequence lists excluded. The event set contains 14 matched-good and 14 severe-onset frames; the two smoke events are from `blackbag`.
- The train-side 12-sequence early-gate replay cohort was excluded from this cohort. No low-AUC test sequence was used to select the train-side events.

## Intervention and preflight

- Conditions: original M1; foreground-query cross-message restriction; and a spatially shifted soft foreground mask used as a non-target query control. The shifted mask preserves the soft mask weights and total mass while moving its support off the target.
- The same tracker snapshot, crop, frame, RNG state and M1 checkpoint were restored before each arm. M1 and foreground arms reproduced the expected fixed-state path within the smoke harness.
- A paired read-only hook was added after MSHA message projection and residual merge to capture token features. Py-compile passed.
- Smoke on two `blackbag` frames: matched-good IoU M1/foreground/shifted = `0.599/0.590/0.572`; severe-onset = `0.316/0.324/0.298`. Foreground-query feature deltas were nonzero, but shifted-query deltas were exactly zero at all three TBSI blocks while the CENTER box changed. Therefore the output hook or branch execution does not currently establish the intended spatial-control effect.

## Interpretation and next allowed action

The original smoke failure was a collector-lifetime bug, not a non-executing shifted branch: the runner's extra M1 forward after an event reused the active `SHIFTED_QUERY` capture dictionary, replacing its feature outputs. The corrected runner disables the active hooks/collector before advancing the tracker. The frozen 28-event replay then showed nonzero projection/fusion/CENTER-input deltas for both interventions on all 28 events. At the final fusion return, the response direction was spatially specific in all 28 events: foreground restriction changed background queries more than foreground queries, while the shifted restriction changed original foreground queries more than background queries.

The target-utility result is weak. On 14 severe-onset events, foreground query restriction improved either GT rank or hard-background margin over parent in 8/14, versus 7/14 for the shifted control. The more direct paired foreground-minus-shifted margin difference was `+0.00082`, with event bootstrap 95% CI `[-0.01234,+0.01324]`; GT rank was 1 win, 13 ties, 0 losses. Foreground-query IoU averaged `-0.00314` vs M1 across all 28 events and did not beat shifted-query IoU on severe events (mean paired delta `-0.00788`). Thus query location causally changes the representation and CENTER input, but target-specific ranking/localization benefit remains unsupported. This is not a tracking-performance result.

Decision: retain P3 as a measurable structural mechanism, but do not train or promote it as a target-selection improvement. Any next step must use an independent cohort and a preregistered target-utility effect threshold; do not revisit the exposed 28-event cohort for tuning.

## Artifacts

- Runner: `tools/run_m1_msha_query_role_fixed_state_20261022.py`, SHA256 `6ed4dd49517064951c4d105f4757d09e1377eaacfb29b680cab04bf95e4eae54`.
- Current `tbsi_layer.py`, SHA256 `46f7fdb0e226e914dbc75c94075b07c9ee01584c5b09693a82072952f70b9c5e`.
- Pre-change library snapshot: `output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261022/provenance/prechange_lib_snapshot.tar.gz`, SHA256 `1922ead875fe06c64b09cbc6656b1b78160b9fa635d8fa8eae2281b882c0763d`.
- Smoke outputs retained under `output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261022_smoke{,_2,_3}/`; the first smoke exposed that the legacy analysis collector ran before the foreground operator. Smoke 3 uses the new post-operator hook but still fails for the shifted control.
- Hardware: 1x NVIDIA GeForce RTX 4090 D; FP32; one CPU thread. No training or benchmark run.

## 2026-10-23 retry

- The runner first failed before model execution because the referenced sequence-list file had been removed during legacy asset cleanup. The list was reconstructed from the retained M1 trace manifest and its SHA256 exactly matched the registered frozen value `e1a976257d71969686c1dc0690596465d5e14d24366ecef567c58f8b129e65f6`; no cohort member or order changed.
- `smoke4` is retained as a failed startup attempt. `smoke5` completed both `blackbag` events. Runtime branch stats confirmed `foreground_queries` and `shifted_queries` at all three TBSI blocks. Shifted-mask overlap with the original mask was `0.0` for RGB and TIR at every block.
- Despite branch execution, `query_feature_deltas.csv` again reports exactly zero shifted-query delta at every block and event. CENTER fixed-state IoU changed from `0.5992` M1 to `0.5717` shifted on matched-good, and `0.3162` to `0.2978` on severe-onset. The 2-event sample does not support an efficacy conclusion; the mismatch means the feature-to-CENTER attribution contract is still unresolved.
- Retry2 (`smoke7`) and retry3 (`smoke8`) fixed the collector-lifetime issue and the role-mask key mismatch in the new boundary summary. `smoke8` showed nonzero projection/fusion/CENTER-input deltas in both interventions and correct role-mask directionality on both smoke events.
- The first retry's startup failure output is `output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261023_smoke4/`; completed smoke artifacts are `output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261023_smoke5/`.
- Full frozen 28-event results: `output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261023_FULL28/{center_results.csv,query_feature_deltas.csv,query_boundary_deltas.csv,branch_stats.csv,run_manifest.json}`; aggregate: `output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261023_ANALYSIS/{summary.json,severe_event_deltas.csv}`.
- Corrected runner SHA256 `a1e2cb82512776abc448f10e43982ddd0f9398731218b6041b275bfe3fd5f1ed`; analysis script SHA256 recorded with the asset. Pre-change snapshot SHA256 `a9fce480d24c0ad017be8447b19fb48634f0f061abd7825216295fbb39232bb6`; instrumentation snapshots `instrumentation_v1.tar.gz` SHA256 `e2a77bf2fefc35c430b20c99564522d84daef6dd274f573036217c3908c5d48a` and `instrumentation_v2.tar.gz` SHA256 `2d06adcb67de697986e7022ad2bdd0c0d36ab92bce78d691b6fcaa91c51c5b9e`.
- The sequence list was restored from retained trace provenance with its registered SHA256 unchanged. Protected M1 YAML/checkpoint hashes were verified after the run. No training or checkpoint writes occurred.
