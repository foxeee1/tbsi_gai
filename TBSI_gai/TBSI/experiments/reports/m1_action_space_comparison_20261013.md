# M1 feature/readout action-space comparison (2026-10-13)

## Question

Under the same frozen M1 state, crop, template, fusion output, and CENTER decode, does useful fixed-state compensation exist in the local fused feature space or the readout space?

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, explicit epoch-15 checkpoint.
- Configuration SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- New outcome-blind train-side cohort: 24 short sequences, 84 selected fixed states, 7 severe-onset states; manifest SHA256 `852266b3ecb0d1145cb6049d8c661e9a450548cde7976ce1348c13c325c95780`; sequence-list SHA256 `5051b66a20ec09e832d5a22360822b65f4a0d06712795c26a95248ee5763970d`.
- Selection: `onset-3-iou-0.5`, with two probe reference frames per sequence. The cohort was excluded from registered prior M1 diagnostic sequence lists; exact M1 training exposure remains unknown.
- Same FP32 saved features, state, template, target, and M1 decode for every action. No tracker state update and no training.

## Actions

### Feature-side

For a fixed 3x3 support centered on the GT cell, replayed

\[
F'=F_0+0.05G(F_m-F_0),\quad m\in\{RGB,TIR\}.
\]

This is an oracle headroom test because the GT cell is unavailable at deployment. It tests the current modality residual contract, not every possible learned feature residual.

### Readout-side

Using the same frozen M1 output maps, replayed four oracle interventions:

- `score`: force the GT cell to win while retaining its M1 offset/size;
- `offset`: retain the M1 peak and replace offset with the GT-consistent offset;
- `size`: retain the M1 peak and replace size with the GT-consistent size;
- `joint`: force the GT cell to win and replace offset/size with GT-consistent values.

The readout interventions are deliberately oracle actions. They quantify action-space headroom and are not deployable predictions.

## Results

| Action | Mean fixed-state utility | Median sequence utility | Positive sequences | Negative sequences |
|---|---:|---:|---:|---:|
| no-op | 0.000000 | 0.000000 | 0 | 0 |
| feature RGB, GT 3x3 | -0.000092 | +0.000020 | 12 | 12 |
| feature TIR, GT 3x3 | +0.000063 | +0.000012 | 12 | 12 |
| readout score oracle | +0.005468 | +0.000745 | 12 | 11 |
| readout offset oracle | +0.007025 | 0.000000 | 7 | 0 |
| readout size oracle | +0.004169 | 0.000000 | 8 | 0 |
| readout joint oracle | +0.195826 | +0.170166 | 24 | 0 |

## Decision

**READOUT ACTION SPACE SUPPORTED; CURRENT FEATURE RESIDUAL CONTRACT NOT SUPPORTED.**

The tested feature-side local residual has no useful aggregate headroom at `rho=0.05`; RGB is slightly negative and TIR is near zero. The readout space has reproducible headroom, with separate contributions from score selection, offset, and size, and a much larger joint oracle effect.

This supports a readout-side selector/correction as the next single-variable direction. It does not prove that a learned readout module will improve tracking, because the readout oracle uses GT cell and GT regression targets. It also does not rule out other feature-side operators, larger or learned supports, or non-modality residual bases.

## Next experiment

Train a matched, frozen-M1 readout selector on an independent sequence split. The selector receives the same joint signal as the feature arm and chooses `no-op`, `score-only`, `offset-only`, `size-only`, or `joint` with bounded strength and uncertainty. Supervision must be fixed-state decoded-box utility relative to no-op; ordinary tracking loss alone is insufficient. Report harmful intervention rate, no-op specificity, selected-action regret, risk-coverage, and center/offset/size errors before any closed-loop rollout.

The feature arm is not promoted to training under this contract. A later feature experiment would require a new action basis and a separate hypothesis.

## R1 top-K readout reachability (completed)

The same 84 states were replayed with the M1 response map's top-K cells (`K=1,4,8,16`). For each K, the offline oracle selected the best candidate cell by decoded IoU. Two regression conditions were compared: the original M1 offset/size at each candidate cell, and GT-consistent offset/size at each candidate cell.

| K | Top-K current-regression utility | Top-K contains GT cell | GT-cell current-regression utility |
|---:|---:|---:|---:|
| 1 | +0.0000 | 14.3% | +0.0091 |
| 4 | +0.0232 | 69.0% | +0.0091 |
| 8 | +0.0394 | 91.7% | +0.0091 |
| 16 | +0.0435 | 96.4% | +0.0091 |

This separates two effects. The current score peak is often not the best candidate: selecting among top-K cells has measurable fixed-state headroom. The candidate set usually contains the GT cell by K=8, so a deployment signal that can rank or reject top-K candidates is plausible. The GT-regression condition is an upper bound and is identical across K in this replay because the GT-consistent offset/size can reconstruct the target from any candidate cell; it is not evidence that a learned regressor will achieve that value.

The next implementation should therefore begin with a **readout candidate ranker plus no-op gate**, using only deployment-visible features. It should first select among top-K cells while retaining the original M1 offset/size. Offset/size correction is a separate nested arm after candidate ranking passes. This ordering keeps score selection and regression correction identifiable.

R1 artifacts: `output/experiments/M1_ACTION_SPACE_COMPARISON_20261013/topk_replay/{summary.json,summary.csv,per_state.csv}`. Replay tool SHA256 is recorded in the asset registry.

## R2 top-K ranker with no-op gate (completed)

The next test fit a ridge utility ranker over deployment-visible candidate features: response statistics, candidate rank and coordinates, response gap, offset, and size. The candidate action was a response peak override at one of the top-8 cells while retaining the original M1 offset/size. A predicted utility below zero selected no-op. The split was sequence-held-out: 16 fit sequences and 8 held-out sequences.

Held-out results (30 states):

- selected mean utility: `-0.0633`;
- sequence-macro utility: `-0.0471`;
- top-8 oracle sequence-macro utility: `+0.0500`;
- oracle fraction: `-0.9425`;
- harmful intervention rate (`utility < -1e-4`): `10%`;
- abstention rate: `50%`.

The selector failed against the no-op baseline. The top-K action space still has headroom, but the current deployment-visible candidate features and linear utility model do not identify useful cells reliably. This is **SIGNAL-NOT-ADMITTED**. Do not train a readout residual from this ranker and do not run closed-loop evaluation from this split.

The next valid diagnostic is a signal learnability audit: compare candidate utility against response rank, score margin, offset/size consistency, template-conditioned evidence, and previous-state variables with nested sequence-held-out models. If these signals remain below no-op after calibration, stop the top-K ranker route and return to state/crop or target-conditioned supervision analysis.

R2 artifacts: `output/experiments/M1_ACTION_SPACE_COMPARISON_20261013/ranker_fit/{summary.json,heldout_selection.csv,per_candidate.csv}`. Script SHA256 `6e00dd41c8be8e036da5e5e45e46cc8b65ca6f45b536103a939c9d5d689a930a`.

## R3 nested signal learnability audit (completed)

Using the same sequence split and top-8 candidate utilities, four nested ridge models were compared. Signals were added in this order: response/rank; local fused and RGB/TIR feature statistics; template-conditioned cosine evidence; state geometry statistics.

| Signal set | Held-out RMSE | Rank correlation | Selected sequence-macro utility | Oracle fraction | Harmful rate | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| response/rank | 0.1797 | 0.2660 | -0.0208 | -41.6% | 3.3% | 10.0% |
| + local fusion | 0.1838 | 0.2588 | +0.0042 | 8.5% | 6.7% | 56.7% |
| + template evidence | 0.1892 | 0.2595 | +0.0108 | 21.6% | 6.7% | 50.0% |
| + state geometry | 0.1982 | 0.2316 | -0.0346 | -69.3% | 16.7% | 50.0% |

The template-conditioned feature group is the only group with a positive held-out sequence-macro utility, but its action ranking remains weak and the cohort has only eight held-out sequences. Lower RMSE does not track better action selection. State geometry is actively harmful in this contract. The result is **TEMPLATE-SIGNAL-PRELIMINARY / NOT-ADMITTED**: run one independent calibration audit for template-conditioned evidence with sequence bootstrap and a pre-registered no-op threshold; do not train a residual yet.

R3 artifacts: `output/experiments/M1_ACTION_SPACE_COMPARISON_20261013/signal_audit/{summary.json,nested_results.csv,candidate_features.csv}`. Audit script SHA256 `2d08a404b21bffb15081250a4164c86266fe4d210d4623f40ba70af4dbd2f8be`.

## Evidence and provenance

- Protocol: `experiments/diagnostics/m1_action_space_comparison_20261013/protocol.md`.
- Trace: `output/experiments/M1_ACTION_SPACE_COMPARISON_20261013/trace_run/`.
- Feature capture: `output/experiments/M1_ACTION_SPACE_COMPARISON_20261013/feature_capture/selected_features.npz`, SHA256 `43ac4f863ffe65451df4ed49d8bcfed36a0c7590933d7bc987696c2f4e8f351e`.
- Oracle replay: `output/experiments/M1_ACTION_SPACE_COMPARISON_20261013/oracle_replay/summary.json`, SHA256 `339248cdb0d09b2d6026b2c39dfb399e9eaaab33a01f3b47394b9aa19e51f2e0`.
- Replay tool SHA256 `9f41f82acc9c468beed68ddddc65400adc02affe5b2c5fa54c81a714ded32741`.
- Git commit at execution: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`.
- Environment: 1x NVIDIA GeForce RTX 4090 D 24 GiB; Python 3.8.20; PyTorch 2.0.0+cu118; CUDA 11.8; FP32; approximately 0.05 GPU-hours across trace, capture, and replay.
- AUC/PR/NPR/OP50/OP75/FPS: `NA`; this was fixed-state diagnostic replay, not tracking evaluation.
