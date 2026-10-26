# M1 low-AUC MSHA foreground cross-message study (2026-10-22)

## Scope and decision

This serial study tests whether the joint template MSHA cross-modal message is a meaningful cause of failure on M1's lowest-AUC sequences, and whether restricting that message to foreground template queries helps without broadly disabling cross-modal exchange. It excludes template updating and evaluates a frame-level MSHA intervention. All test-set results are exploratory diagnostics, not generalization claims.

**Decision: do not train or promote the current foreground-query restriction.** Removing all cross-modal attention helps a few low-AUC events substantially and harms others. Restricting cross-modal messages to foreground template queries produces a small positive 128-frame cohort mean, but the sequence-bootstrap interval crosses zero and the fixed-event median is approximately zero. The evidence supports cross-modal interaction as a causal influence with heterogeneous direction; it does not yet identify a stable selective correction.

## Parent and cohort

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- M1 config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- M1 checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Cohort: lowest 20 M1 sequence AUC values from the official full-244 sequence audit, fixed before this intervention; list SHA256 `e383fc1fd66cfb86c4909fe85b49c40f3a818ab48fc3b4ee3bb66ac2a355ec1b`.
- Data: LasHeR `testingset`; the cohort was selected by M1 test-set performance, so the study is diagnostic only.
- Environment: 1x NVIDIA GeForce RTX 4090 D; Python 3.8.20; PyTorch 1.9.0+cu111; CUDA 11.1; FP32; OMP/MKL threads 1.
- Protected YAML and checkpoint hashes were rechecked after all runs and match the registered hashes.
- Source snapshot: `experiments/snapshots/M1_MSHA_FG_CROSS_20261022/final_sources.tar.gz`, SHA256 `02d21db10d2dc9e19948c1f6706f169037c0fa6dfa549ce81d1c4bee56e11070`.

## Serial results

### 1. Baseline interaction audit and cross-modal ablation

First, standard MSHA statistics were sampled over up to the first 48 frames of each sequence (20 sequences, 2,880 layer-frame rows). Mean cross-attention probability mass was small and directional: RGB-query to TIR-key was `0.00533/0.00567/0.00526` at TBSI locations 3/6/9; TIR-query to RGB-key was `0.01174/0.01110/0.01024`. Mean foreground key mass increased from about `0.27` at loc3 to `0.43-0.47` at loc9. This confirms interaction exists, but attention mass alone does not establish useful target selectivity.

At each sequence's first sustained low-IoU event, `TBSI_TEMPLATE_CROSS_SCALE=0` was compared with scale 1 from the same replay state, crop, template and frame. Across 20 events, disabling all cross-modal attention changed mean IoU from `0.2744` to `0.3515` (`+0.0771`); 8 events improved, 3 tied and 9 declined. Large improvements were concentrated in `3bike1` (`0.000 -> 0.680`), `rightgirltakingcup` (`0.000 -> 0.683`) and `boyunder2baskets` (`+0.277`). Clear counterexamples included `ab_bikeoccluded` (`-0.036`), `girl'sblkbag` (`-0.026`) and `leftunderbasket` (`-0.024`).

Interpretation: MSHA cross-modal routing can alter the selected box at a fixed state and is therefore causally relevant on this cohort. This intervention sets cross logits to a large negative value before the joint softmax, so it also reallocates probability mass to self-modal keys; it does not isolate the effect of the cross Value content alone. The mixed signs rule out globally removing it. The large event jumps need independent replay confirmation before being treated as representative.

### 2. Foreground-query cross-message restriction

Added a default-off inference switch, `TBSI_MSHA_FG_CROSS_MODE=foreground_queries`. It retains the original joint Q/K, joint softmax, Value projections and self-modal messages. For each modality, it multiplies only the cross-modal message received by each template query by that modality's fractional foreground coverage. Foreground queries retain the cross-modal message; background queries receive less or none. No parameters or losses were added.

### 3. Fixed-state CENTER outcome

The same 20 event states were compared under M1 and the foreground-query switch. The observable was the final predicted box against the event GT, which tests whether changes reach the CENTER output.

- Mean event IoU: `0.2744 -> 0.3106`, delta `+0.0362`.
- Median event delta: approximately `-0.0001`.
- Per-event outcomes: 6 improved, 10 declined, 4 tied within numerical tolerance.
- Mean center-error change: `-0.93 px`, with high sequence variance.
- The largest gains/losses remain sequence-specific; no stable event-level advantage was established.

Thus this restricted message can reach CENTER, but it does not reliably improve the GT/background decision.

### 4. Closed-loop 128-frame screen

The same 20 sequences were replayed from their initial state under M1 and with the restriction active at every frame, capped at 128 tracked frames per sequence. This is a paired closed-loop prefix diagnostic, not official sequence AUC or full-sequence evaluation. The restriction was proposed after inspecting this lowest-AUC test cohort, so this result is not an independent confirmation set.

- Sequence-macro mean IoU: `0.23638 -> 0.24844`, delta `+0.01206`.
- Sequence bootstrap 95% interval for mean delta: `[-0.00840, +0.03784]`.
- Paired sequence outcomes: 11 improved, 5 declined, 4 were within `0.001`.
- Mean change in per-sequence IoU<0.5 frame rate: `-0.0184`; 11 sequences improved, 5 worsened, 4 were unchanged.
- Largest positive changes: boy`sheadingreycol `+0.2040`, `leftboy2jointhe4` `+0.0531`, `middrillmaster` `+0.0381`.
- Largest negative change: `basketball849` `-0.1009`; `lastleftgirl` was `-0.0214`.

The mean gain is not statistically separated from zero under sequence resampling and is offset by material regressions. Do not interpret it as a performance gain.

## Implementation and execution notes

- `lib/models/layers/tbsi_layer.py`: default-off foreground-query restriction; default-off route retains the original `(attn @ v)` computation. Also guarded early-gate diagnostic statistics when that unrelated gate is disabled; this fixed an existing `UnboundLocalError` in the statistics-only default path.
- `tools/probe_mhsa_effective_message_v2.py`: added `FG_CROSS` paired condition and center-error output.
- `tools/build_m1_lowauc_msha_events_20261022.py`: creates fixed onset events from saved M1 traces.
- `tools/run_m1_lowauc20_msha_closed_loop_20261022.py`: runs paired 128-frame closed-loop prefixes.
- The historical wrapper `run_mhsa_causal_probe_v2.sh` stopped before inference because it checks for a deleted historical v1.0.0 YAML. No model was loaded in that attempt; it is not part of the result.
- The first attention-statistics invocation exposed an uninitialized diagnostic variable and stopped on sequence 1. The statistics guard was fixed and the same 20-sequence collection completed. The first closed-loop script invocation stopped before model construction due to a missing repository import path; the path was added and the full paired prefix run completed.
- Compile checks passed for the changed Python files. The active feature switch is off by default; the M1 YAML and checkpoint hashes remain unchanged.

## Next action

Do not start quick training. The remaining uncertainty is whether the result comes from limiting background-query cross messages specifically or from a wider change to cross/self message balance and closed-loop sampling. If this line is resumed, the next comparison should be an independent train-side fixed-state cohort with a matched non-target-template-query control, recording GT rank, strongest hard-background rank and CENTER output under identical crops. Training is justified only if foreground locations show a consistent margin/rank benefit and the non-target control does not.

## Evidence paths

- Attention statistics: `output/experiments/M1_LOWAUC20_MSHA_STATS_20261022_retry/`.
- Event manifest: `output/experiments/M1_LOWAUC20_MSHA_CAUSAL_20261022/manifest/`.
- All-cross ablation and foreground fixed-event comparison: `output/experiments/M1_LOWAUC20_MSHA_CAUSAL_20261022/probe/event_results.csv` and `output/experiments/M1_LOWAUC20_MSHA_CAUSAL_20261022/fg_cross_probe/event_results.csv`.
- Closed-loop paired results: `output/experiments/M1_LOWAUC20_MSHA_CLOSED_LOOP_20261022/{per_sequence.csv,per_frame.csv}`.
