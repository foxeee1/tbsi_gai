# M1 Fixed-State Information Flow Calibration (2026-10-22)

## Question

At which measured stage does crop-contained target-versus-background ranking become less reliable before the first persistent low-IoU event, and does `tbsi_fuse_search` appear to be the first observable loss point? This is a calibration-cohort diagnostic of the protected M1 parent, not a causal attribution or a tracking-gain result.

## Contract and provenance

- Parent: `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Protected config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Protected checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Source snapshot: `experiments/snapshots/M1_SCREENING_PARITY_20261022/fixed_state_flow_sources_v9.tar.gz`, SHA256 `7692645f316b291f4351504618cfa7df4eede7e5a02b59a8279f220aafbd437e`.
- Final runner SHA256: `e08aa6159b08c84a79f2fb13b2f8862529dbf4f166a4eba19ec257031cac7107`.
- Dataset: LasHeR official `trainingset`; 23 calibration sequences with crop-contained first persistent failure onset and the immediately preceding frame; 46 fixed crop/template/state/frame observations.
- Runtime: 1x NVIDIA GeForce RTX 4090 D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1; FP32, one Torch/BLAS CPU thread, one serial tracker process.
- Output: `output/experiments/M1_SCREENING_FIXED_STATE_FLOW_CALIBRATION_V8_20261022/`.
- `per_event.csv` SHA256: `c998f93b9a1de9f20b6c87caf8713673b517b6afce78f512a25ce4a31c35ffee`.
- `stage_metrics.csv` SHA256: `d31347a031ec47ea42ab6bf1a34597db45034717b5556892a8ecf4fbd59bafba`.
- `run_manifest.json` SHA256: `74e10bb01f1e387af41306253d097683f270012476a6c7c1a587e704619ffc5f`.

Each query score uses the same operation as the existing `query_map`: L2-normalize every template/search token; compute weighted foreground and background template prototypes; normalize each prototype; score each search token as foreground cosine minus background cosine. The fixed hard-background coordinate is the strongest CENTER×Hann location outside GT, held constant across feature stages. GT rank counts all non-GT cells above the GT query peak. The report also retains the query map's own strongest non-GT peak separately.

## Validation

The first two metric attempts were not used for attribution. V4 pooled a template into one prototype while comparing unrelated feature sources; V5 used token-max matching and mixed `backbone.norm` template tokens with `tbsi_fuse_search` inputs. The V6 smoke also exposed that its `hard_bg` margin initially meant the query map's own strongest background, unlike the stage metric's fixed CENTER-selected background. These outputs remain as measurement-revision evidence.

V7 smoke validated the corrected formula against the existing `backbone_norm` query metrics: on both modalities, GT peak and fixed-background score differed by less than `9e-8`, GT margin by less than `8e-8`, and rank matched exactly. V8 then completed all 46 states. Replayed fixed-state IoU equals the calibration trace exactly for all 46 observations (maximum absolute difference 0).

## Results

| Observable | Pre-onset | Persistent-failure onset |
|---|---:|---:|
| CENTER GT rank = 1 | 22/23 | 11/23 |
| CENTER mean GT rank | 1.04 | 2.13 |
| CENTER mean GT minus fixed hard-background response | +0.293 | +0.088 |
| RGB pre-fusion query rank = 1 | 17/23 | 14/23 |
| TIR pre-fusion query rank = 1 | 13/23 | 9/23 |
| `tbsi_fuse_search` post-fusion query rank = 1 | 20/23 | 11/23 |

Across the 23 paired pre-onset/onset events, CENTER rank worsened in 12, stayed the same in 11, and improved in none. Its fixed hard-background margin fell by 0.204 on average; 11 events changed from positive margin to negative margin.

At onset, mean stage ranks and fixed-background margins over RGB/TIR observations were:

| Stage | Mean rank | Mean margin | GT rank = 1 |
|---|---:|---:|---:|
| block02 | 3.61 | +0.0961 | 23/46 |
| TBSI loc3 | 3.04 | +0.0986 | 23/46 |
| TBSI loc6 | 2.63 | +0.0894 | 27/46 |
| TBSI loc9 | 2.70 | +0.0678 | 28/46 |
| backbone norm | 3.50 | +0.1066 | 23/46 |

The largest observed late rank deterioration occurs from block10 to block11: mean rank worsens by 0.63 across 46 modality/state observations, while seven rank-1 cases are lost and three gained. From block11 to `backbone.norm`, fixed-background margin rises by 0.064, but rank worsens slightly (+0.15); therefore margin alone does not describe the ordering change. Earlier, block02 to loc3 improves mean rank (-0.57); loc3 to loc6 also improves it (-0.41). Loc6 to loc9 reduces fixed-background margin by 0.022, while mean rank is nearly unchanged (+0.07).

The post-fusion query has 11/23 rank-1 cases at onset, matching CENTER's count, but the ranks agree exactly in only 11/23 cases. This is not enough to establish that search fusion is the first cause. It also does not prove a head defect: query-map and CENTER scores are different readouts.

## Interpretation

- **Supports:** the dominant observable event is a paired collapse of CENTER target/background ordering between pre-onset and onset, despite the GT remaining inside the crop. Crop-contained single-frame ranking failure is therefore a real part of the observed failure chain.
- **Supports a location test:** block10→block11 is the strongest measured late rank deterioration among the inspected adjacent stages. A reversible block11 intervention can now test whether this is an operation-level amplifier.
- **Does not support:** `tbsi_fuse_search` as the first or sole source of target/background loss. At onset, unimodal rankings are mixed; post-fusion rank-1 frequency is lower than RGB and higher than TIR, which is not a uniform cross-modality result.
- **Does not establish:** a causal role for block11, TBSI, fusion neck, or CENTER. The current run is observational, has 23 calibration sequences, and examines only one onset pair per sequence.
- **Still unresolved:** whether late block11 changes CENTER ordering through useful target evidence, distractor amplification, or representation/readout alignment; whether those effects generalize across scene/attribute strata; and whether closed-loop state update compounds the first-frame ranking loss.

## Next bounded diagnostic

On the same 46 states, use a forward hook to make one reversible intervention per run: (1) replace block11 output with its input; (2) apply the same operation at block05 as a non-target-position control. Keep template, crop, state, data, precision, and tracker path fixed. Record CENTER GT rank, fixed hard-background margin, top-1 location, fixed-state IoU, and the intermediate query maps. The block11 hypothesis is supported only if its intervention improves target rank/CENTER ordering more often and more selectively than the block05 control. Otherwise stop location-specific fusion repair and inspect the CENTER readout/target-grid contract before training.

### Intervention preregistration

- Target hypothesis: the observed block10→block11 rank deterioration means block11's transformation can amplify an already ambiguous crop-level target/background ordering.
- Target intervention: at block index 11, return the block input in place of its output. This changes one operation and is removed after each event.
- Non-target-position control: perform the same reversible bypass at block index 5, selected before intervention results and outside the observed late deterioration interval.
- Fixed conditions: same calibration sequences, event frames, initial template, tracker state, crop, GT, CENTER decoding, M1 weights, FP32, and single-process runtime as V8.
- Decision observables: CENTER GT rank, CENTER×Hann GT-minus-fixed-hard-background margin, top-1-in-GT, distance of peak from GT center, replay IoU, and per-stage query ranking.
- Support rule: block11 bypass must improve CENTER rank/top-1 more consistently than block05 and must not owe its apparent effect to changed crop/state inputs. A common effect from both bypasses is not location-specific evidence.
- Stop rule: if block11 is not selectively better than the control, do not train a fusion repair from this evidence; investigate CENTER/readout alignment and temporal feedback.

## Reversible intervention results

Both arms replayed the same 46 states, and the affected block's captured query maps equal the preceding block output exactly (92/92 modality-state comparisons per arm). This verifies that the hook was applied at the intended location. The protected M1 config and checkpoint hashes remained unchanged.

| Arm | Onset CENTER rank improves / worsens / same | Onset top-1 gain / loss | Mean / median fixed-state IoU delta | Pre-onset mean / median IoU delta |
|---|---:|---:|---:|---:|
| block11 output → input | 8 / 1 / 14 | 4 / 1 | +0.126 / +0.012 | -0.076 / -0.020 |
| block05 output → input control | 5 / 1 / 17 | 1 / 0 | +0.025 / 0.000 | -0.099 / -0.028 |

At onset, block11 bypass wins over the block05 control on 12/23 events, loses on 8/23, and ties on 3/23 for IoU delta. Its mean IoU gain is skewed by a few events (`rightof2boys` +0.872, `Amidredgirl` +0.797, `Aboydownbike` +0.534); the median gain is only +0.012. The intervention also harms many pre-onset frames. Thus:

- **Supports:** block11 can amplify some crop-contained failure events; a target-conditioned correction around this transformation remains a testable idea.
- **Does not support:** globally bypassing block11 or starting an unconditional block11 replacement. The control also helps some failures, and both bypasses hurt pre-onset frames.
- **Not yet distinguished:** whether a reliable online signal can identify the subset helped by block11 correction; whether the apparent gain replicates on the untouched development cohort; and whether the effect survives training.

The calibration intervention was `SUPPORTS-NEXT-REPLICATION / NOT-TRAINING-ADMITTED`; the one-time development replication is now complete.

## Development replication

The frozen event rule found 15 first persistent events in the 24-sequence development cohort (11,422 trace frames). The same 15 events plus their prior frames (30 states) were run through baseline, block11 bypass, and block05 control. Baseline fixed-state replay IoU matched the mined M1 trace at all 30 states.

| Observable | Pre-onset | Persistent-failure onset |
|---|---:|---:|
| CENTER GT rank = 1 | 14/15 | 7/15 |
| CENTER mean GT rank | 1.13 | 2.20 |
| CENTER mean GT minus fixed hard-background response | +0.221 | +0.018 |
| Baseline mean fixed-state IoU | 0.649 | 0.231 |

On the 15 onset states, block11 bypass improves/worsens/keeps CENTER rank in 3/4/8 cases, gains/loses top-1 in 2/2 cases, and changes IoU by mean `+0.060` and median `-0.007` (4 wins, 3 ties, 8 losses). Block05 control improves/worsens/keeps rank in 4/4/7 cases, gains/loses top-1 in 3/2 cases, and changes IoU by mean `+0.130` and median `0.000` (7 wins, 2 ties, 6 losses). The paired block11-minus-control IoU effect averages `-0.070`; it is positive on 7 events, negative on 6, tied on 2.

### Decision

- **Supported across calibration and development:** the event class is real in this protocol: with GT still inside the search crop, target-vs-background CENTER rank and response margin often collapse at the first persistent low-IoU onset.
- **Not supported:** block11 as the repeatable first cause or a privileged repair location. Its calibration advantage over block05 does not replicate; development favors the non-target control on mean and win count.
- **Not training-admitted:** do not train a block11 bypass, block11 replacement, or conditional block11 adapter from these results. The calibration effect is skewed and the development differential fails.
- **Remaining route:** keep template updating out of scope. For a future M1 improvement experiment, formulate one candidate around target-conditioned frame-local fusion/readout alignment, but first use calibration-only diagnostics to define an online signal that identifies the ambiguous target/hard-background state. Do not tune against development again; locked holdout stays closed until a candidate, training contract, and success rule are frozen.

This closes the first evidence-location chain. The durable claim is about the failure scenario, not a proven culprit operator. No candidate training or locked-holdout evaluation has started.

## Remaining location interventions

The earlier user-requested localization contract also calls for reversible tests at the early block, TBSI locations, and search fusion. These will use only the 23 calibration events and their previous frames (46 states); development will not be reused.

- `block02` bypass returns the block input in place of its output.
- Each TBSI bypass (`loc3`, `loc6`, `loc9`) returns the modality inputs in place of that layer's outputs.
- Search-fusion replacement runs the existing `tbsi_fuse_search` with either RGB or TIR input and zeros in the other modality. Its template tokens use the same single-modality projection so query features remain source-aligned.
- The existing M1 baseline is the direct comparator; block05 is retained as the non-target-position control.
- Per state, compare CENTER GT rank/top-1, GT-minus-fixed-hard-background margin, top-1 distance from GT center, fixed-state IoU, and stage query ranking. A site is only an error amplifier candidate if its reversible change restores GT ordering at CENTER more than the control without a corresponding loss on the pre-onset state.
- These are intervention diagnostics only. No gate magnitude is tuned, no learned candidate is trained, and no development/locked-holdout results are used to choose among these locations.

## Remaining location intervention results

All six remaining calibration-only interventions replayed the same 46 fixed states. Each intervention changed the captured feature at its declared site; baseline replay remained identical to the mined M1 trace. IoU deltas below are paired to the V8 baseline at the same state. Positive rank delta means a worse (larger) GT rank.

| Intervention | Onset mean / median IoU delta | Onset IoU wins / ties / losses | Onset mean GT-rank delta | Onset CENTER top-1 gains / losses | Pre-onset mean / median IoU delta |
|---|---:|---:|---:|---:|---:|
| block02 output → input | +0.0762 / 0.0000 | 11 / 4 / 8 | -0.348 | 2 / 0 | -0.1694 / -0.0073 |
| TBSI loc3 bypass | +0.0216 / 0.0000 | 9 / 4 / 10 | -0.174 | 1 / 0 | -0.0304 / -0.0041 |
| TBSI loc6 bypass | +0.0495 / 0.0000 | 10 / 4 / 9 | -0.304 | 2 / 0 | -0.0600 / -0.0033 |
| TBSI loc9 bypass | +0.0560 / +0.0003 | 13 / 4 / 6 | -0.174 | 2 / 0 | +0.0041 / +0.0013 |
| Search fusion, RGB-only | +0.0700 / 0.0000 | 10 / 3 / 10 | +0.391 | 4 / 2 | -0.2034 / -0.0951 |
| Search fusion, TIR-only | +0.0452 / -0.0023 | 11 / 0 / 12 | -0.609 | 8 / 1 | -0.1982 / -0.1410 |

The single-modality search-fusion arms show that changing the fusion input can restore CENTER top-1 on some onset states, especially TIR-only (8 gains, 1 loss), but its median IoU delta is negative and the intervention harms preceding states. This is evidence that the fusion neck can change the selected peak, not evidence for a generally useful modality gate. The block02 and TBSI bypasses also have zero median onset IoU change and substantial pre-onset harm; none identifies a stable preferred repair site. These results are calibration-only and are not independent performance estimates.

Across the 23 onset states, an oracle that selects whichever single-modality intervention has higher fixed-state IoU beats baseline on 16/23 states (mean delta `+0.1402`, median `+0.0773`) and loses on 7/23. The oracle selects RGB on 13 events and TIR on 10; neither modality is globally preferable. This estimates available per-event decision headroom only: it uses the observed outcome and cannot be implemented as an online rule. Together with the weak early-margin correlation, it motivates learning a utility predictor from train-side supervision rather than hand-tuning a margin threshold.

## Early signal admissibility

The existing early target/local-hard-background margin candidate was audited at fixed state on 25 events. The statistic itself had nontrivial values (mean RGB/TIR margins 0.6571/0.7342), but the learned gate changed by only `5.85e-6`/`2.27e-6` on average. Margin-to-gate correlations were 0.0252/0.0061, CENTER margin changed by `2.83e-5`, and CENTER GT rank changed on 0/25 events. A separate direct-margin replay had mean CENTER margin delta `1.29e-5`, median `1.20e-5`, and rank change 0/25. Thus the current estimator-to-gate path is functionally inactive; it does not establish that more training will make the statistic target-specific.

An early block02 RGB/TIR foreground-background margin also failed to predict whether RGB-only or TIR-only search fusion helped fixed-state IoU on the 23 onset events (Spearman `-0.079`, `p=0.72`). The current measurements therefore do not admit a learned online gate based on this margin. This is a signal-contract failure, not evidence that all conditional fusion is impossible.

## Current decision and experiment entry gate

- **Established problem class:** first persistent low-IoU events frequently begin while the GT remains inside the crop, with CENTER GT top-1 falling from 22/23 to 11/23 in calibration and 14/15 to 7/15 in development.
- **Not established:** a unique first-cause operation, a consistently beneficial bypass location, or an online observable that can choose when/which modality to correct.
- **Training admission:** none of the tested location interventions or early-margin estimators passes the candidate gate. Do not launch a learned correction from these measurements alone.
- **Next M1 candidate, now preregistered:** an early-feature-conditioned modality-utility router at the MSHA cross-modal Value interaction. A small predictor reads Stage-2 RGB/TIR and template foreground/background features, then predicts a two-modality utility distribution. Training uses one auxiliary KL objective whose detached teacher is the per-modality CENTER GT-minus-hard-background utility computed from labels on candidate-train samples. This replaces hand-designed cosine/margin thresholds; it does not use calibration or development labels for updates.
- **Preflight before training:** zero-gate parity; shape and save/load; gradient flow into router with frozen parent; parent tensor drift check; and a fixed-small-sample fit test. Then evaluate predicted utility ranking/regret on calibration once, with a predeclared metric and no threshold fitting. If prediction does not beat the constant-prior predictor or its selected modality is no better than chance against the observed modality utility, stop before quick training. If it passes, run one 176-sequence, 1,264-optimizer-update candidate and a matched direct-parent control; add no other architectural change or loss.
- **Data roles:** candidate quick training uses the fixed 176-sequence stratified manifest; calibration 48 defines diagnostics, development 24 is used once for the frozen hypothesis, and locked holdout 24 remains sealed until design freeze. Full-244 LasHeR test is selection-exposed historical evaluation, while the four paper cases are retrospective illustrations only.

This completes the current chain's diagnosis, data-role freeze, and candidate preregistration. No candidate code or training has started. The next implementation is limited to the preregistered utility router and its preflight; no architecture sweep is admitted.

No candidate operator training has started. The 48/24/24 data-role protocol remains intact; the 24-sequence development and locked holdout cohorts have not been used. Historical full-244 remains selection-exposed and cannot be described as a fresh holdout.
