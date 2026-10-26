# M1 temporal trace v1 and replay-contract audit (2026-10-06)

## Purpose

This run validates the data contract required before a causal TAM G0. It uses
the frozen M1 epoch-15 checkpoint on three fixed LasHeR **train** sequences:
`abreastinnerboy`, `right2ndfarboytakinglight2left`, and `biketurndark`.
The tracker remains FP32 (`TBSI_INFER_FP16=0`), TAM starts at gamma zero, and
the trace has an independent output directory.

Checkpoint SHA256:
`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`

Canonical baseline YAML SHA256:
`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`

## Trace validation

The export produced 1,374 frame rows and 1,371 history rows. All rows contain
the actual integer tracker crop, affine transform, state-before/state-after,
prediction, and raw head token shape `[1,256,768]`. The validator found:

- causal errors: `0`;
- cross-sequence errors: `0`;
- crop-link errors: `0`;
- duplicate frames/gaps: `0`;
- raw file/shape/checksum errors: `0`.

The raw head token is now stored as one float32 `.npy` file per frame. This is
an audit asset, not a new model result.

## Replay-contract audit

The read-only audit checked that each row can be linked to its immediately
preceding row without fabricating history. Results:

- raw history links: `1,371/1,371`;
- tracker state links: `1,371/1,371`;
- crop-transform formula checks: `1,374/1,374`;
- valid GT rows: `1,374`;
- GT intersecting current crop: `1,123/1,374 = 81.73%`;
- mean GT-center distance: `4.224` tokens;
- mean GT/crop area ratio: `0.05657`.

The decision is `REPLAY_CONTRACT_PASS`. This is a provenance and geometry
check only: `model_reforward_performed=false`. It does not establish that TAM
improves tracking or that the training sampler has reconstructed these crops.

The follow-up static actor audit remains
`STOP_TRAINING_HISTORY_CONTRACT`: all explicit temporal keywords are present,
but `crop_trace_available=false`, `history_source_verified=false`, and
`real_batch_count=0`. The frozen trace is therefore ready as an input asset,
but has not yet been wired into a real training batch.

## Real-trace adapter batch audit

As an intermediate check, `audit_causal_tam_trace_batch.py` consumed one
consecutive pair from each of the three fixed sequences. It used the stored
float32 M1 raw head tokens and the actual current/previous crop rectangles.
All three pairs passed finite-value checks, causal IDs, gate gradient, Q/K/V/out
gradients, and the `0.10` residual cap. Coverage was `0.8203`, `0.7656`, and
`0.8789`; the largest actual residual ratio was `0.03541`.

Decision: `TRACE_BATCH_ADAPTER_PASS`. This is not a training-actor pass:
`training_actor_exercised=false`, `optimizer_step_performed=false`, and
`tracking_effect_claim=false`. G0 and TAM training remain blocked until the
frozen trace is consumed by the actual training sampler/actor.

## Remaining gate

The training sampler still samples ordinary processed frame pairs and does not
yet consume a frozen M1 state/crop trace. Therefore the training actor has not
passed a real three-batch history audit. G0 feature-source screening, TAM
training, and full-244 remain prohibited until a trace-backed training batch
is implemented and passes checks for sequence/frame IDs, actual crops, raw
history tensors, coverage, finite values, gradients, residual ratio, and
non-TAM state parity.

## Evidence

- Trace output: `output/experiments/m1_temporal_trace_v1_20261006/`
- Trace validator: `tools/validate_m1_temporal_trace.py`
- Replay audit: `tools/audit_m1_temporal_trace_replay.py`
- Export command: `bash tools/export_m1_temporal_trace.sh`
- Decisions: `trace_validation.json`, `replay_contract_audit.json`
- Actor audit: `training_history_static_audit.json`
- Real-trace adapter audit: `trace_batch_adapter_audit.json`

## Trace-backed training actor audit

`audit_causal_tam_training_actor.py` loaded the frozen M1 epoch-15 checkpoint,
the existing CenterPredictor head, and three one-sample batches from distinct
train sequences. The actor transforms image-coordinate GT through each actual
image-to-crop affine and uses the repository's native `generate_heatmap`,
GIoU, L1, and focal losses. The optimizer contains only
`model.causal_tam.parameters()`.

All three optimizer steps passed: same-sequence rate `1.0`, causal ordering,
`[1,256,768]` raw-token shape, finite loss/output/gradients, gamma gradients,
post-gate Q/K/V/output gradients, residual ratio cap, and frozen box-head eval
state. At step 0, gamma was zero so Q/K/V/output gradients were zero as
expected; after the first gate update, all branch gradients were nonzero on
steps 1 and 2. The non-TAM model state SHA256 was identical before and after:
`9877c58997160bee81027f55b35507fbfc925e3d26a13530e95366f054bbe4ad`.

Decision: `TRACE_BACKED_ACTOR_PASS`. The actor audit performed three optimizer
steps in memory but saved no checkpoint and makes no tracking-effect claim.
The three-sequence trace is not a training set. Expanding to the registered
train/dev trace manifests is the next stage; disk space must be checked before
that export because the current full-precision raw-token trace already uses
about 1.1 GiB.

The fixed FFM manifests contain 152 train and 24 development sequences and are
sequence-disjoint. Their combined LasHeR train videos contain 114,887 frames.
Saving one float32 `[1,256,768]` raw tensor per frame would require about
84.15 GiB before JSON and filesystem overhead, while the workspace currently
has about 5.7 GiB free. No expanded trace export was started. The development
manifest is also previously used for module selection, so it remains a
development set and cannot serve as independent confirmation.

An initial audit invocation completed the computation but failed while
serializing a tensor-valued JSON field. It is retained as a tooling failure;
the serializer was fixed and the audit was rerun from the original M1
checkpoint with seed 42 fixed to produce the accepted result. Final maximum
actual residual ratio was `6.83e-5`.

Status: `DIAGNOSTIC-COMPLETE / TRACE-READY-FOR-TRAINING-ACTOR-AUDIT`.
No G0, TAM training, or full-244 was started by this run.
