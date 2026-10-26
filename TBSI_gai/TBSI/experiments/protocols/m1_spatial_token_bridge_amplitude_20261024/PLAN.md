# Candidate 2 Amplitude Diagnostic Plan

## Selected idea

Use the learned E10 bridge vectors unchanged and vary only their post-bridge amplitude on the same fixed states. Target and equal-density shifted arms at matched scales distinguish a weak but useful direction from a direction that carries no target-specific localization information.

## Minimal change map

- Add a scale argument and temporary forward hook to the fixed-state runner only.
- Add an inference-only config with the bridge architecture matching the E10 checkpoint.
- Add a wrapper that derives the exact fixed-state event manifest from the saved E10 off replay and runs seven serial arms.
- No model parameter, training code, optimizer, checkpoint, or evaluator metric definition changes.

## Contract

- Tier: auxiliary/dev mechanism diagnostic.
- Parent: Candidate 2 E10 bridge-only checkpoint and its 46-state off/target/shifted replay.
- Arms: off; target and shifted at `1x`, `32x`, `128x` post-bridge output delta scale.
- Dataset: LasHeR train-side calibration cohort, 23 sequences, 46 fixed states; inherited state data are hash-recorded.
- Precision/throughput: FP32, serial, one tracker process, torch threads 1.
- Primary: CENTER margin/rank; post-fusion query margin/rank; fixed replay IoU.
- Secondary: pre-onset/severe-onset and sequence-level paired changes; recorded bridge residuals.
- Minimum evidence: scale-1 replication and complete 46-state outputs for every arm.
- Solid evidence: target-specific scale response passing preregistered `>1e-3` and majority sequence-sign gate without rank/IoU regression.
- Maximum evidence: not part of this diagnostic; would require a new held-out or M1-aligned experiment.
- Stop: any scale-1 parity failure; missing states; non-finite outputs; or process/resource conflict.

## Expected signatures

- Amplitude-limited: target gains relative to shifted as scale rises, with a localization-facing rank or replay change.
- Content/interface-limited: target-specific readouts remain tied through the stress scale.
- Overdrive: high scale changes both controls similarly or causes target-specific and shifted regressions.

## Planned decision

Use the paired response and validity check to decide whether native bridge magnitude merits a separately designed operator revision. This run itself cannot justify full training or a performance claim.

## Execution incident

The first attempt stopped before model initialization because the event file was serialized per state while the runner contract expects one grouped pre/onset record per sequence. It evaluated zero states. The failure log and manifest are retained under `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_20261024/`. Retry 1 rebuilds the same 46 states in the runner's expected schema and writes to a separate output directory; the model intervention and arms are unchanged.

## Outcome

- Retry 1 completed all seven arms and 322 fixed-state evaluations.
- Off and both 1x arms exactly reproduced the E10 off/target/shifted replay on all three primary metrics.
- At 128x, target-minus-shifted means were CENTER `-1.996e-4`, query `+5.228e-4`, and replay IoU `-0.02054`; CENTER/query rank changed on 3/46 and 2/46 states with mixed direction.
- The preregistered `>1e-3` target-specific localization gate failed. Amplitude alone is not an adequate explanation for this bridge's null effect; stop scaling and do not proceed to full training. Results are train-side diagnostics, not benchmark evidence.
- Detailed report and sequence-level outputs: `experiments/reports/m1_spatial_token_bridge_amplitude_fixed46_20261024.md` and `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024/`.
