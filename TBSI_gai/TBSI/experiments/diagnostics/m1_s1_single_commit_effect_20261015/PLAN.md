# M1-S1 Single State Commit Effect

## Research question

At an actual S1 correction event, does committing the corrected box to the next-frame tracker state improve or harm subsequent tracking, holding the event-frame output and all pre-event history fixed?

## Hypothesis and null

- Alternative: committing an S1 correction has a nonzero, sequence-dependent effect on future localization, and event-visible signals can help distinguish benefit from harm.
- Null: the effect is near zero or inconsistent, leaving no useful event-level decision target for a gate.

## Run contract

- Parent: protected M1 epoch15; frozen S1 predictor/policy; no training.
- One intervention variable: on one selected frame, internal state commit is either corrected S1 box (`commit`) or uncorrected M1 box (`fallback`). The reported event-frame box remains S1 in both branches.
- Event selection: first/nearest non-noop S1 action around 25% and 70% of each sequence, selected from an S1-only discovery rollout without GT. Up to two events per sequence.
- Data: 12 whole sequences from the frozen train-side 72 list, excluding the prior 12-sequence branch-association diagnostic; chosen by deterministic SHA256 order. GT is used only for future outcome labels.
- Pairing: replay the same S1 prefix twice; at the event frame, require identical reported boxes and identical pre-event states. Change only commit mode for that frame. Restore coupled S1 mode on the next frame in both branches.
- Primary labels: same-frame IoU difference `S1 reported box - M1 raw box`, plus paired mean IoU difference `commit - fallback` over the next 10 and 30 frames. Secondary: future frames below IoU 0.1 and recovery time to 5 consecutive frames at IoU >= 0.5.
- Event-visible predictors: selected action, peak margin, normalized response entropy, top-5 margin, and correction magnitude. No predictor fitting or threshold tuning.
- Unit of inference: sequence; events are clustered by sequence. Report sequence bootstrap intervals and event-level distributions without treating frames as independent.
- Promotion rule: proceed to a train-side risk model only if event-frame output parity passes, future effect is nontrivial and heterogeneous, and event-visible scores show sequence-held-out ranking beyond chance. Otherwise test a fixed short state hold or reject state-gating as the route. Judge readout correction separately from state commit effects.
- No official test use in this diagnostic. Any later full-test confirmation must use a frozen rule and the fixed M1 parent.

## Minimal code map

- Add one standalone experiment runner. Do not edit tracker/model code; use existing `coupled` and `output_only` modes for the one-frame intervention.
- Save the fixed sequence manifest, discovery actions, event list, paired outcomes, parity checks, manifest and report under independent directories.

## Evidence targets

- Minimum: at least one eligible correction event, parity at each intervention, complete future labels and immutable provenance.
- Solid: events from at least 8 distinct sequences, sequence-cluster bootstrap, action-type and signal summaries.
- Maximum: broader train-side event cohort; out of scope for this run.
