# Synthetic joint-interference v2 robustness and closed-loop diagnostic

This campaign is a fixed-event diagnostic follow-up to SJI-v1. It does not read blind-review labels, the 169-row queue, or any D/R/G labels. It does not change the canonical baseline YAML, model weights, attention, template update, or tracker logic.

## Protocol

- Fixed source: the 72 events from `synthetic_joint_interference_v1/batch/selected_events.json`.
- Audit: event and sequence statistics, trimmed means, leave-one-sequence-out analysis, sequence-cluster bootstrap, paired permutation, and IoU-form interaction `J`.
- Controls: `00`, `10`, `01`, `11_same`, `11_independent`, `11_texture`, `11_shift`.
- Closed loop: conditions `00`, `10`, `01`, `11_same`, `11_independent`, `11_texture`; horizons H=1,5,10; recovery threshold tau=0.5.
- Runtime: 1x RTX 4090 D, PyTorch 1.9.0+cu111, FP32 tracker inference, canonical epoch-15 checkpoint.

## Audit

- Mean I: `0.023641`; median I: approximately zero.
- Sequence-bootstrap 95% CI: `[-0.000649, 0.058274]`.
- Paired sequence permutation p-value: `0.19694`.
- Dropping the largest three events: mean I `-0.000447`.
- I>0.05: `3/72`; positive-I sequences: `>=8`; largest positive sequence contribution: `47.4%`.

The original effect is therefore not robust under the pre-registered A2/A3 checks.

## Controls and closed loop

- Same-location interaction mean: `0.023962`.
- Independent-location interaction mean: `-0.004315`.
- Texture-control interaction mean: `-0.011543`.
- H=5 clean-minus-same mean IoU: `0.005910` across 71 eligible events.
- H=10 clean-minus-same mean IoU: `0.002662` across 71 eligible events.
- Alpha sensitivity mean IoU: `0.802135`, `0.798281`, `0.796186`, `0.756676` for alpha `0.25`, `0.50`, `0.75`, `1.00` respectively.
- Position sensitivity mean IoU: `0.762112`, `0.797126`, `0.800929` for `P0`, `P1`, `P2` respectively. The position result is not monotonic and should not be generalized as a distance law.

The five-item automatic continuation gate passed: sequence bootstrap support, same>independent, texture below same scale, and positive H=5/H=10 degradation. However, A2 and A3 remain false, so this is conditional evidence rather than a stable universal failure claim.

## Corrected offline reanalysis

The original automatic gate was not used as training permission. A CPU-only paired reanalysis corrected three definitions: A1 no longer treats a CI upper bound above zero as support; texture is reported as a direct same-event `11_same` vs `11_texture` pairing rather than an incomplete interaction term; and closed-loop means exclude intervention frame `offset=0`.

- `same` vs `independent`: event mean IoU difference `+0.028276` (71 events); sequence-cluster bootstrap 95% CI `[0.000980, 0.076114]`; paired sequence permutation `p=0.0758`.
- `same` vs `texture`: event mean IoU difference `+0.035505`; sequence-cluster bootstrap 95% CI `[0.002471, 0.085882]`; paired sequence permutation `p=0.0704`. This is a direct pairing, not a texture interaction statistic.
- Post-only H=5: clean-minus-same mean `-0.001246`; sequence-cluster CI `[-0.003692, 0.001157]`; 5/71 events unrecovered under the fixed recovery rule.
- Post-only H=10: clean-minus-same mean `-0.001241`; sequence-cluster CI `[-0.003740, 0.001029]`; 5/71 events unrecovered.

The corrected result does not support persistent downstream degradation. The evidence remains a conditional single-frame synthetic phenomenon, concentrated in a few sequences, and is insufficient for C/A training or a false-consensus main claim. Decision: `STOP_NO_TRAINING`; return effort to the real-image blind review and preserve SJI-v1/v2 as diagnostic evidence.

Evidence: `output/experiments/synthetic_joint_interference_v2/`; corrected JSON: `paired_post_only_reanalysis_v1.json`; corrected event files: `paired_same_independent_event.csv`, `paired_same_texture_event.csv`, `closed_loop_post_h5_event.csv`, `closed_loop_post_h10_event.csv`; original audit/controls/sensitivity files remain unchanged.

The three extreme SJI events were checked offline within the fixed two-hour audit scope. Their contact sheets are in `output/experiments/synthetic_joint_interference_v1/batch/extreme_event_inspection/`; no obvious coordinate overlap bug was found, but the strongest two events are crowded/low-contrast scenes and remain synthetic, event-specific failures. This does not change `STOP_NO_TRAINING`.
