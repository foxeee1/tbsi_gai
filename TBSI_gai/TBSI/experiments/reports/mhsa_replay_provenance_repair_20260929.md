# MHSA replay provenance repair and P2 gate (2026-09-29)

## Diagnosis

The original parity failure had two execution-contract mismatches:

1. The probe did not pass `previous_output` exactly as `lib.test.evaluation.tracker._track_sequence` does.
2. The probe forced `torch.backends.cudnn.benchmark=False` and `deterministic=True`, although the stored M1 full-test predictions were produced through the regular test entrypoint.

The first mismatch was corrected in `probe_mhsa_effective_message_v2.py`. The second was removed. A separate audit with the forced deterministic settings reproduced a first divergence at `besom3` frame 38 and reached a 6.24 px event-frame difference at frame 635, confirming that the global cuDNN override was material to the closed-loop trajectory.

## Gates

- P0 three-sequence replay: `besom3` to 635, `lefthyalinepaperfrontpants` to 702, and `blkboyback` to 103. No frame exceeded 1 px; actual `use_fp16=False` was recorded. The non-deterministic regular-entrypoint replay had maxima `0.99965`, `0.99908`, and `0.99728` px.
- P1 frozen 48-event F-only replay: 48/48 events, maximum floating-box difference `0.992050 px`, mean per-event maximum `0.801937 px`, `over_1px=0`, and integer-truncated boxes equal for 48/48 events.
- P2 same-state split: `F -> C_OFF -> C_HALF -> C_1.5 -> F_POST`, 720 rows. Across 48 events x 3 layers, `F` and `F_POST` had maximum box difference `0.0` and all state hashes matched.

## Corrected probe analysis

The corrected P2 run is finite and complete. The regression-minus-gain mean intervention differences are `+0.025835` (`C_OFF`), `+0.008360` (`C_HALF`), and `+0.020750` (`C_1.5`). Event-level bootstrap intervals from the current analysis cross zero for all group effects; the reported 48 events are repeated measurements clustered within sequences and are not 576 independent samples. The cross-scale logit-bias intervention is also not a linear effective-message multiplier.

These values are now valid diagnostic evidence under the repaired execution contract, but they do not establish a trained intervention or generalization claim. The events remain post-hoc test-set failure/gain events. No training or module promotion is authorized from this probe alone.

## Evidence

- Audit runner: `tools/run_mhsa_replay_audit_v1.sh`
- P1 runner: `tools/run_mhsa_f_parity_v2.sh`
- P2 runner: `tools/run_mhsa_causal_probe_v2_p2.sh`
- P0/P1 outputs: `output/experiments/mhsa_replay_audit_v1_20260929/`, `output/experiments/mhsa_f_parity_v2_20260929_b/`
- P2 output: `output/experiments/mhsa_causal_probe_v2_p2_20260929/`
- Checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`
- Canonical YAML SHA256: `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
