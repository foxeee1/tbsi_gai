# Synthetic joint-interference diagnostic v1

This is a diagnostic-only pre-experiment. It does not use the blind-review labels or the 169-row queue, and it does not alter network weights, attention, template updates, or the canonical baseline.

## Protocol

- Parent/checkpoint: canonical TBSI checkpoint `TBSITrack_ep0015.pth.tar`, seed 42.
- Data: 24 fixed LasHeR training sequences outside the 30 blind-review sequences; three event frames per sequence; H=1.
- Conditions: `00`, `10`, `01`, `11`, plus `bg10`, `bg01`, `bg11` background-block controls.
- Smoke gate: 8 events. Unmodified `00` parity, geometry, CUDA/Python/NumPy RNG restoration, and forward/reverse condition order all passed.
- Hardware: 1x NVIDIA GeForce RTX 4090 D, Python 3.8.20, PyTorch 1.9.0+cu111, `TBSI_INFER_FP16=0`.

## Result

The batch completed 72 events and 504 condition rows with zero recorded geometry/runtime failures. Event-level `I` had mean `0.023641`, median approximately `0`, range `-0.024072` to `0.857087`, and was positive for 33/72 events. Three events exceeded `I > 0.05`: `SJI-00005`, `SJI-00014`, and `SJI-00030`. These are synthetic copy-paste diagnostics, not evidence of real-world false consensus or an end-to-end tracking improvement.

## Evidence

- Smoke: `output/experiments/synthetic_joint_interference_v1/`
- Batch: `output/experiments/synthetic_joint_interference_v1/batch/`
- Event rows: `event_results.csv`, `event_summary.csv`
- Sequence aggregation: `sequence_summary.csv`
- Failure examples: `failure_examples/`
- Provenance: `metadata.json`
