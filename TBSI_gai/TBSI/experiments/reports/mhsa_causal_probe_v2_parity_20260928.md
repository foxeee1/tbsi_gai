# MHSA-CAUSAL-PROBE-V2 parity follow-up (2026-09-28)

This follow-up validates execution provenance for the frozen 24 regression and 24 gain event probe. No training started and no checkpoint or canonical result directory was modified.

- Parent: `TBSI-E-MHSA-FULL-20260927`
- Config: `TBSI_E_MHSA_FULL_4090D_seed42`
- Code: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` + working tree
- Checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`
- Canonical YAML SHA256: `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
- Runtime: 1x RTX4090D 24GB, Python 3.8.20, PyTorch 1.9.0+cu111, FP32 tracker

The single-sequence checker used `besom3`, frame 635, with 0-based indexing. Current default, `TBSI_TEMPLATE_CROSS_SCALE=1.0`, and a no-op analysis collector agreed; the current box differed from integer-serialized M1 by less than 1 px per coordinate (`IoU` difference `+0.004094`).

The full serial rerun completed 48 events x 4 conditions x 3 layers = 576 finite rows with one state hash per event. Its `F` boxes nevertheless failed comparison with stored M1 results: maximum coordinate difference `77.9545 px`, mean per-event maximum difference `11.0176 px`, and 38/48 events above 1 px. The rerun is therefore `FAILED-PARITY`; its intervention effects are diagnostic-only. The numerical regression-minus-gain effects were `-0.005878` (`C_OFF`), `-0.008748` (`C_HALF`), and `+0.000385` (`C_1.5`), but are not causal evidence.

Evidence: `tools/check_mhsa_f_parity_v2.py`, `output/experiments/mhsa_causal_probe_v2/parity/`, `output/experiments/mhsa_causal_probe_v2_rerun_20260928/`, and the preserved original `output/experiments/mhsa_causal_probe_v2/`.

Decision: do not train or promote an MHSA intervention. First isolate the tracker state/provenance mismatch, then rerun the frozen causal probe.
