# M1 Current Source Replay Parity (2026-10-22)

## Result

**PASS.** The current working-tree M1 default path reproduces the archived M1 predictions on the frozen 48-event parity cohort. All 48 integer-serialized boxes match exactly; no floating coordinate differs by more than 1 px.

| Measure | Result |
|---|---:|
| Events | 48 |
| Maximum absolute coordinate difference | 0.992050 px |
| Mean event maximum coordinate difference | 0.801937 px |
| Events over 1 px | 0 |
| Integer-truncated boxes equal | 48/48 |

The worst event is `carcomingfromlight` frame 88 at 0.992050 px. This matches the earlier corrected replay result and resolves the concern that current diagnostic code changes necessarily invalidate M1 replay.

## Contract

- Parent checkpoint: `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15, SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Protected config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`.
- Code: current working tree based on HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; source snapshot `experiments/snapshots/M1_SCREENING_PARITY_20261022/current_sources.tar.gz`, SHA256 `a0ee51be30a0a85bf8896816c548888cf88ca0875ab94b98c3aad7a89dc8ee3c`.
- Data: historical frozen parity events in `output/experiments/mhsa_causal_probe_v2/manifest`; archived predictions in the protected M1 full-244 output.
- Runtime: 1x NVIDIA GeForce RTX 4090 D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1; FP32; OMP/MKL/OpenBLAS single-thread; one tracker process.
- Command from `TBSI_gai/TBSI`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TBSI_INFER_FP16=0 \
  /root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/probe_mhsa_effective_message_v2.py \
  --config TBSI_E_MHSA_FULL_4090D_seed42 \
  --checkpoint output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar \
  --events-dir output/experiments/mhsa_causal_probe_v2/manifest \
  --output output/experiments/M1_SCREENING_CURRENT_SOURCE_PARITY_20261022/probe \
  --conditions F
```

Raw output: `output/experiments/M1_SCREENING_CURRENT_SOURCE_PARITY_20261022/probe/event_results.csv`, SHA256 `d1e8bffce27b7a85af7cb4cd08062009d1dd6ac351d44c306c5faf1235c510d5`. Parity was computed by joining each F row to the archived per-frame bbox by sequence and zero-based frame index.

## Interpretation

The earlier September mismatch came from a different probe execution contract: missing `previous_output` propagation and a forced deterministic cuDNN override. The corrected September runner documented these issues and passed. The present run reconfirms the corrected contract with the current source snapshot. Fixed-state action replay reports that its own no-op reproduces its saved feature-replay baseline; that report did not claim archived full-sequence parity.

This is an execution-contract check only. The test events are historically exposed and were not used to choose a candidate. New mechanism discovery proceeds on the frozen train-side cohorts in `m1_screening_v1_20261022`; no training has started.

## Next Step

Run protected M1 on the 48-sequence calibration cohort, mine the preregistered first persistent low-IoU events with at least 99.9% GT crop coverage, and inspect target/hard-background ordering at matched crop, template, and state. Calibration may repair extraction definitions only. Development and locked holdout remain untouched by training; full-244 stays historical reporting evidence.
