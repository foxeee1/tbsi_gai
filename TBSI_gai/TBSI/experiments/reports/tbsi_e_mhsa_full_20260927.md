# TBSI-E simplified MHSA full-run attempt

- ID: `TBSI-E-MHSA-FULL-BLOCKED-20260927`
- Parent: direct simplified-fusion design; no IGF, TPTU, or HMoE
- Config: `TBSI_E_MHSA_FULL_4090D_seed42`
- Commit/snapshot: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` + working tree
- Config SHA256: `c0a279c2901cbf7e2d07977f8be6cebcf86c718a24b1ab21abc0e0091caf853b`
- Canonical baseline YAML SHA256: `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
- Seed: 42
- Dataset intended: LasHeR train and full LasHeR test (244 sequences)
- Hardware: 1 x NVIDIA GeForce RTX 4090D, 24 GiB
- Software: Python 3.8.20; PyTorch 1.9.0+cu111; CUDA 11.1; FP32 tracker inference

## Implementation checks

- `tbsi_e` constructs one template-only joint MHSA per configured TBSI location.
- The six legacy CAST paths are absent in this mode.
- Model construction passed for all three configured locations `[3, 6, 9]`.
- Synthetic forward passed with RGB/TIR tensors `[1, 320, 768]` and `lens_z=64`.
- Python syntax and config loading passed.

## Full-run result

The user authorized bypassing the disk threshold. Training completed for 15 epochs with
rolling checkpoints enabled, leaving only `TBSITrack_ep0015.pth.tar`. Full FP32 tracking
completed on all 244 LasHeR test sequences, followed by analysis.

| AUC / Success | OP50 | OP75 | Precision | Norm Precision |
|---:|---:|---:|---:|---:|
| 56.21 | 68.33 | 47.13 | 70.21 | 66.20 |

- Result files: 244 sequence files
- Checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`
- Config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`
- Training/test log: `output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/logs/full_pipeline.log`
- Tracking results: `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42/`

Status: `VALID-ONE-SEED`.

This is a one-seed result on 1 x RTX 4090D using an independent 4090D configuration.
It is not a canonical-baseline reproduction and does not establish a multi-seed method
claim. Relative to the fixed canonical reference, the raw differences are AUC +1.39,
OP50 +1.81, OP75 +1.33, Precision +1.68, and Norm Precision +1.61; these are not a
causal comparison because hardware/configuration/code provenance differ.
