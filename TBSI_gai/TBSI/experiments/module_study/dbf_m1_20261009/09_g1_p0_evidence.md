# DBF G1 P0 evidence (2026-10-09)

Parent: `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15. Protected config SHA256
`fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`;
checkpoint SHA256
`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` plus the full runtime
source snapshot `experiments/snapshots/dbf_g1_ready_20261009.tar.gz`, SHA256
`3c095c66d11a8d058f8a7025918b87db4ec337a6f6accb917bc1c9634b2ccf11`.
The prechange edited-source snapshot is
`experiments/snapshots/dbf_g1_prechange_20261009.tar.gz`, SHA256
`2524f0b790dda3466e99f84773d01cf52d8c2172d63bc025c5a837e2294e63b6`.

Environment: 1x NVIDIA GeForce RTX 4090 D 24 GiB, Python 3.8.20,
PyTorch 1.9.0+cu111, CUDA 11.1. LasHeR train path
`/root/autodl-tmp/TBSI_gai/data/lasher/trainingset`. Frozen 176-sequence
short-train list SHA256 `0ee927326adc60aae662dd09ff780cab1b25e4340651fdc70d3630923de4e1a0`;
new 30-sequence closed-loop list SHA256
`37ad5c642c246d49dbdf7d05bcaab8551bff0a0c75aaa92aa57b5570779c33f2`.

The first invocation of `python -u tools/preflight_dbf_g1.py` stopped before
model construction with `ModuleNotFoundError: No module named 'lib'` because
direct `tools/` execution did not add the repository root to `sys.path`.
This was corrected in the P0 script. No training, tracker, checkpoint write,
or GPU model run occurred in that failed invocation.

The corrected preflight used:

```bash
/root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/preflight_dbf_g1.py
bash tools/run_dbf_g1_train.sh --check-only
```

Both completed with exit code 0. The full JSON is
`output/experiments/DBF_G1_P0_20261009/preflight.json`. The real first
training batch supplied RGB/TIR template `[1,3,128,128]` and search
`[1,3,256,256]`; all 30 evaluation sequences have equal visible/infrared
frame and annotation counts. Each model produced `[1,768,16,16]` head
features and `[1,1,4]` finite boxes. All shared M1 backbone, head, and
readout weights matched the checkpoint exactly. Both bridge arms had exact
zero-gamma fused-feature and box parity with Legacy, and nonzero-gamma
output differences. At gamma 0.001, q/k1/v/out/gamma gradients were finite
and nonzero in both arms; Differential also had finite nonzero k2 and lambda
gradients. The optimizer assigned all G1 parameters LR 1e-4 and changed gamma
after a transient FP32 optimizer step. No updated weights were saved.

On one real crop, Legacy feature mean/RMS was `0.42145/0.78116`; raw mean
feature mean/RMS was `-0.01295/0.32165`. Initial V/D to Legacy relative
feature norms were `0.000461/0.000418`. These are one-batch numerical
diagnostics, not tracking improvement evidence. No three-arm training,
closed-loop tracking, or full test has run in this preparation step.
