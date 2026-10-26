# DBF Post-Backbone Replacement Short Screen (2026-10-09)

## Decision

Stop before perturbation screening and full training. The Differential arm is
below both the matched Vanilla bridge and Legacy fusion on the frozen 60
sequence proxy. This is an exploratory, single-seed proxy result, not a full
LasHeR result and not a comparison against the paper's original IGF.

| Arm | AUC | PR | NPR | OP50 | OP75 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Legacy | 52.4663 | 67.3811 | 62.1222 | 63.9471 | 41.6184 | 73.73 |
| Mean | 49.5412 | 63.4810 | 58.4137 | 60.5221 | 38.7916 | 76.06 |
| Vanilla | 50.5768 | 64.5586 | 59.5616 | 61.6199 | 40.1012 | 73.06 |
| Differential | 49.0681 | 62.8618 | 57.7138 | 59.8113 | 38.7757 | 70.85 |

Differential minus Vanilla is -1.5087 AUC, -1.6968 PR, -1.8478 NPR,
-1.8085 OP50, and -1.3256 OP75. Differential minus Legacy is -3.3982 AUC,
-4.5193 PR, -4.4083 NPR, -4.1358 OP50, and -2.8427 OP75. The proxy gate
therefore fails; the pre-registered perturbation screen was not run.

## Run Contract

- Parent: protected M1 epoch 15, config SHA256
  `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`,
  checkpoint SHA256
  `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Four matched arms used explicit epoch-3 checkpoints, FP32 inference, fixed
  seed 42, frozen RPP-v1 train proxy (176 sequences) and test proxy (60
  sequences). Train sequence manifest SHA256:
  `0ee927326adc60aae662dd09ff780cab1b25e4340651fdc70d3630923de4e1a0`; test
  sequence manifest SHA256:
  `b1a91b0f806da7119940bedea9827c7a54657ddd1acc4964b7808d7d921eb46d`.
- Hardware/software: 1x NVIDIA GeForce RTX 4090 D 24 GiB; Python 3.8.20;
  PyTorch 1.9.0+cu111; CUDA 11.1; `TBSI_INFER_FP16=0`.
- Training command: `bash tools/run_dbf_igf_short_train.sh`; four independent
  configs, 3 epochs each, `NUM_WORKER=1`, AMP off, one training run at a time.
- The protected M1 YAML diff was empty after the experiment. Its recorded
  config and checkpoint SHA256 values were rechecked before the runs.
- Code base: Git `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` plus working-tree
  changes. Relevant source, scripts, configs, and manifests are preserved in
  `experiments/snapshots/dbf_igf_replacement_20261009.tar.gz`.
- Clean evaluator command: `bash tools/run_dbf_igf_short_eval.sh`; these four
  gate runs used `--threads 0`, one evaluator at a time. Two-process shard
  outputs are retained as diagnostic-only `parallel_probe` artifacts and are
  excluded from the comparison because prior scheduling changed predictions.
- The proxy run completed 3 short-training epochs per arm, then 60/60 clean
  tracking sequences per arm. The exact aggregate and per-sequence output is
  `output/experiments/dbf_igf_replacement_short_eval_20261009/comparison.json`.

## Thread Probe

After the clean runs, Differential `bike2left` was replayed from the same
epoch-3 checkpoint using `--threads 0` (run 604) and `--threads 1` (run 605),
with `TBSI_DIAGNOSTIC_MAX_SEQUENCES=1`. Prediction files compare byte-for-byte
equal; tracker FPS was 69.83 and 70.81 respectively. This verifies sampled
output consistency only; one sequence does not establish a full-dataset speed
gain. The runner now accepts `DBF_IGF_TEST_THREADS=0|1`, defaults to 0, and
rejects the former two-shard parallel mode. Do not exceed one evaluator worker
under the repository's global process limit.

## Config And Checkpoint Hashes

| Arm | Config SHA256 | Epoch-3 checkpoint SHA256 |
|---|---|---|
| Legacy | `eca5a1243c1bb34af21c6ef0e9ab386a261c25346574e267f2f0566f69b56952` | `0fe4171003c27c628628ae09c663cbb7a0b2f40cb4e96e394f487173b64bdfa5` |
| Mean | `1edc55171c16acfd131938bfec0f2f1b5ce0373f8eeb06c733ceebcfad262032` | `02615162c14efda0f9e4b09f72fc7b83bb5bc9918e9d610b737db7a4dffaa128` |
| Vanilla | `0105b7a0909533a7709f38590eff3e37feefaeb54b9c4af2dd0d8513ab5e0e81` | `673851ea2238491fd80c654ac1197673499fd32e189f827b777273ecfcb527bc` |
| Differential | `0e8e3bc5919101e93fa3e1cdafb2a3c60968cc7a2915ec64bcb4eb158ab17453` | `f50886904730e54843957bfaa51ad56ec0b1994147f4ed8d13ea41c0154f04d8` |

Full-sequence predictions and timing files are in
`output/test/tracking_results/tbsi_track/DBF_IGF_REPLACE_SHORT_*_seed42_{400,401,402,403}`.
