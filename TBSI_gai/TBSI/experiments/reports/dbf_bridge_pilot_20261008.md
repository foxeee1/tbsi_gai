# DBF Bridge Pilot: P0 and Stage Gates (2026-10-08)

## Scope

This record continues the requested DBF pilot against the protected M1 epoch-15 checkpoint. It covers the model-level P0 run and documents why the short training and failure-segment rounds have not started. The canonical 3090 baseline YAML was not changed; its Git diff is empty.

Parent: `TBSI_E_MHSA_FULL_4090D_seed42`, seed 42, checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.

The source tree already contained uncommitted research changes before this continuation. The P0 runner/probe edits are therefore recorded as a working-tree delta on top of Git commit `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; the pre-edit source snapshot is `experiments/snapshots/dbf_pilot_prechange_20261008.tar.gz`, SHA256 `d599487252431f4e1f5cac30177312fe3521eab74531aa2f2cc5f7912c520eda`.

## P0 Result

Run `DBF_P0_20261008_R5` used Python 3.8.20, PyTorch 1.9.0+cu111, CUDA runtime 11.1, one NVIDIA GeForce RTX 4090 D (24564 MiB), and FP32 inference semantics. Inputs were synthetic tracker tensors, not LasHeR frames. The DBF call contract was query `[1,256,768]` and context `[1,320,768]`, at TBSI location 9.

| Arm | Exact max output delta vs M1 | Median model forward | Peak allocated |
|---|---:|---:|---:|
| M1 | 0 | 12.49 ms | 482.0 MiB |
| DBF gamma=0 | 0 | 13.38 ms | 491.6 MiB |
| DBF gamma=1e-3 | backbone 7.73e-4; boxes 8.29e-6; score map 2.09e-5 | 13.91 ms | 491.6 MiB |

All outputs were finite. The DBF adds 1,972,740 parameters, increasing the whole model from 110,229,381 to 112,202,121 parameters. Gamma zero preserves exact outputs. On a synthetic training-mode loss backward, gamma has nonzero gradients at gamma=0 while q/k/v projection gradients are zero on the first backward; gamma=1e-3 produces nonzero q/k/v gradients. This is a one-backward mechanism check, not an optimizer-step or convergence result. The measured model-forward median latency overhead for gamma=1e-3 relative to M1 is about 11.4%; this is not end-to-end tracker FPS.

Evidence: `output/experiments/DBF_P0_20261008_R5/p0_evidence.json` (SHA256 `a7061af152325c669179af110496478d136236a444e6bde7881ec1cffef40861`). Failed R1-R4 logs are retained; R5 corrected import path, dual-modality input shape, and result-shape serialization issues.

## Requested Later Rounds

The requested short comparison arms are M1 continuation, Vanilla Cross, Additive Dual, and DBF for 2-3 epochs. They have not run. The repository's RPP-v1 rule requires a qualifying historical anchor backtest before proxy-based candidate screening; no admission artifact meeting that rule was found. Its training protocol is 15 epochs, and the only explicit 3-epoch exception is the separately scoped CTVM diagnostic. The DBF pilot does not qualify for that exception. No alternate small-data sampling strategy is allowed.

The requested failure-segment and modality perturbation round has also not run. M1's archived 244-sequence failure map includes 430 persistent IoU<0.5 segments, but it lacks per-frame crop transforms. There is no trained DBF checkpoint to compare on fixed states/crops. Running an untrained gamma=1e-3 model on selected test failures would not test the learned DBF claim and would improperly use test failures for candidate screening.

Consequently, there is no evidence yet that DBF mitigates RGB failure, TIR failure, hot-background distractors, local misalignment, or modality conflict. P0 only establishes interface correctness, exact zero-gamma parity, measurable nonzero-gamma behavior, resource overhead, and initial gradient behavior.

## Gate to Resume

Before P2, complete and archive the required RPP-v1 historical anchor backtest and serial/parallel consistency audit under `AGENTS.md` section 7.2. Then use the frozen manifests, two registered seeds, independent configs/output directories, matched parent checkpoint and training controls. P3 requires the resulting trained checkpoints and a predeclared fixed event/crop replay manifest that does not select events by candidate outcomes. Keep the result labeled diagnostic until those steps are complete.

## Continuation Update (2026-10-08)

Following the user's explicit instruction to proceed with an exploratory short run, four matched-seed arms completed 3 epochs on the frozen RPP-v1 176-sequence train manifest: M1 continuation, Vanilla, Additive, and Differential. Their epoch-3 proxy validation mean IoU was 0.66906, 0.66882, 0.66870, and 0.66842, respectively. These are training-validation diagnostics only; they are not tracking metrics or evidence of a robust DBF benefit. Checkpoints and per-run evidence remain in their independent output directories and are registered as `TRAINING-ONLY`.

Clean proxy tracking was interrupted after 8/60 sequences for the M1 continuation arm at run ID 300. Those sequence outputs are retained and the serial runner now skips a sequence only when both its prediction and timing files exist. A one-sequence smoke run for `bike2left` completed with run ID 399 in 10.5 seconds at 71.09 FPS; it loaded the explicit epoch-3 checkpoint and wrote 257 prediction and 257 timing rows. The full clean and perturbation evaluations remain pending. The runner uses `--threads 0` and executes one sequence at a time; it does not overwrite the retained run-300 files.

The 3-epoch results are exploratory and do not establish full-training performance, candidate admission, or failure-mode mitigation. The prior gate record is retained as historical context; the user-authorized continuation is tracked separately in the asset registry.
