# M1 Early-Conditioned MSHA Gate Proxy

## Run contract

- ID: `M1-EARLY-CONDITIONED-MSHA-GATE-PROXY-20261021`
- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, explicit epoch-15 checkpoint.
- Hypothesis: early block-2 RGB/TIR template-search evidence can condition only the cross-modal Value message in TBSI-E template MHSA, reducing background propagation without changing self message, residual, search fusion, CENTER, loss, optimizer, dataset or tracker state.
- Config: `experiments/tbsi_track/M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021.yaml`.
- Tier: auxiliary/dev proxy screening; not a full-244 claim.
- Training: seed 42, 3 epochs, 512 samples/epoch, FP32, worker 4, gate-only parameters (15 tensors, 585 parameters), initialized from M1 epoch 15.
- Evaluation: after training, explicit epoch-3 checkpoint on the fixed proxy protocol available for this experiment; report paired metrics and mechanism diagnostics separately.

## Preflight

- Python compile: passed.
- Config parse: passed; AMCM disabled and early gate enabled.
- Model construction/checkpoint load: passed; only new gate keys are missing from the parent checkpoint.
- Random forward: passed; `pred_boxes [1,1,4]`, `score_map [1,1,16,16]`, `size_map [1,2,16,16]`, `offset_map [1,2,16,16]`, `backbone_feat [1,640,768]`, all finite.
- Two-step gradient/update check: passed; all 15 intended tensors had finite nonzero gradients and changed.
- Zero-gate parity: passed within FP32 numerical tolerance; maximum output difference about `1.97e-6`.
- First evaluation launch: failed before dataset construction because an obsolete proxy manifest path was supplied; no prediction files were created. The corrected frozen manifest is `output/proxy_runs/rpp-v2/proxy_test_sequences.txt` (60 sequences, SHA256 `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8`).

## Resource and provenance

- GPU: 1x NVIDIA GeForce RTX 4090 D 24 GiB.
- Python/PyTorch/CUDA: Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1.
- Git: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` (working tree contains unrelated historical changes).
- Config SHA256: `6c1281c68715159f8fef5fe8b65b7e8b30302a64273c834e408e66cb0a2c905b`.
- Parent checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Output: `output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/`.

## Acceptance and stop rules

Stop before any larger run if the gate-only contract, finite gradients, checkpoint output, or paired evaluation is incomplete. A short proxy gain is exploratory; promotion requires target/background mechanism evidence and no concentrated regression on the paired sequence set.

## Status

## Completed proxy result

- Tracking command: `TBSI_RPP_TEST_LIST=/root/autodl-tmp/TBSI_gai/TBSI/output/proxy_runs/rpp-v2/proxy_test_sequences.txt TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u tracking/test.py tbsi_track M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021 --dataset_name rpp_lasher_test --threads 0 --num_gpus 1`.
- Evaluation command: `python -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021 --dataset_name rpp_lasher_test`.
- Coverage: 60/60 sequences; 120 bbox/time files.
- Metrics: AUC `51.46`, OP50 `61.10`, OP75 `43.59`, Precision `62.71`, Norm Precision `59.82`.
- Checkpoint SHA256: `1704b42f0cf0bb8dbb598b85c55a3eec0c303516c9fe7f055d4062a540bfb8c5`.
- Gate parameter audit: all three learned `early_gate.2.weight` tensors moved from zero initialization to nonzero norms `0.1854/0.1785/0.1496`; learned `early_gate_scale` remained close to its bounded initialization (`0.1001/0.1000/0.0997`). This confirms the gate was optimized, but does not establish target selectivity.
- Interpretation: `PROXY-COMPLETE / ABSOLUTE-RESULT-ONLY`. The matching M1 parent was not tracked on this RPP-v2 manifest, so no paired gain or loss claim is made. The candidate remains eligible for a matched parent rerun or is rejected if that comparison is not worth the cost.

## Full-244 evaluation (2026-10-22)

- Full tracking command: `TBSI_CHECKPOINT_OVERRIDE=/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/checkpoints/TBSITrack_ep0003.pth.tar TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u tracking/test.py tbsi_track M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021 --runid 244 --dataset_name lasher_test --threads 0 --num_gpus 1`.
- Full evaluator: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021 --dataset_name lasher_test --runid 244`.
- Coverage: 244/244 sequences; 488 result files (244 bbox + 244 time); serial tracking completed without NaN, OOM, or restart.
- Full metrics: AUC/Success `55.99`, OP50 `68.05`, OP75 `47.19`, Precision `69.85`, Norm Precision `66.02`.
- Protected M1 reference: AUC/Success `56.21`, OP50 `68.33`, OP75 `47.13`, Precision `70.21`, Norm Precision `66.20`.
- Paired deltas (candidate - M1): AUC `-0.22`, OP50 `-0.28`, OP75 `+0.06`, Precision `-0.36`, Norm Precision `-0.18` percentage points. These are single-seed, matched full-test observations, not multi-seed claims.
- Per-sequence paired file: `output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/full244_per_sequence_m1_compare.csv`.
- Corrected per-sequence AUC outcome: candidate wins 122, loses 115, ties 7. The first generated CSV had the M1 and candidate columns reversed; the full-set aggregate row above came directly from the evaluator and was correct. The CSV has been regenerated with verified tracker ordering. Largest candidate losses are `10runone` `-65.37`, `bluegirlbiketurn` `-34.98`, `whiteridingbike` `-26.63`; largest gains are `blktribikecome` `+58.75`, `3rdfatboy` `+54.29`, `leftfarboycomingpicktheball` `+42.39`.
- Previously tracked low-AUC cases after correction: `leftunderbasket` `+1.50`, `rightbottlecomes` `+2.14`, `boyunder2baskets` `+0.02`, `whitesuvturn` `-0.03`, `ab_bikeoccluded` `-2.93` AUC points. Most changes are small on these named cases; they do not explain the full-set aggregate.
- Across the 244 paired sequence AUC deltas, mean `-0.213`, median `+0.0068`, standard deviation `8.72` points; a sequence bootstrap 95% interval for the mean is `[-1.30, +0.88]`. There are 17 losses below `-5` and 11 gains above `+5`; the 115 negative deltas sum to `-378.17`, while the 122 positive deltas sum to `+326.20` points. Thus a small number of large losses outweigh a broader set of small gains.
- The lowest M1-AUC quartile (61 sequences; M1 mean `21.89`) gains `+2.54` AUC points on average, while the next quartiles average `-1.93`, `-0.42`, and `-1.04`. The Pearson correlation between M1 AUC and candidate delta is only `-0.130`; low-baseline sequences tend to benefit slightly, but baseline AUC does not reliably identify which sequences benefit.

## Full-run interpretation

The full test does not support a stable aggregate improvement for this gate under the current contract. The paired aggregate delta is small and its sequence-bootstrap interval includes zero, while per-sequence variance is high. A temporal audit of the paired bbox files corrects the earlier first-update hypothesis: among the 17 sequences with AUC drops below `-5` points, mean IoU change is only `+0.002` over the first five tracked frames, and only one has a sustained five-frame IoU gap below `-0.25` within its first 20 frames. Several large losses diverge later: `10runone` is nearly tied through frame 20, begins a sustained gap near frame 51, then candidate IoU remains below 0.1 for 83% of tracked frames; `bluegirlbiketurn` begins the sustained gap near frame 38; `4thboywithwhite` near frame 73. This supports delayed closed-loop amplification, not an initialization-time peak failure. Per-sequence early IoU change has essentially no correlation with final AUC delta (`r=0.012`).

The candidate remains diagnostic, not promoted. The module uses three local cues (two early template-to-search maxima and RGB/TIR agreement) to predict a per-template-token, per-direction Value multiplier. Its scalar `tanh(early_gate_scale)` starts at 0.1 and the `exp(tanh(...))` map limits the initial learned modulation to roughly a 10% range; this modest perturbation can still compound through recurrent crop/state feedback. The current input contains no explicit hard-background margin, spatial peak ambiguity, prediction quality, or temporal stability signal. The training loss is the existing box/center objective and contains no direct target-vs-hard-background calibration for this gate. These are plausible limitations, not proven causes of any individual lost sequence.

The next experiment should be a **train-side closed-loop strength and bypass replay**, with one variable only: multiply the learned gate delta by a fixed coefficient `lambda` in `{0, 0.5, 1}` (where 0 is exact M1/no-op) on independent train-side sequences selected before outcomes are inspected, using the same initial state, crop protocol, and candidate checkpoint. Include a stable control cohort and a difficult cohort; record first divergence time, crop-center error, CENTER GT rank/top-1 margin, and IoU for the full sequence. This tests whether reducing exposure to the learned correction recovers drift while retaining gains. Do not add a new quality head or loss in this run. If `lambda=0.5` avoids delayed collapse while preserving the corrected difficult-sequence gains, the next module version can use a confidence-derived attenuation toward identity. If only `lambda=0` restores stability, the current learned direction is not safe to keep; return to a causal fixed-crop audit of cross-modal Value, `tbsi_fuse_search`, and CENTER before redesigning its inputs. Test-set sequence outcomes must not be used to tune lambda.
