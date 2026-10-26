# M1 official LCGA matched short continuation (2026-10-10)

## Locked protocol

The user explicitly prioritized testing whether local spatial modeling improves center localization over the former 10% FPS stop line. This is a **new decision**, not a retrospective pass of the previous latency gate. The two prior measurements remain 11.92% and 19.17% FPS losses. Budget is one matched 3-epoch proxy pair and one evaluation of each arm on the previously unused train-side dev30. No full training/test, parameter search, modality-only variants or dev-driven retraining is included.

Both arms start from protected M1 epoch-15 checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`. The protected config SHA256 is `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`. Control `M1_LCGA_FAITHFUL_CONTROL_seed42.yaml` SHA256 `788b9b3c859b5cb19b56fba15303a4a93f8e9b76ada7a80862de05296409973b` and candidate `M1_LCGA_FAITHFUL_PROXY_seed42.yaml` SHA256 `a9de1cdf162a1b08fa35801eeaf228f736948a16e14f5655621b477a95089d6d` differ structurally only in `SEARCH_REFINER.ENABLE`. Both use seed 42, batch 16, accumulation 8, worker 1, FP32, three epochs and the same 176-sequence RPP-v1 training list SHA256 `0ee927326adc60aae662dd09ff780cab1b25e4340651fdc70d3630923de4e1a0`. Each receives 2016 training batches / 252 optimizer steps if all three epochs complete. New dev30 SHA256 `4739caca72bb3f8d2ec6dce71a7e1abb9be4c7cac6a8ee354e584a6378bab6d2` is disjoint from train176 and old dev30 and is evaluated only after training.

The operator remains CGA official commit `8941ace6065fdb9bddb7b56e6c00152f67defd78`; source snapshot before launch `output/experiments/M1_LCGA_FAITHFUL_SHORT_20261010/provenance_source_before_launch.tar.gz` SHA256 `aafb1b64ea1d8eb9b82428af6422e03ed7f552a01908608569302d098e8f5e66`, Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`. Machine: 1x NVIDIA GeForce RTX 4090 D 24 GiB, Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1.

Primary mechanism screen: paired per-sequence AUC and median center-error changes, center-error win count, AUC median, win/loss count, and AUC mean after removing the largest-gain sequence. PR/NPR/OP50/OP75 and FPS are supporting metrics. A gain confined to one or two sequences, center error not improving across sequences, or intervals spanning zero will **not** justify full training. Joint training means a positive result cannot isolate local representation from head adaptation; response-peak position and target-background margin require a separate trace and must not be inferred from boxes.

Launcher: `bash tools/run_lcga_faithful_short_20261010.sh`. It trains and evaluates the control, then trains and evaluates the candidate, then writes `output/experiments/M1_LCGA_FAITHFUL_SHORT_20261010/comparison.json`. Run IDs 813/814 and all output directories are independent. Preflight passed at 2026-10-10 10:27 +08 with 9.8 GiB free and no competing experiment process.

## Historical proxy cleanup

At user request, only the two old `LCGA-style-unmasked-shift` A0/A1 epoch-3 checkpoint files were removed after SHA256 verification: A0 `98de0c2ab2c3ed21ddc48629cef4beaf2562b37198052c7f48e9b843d3fc99d2`, A1 `584cc2ce405e360fc65ebc671805c919ffaa31fcea9c9ae557942e3180943696`. This released 2,700,379,174 bytes. Reports, configs, logs, archived predictions, metrics and attribution JSON remain; old checkpoint-dependent replays are no longer possible without retraining. The protected M1 assets were not changed.

## Results and decision

The serial launcher finished `ALL_FAITHFUL_SHORT_COMPLETE` at 2026-10-10 11:50 +08. Both arms completed three epochs (2016 batches and 252 optimizer updates per arm), and all 30 new-dev sequences were evaluated once per arm. Control checkpoint SHA256 `d24faa0fa97c9f7f83c12ac6641c9091c6e0cd699a3bb87a174f41401e5a1920`; candidate SHA256 `f628599219d954e4b9f4b57263b04c03baa5f2f0e49c95a88f39018ddbae97ff`. Configs, protected M1 files and new-dev manifest hashes remained unchanged. Actual GPU active hours were not independently measured (NA); wall-clock training was about 34 minutes per arm, plus approximately 6-7 minutes evaluation per arm.

| New dev30 canonical metric | Control | Official-LCGA candidate | Candidate minus control |
|---|---:|---:|---:|
| AUC | 74.0419 | 72.9775 | -1.0643 |
| PR | 91.9442 | 90.5521 | -1.3921 |
| NPR | 88.2979 | 86.8891 | -1.4089 |
| OP50 | 89.8970 | 88.4513 | -1.4457 |
| OP75 | 70.4296 | 69.5584 | -0.8712 |
| Tracker FPS | 72.8714 | 60.1457 | -17.46% |

Paired AUC: 17 gains, 13 losses, median delta +0.0433, mean delta -1.0643, sequence bootstrap 95% interval [-3.9224, +1.1858]. Removing the single largest gain (`boyputtinghandup`, +15.7867) worsens the mean to -1.6454. The two largest losses are `leftwhiteblack` -33.7456 and `girlbikeinlight` -16.4972; excluding them would reverse the mean, which demonstrates outlier sensitivity rather than providing a valid alternative headline. Median center error improved in 15/30 sequences; median *sequence* delta was -0.0051 px, effectively zero. Across 12,547 paired valid frames, median center error was 4.1231 px control versus 5.3852 px candidate. `leftwhiteblack` median center error worsened by 243.09 px. Canonical per-sequence metrics and the independent paired script agree to rounding.

The trained candidate's outer RGB/TIR gates were -0.0002750 / +0.0002719. Its four internal curvature-alpha values were +0.0002517, -0.0003899, +0.0000706, -0.0000238. On the *old* dev's first three sequences, a same-checkpoint intervention zeroing only those internal curvature alphas changed many floating-point box coordinates by at most 0.000794 px but changed no sequence AUC. This is a narrow immediate-effect check, not a head-only or training-trajectory control. It indicates the actual curvature guidance learned in this short run was too weak to account for the new-dev changes; the comparison tests adding the full official-operator branch with joint continuation, **not an isolated curvature benefit**.

**Decision: stop this candidate; no full training/test.** The preregistered primary center-localization signal is absent across sequences, AUC uncertainty spans zero with severe outlier sensitivity, and the FPS cost remains material even under the relaxed speed budget. Some individual sequences improve, so neither local spatial modeling in general nor the paper's curvature mechanism is disproved. This short, single-seed joint continuation also cannot separate feature changes from CENTER-head adaptation or training trajectory. A future attempt needs a new method identity and an explicit mechanism for activating curvature; do not retune this checkpoint on the already-used dev30.

Evidence: `output/experiments/M1_LCGA_FAITHFUL_SHORT_20261010/logs/serial.log`, `metrics/`, `comparison.json` SHA256 `66f315432ebdc8417b7e754788c4c1a0ac927f53cf07f73174063c3238af6a9c`, and `curvature_old_dev3.json` SHA256 `6c2fa212cbb35b7c06c2bb93f0abc8134a0e914708726c6ced4851b2901261b9`. Post-launch curvature diagnostic source `tools/diagnose_lcga_faithful_curvature_20261010.py` SHA256 `14e59cc9cd9ebb409802d53f13b81a0e798ae04ff8d57c2f27a6e05fce906853` (not part of the before-launch snapshot). The curvature diagnostic command was:

```bash
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python tools/diagnose_lcga_faithful_curvature_20261010.py
```
