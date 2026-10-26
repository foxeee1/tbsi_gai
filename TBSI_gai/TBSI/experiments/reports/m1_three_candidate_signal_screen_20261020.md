# M1 三候选模块初步筛选（2026-10-20）

## Protocol

Using `M1_PRECISION_SCREEN_MANIFEST_20261020`: 20 persistent low-AUC/low-IoU failure events and 5 high-AUC controls. M1 epoch15, exact template initialization, saved `state_before`, crop and FP32 forward were frozen. No candidate weights were trained and no candidate output was injected into `tbsi_fuse_search`; this is a signal-availability screen, not a module performance experiment.

The script computes three paper-derived proxies at the existing RGB/TIR search feature boundary:

- `CMX`: mean RGB/TIR template similarity minus cross-modal disagreement, representing cross-modal feature rectification.
- `FiLM`: mean RGB/TIR template similarity, representing template-conditioned modulation strength.
- `MMTM`: channel-only cross-modal statistics, represented as a spatially constant modulation map. It must be AUC `0.5` if no spatial information is available.

The proxies deliberately answer a narrow question: does the current M1 feature contain a target-selective signal that the candidate could plausibly modulate? They do not reproduce the original papers' full architectures or losses.

## Results

| Candidate | Group | Events | Mean signal AUC | Mean target-background margin | Mean abs target/background ratio | AUC above chance |
|---|---|---:|---:|---:|---:|---:|
| CMX-style rectification | 20 failures | 20 | 0.118 | -0.2243 | 0.463 | 0/20 |
| CMX-style rectification | 5 controls | 5 | 0.008 | -0.4231 | 0.258 | 0/5 |
| FiLM-style template condition | 20 failures | 20 | 0.160 | -0.1119 | 0.814 | 0/20 |
| FiLM-style template condition | 5 controls | 5 | 0.021 | -0.2637 | 0.613 | 0/5 |
| MMTM-style channel reliability | 20 failures | 20 | 0.500 | 0.0000 | 1.000 | 0/20 |
| MMTM-style channel reliability | 5 controls | 5 | 0.500 | 0.0000 | 1.000 | 0/5 |

All values are exploratory signal scores, not tracking metrics. The very low AUC values mean the candidate score ranks GT cells below background cells on these frozen M1 features. MMTM's `0.5` is a control result: channel-only modulation has no spatial target/background evidence by construction.

## Screening decision

1. **MMTM: reject as a primary candidate.** It can test global modality quality, but cannot address M1's spatial peak competition by itself. Keep only as a cheap channel-only control.
2. **CMX: do not train yet.** Its cross-modal disagreement proxy is strongly background-favoring (`AUC 0.118`, target/background ratio `0.463`). A trained rectifier could in principle learn a different mapping, but the current input signal does not support a simple no-label transfer. It needs an explicit target-conditioned signal or hard-negative supervision before implementation.
3. **FiLM: closest of the three, but not admitted.** Template similarity is less poor than CMX, yet still below chance (`AUC 0.160`) and target/background ratio is below one. A template affine modulation alone is therefore unlikely to fix the current problem; it becomes viable only if the template condition is changed from global mean to a foreground token/target-context representation and its directional loss is rechecked.

The common result is more important than the ranking: **the M1 search features plus the current template representation do not expose a usable target-selective signal to these simple candidate proxies.** This agrees with the LCGM audit, where the residual direction was loss-increasing in the GT region. It argues against immediately training all three modules and against increasing modulation strength.

## Recommended next step

Run one additional, still training-free diagnostic before architecture training: compute token-level template-to-search matching using an explicit initial-template foreground mask, and compare it with the all-token/global-mean scores on the same 25 events. If foreground-token matching remains below chance, stop replacing AMCM modules and investigate template representation/crop state. If it becomes above chance, only then implement a minimal FiLM or CMX rectifier using that signal, with one variable changed and the same 25-event gradient contract.

## Provenance

- Selector: `tools/select_m1_precision_screen_sequences_20261020.py`.
- Candidate screen: `tools/screen_m1_nontracking_candidates_20261020.py`, SHA256 `7acb202d8819288e97f2d6b0cb11eb721fb6e51b0bceaebbee1ebb90264dc9d0`.
- Output: `output/experiments/M1_NONTRACKING_CANDIDATE_SCREEN_20261020/{per_event_candidate.csv,summary.csv,protocol.json}`.
- Candidate screen SHA256 is recorded with the output; no model, tracker, M1 config, protected checkpoint, or tracker state code was modified.

## Expanded fourteen-candidate screen

The same 25 fixed events were used to screen eleven additional non-tracking candidates. The screen remains parameter-free: it does not reproduce the papers' trainable networks, losses, or data pipelines, and it does not inject any candidate output into `tbsi_fuse_search`.

| Candidate | Failure AUC | Control AUC | Failure-control gap | Failure target/background ratio | Interpretation |
|---|---:|---:|---:|---:|---|
| CBAM | 0.886 | 0.994 | -0.108 | 1.086 | Strong spatial saliency signal, but stronger on controls; probe only |
| SwinFusion | 0.401 | 0.248 | +0.153 | 0.974 | Best absolute score among context candidates, still below chance |
| Focal Modulation | 0.388 | 0.210 | +0.178 | 0.960 | Largest failure/control separation, still below chance |
| SKNet | 0.372 | 0.250 | +0.122 | 0.956 | Scale-selection probe, below chance |
| Dynamic Convolution | 0.333 | 0.170 | +0.163 | 0.911 | Dynamic local operator, below chance |
| TokenFusion | 0.256 | 0.060 | +0.196 | 0.865 | Failure/control gap without target ranking |
| GMU | 0.256 | 0.060 | +0.196 | 0.865 | Same proxy as TokenFusion; no independent evidence yet |
| SeAFusion | 0.243 | 0.054 | +0.189 | 0.857 | Modality interaction signal remains background-biased |
| CrossViT | 0.164 | 0.023 | +0.141 | 0.916 | Cross-scale token proxy below chance |
| FiLM | 0.160 | 0.021 | +0.140 | 0.814 | Template affine condition below chance |
| DIDFuse | 0.154 | 0.018 | +0.135 | 0.806 | Decomposition proxy below chance |
| CMX | 0.118 | 0.008 | +0.110 | 0.463 | Cross-modal rectification is strongly background-biased |
| CDDFuse | 0.118 | 0.008 | +0.109 | 0.521 | Decomposition/fusion proxy is background-biased |
| MMTM | 0.500 | 0.500 | 0 | 1.000 | Spatially constant channel control, not a localization signal |

### Priority decision

1. **CBAM-style spatial probe: first priority for a contract audit.** It is the only candidate with above-chance target/background ordering on every fixed event. The control-group score is even higher, so this is evidence that the input contains generic spatial saliency, not evidence that CBAM will repair M1 failures. The next test must check whether a target-conditioned version preserves the signal specifically on failure frames and whether its residual direction lowers the tracking loss.
2. **Focal Modulation: second priority.** It has the largest failure/control gap (`+0.178`) and is a plausible test of whether multi-scale context helps the crop-local ambiguity. Its absolute failure AUC (`0.388`) is still below chance, so it is admitted only to the same identity, gradient, and fusion-transfer contract audit.
3. **SwinFusion: third priority.** It has the highest absolute failure AUC among the non-CBAM context candidates (`0.401`) and a positive failure/control gap. It remains below chance and therefore does not justify training before the contract audit.
4. **SKNet and Dynamic Convolution: reserve scale/operator controls.** Their signals are weaker than Focal Modulation and SwinFusion. Use one at a time only if the preceding audit separates a scale-selection bottleneck from a spatial-saliency bottleneck.
5. **TokenFusion/GMU/SeAFusion: defer.** Their subgroup gaps are not accompanied by target ranking, and TokenFusion and GMU produce identical proxies in this screen. They do not currently provide a discriminating next experiment.
6. **CrossViT, FiLM, DIDFuse, CMX, and CDDFuse: do not train under the current interface.** Their frozen proxies rank background above GT on essentially all events. A different foreground-token representation or an explicit target-conditioned loss would be required before reconsidering them.
7. **MMTM: retain only as a channel-only negative control.** Its 0.5 AUC is the expected spatially constant result and cannot address peak competition.

**Admission rule:** no candidate is promoted to a performance experiment from this proxy screen alone. For CBAM, Focal Modulation, and SwinFusion, run the same fixed-event contract checks: zero-gate identity, shape/state parity, target-versus-background residual map, loss-gradient directional derivative, and finite-difference transfer through `tbsi_fuse_search`. Only a candidate with target-selective change, improved peak margin/GT rank, and matching loss direction should enter a short training run.
