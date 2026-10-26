# M1 2025–2026 非跟踪模块 AMCM 接口初筛（2026-10-20）

## Protocol

- Selector: `tools/select_m1_precision_screen_sequences_20261020.py` and its fixed manifest: 20 M1 failure events + 5 high-AUC controls.
- Parent: frozen M1 epoch15 checkpoint, exact template/state/crop, FP32.
- Interface: frozen backbone output is unpacked as RGB/TIR search maps `[B,C,16,16]` and template tokens `[B,64,C]`. Each candidate computes a parameter-free mechanism proxy with the same spatial output `[B,1,16,16]`, which is the shape an AMCM residual route would consume.
- No candidate was inserted into `tbsi_fuse_search`, no weights were trained, and no loss or tracker state was changed. Therefore the result is an end-to-end signal-availability screen, not a tracking performance result.

## Results

| Candidate | Failure AUC | Control AUC | Failure margin | Failure target/background ratio | Decision |
|---|---:|---:|---:|---:|---|
| N-gram Local Fusion | 0.427 | 0.264 | -0.0059 | 0.990 | Highest, still below chance; retain for one contract audit |
| CALA Local Attention | 0.337 | 0.119 | -0.0317 | 0.904 | Weak local signal; reserve |
| Edge-guided Cross-modal | 0.321 | 0.150 | -0.3401 | 0.615 | Edge proxy strongly background-biased; reject |
| Text-guided Modulation | 0.277 | 0.115 | -0.0863 | 0.887 | No target-selective signal; defer |
| TCTFusion | 0.254 | 0.085 | -0.0945 | 0.873 | Triple branch proxy below chance |
| Uncertainty Modulation | 0.246 | 0.036 | -0.0451 | 0.752 | Current perturbation proxy fails; do not train directly |
| Morphology Region | 0.179 | 0.012 | -0.0888 | 0.801 | Morphology proxy background-biased |
| Grounding Adapter | 0.205 | 0.045 | -0.1098 | 0.847 | Template condition insufficient |
| Dynamic Meta Fusion | 0.205 | 0.045 | -0.1098 | 0.847 | Same proxy as grounding adapter; no independent evidence |
| CAFF Cross-modal | 0.196 | 0.023 | -0.1552 | 0.852 | Diffusion-like smoothing suppresses target signal |
| CTGAN Cross-modal | 0.158 | 0.028 | -0.1048 | 0.776 | Attention proxy background-biased |
| Spatial-spectral Collaborative | 0.157 | 0.007 | -0.0761 | 0.723 | No useful spatial signal |
| Orthogonal Sequential | 0.136 | 0.004 | -0.1331 | 0.635 | Orthogonal residual removes target evidence under frozen features |
| Spectral-spatial Decoupled | 0.119 | 0.009 | -0.1205 | 0.470 | Strongly background-biased |

`AUC > 0.5` event counts on the failure group were: N-gram `5/20`, CALA `3/20`, Text-guided `4/20`, TCTFusion `3/20`, and all other candidates at most `2/20`; none had a positive mean target/background margin.

## Interpretation

1. **No candidate passes the promotion threshold.** The current M1 features and template tokens do not provide a reliable target-selective input for these parameter-free transfers. A below-chance proxy cannot be repaired responsibly by increasing residual strength.
2. **N-gram Local Fusion is the only candidate worth a minimal follow-up.** Its failure AUC `0.427` is closest to chance and its mean margin is nearly zero, suggesting local token aggregation may preserve more target evidence than the other proxies. It still fails the target/background ratio and must first pass identity, gradient, and fusion-transfer checks.
3. **CALA is a reserve local operator.** Its absolute score is second, but the target/background ratio remains below one. It should only be checked if N-gram reveals a usable local signal.
4. **Uncertainty modulation is not validated by this screen.** The proxy used perturbation sensitivity on frozen M1 features; its failure AUC `0.246` means that this particular uncertainty definition is background-biased. This does not prove that a trained uncertainty estimator is impossible, but it does rule out direct no-label reuse of the current input statistics.
5. **Orthogonal fusion is currently contradicted by the signal.** Removing the projection of one modality onto the other produces a strong background bias. It needs a target-conditioned orthogonal decomposition before reconsideration.
6. **Edge, morphology, diffusion smoothing, and generic template conditioning are not suitable direct replacements.** Their proxies either suppress local target evidence or enhance background structure.

## Recommended next step

Do not connect all candidates to M1 training. Admit only **N-gram Local Fusion** to a single AMCM contract diagnostic, with a token-level target condition added but no new loss: zero-gate parity, residual target/background map, loss directional derivative, and finite-difference transfer through `tbsi_fuse_search`. If it remains below chance, stop this candidate family and return to the template representation rather than searching for another operator. CALA can be the one reserve control; the other 12 candidates are rejected or deferred under the current interface.

## Provenance

- Output: `output/experiments/M1_NONTRACKING_NEW_MODULES_SCREEN_20261020/{per_event_candidate.csv,summary.csv,protocol.json}`.
- Script SHA256: `3b07768a047136332639a52cb694e04f5ef60894cce1b32cc06d53efb61477eb`.
- This screen is exploratory and does not reproduce the original papers' trainable architectures or losses.
