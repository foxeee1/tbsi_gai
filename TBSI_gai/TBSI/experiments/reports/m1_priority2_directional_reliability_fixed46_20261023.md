# M1 Priority 2 Direction-Separated Reliability: Fixed-State Result

Date: 2026-10-23  
Decision: `MECHANISM-NOT-SUPPORTED / NO-TRAINING`

## Question and scope

Test whether modality-specific foreground-template evidence should scale only its own outgoing MSHA cross-modal Value message, and whether this independent mapping is better than shared or swapped evidence on identical fixed inputs. This was a no-training mechanism probe on the already exposed LasHeR train-side calibration cohort: 23 sequences, one pre-onset and one severe-onset state per sequence (46 states). It is not a fresh generalization cohort and does not estimate full-sequence tracking performance.

Parent was protected M1 `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15. Protected YAML and checkpoint hashes were verified unchanged after the runs: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1` and `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.

## Mechanism chain

At block02, each modality independently matched its initialized foreground template tokens against search tokens, selected its own peak and strongest competitor outside the peak's 3x3 neighborhood, and converted the peak-minus-competitor margin to a bounded reliability scalar. Four fixed-input arms were captured: parent/off, independent source-to-direction mapping, shared scalar control, and swapped mapping. No optimizer or checkpoint writes were used.

## Evidence

The local message-scaling contract worked. Observed directed Value-message norm ratios matched the specified gates to at most `1.9e-7` absolute error. Both directions changed in all 46 states for each intervention arm. Mean RGB->TIR / TIR->RGB norm ratios were `1.1446 / 1.1381` for independent, `1.1413 / 1.1413` for shared, and `1.1381 / 1.1446` for swapped.

The evidence itself was not a reliable target indicator. RGB candidate peaks fell inside the GT region in 22/46 states and its evidence-margin AUC was `0.608`. TIR candidate peaks fell inside GT in 17/46 states and its margin AUC was `0.440`. These are weak and modality-asymmetric signals on an exposed calibration set.

No downstream criterion favored independent mapping. Mean fixed-state IoU was `0.46917` for parent, `0.47068` independent, `0.46937` shared, and `0.46950` swapped. Paired mean deltas vs parent were `+0.00151`, `+0.00020`, and `+0.00033`; independent had 20 wins, 4 ties, and 22 losses. CENTER top1-in-GT was 33/46 in all arms. Independent had slightly worse mean GT rank (`+0.0217`, where positive is worse) and slightly lower mean CENTER margin (`-0.00115`). Its output box shift averaged 1.34 px L1, but movement did not produce consistent localization improvement.

Parent trace-vs-fixed-replay IoU differed by at most `4.46e-4`. The captured artifacts do not provide a direct max score-map and decoded-box parity check at the preregistered `1e-4` tolerance, so that strict disabled-parity criterion is unverified. This gap does not rescue the candidate: the evidence-to-target alignment and independent-vs-control criteria already fail.

## Judgment

The intended directional mapping is implemented and its message-level arithmetic is verified. The proposed mechanism is not supported because the per-modality margin is not sufficiently target-specific and independent mapping does not beat shared or swapped controls on downstream fixed-state readouts. Stop this operator: no training, coefficient sweep, or architecture expansion is admitted from these results.

This result does not establish that all modality-specific reliability methods are ineffective. It only rejects this foreground-token LSE peak/competitor scalar under the current M1 block02 interface and calibration states. It is not a LasHeR or RGBT234 tracking gain or regression claim.

## Provenance

- Preregistration: `experiments/protocols/m1_priority2_directional_reliability_20261023/preregistration.md`
- Analysis: `tools/analyze_m1_priority2_directional_reliability_20261023.py` (SHA256 `3f5fa24f380a5a1a3143b583a4842eb3c3d194bf72928d53c4cc3298ee2c3faa`)
- Summary and per-state evidence: `output/experiments/M1_PRIORITY2_DIRECTIONAL_RELIABILITY_ANALYSIS_20261023/{summary.json,evidence_per_state.csv}`
- Arms: `output/experiments/M1_PRIORITY2_DIRECTIONAL_RELIABILITY_{PARENT,INDEPENDENT,SHARED,SWAPPED}_CAPTURE_FIXED46_20261023/`
- Pre-change source snapshot SHA256: `718761128ed8edfb0dff3560531da4bb17e8f245ae05b43190a25c89ea9c5fe2`
- Environment: 1x NVIDIA GeForce RTX 4090D; FP32 serial capture, one Torch thread; analysis on CPU. No training, no new checkpoint, AUC/PR/NPR/OP/FPS not applicable.
