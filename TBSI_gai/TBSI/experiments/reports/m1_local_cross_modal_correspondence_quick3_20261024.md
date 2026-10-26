# M1 Local Cross-Modal Correspondence Quick3

## Decision

**PROXY-COMPLETE / NEGATIVE-POINT-ESTIMATE / CI-CROSSES-ZERO / STOP-CURRENT-OPERATOR.** The candidate lowered all five macro metrics relative to its matched 3-epoch control on the locked RPP-v2 60-sequence proxy. The AUC point estimate was -1.113 percentage points, but its sequence-bootstrap 95% interval crossed zero. Effects varied sharply by sequence, so this run does not establish a robust population-level decline or a target-specific mechanism. Do not promote this operator to full training from this result.

## Question and Operator

The candidate adds a bidirectional local correspondence exchange to the two search-token streams at backbone block 2, before later TBSI fusion. For each direction, normalized query features attend over a valid 3x3 neighborhood in the other modality. The resulting local context is combined with foreground-pooled template descriptors to estimate per-token relevance, then added as a bounded residual (`max_delta=0.05`). Hidden size is 96 and relevance hidden size is 64. The control uses the same parent, training schedule and data with this operator disabled.

The hypothesis was that target-conditioned, spatially local cross-modal evidence would change search features before fusion and improve localization/closed-loop tracking. The ablation changes only the operator enable switch between the paired arms.

## Protocol

- Parent: Early-conditioned MSHA gate joint-training epoch 15 checkpoint, SHA256 `e38d18fe50c2895e75e4b9392bf22d9323bb8e6ad31ccbe39ad8c038e3fc1811`.
- Training: LasHeR RPP-v1 fixed 176-sequence manifest (SHA256 `410f7197a6d6a9d6e1886c0511e93d5a4ee1b4729fe77541e9d911aeaa82d8c3`), 10,112 samples per epoch, 3 continuation epochs, seed 42, 237 optimizer updates per arm. Candidate and control used matched settings; candidate trained jointly with the parent model under the same freeze contract.
- Evaluation: the same locked RPP-v2 60-sequence manifest (SHA256 `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8`), 97,908 frames, FP32, serial `threads=0`, template update off. Both arms completed 60/60 sequences.
- Environment: 1x NVIDIA GeForce RTX 4090 D; Python 3.8.20; PyTorch 1.9.0+cu111; CUDA 11.1.
- Training wall times: control 7m29s; candidate 7m24s. Test wall times: control 54m35s; candidate 58m30s. GPU telemetry was not captured, so GPU-hours are `NA`.
- A zero-output preflight passed: module and full-model identity max difference were 0; output projection gradient L1 was 0.140122 and multi-step upstream gradient L1 was 1.5591e-5. The trained candidate checkpoint contains nonzero output projections, confirming that the added branch updated.

## Results

| Metric | Control | Candidate | Candidate - control | 95% sequence bootstrap CI |
|---|---:|---:|---:|---:|
| AUC / Success | 51.086 | 49.972 | -1.113 | [-3.538, +0.564] |
| Precision | 62.503 | 60.828 | -1.674 | [-4.874, +0.636] |
| Normalized Precision | 59.475 | 57.805 | -1.670 | [-4.757, +0.444] |
| OP50 | 60.582 | 59.278 | -1.304 | [-4.326, +0.734] |
| OP75 | 42.543 | 41.731 | -0.812 | [-3.102, +0.565] |

Sequence AUC deltas: 26 wins, 34 losses; mean -1.113, median -0.073. Sequence mean-IoU delta was -0.01131 (95% CI [-0.03600, +0.00573], 28 wins, 32 losses). All five benchmark-metric bootstrap intervals cross zero. The point estimates lean negative, but the proxy does not resolve a stable population effect.

Sequence effects are heterogeneous. The largest AUC declines were `3rdfatboy` (-58.121 points), `shinybikeboy2left` (-13.376), and `carcomingfromlight` (-8.428). The largest gains were `middrillmaster` (+19.464), `leftblkTboy` (+3.292), and `leftmirrorside` (+3.178). These metric changes identify sequences for diagnosis; they do not establish visual causes or mechanism attribution.

Aggregate throughput from recorded per-frame timings was 60.86 FPS for control and 53.49 FPS for candidate, a 12.1% reduction. Thus this operator did not buy an aggregate accuracy improvement and added measurable inference cost in this configuration.

## Interpretation and Next Step

The experiment answers the immediate screening question: the tested local 3x3 exchange is not a promotion candidate under the current train/eval contract. It does not prove that cross-modal spatial interaction is generally ineffective. The negative mean is influenced by a few large sequence changes, confidence intervals cross zero, and the run has one seed, three continuation epochs, and a 60-sequence proxy rather than full LasHeR evaluation.

The informative next step is a bounded, no-training paired replay on the most harmed and helped sequences, using identical saved crops and tracker states. Compare target-versus-shifted correspondence, per-frame feature residual magnitude, CENTER ranking, decoded box, and first state divergence. This can determine whether the branch changes target-specific localization or injects non-target local evidence. Do not tune `max_delta` or train another variant until that distinction is measured. If replay cannot separate those explanations, stop this operator family rather than expanding parameter search.

## Provenance

- Run snapshot: `experiments/snapshots/m1_local_cross_modal_correspondence_20261024/` (`provenance.txt`, `preflight.json`, `prechange_sources.tar.gz`, locked manifests).
- Configs: `experiments/tbsi_track/M1_LOCAL_CROSS_MODAL_CORRESPONDENCE_CONTROL_QUICK3_20261024.yaml` SHA256 `65c21b38516bb7aec9c4d2d8c0fe4583c2dcaefb7926eadc025dd592c2e1ef45`; candidate config SHA256 `1f8bef2074ab68fc7d8d2c573fe81580fae971389a503da24f2f26c1edc7b7e5`.
- Operator source SHA256 `599d8197060528c5a6b41833e4f82f108739fcc48069c4c3a5d352b6466410ab`; tracker integration source hashes are recorded in `provenance.txt`.
- Control checkpoint SHA256 `b139979420fd255f02ba5843ee80f804f9adabaf5134c350e8d8a6c6afd45a01`; candidate checkpoint SHA256 `4827fc86e537c620174ff460060e60c7ccd749a52fab4afd9ce4a9ed317d17da`.
- Paired summary and per-sequence rows: `output/experiments/M1_LOCAL_CROSS_MODAL_CORRESPONDENCE_QUICK3_20261024/paired_rpp60_summary.json` and `paired_rpp60_per_sequence.csv`.
- Control/candidate tracker outputs and complete logs are retained in their independent experiment directories.
