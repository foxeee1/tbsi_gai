# M1/MSHA feature operator: 2025-2026 literature scout

## Research question

Can the next M1-derived contribution move beyond a scalar decision gate and change the target-bearing template representation through a more expressive, still testable cross-modal operator?

## M1 framing

M1's archived failure analysis describes a system-level pattern: many sustained severe failures begin while the target is still inside the approximate search crop, then later crop feedback can push it out. This does not establish MSHA as the first cause. The current Early-Conditioned Gate scales per-token cross-modal Value messages, so it can alter downstream features, but it does not rewrite stored template tokens or rotate the channel content of a message; its main action is a bounded scalar amplitude per token and direction.

Existing M1 evidence argues against another amplitude-only or local-confidence-only expansion: the strength replay's aggregate gain was dominated by one sequence; a learned local-margin correction regressed against gate-only; direct local margin barely changed CENTER rank; and native direction amplitude controls did not yield stable localization benefit. Those results do not reject feature transformation or token correspondence mechanisms, which were not the same intervention.

## Recent methods worth borrowing from

| Work | Date/status | Mechanism relevant to M1 | Transfer judgment |
|---|---|---|---|
| [BTMTrack: Robust RGB-T Tracking via Dual-template Bridging and Temporal-Modal Candidate Elimination](https://arxiv.org/abs/2501.03616) | arXiv 2025-01; verify venue/status from the paper before claiming publication | Filters target-relevant tokens using temporal and modal correlation, then bridges dual templates with dynamically filtered cross-modal interaction | Closest task match. Borrow the idea that interaction should be formed from selected target-bearing tokens, not a global reliability scalar. Reproduce only the minimal token-selection/bridging behavior before claiming novelty. |
| [GDSTrack: Modality-Guided Dynamic Graph Fusion and Temporal Diffusion for Self-Supervised RGB-T Tracking](https://arxiv.org/abs/2505.03507) | arXiv 2025-05; abstract says accepted at IJCAI 2025 | Builds dynamic adjacency over coherent regions; neighboring-frame signals are treated as distractor noise and denoised temporally | Useful for a later temporal extension. A graph over current template/search tokens is a more direct M1 transfer than its diffusion pipeline; avoid adopting both graph and temporal diffusion at once. |
| [Group Orthogonal Low-Rank Adaptation for RGB-T Tracking (GOLA)](https://arxiv.org/abs/2512.05359) | arXiv 2025-12; abstract identifies AAAI 2026 | Decomposes adaptation into rank groups and encourages complementary feature directions with inter-group orthogonality | Supports feature-space complementarity as a design family, not a per-frame gate. However, M1 has already seen several residual/projection experiments; an orthogonal penalty alone is not a sufficiently distinct contribution. |
| [Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking (APRTrack)](https://arxiv.org/abs/2606.26455) | arXiv 2026-06 preprint; RGB-Event, not RGB-T | Uses footprint-guided, channel-calibrated Hopfield retrieval to compensate structured missing target evidence from a historical memory | Strong cross-domain idea if M1 failures include partial target visibility: retrieve/restore target features instead of merely reducing a message. This requires explicit target memory and is a larger state-contract change, so not the first implementation. |
| [End-to-End Self-Supervised RGB-T Tracking without Modality Misleading (ESMTrack)](https://arxiv.org/abs/2609.37162) | arXiv 2026-09 preprint | Compares fused and unimodal responses to estimate contribution and uses modality decoupling plus temporal/grounding objectives during training | Useful evidence that reliability can shape learned representations and objectives, not just inference gates. It is a whole training framework and should not be transplanted wholesale into M1. |

These are primary-source abstract-level findings, not independent reproductions. The latest three 2026 entries are preprints unless the source itself establishes a venue; do not describe them as validated improvements to M1.

## Recommended M1 operator hypothesis

The most tractable next candidate is a **target-conditioned cross-modal template reconstruction / token bridge**, not another reliability gate. For each target-bearing template token, form a correspondence-weighted message from the other modality and a complementary residual, then update the token embedding itself:

`T'_m = T_m + alpha * W_o([C_{m<-n}, D_m, P_m])`

`C_{m<-n}` is cross-modal correspondence gathered from a small set of aligned template/search candidates; `D_m` preserves modality-specific evidence instead of collapsing it into a shared score; `P_m` is a target-conditioned local search context. `alpha` starts at zero for exact identity behavior. The new operation changes token channel content and target representation, while the existing gate only rescales message amplitude. Restrict computation to target template tokens and top-k spatial candidates; use low-rank projections if needed. Do not add online template updates in the first test.

This operator is only a candidate, not a claim of novelty. BTMTrack already makes token filtering and dual-template bridging salient in RGB-T tracking, so differentiation must come from the M1-specific failure condition and the precise reconstruction contract (e.g. preserving complementary modality evidence under target/distractor ambiguity), with related work checked in full before naming the contribution.

## Minimum evidence chain

1. Use existing M1 assets to define a fixed failure phenotype such as crop-inside sustained failure followed by feedback drift. Stratify only if existing annotations and frames support a reproducible condition; do not infer causes from sequence names.
2. On the exact M1 template/search features at fixed states, compare baseline, identity-initialized feature reconstruction, and a spatially shifted/non-target token control. Measure target-token representation change, target-versus-distractor feature separation, and downstream same-state localization. This tests whether the operator changes useful target evidence rather than merely perturbing features.
3. Only after fixed-state mechanism evidence is positive, train one new M1 child configuration with the feature operator as the only architectural change. Evaluate the prespecified failure cohort and matched controls, then full LasHeR. Keep protected M1 and all prior runs immutable.

Expected signature: target-conditioned representation becomes more separable from distractor/background features on the target failure phenotype; the shifted/non-target control does not produce the same gain; stable controls do not incur broad damage. If the representation changes but the target/distractor gap and fixed-state output do not improve, reject the operator before training.

## Decision

The concern that a multi-source scalar gate has limited expressive power is valid. The current formula can decide message magnitude but cannot synthesize or reorient the template feature. A token-level feature transformation is a meaningful next mechanism family and is more aligned with the goal of improving MSHA's representation. The first choice should be a bounded, identity-initialized target-conditioned token bridge, borrowing token filtering/correspondence from recent tracking methods while keeping the M1 test to one operator. Memory retrieval and temporal propagation are promising later branches only if the fixed failure phenotype demonstrates missing or stale target evidence.
