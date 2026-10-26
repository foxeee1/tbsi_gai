# DBF x M1: minimum-cost decision plan

Status: `DESIGNED / NOT RUN` (2026-10-09). This is a future protocol, not permission to start training or tracking. The protected M1 remains `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15. The target question is whether search-token differential fusion can improve this M1 at the post-backbone fusion position. The TPAMI paper PDF is now available under `papers/`, but comparison to its **original IGF** still requires faithful GIE/CA/SA implementation and GIE weights, which this repository does not contain. The problem-fit analysis is in `07_problem_fit.md`.

## 1. What the existing result already decides

The completed template-context, post-backbone 3-epoch proxy screen (one seed, 60 LasHeR test sequences) gives AUC `L 52.4663`, `M 49.5412`, `V 50.5768`, `D 49.0681`; D-V is `-1.5087`, D-L is `-3.3982`. See `reports/dbf_igf_replacement_short_20261009.md` and its saved predictions. **Stop that exact configuration**; do not pay for a full run. L uses trained `tbsi_fuse_search` (3x3 Conv-BN-ReLU), while M/V/D bypass it. V and D share this bypass, so D-V is the most informative existing contrast, but it lacks gradient, effective-residual, and uncertainty records. The new module was placed in a `1e-5` optimizer group and trained for only 3 epochs. These facts cannot establish that the paper's differential attention is ineffective.

The paper-aligned mechanism is also absent: local D shares Q across maps, uses sigmoid scalar lambda, and lacks headwise output normalization and the fixed scale. The paper's Table 6 specifically reports worse validation loss and unstable training when its headwise normalization is removed. This motivates **measurement first**, not an immediate architecture bundle of search-only context, two Q branches, new lambda, normalization, and LR changes.

## 2. Claim ladder and minimal arms

| Claim | Cheapest necessary comparison | Permitted wording |
|---|---|---|
| The current local D learns a useful residual | Existing checkpoints/logs plus one fixed-batch numerical diagnosis; then matched L/V/D short runs only if healthy | Differential-inspired M1 fusion diagnostic |
| Subtracting attention adds value beyond ordinary cross-modal reading | V versus D under the same search-only wrapper, readout, initialization, data, and schedule | Differential-inspired search fusion helps this M1 if replicated |
| The module replaces M1 post-backbone fusion | D versus matched L continuation, with a genuine standalone fused-search output | M1 post-backbone fusion replacement |
| The method is better than paper IGF | Faithful original IGF implementation and frozen GIE weights, matched training/evaluation | Comparison to original IGF, only after that arm qualifies |

`TPTU` is an inference-time paper component, and its benefit must not be attributed to DBF. The repository M1 config selects TBSI-E. No explicit MSHA implementation name was located in the inspected model/config files; do not present the user's local phrase “TBSI first version + MSHA” as a verified execution trace.

The prospective paper narrative is **search interaction contrast without an explicit illumination estimator**: TBSI supplies target-related context upstream, while a post-backbone DBF reads the two search streams and tests whether subtractive attention suppresses irrelevant cross-modal responses. This is a hypothesis, not an observed mechanism. It is falsified if a matched Vanilla bridge performs as well or better, if target-region attention is also suppressed, or if the extra latency outweighs any robust tracking gain. A positive proxy score alone would not show that background cancellation caused the gain; target/background attention diagnostics and condition-specific tracking results would still be needed.

## 3. Gate G0: reuse evidence and perform static/numerical checks

No training. Before model-code changes, preserve a Git commit/branch or equivalent source snapshot; new features use a default-off gate and independent config/output names. Do not touch protected M1 files. The future implementer should use one fixed training-side batch and the explicit protected checkpoint, never an auto-selected latest checkpoint.

1. Confirm the effective M1 config (`SEARCH_SIZE=256`, TBSI-E, temporal tokens off, degradation-aware off, in-backbone DBF off), final `[B,640,768]` layout `[RGB template 64 | RGB search 256 | TIR template 64 | TIR search 256]`, head grid `16x16`, and training/online tracker paths. Assert `Ns == feat_sz_s**2` rather than relying on the default config's search size of 320.
2. Under `legacy` with the new gate disabled, check bitwise or tolerance-defined parity at fused head input and predicted boxes against the protected M1 path. Check shared backbone/head weight loading and list expected new-module missing keys.
3. On the fixed batch, record head-input mean/std/norm/positive fraction for L, raw M, existing V/D, and the proposed search-only path. Record `gamma`, `lambda`, each direction's `||gamma*delta||/||query||`, per-head output spread, map entropy, and output projection bias contribution. A raw mean-to-L distribution gap is a **measured signature**, not itself a causal proof of AUC loss.
4. Check parameter membership, group LR, and FP32 finite, nonzero gradients/updates for q/k1/k2/v/out/gamma/lambda with nonzero gamma. At gamma zero, only gamma learning initially is expected. Keep AMP off; if later enabled, check k2/lambda gradients under the actual scaler path.

Failing shape, parity, checkpoint, or optimizer checks marks the implementation invalid and blocks training. A lack of saved old gradient traces is recorded as unknown, not inferred from aggregate AUC.

## 4. Gate G1: inexpensive mechanism screen

The primary new input contract is `fusion(rgb_search_frame, rgb_search, tir_search) -> [B,Ns,C]`. For DBF, the RGB frame is deliberately unused; each direction reads **only the other modality's search tokens**, with no template input. The common prediction head remains unchanged. This tests an illumination-free alternative to IGF's search fusion idea; it does not replicate IGF's GIE or CA/SA.

To avoid paying for another raw-mean head distribution break in the first screen, use an **explicit diagnostic readout**:

```text
R' = R + gamma_r * delta(R <- T_search)
T' = T + gamma_t * delta(T <- R_search)
F  = trained_M1_tbsi_fuse_search(concat(R', T'))
```

At `gamma=0`, F equals the trained M1 fused feature, so legacy parity and head compatibility can be tested exactly. This G1 architecture **augments** the M1 fusion operator; it is not yet a standalone IGF/legacy replacement. Keep this naming in logs and any paper draft. It is a cheap gate for whether cross-modal differential residuals have value when the trained readout is held fixed.

Only three matched arms are needed initially: `L` (gamma-free original fusion, matched continuation), `V` (one-softmax bridge), and `D` (local two-softmax subtraction). The old M/ V/ D proxy scores cannot replace these controls because context, readout, and optimizer contract change. Use the same protected M1 checkpoint, same head initialization, trainable parameter scope, data order, optimizer steps, seed, FP32 precision, and explicit per-group LR. V/D shared projection weights should start identically; both use the same small nonzero gamma (e.g. 0.001), with a temporary gamma-zero parity check. Choose a single short budget and an LR schedule without a near-immediate decay; record actual optimizer updates. Do not silently treat equal global LR as equal module LR.

Choose a **frozen training-side screening manifest** before predictions, record members/path/hash and frame rules, and use it for all three arms. The earlier 60 LasHeR test sequences already influenced design, so do not reuse them for further model selection. Since M1 was trained on LasHeR training data, this screen tests continuation behavior, not independent generalization. Track closed-loop AUC as primary, plus PR/NPR/OP50/OP75, per-sequence paired differences, predefined low-light/occlusion subsets if labeled, FPS, peak memory, gamma/lambda and effective-residual trajectories. `L` continuation is needed to interpret D-L after equal extra training.

**Stop early** if D is no better than V with finite gradients, meaningful residuals, and no diagnosed per-head instability, or if D cannot beat matched L within the prespecified proxy uncertainty/cost tolerance. If gradients/effective residuals are negligible or per-head statistics are unstable, make at most one diagnosis-driven intervention before repeating this gate: correct the module LR **or** add per-head output normalization, according to the observed failure. If normalization is selected, compare D+norm against D first; if positive, add V+norm before assigning any gain to subtraction. A paper-aligned two-Q/lambda/normalization version requires its own identity and a capacity-matched control; never relabel local D as the ICLR implementation.

## 5. Gate G2: actual replacement and paper comparison

Enter only if G1 supports the mechanism and cost. Remove the trained legacy readout from the candidate path so DBF itself emits one fused search stream. For the standalone path, choose one shared output adapter/calibration policy for V and D **before** training; any adapter must be counted as part of the candidate. Use a matched L continuation and V/D with identical readout, training and head contracts. A separate mean control is added only if the initial head-input distribution or D/V interpretation requires it. This is the earliest gate at which “replaces M1 post-backbone fusion” is accurate.

Only after G2 success should an `I=original_igf` arm be attempted. Qualification requires an implementation checked against the available TPAMI method/Fig. 5, GIE pretraining source and hash, its frozen status, RGB frame preprocessing, CA/SA design, and train/online inference parity. If any are missing, `D-I` is `NA`; the project may still report comparison to M1 legacy but not superiority to original IGF. No new full-data test, multi-seed confirmation, or paper headline should be triggered by this plan alone.

## 6. Decision and resource rules

The 60-sequence historical proxy is exploratory only. Any positive G1/G2 result must be checked against uncertainty, paired sequence behavior, latency, and a frozen evaluation set not used for selecting the design. Independent parent-seed claims require independent M1 parents; continuation seeds from the same seed-42 checkpoint are not independent model-seed evidence. At any future run, use a unique ID/config/output directory, explicit checkpoint, source/config/data hashes, and incrementally update both asset tables. All training/tracking/worker processes together must respect the repository-wide limit of two; thread and worker values are measured choices, not a blanket maximum-speed setting. Prior two-shard tracking changed predictions; any faster schedule must pass a fixed-sequence numerical equivalence check before its FPS or AUC is used.

Current decision: **do not train**. First close G0 evidence, then decide whether the narrow G1 mechanism screen is worth its three short runs. The missing faithful IGF implementation/GIE weights block only the original-IGF comparison, not the M1 mechanism diagnosis.
