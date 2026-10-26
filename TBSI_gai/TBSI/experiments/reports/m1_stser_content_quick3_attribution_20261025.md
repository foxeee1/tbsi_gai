# M1 STSER Content Quick3 Retry1: Attribution

## Scope and verdict

This is an assets-only analysis of the existing retry1 checkpoints, logs, and paired RPP-v2 results. No training, inference, or evaluation was run for this report.

The candidate was initialized from the shared SOT-pretrained state, not from the learned Early-conditioned MSHA gate checkpoint. It is therefore a conceptual follow-up to that MSHA line, not a fine-tune or a direct checkpoint extension of the gate. Its native MSHA path remains in place; STSER adds a separate search-conditioned T2S residual at loc9.

Verdict: `MECHANISM-COMPUTES / EFFECTIVE-WRITEBACK-TINY / PERFORMANCE-BENEFIT-NOT-SUPPORTED`. The correction is nonzero and receives gradients, but the sampled foreground attention mass slightly decreases, and the complete correction path is reduced to a negligible search-token residual by router suppression, output projection, bounded residual, and the near-zero outer gamma. The paired 60-sequence metrics are mixed and statistically inconclusive.

## Relation to the strongest prior MSHA experiment

The closest prior MSHA candidate is `M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023`. It uses early RGB/TIR template-search match maxima plus template agreement to produce per-template, per-direction multipliers for cross-modal Value messages in native TBSI template MHSA. It leaves QK attention, self messages, and the CENTER head unchanged. Its full-test point estimate was directionally positive against M1, but bootstrap intervals crossed zero; it had no matched gate-off full-training control. A later provenance audit invalidated its archived sequence-level comparison and found the available current-output recomputation inconsistent with the protected M1 reference. It is not established as a reliable performance gain.

The conceptual design progression was:

1. The early gate asks which modality's template Value should contribute, using a pooled match signal. It does not retain the search position that produced the maximum.
2. STSER asks a more local question at each search token: which opposite-modality template content should this search location receive, relative to nearby target-scale candidates and the rest of the search grid?
3. The content variant uses the first-frame template foreground mask as a prior and adds a local-versus-outside-search contrast to the attention logits. It changes the content distribution, not just a scalar modality gate.

This is an operator-level follow-up to the early gate's lost spatial axis, but the retry1 checkpoint does not inherit the gate's learned weights or isolate the gate as its parent. The valid claim is “new search-conditioned content redistribution added alongside native MSHA,” not “the best MSHA checkpoint was further optimized.”

## Information flow

For each of RGB-search and TIR-search, the implemented path is:

1. **Native MSHA remains intact.** At `tbsi_e`, RGB and TIR template tokens still pass through the native joint template attention and produce fused template features. This path is not replaced by STSER.
2. **Search-conditioned cross-modal read.** At loc9, RGB search tokens form queries over TIR template keys/values; TIR search tokens form queries over RGB template keys/values. A learned null key/value lets the base attention abstain.
3. **Target prior.** The first-frame box becomes a soft foreground coverage mask over template tokens. A floor prevents background template tokens from receiving exactly zero prior.
4. **Position-aware contrast.** For each search token and template token, normalized QK similarity is compared against an object-scale local search window's outside-region reference. The local window size is derived from the foreground mask's width/height. Unlike the earlier max-pooled gate, this computation retains the search-token axis.
5. **Redistribution.** `base = softmax(score + log(prior))`; `changed = softmax(score + log(prior) + lambda * contrast)`. Their difference multiplied by the opposite-modality Value gives the content correction. This is a redistribution delta, not a second full attention message.
6. **Competition experts and null route.** The ordinary T2S message is routed through shared/private/null experts using search query, cross-modal message, non-neighbor competitor descriptor, and modality agreement. The content correction is added to that result, scaled by `1 - route_null`; it does not use the shared/private weights directly.
7. **Projection and bound.** The summed expert/content message passes through `out_rgb/out_tir`, then is clipped to at most 10% of the input search-token norm.
8. **Residual writeback.** A zero-initialized learned `gamma` scales the bounded delta, which is added to the search token. The updated search features continue to downstream backbone/head processing and ultimately supply CENTER's score, size, and offset maps.

In compact form, the new path is:

```text
native template MSHA -> fused templates
search tokens + opposite fused template -> QK attention
foreground prior + local/outside contrast -> changed attention
(changed - base) @ Value -> content correction
expert route + (1 - null) * correction -> output projection
10%-bounded delta -> gamma * delta -> updated search tokens -> CENTER
```

## Evidence and attribution

The retry1 pair is well matched for its single tested variable: shared SOT initialization, seed 42, 176 RPP train sequences, 3 epochs, and 60 paired RPP-v2 test sequences. The only config change was content strength 0 versus 1; the sixth RGB-T standard-deviation entry was corrected in both retry1 configs after the earlier data-contract failure.

Performance is not a reliable improvement: ΔAUC `-0.077 pp` with sequence-bootstrap 95% CI `[-1.068, +0.929]`; ΔOP75 `+0.213 pp` with CI `[-0.633, +1.095]`. AUC improves on 31 sequences and declines on 29. Failure-episode count rises on 22 sequences, falls on 8, and is unchanged on 30; aggregate failure-episode frames fall by 173. These are descriptive proxy diagnostics, not a causal explanation of the aggregate metrics. FPS drops from `56.496` to `47.982` (about 15%).

The sampled mechanism logs establish computation but not useful target transfer:

- `source_shift_l1` is about `0.018-0.020`, and `content_message_norm` about `0.30-0.36`; the correction is not mathematically zero.
- The sampled foreground attention mass changes from roughly `0.69-0.73` to a value lower by about `0.00017-0.00030`. Thus this version does not show the intended increase in foreground allocation.
- The correction/base-message cosine is only `0.04-0.06`, so the correction mostly points in a different direction. This proves neither that it is useful nor harmful.
- The learned content coefficient stays near `0.1`; its gradient exists but is small (`about 5e-9` to `1e-7` at recorded probes). The probes are sparse samples, not epoch averages.
- At the final recorded training probe (epoch 3, step 600), relative bounded delta before gamma is `1.8e-7` for RGB and `4.3e-6` for TIR. Multiplying by the recorded gamma (`3.77e-4` and `-1.77e-3`) makes the estimated final relative writes approximately `6.6e-11` and `-7.6e-9` on that sampled batch. This is a strong indication that the trained residual is effectively suppressed at that point, not proof for every sample.
- The route is asymmetric at that probe: RGB null mass is about `0.673`, while TIR is almost entirely shared (`shared about 0.988`, null about `0.003`). Because the content correction is only multiplied by `1-null`, its effective modulation differs substantially by direction. The content path is also not conditioned on the shared/private route weights, which leaves a plausible interface mismatch to test.

The evidence therefore narrows the current issue to two unresolved stages: (a) the contrast redistribution does not measurably increase foreground mass in the recorded probes; and (b) even a nonzero correction is heavily attenuated before it reaches CENTER. No existing log traces candidate CENTER rank or decoded boxes under content-on/content-off fixed states, so we cannot say which stage caused each sequence's change.

## Next direction

Do not add another attention block, temporal updater, or reliability head yet. The smallest useful next step is a fixed-state, same-crop inference audit on a prespecified train-side event set. At loc9, record the content delta after each stage: raw redistribution, after null factor, after output projection, after bound, after gamma, and final search-token difference. On the same crops, compare content on/off for CENTER Hann top-1, GT-vs-hard-background rank/margin, and decoded box. This is a diagnostic, not a performance claim.

Decision rule:

- If the post-gamma write is negligible, test one isolated interface change: give the content branch its own small identity-preserving residual scale instead of passing it through both the expert null factor and shared outer gamma. Keep the same contrast formula and everything else fixed.
- If a measurable write reaches the search tokens but foreground mass and CENTER rank do not improve, stop tuning amplitude. Replace the scalar local/outside contrast with a foreground-normalized target-versus-background logit contrast, then test it on the same fixed states before any training.
- If the same-crop rank and decoded box improve across prespecified events, run one matched quick training pair from the same SOT initialization. Only promote after paired sequence evidence supports an aggregate gain and the runtime cost is acceptable.

This order tests the causal link from target evidence to content redistribution to search-token writeback to CENTER, while keeping the next model change to one variable.

## Evidence paths

- Quick3 paired result: `output/experiments/M1_STSER_CONTENT_QUICK3_CANDIDATE_20261025_RETRY1/paired_rpp60_summary.json` and `paired_rpp60_per_sequence.csv`.
- Flow logs: `logs/stser_content_flow.csv`, `logs/stser_flow.csv`, `logs/stser_content_preclip_grad.csv`, and `logs/feature_changes.csv` in the candidate retry1 run directory.
- Existing paired protocol report: `experiments/reports/m1_stser_content_quick3_retry1_20261025.md`.
- Early-gate experiment and provenance audit: `experiments/实验资产数据库.md`, entries `M1-EARLY-CONDITIONED-MSHA-GATE-FULL15-JOINT-20261023-RETRY1` and `M1-EARLY-GATE-FULL244-PREDICTION-PROVENANCE-AUDIT-20261024`.
