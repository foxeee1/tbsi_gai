# M1 MSHA Query-Role Fixed-State Mechanism Check

Status: `COMPLETE / SPATIAL-QUERY-EFFECT-SUPPORTED / TARGET-UTILITY-INCONCLUSIVE / NO-TRAINING`  
Date: 2026-10-23

## Question

Does restricting MSHA cross-modal messages to initialized-template foreground queries produce a target-aligned effect that a matched-mass spatially shifted non-target query control does not?

## Parent, cohort, and intervention

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15; YAML SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`; checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Cohort: the existing fixed train-side hash cohort and its 14 matched-good plus 14 severe-onset events from `output/experiments/M1_CENTER_READOUT_FRESH_DIAG_20261010/analysis/per_event.csv`. Sequence membership/order is frozen at SHA256 `e1a976257d71969686c1dc0690596465d5e14d24366ecef567c58f8b129e65f6`. This cohort is previously exposed; no holdout or new sequence selection.
- Arms: parent M1; `foreground_queries`; `shifted_queries`, which cyclically shifts the same soft mask by half the square token-grid side, preserving its mass and weights.
- Every arm restores the same tracker state and RNG at each event. In-between frames and state progression use M1. FP32, one Torch thread, serial. No optimizer, training, or checkpoint writes.
- Model/checkpoint and M1 feature behavior are unchanged. Only the offline runner's collectors are modified. Before the post-event M1 state-advance forward, event collectors are detached to prevent its output from overwriting the final `SHIFTED_QUERY` record.

## Gate and measurements

Two-event smoke on `blackbag` passed the instrumentation gate: both interventions changed projection input/output and fusion returns at all three TBSI blocks; CENTER head input changed; shifted mask overlap with the original foreground mask was zero; arm crops and states matched. Query-role spatial response reversed as intended: foreground restriction changed background queries more than foreground queries, while shifted restriction changed the original foreground queries more than background queries.

For the full 28 events, record per arm and event: mask/branch stats, projection input/output and fusion-return differences by original foreground/background query, actual CENTER input difference, crop/state, GT rank, GT-vs-hard-background margin, selected peak, predicted box, and fixed-state IoU.

Mechanism support requires:

1. All event arms have identical crop and pre-action state; tensors are finite and shapes stable.
2. Both interventions produce nonzero projection/fusion-return and CENTER-input deltas on at least 90% of events.
3. The query-role spatial response is directionally specific at the final fusion return: FG_QUERY has larger background than foreground delta, while SHIFTED_QUERY has larger original-foreground than background delta, on at least 70% of events.
4. On severe-onset events, FG_QUERY improves GT rank or hard-background margin vs parent on more events than SHIFTED_QUERY; report paired counts and event-level deltas, without requiring a tracking benchmark gain.

Failure of items 1-2 is an instrumentation/interface failure and stops the campaign. Failure of item 3 means the operation is not spatially selective as intended. Failure of item 4 means the structural query effect is measurable but target-ranking benefit is unsupported. In all cases, no training is admitted by this diagnostic.

## Provenance

- Original pre-change source snapshot SHA256 `a9fce480d24c0ad017be8447b19fb48634f0f061abd7825216295fbb39232bb6`.
- Offline runner instrumentation snapshots: `experiments/snapshots/m1_msha_query_role_capture_20261023/instrumentation_v1.tar.gz` and `instrumentation_v2.tar.gz` (hashes and changes documented in the asset report).
- Full output directory: `output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261023_FULL28/`.

## Result

- All 28 paired events had matching pre-action tracker states and crops. Both intervention arms produced nonzero projection-input, projection-output, fusion-return, and CENTER-input deltas on 28/28 events; all captured deltas were finite and shapes remained stable.
- At the final fusion return, FG_QUERY changed background queries more than foreground queries on 28/28 events; SHIFTED_QUERY changed the original foreground queries more than background queries on 28/28 events. This supports the intended spatially selective structural effect.
- Severe-onset target readout was weak: FG_QUERY improved GT rank or hard-background margin vs parent on 8/14 events, SHIFTED_QUERY on 7/14. Direct FG-vs-shifted margin difference averaged `+0.00082` with event bootstrap 95% CI `[-0.01234,+0.01324]`; GT rank was 1 win/13 ties/0 losses. Mean FG_QUERY IoU delta vs M1 over 28 events was `-0.00314`; on severe events FG_QUERY IoU was below SHIFTED_QUERY by `0.00788` mean.
- The capture bug is identified: the runner's post-event M1 state-advance forward overwrote the last arm's feature cache. Collector detachment fixed the attribution; it did not change model behavior.
- Final decision: support the structural claim that query role changes the MSHA output and CENTER input; target-ranking/localization utility is inconclusive and not supported. Do not train this operator or claim tracking gain. Any further test requires an independent cohort and a preregistered minimum target-utility effect; the exposed 28-event set is closed to tuning.
