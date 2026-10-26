# M1-S1 center/crop reachability audit (2026-10-16)

## Parent

Protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15, fixed-state replay only.

## Hypothesis

The remaining S1 failure may be caused by crop-internal center-cell selection rather than by offset/size correction. If the GT cell is absent from the high-ranked response cells, a readout-side residual cannot recover the target reliably.

## Single change

No model or tracker change. Re-aggregate the existing M1 candidate-recall audit with explicit center oracle, top-k recall, and sequence-level outputs.

## Measurements

- actual top-1 decoded IoU;
- GT cell rank and top-k recall;
- IoU after replacing only the center cell with GT, keeping size/offset unchanged;
- 3x3 and all-map offline oracle bounds;
- matched-good, pre-onset, severe-onset, and per-sequence aggregates.

## Decision

Promote a center-cell/reranking or crop-coverage experiment only if severe-onset retains material oracle headroom. Do not treat GT-cell or all-map oracle as deployable performance.

## Data boundary

LasHeR trainingset, 60 deterministic sequences and 123 fixed states from the source audit; no full-test claim and no closed-loop state update.
