# False-Consensus Week 1: v1.1 Pre-Review Screen

Date: 2026-09-25  
Phase: pre-review geometry screening only  
Input: original `candidate_events.csv` from the 30-sequence, 180-frame discovery run  
Intervention outcomes read: no  
Human annotations read: no

## Frozen screening rule

For each original candidate pair, expand the GT box by 10% of its width and height on each side. Keep the pair only when both RGB and TIR candidate centers are outside the expanded GT box and their center distance is at most `min(64 px, 0.5 * GT diagonal)`. Also compute candidate-box coverage by the original GT box, normalized pair distance, and explicit rejection reasons.

The rule was fixed before any intervention or human result. The original 466-row candidate table and original annotations were preserved unchanged.

## Counts

- Input: 466 automatic proposals.
- Screen pass: 129 proposals.
- Screen-rejected: 337 proposals.
- Screen-pass independent frames: 79.
- Screen-pass sequences: 29.
- Random rejected review sample: 40 proposals, seed 3407.
- Review queue: 169 proposals total.
- Verified P/S/N events: 0; all queue labels remain `uncertain`.

This is a review-priority screen, not evidence that any proposal is a P event. Rejected proposals are deliberately sampled for false-negative auditing.

Evidence: `output/experiments/false_consensus_week1/v1.1_pre_review_screen/candidate_v1.csv`, `review_queue_v1.1.csv`, `screening_rules.json`, `review_images/`.

## Decision state

`AWAITING_BLIND_REVIEW`. Do not run D-RGB, D-TIR, D-both, R, G, or T until the queue is reviewed, boxes can be corrected, and P/S/N/uncertain labels are frozen.
