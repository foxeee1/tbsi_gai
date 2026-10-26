# False-Consensus Week 1: Blind-Review Tool Run

Date: 2026-09-25  
Phase: review-manifest export, label-integrity audit, and mask-geometry smoke test  
Intervention outputs read: no

## Inputs and outputs

- Frozen source: `output/experiments/false_consensus_week1/v1.1_pre_review_screen/candidate_v1.csv`
- Frozen queue: `output/experiments/false_consensus_week1/v1.1_pre_review_screen/review_queue_v1.1.csv`
- Reviewer copy: `output/experiments/false_consensus_week1/v1.1_pre_review_screen/reviewer_manifest.csv`
- Audit: `output/experiments/false_consensus_week1/review_audit.json`
- Tools: `experiments/false_consensus_week1/review_tools/`

The reviewer copy contains exactly the intended 14 review fields. It excludes baseline boxes, response scores, screening reason, and review priority. The frozen source and queue were not overwritten.

## Image isolation audit

The first gallery was rejected. Six representative images from `v1.1_pre_review_screen/review_images/` visibly contained response heatmaps and text identifying `GT`, `baseline`, and `candidate`; they are retained as provenance but must not be used for blind labeling. The source was `tools/false_consensus_week1_collect.py:draw_review`.

`review_tools/build_clean_review_images.py` rebuilt 169 panels from the raw LasHeR visible/infrared frames. The replacement directory is `v1.1_pre_review_screen/review_images_clean_v1/`, and `review_gallery_r2.html` indexes it. Six representative replacement images were visually checked: they show only the two raw modalities and neutral candidate ROI outlines, with no response heatmap, GT/baseline box, score, or intervention output.

## Computed checks

- Exported rows: 169.
- Source candidate rows: 466.
- Screen-pass rows: 129.
- Queue labels: `UNCERTAIN=169`.
- Reviewed rows: 0.
- Verified P rows: 0; distinct frames: 0; distinct sequences: 0.
- Audit integrity errors: 0.
- Geometry identity and translated-transform smoke tests: passed.

Frozen hashes:

- `candidate_v1.csv`: `808727424f1a8e86eef81086ef47bfc2c6698c417f647a8211295e6cee055215`
- `review_queue_v1.1.csv`: `851cd32749fcbb64d2c4e43c6dc5b0aa2a86c2a6eaec60445469009642f1ef18`

## Decision state

`AWAITING-BLIND-REVIEW`. No P event is verified. The old image package is invalid for review; use only `review_gallery_r2.html`. Do not start spatial intervention, differential attention, or training. Human labels must be merged into a new queue copy and audited before any intervention decision.
