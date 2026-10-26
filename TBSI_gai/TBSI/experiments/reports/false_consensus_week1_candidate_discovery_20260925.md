# False-Consensus Week 1: Candidate Discovery

Date: 2026-09-25  
Phase: candidate discovery and blind review only  
Config: `v1.0.3-单卡4090D-24G-等效全局128-seed3407`  
Checkpoint: `TBSITrack_ep0015.pth.tar`  
Data: 30 fixed LasHeR training sequences from RPP-v1, 6 uniformly spaced frames per sequence  
Layer: TBSI index 0, configured location 3  
Inference: FP32, seed 3407

## What was run

At the first TBSI layer, before its first cross-modal interaction, RGB and TIR template/search tokens were captured through a forward pre-hook. For each modality, the diagnostic response was the mean of the top 8 cosine similarities from initial template foreground tokens to each search token. Candidate components used a fixed 0.90 response quantile, maximum RGB/TIR center distance of 64 pixels, and maximum GT overlap of 0.10.

The hook only observed tensors. No attention, value, key, model weight, tracker state, or checkpoint was modified. No intervention outcome was written into review artifacts.

## Output

- 30 sequences and 180 fixed frames processed.
- 466 automatically proposed RGB/TIR paired candidate events.
- Candidates occurred in all 30 sequences.
- 466 blind-review images and 180 per-frame feature records were written.
- `annotations.csv` contains 466 rows initialized as `uncertain`; no event is yet counted as P, S, or N.
- Mean automatic response scores: RGB 0.529, TIR 0.624; mean paired center distance 36.1 pixels.

These counts describe automatic proposals only. They do not establish that RGB and TIR selected the same physical non-target object. Human review must confirm `same_physical_object`, visibility, and alignment before any P-event count is reported.

Evidence: `output/experiments/false_consensus_week1/metadata.json`, `candidate_events.csv`, `annotations.csv`, `review_images/`, and `feature_records/`.

## Next gate

Freeze the candidate list, complete blind P/S/N/uncertain annotation and a second review subset, then decide whether at least 20 verified P events exist. Until that gate passes, do not run D-RGB, D-TIR, D-both, R, G, or T interventions.
