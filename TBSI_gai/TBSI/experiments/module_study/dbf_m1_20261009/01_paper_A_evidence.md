# 01 Paper A and M1 Evidence

Paper identity: Bo Li et al., *RGB-T Tracking With Template-Bridged Search Interaction and Target-Preserved Template Updating*, IEEE TPAMI 47(1), 634-649 (2025), DOI `10.1109/TPAMI.2024.3475472`, PMID `39374289`. The user's 16-page TPAMI PDF is now available as `papers/TBSI_Target_Preserved_Template_Updating.pdf`, SHA256 `8d21db472dcdb1eda98f4b88c3cbef5d1ceae05e6fc1965b96e7351d596b4efe`. The claims below were checked against its printed page numbers, sections, and figure captions. Text extraction did not recover numerical table cells; table-related statements below use the authors' surrounding prose, not invented cell values.

## Paper-level evidence

| Claim | Type | Evidence | Boundary |
|---|---|---|---|
| TBSI uses templates as a medium to exchange target/environment context between RGB/TIR search regions and reduce distractor background noise. | Paper explicit | Abstract and Introduction, pp. 634-635; Section III-B, pp. 638-639. | This capability is already a stated purpose of TBSI; DBF must show an incremental benefit. |
| TBSI-E replaces the conference version's six MHCA operations with one joint-template MHSA and subsequent ViT interactions. | Paper explicit | Section III-B and Figs. 3-4, pp. 638-639; Table V discussion, p. 644. | The protected repository config independently selects `tbsi_e`. |
| IGF adapts RGB/TIR search fusion to current lighting using the RGB search frame, GIE global illumination factor, then CA and SA. | Paper explicit | Introduction p. 635; Section III-C and Fig. 5, pp. 639-640. | The current repository `legacy` and local DBF do not implement this. |
| IGF trains CA/SA from scratch and initializes GIE from reference [23], freezing it during training. | Paper explicit | Section III-C, p. 640. | Exact reference-[23] weights, preprocessing, and official IGF code remain unverified locally. |
| TPTU updates a dynamic template at inference using a classification threshold and reverse-tracking preservation IoU. | Paper explicit | Section III-D/Algorithm 1, p. 640; Section IV-D/Table VII discussion, pp. 644-645. | A stateless post-backbone DBF cannot directly implement this temporal decision. |
| The paper's component ablation prose reports positive contributions from TBSI, IGF, and TPTU. | Paper explicit, prose only | Section IV-D, Table IV discussion, pp. 642-643. | The PDF text extractor did not recover Table IV cell values; no exact deltas are claimed here. |
| The source and exact preprocessing of the frozen GIE weights are available in this repository. | Not established | No IGF/GIE implementation or weights found in `TBSI_original` or the active model tree. | Obtain and verify before labeling an arm faithful original IGF. |

## Repository M1 evidence

| Claim | Type | Evidence | Confidence / gap |
|---|---|---|---|
| Protected M1 config selects `TBSI_FUSION: tbsi_e`, locations `[3,6,9]`, and a CENTER head. | Code/config confirmed | `experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml`; hash recorded in `00_intake.md`. | High for this repository snapshot. |
| M1 epoch-15 checkpoint is the protected parent for the four post-fusion arms. | Artifact confirmed | `output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar`; hash recorded in `00_intake.md`. | High. Do not replace with another checkpoint. |
| The active protected M1 label “TBSI first version + MSHA” matches this config. | Not established | No `MSHA`/`msha` string found in the searched configs/model code; protected config explicitly selects `tbsi_e`. | Do not use this label as a verified repository fact. |
| M1 legacy search fusion is `tbsi_fuse_search`, implemented by the project's `conv` helper. | Code confirmed | `lib/models/tbsi_track/tbsi_track.py:188,417-435`; helper in `lib/models/layers/head.py:8-23`. | Helper defaults to 3x3 Conv2d + BatchNorm2d + ReLU. |

## M1 current limitation

The Legacy head input comes from a trained spatial fusion operator. `mean` and both DBF replacement modes bypass it. Their `[B,C,H,W]` shape matches, but operator, normalization, activation, and learned spatial filtering do not. This is a concrete feature-distribution and function change, not merely a shape-compatible replacement.
