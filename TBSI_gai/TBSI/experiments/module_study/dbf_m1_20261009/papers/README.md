# Paper sources and durable reading notes

## B: Differential Transformer

- Ye et al., *Differential Transformer*, [ICLR 2025 Oral](https://openreview.net/forum?id=OvoCm1gGhN); [official arXiv record](https://arxiv.org/abs/2410.05258), [v1 HTML](https://arxiv.org/html/2410.05258v1).
- Conference PDF: `Differential_Transformer_ICLR2025_OpenReview_OvoCm1gGhN.pdf`, from [OpenReview attachment API](https://api2.openreview.net/notes/attachment?id=OvoCm1gGhN&name=pdf); SHA256 `8efe33ae186b2a2190d5e234498391b7527342c07cd0520b5cf67da2c2568ec3`.
- arXiv PDF: `Differential_Transformer_arxiv_2410.05258.pdf`; SHA256 `1f38ccd2b4e6459ab76cb949a963feeb6a6d2e354d3d5ab59aa5e4b23d23eaa3`. Local HTML: `Differential_Transformer_arxiv_2410.05258v1.html`.
- Read from arXiv v1 HTML: 2.1/Eqs. (1)-(3), 3.8/Table 6, Appendix A. Two independent Q/K attention maps share V; the second softmax is scaled by learned lambda and subtracted. Each head's output is normalized and scaled by `(1 - lambda_init)` before concatenation. Table 6 reports validation loss 3.062 for default Diff Transformer and 3.122 without headwise GroupNorm in the paper's language-model setup. The conference PDF is saved but was not text-extracted here; version-level textual differences remain unverified. Details and local-code comparison: `../02_paper_B_evidence.md`.
- Scope: language-model evidence. It does not establish RGB-T tracking improvement.

## A: RGB-T Tracking With Template-Bridged Search Interaction and Target-Preserved Template Updating

- Li et al., IEEE TPAMI 47(1), 634-649 (2025), DOI `10.1109/TPAMI.2024.3475472`, PMID `39374289`; [Europe PMC bibliographic record and abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:39374289%20AND%20SRC:MED&format=json&resultType=core).
- User-supplied TPAMI PDF: `TBSI_Target_Preserved_Template_Updating.pdf`, SHA256 `8d21db472dcdb1eda98f4b88c3cbef5d1ceae05e6fc1965b96e7351d596b4efe`. This is the TPAMI extension, not the CVPR 2023 predecessor.
- Directly read: Introduction pp. 634-636, Section III-B-D pp. 638-640, Section IV-D/E pp. 642-646. TBSI reduces coarse search interaction/background interference through template mediation; IGF uses RGB-frame GIE illumination, weighted RGB/TIR search fusion, CA/SA; TPTU uses inference-time reverse-tracking preservation checks. GIE is initialized from reference [23] and frozen, while CA/SA train from scratch. See `../01_paper_A_evidence.md` and the problem-fit verdict in `../07_problem_fit.md`.
- Numerical table cells were not recovered by the local text extractor, so exact Table IV component deltas are not recorded. The source code, reference-[23] GIE weights, and exact preprocessing still need verification before certifying an original-IGF experimental arm.
