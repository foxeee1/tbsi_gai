# 00 Intake: DBF x Repository M1

- Date: 2026-10-09
- Project: `/root/autodl-tmp/TBSI_gai/TBSI`
- Goal: analyze the existing DBF evidence and design a follow-up in which a Differential Transformer-derived module occupies the paper IGF's semantic position after the backbone and before the prediction head.
- Authorization: static source/config/report review and experiment design. No training, tracking, checkpoint load, or model implementation change in this pass.
- Candidate: the local DifferentialBridge adaptation inspired by *Differential Transformer*. DBF/DifferentialBridge is a project adaptation name, not the paper's module name.

## Baseline identity

The protected repository anchor is `TBSI_E_MHSA_FULL_4090D_seed42`, configured with `TBSI_FUSION: tbsi_e`, locations `[3, 6, 9]`, and checkpoint epoch 15. Its recorded config and checkpoint hashes are verified in the current workspace: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e` and `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.

The user's phrase “TBSI first version + MSHA” is not established by this repository anchor. The protected YAML selects TBSI-E; no `MSHA`/`msha` implementation or config key was found in the searched model/config paths. This is a repository-bounded finding, not proof that no external MSHA source exists.

## Evidence boundaries

- Source/config facts below refer to this working tree, whose base commit is `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` plus extensive existing working-tree changes.
- Existing proxy metrics and provenance are reused from `reports/dbf_igf_replacement_short_20261009.md` and `reports/dbf_igf_contract_audit_20261009.md`; this pass did not rerun them.
- The named paper identities were checked against DOI `10.1109/TPAMI.2024.3475472` (TBSI) and the official arXiv record `2410.05258` (Differential Transformer). Both the user's TPAMI PDF and official Differential Transformer sources are now present under `papers/`. TPAMI Section III/Fig. 5 and its component ablation discussion have been read directly. A faithful `original_igf` arm still requires source code, GIE weights, and preprocessing verification.
- The short-screen comparison covers one seed and a fixed 60-sequence RPP-v1 test proxy. It is not a full LasHeR result or a comparison against a faithful original IGF implementation.
- Existing `AGENTS.md` protects the M1 YAML, checkpoint, logs, and full-244 results. They were not changed.

## Stage gate

Status: `ITERATE / causal attribution incomplete`.

- Passed: repository M1 identity, both DBF insertion points, post-backbone tensor contract, short-screen provenance, and major training-contract confounds are documented.
- Not passed: no matched learning-rate/head-distribution diagnosis isolates the Differential operation; the local bridge omits paper components including the independent second Q and headwise normalization; no verified original IGF implementation or GIE weights were found in this repository; no multi-seed or full-data post-backbone DBF evidence exists. The existing 60-sequence attribute summaries do not show a reliable DBF benefit for illumination, background clutter, or thermal crossover; details are in `07_problem_fit.md`.
- Current decision: do not claim DBF benefit or general Differential failure from this evidence. Keep the existing proxy no-go as a result for the tested implementation and contract.
