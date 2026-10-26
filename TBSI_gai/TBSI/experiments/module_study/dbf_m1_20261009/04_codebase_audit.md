# 04 Codebase Audit

## Repository and protected baseline

- Base commit: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` plus pre-existing working-tree changes.
- Protected config: `experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml`.
- Protected checkpoint: `output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar`.
- Protected hashes: config `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`; checkpoint `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Current worktree has extensive user/research changes. This analysis did not alter model source, configs, checkpoints, or results.

## Execution path and shape trace

| Node | Input | Output | Evidence |
|---|---|---|---|
| TBSI-E backbone final tokens | Two modalities, each 64 template + 256 search tokens, hidden dim 768 | `[B,640,768]` in the configured path | `lib/models/tbsi_track/tbsi_track.py:384-400`; short-screen assertions/logs. |
| Search token split | RGB `64:320`, TIR final `256` | Two `[B,256,768]` search streams | Same file, lines 400-401. |
| Template-context DBF wrapper | RGB/TIR templates `[B,64,768]`, searches `[B,256,768]` | Fused `[B,256,768]` | `lib/models/layers/dbf_igf_replacement.py:21-41`. |
| Head feature conversion | `[B,256,768]` | `[B,768,16,16]` | `tbsi_track.py:403-415`; current head grid assertion. |
| Legacy search fusion | Concatenated two search maps | `tbsi_fuse_search` output `[B,768,16,16]` | `tbsi_track.py:417-435`; helper `layers/head.py:8-23`. |

## Config and training contract in the completed short screen

- Parent: fixed protected M1 epoch-15 checkpoint, seed 42.
- Four arms: Legacy, Mean, DBF Vanilla, DBF Differential; all three epochs, same frozen RPP-v1 train proxy 176 and clean test proxy 60; FP32 tracking evaluation.
- Differential config: gamma init `0.001`, temporal tokens off, degradation-aware path off, in-layer DBF off, AMP off, LR `1e-4`, backbone multiplier `0.1`, scheduler drop at epoch 2.
- `post_fusion.*` parameters were present in optimizer, but allocated at `1e-5`; `tbsi_fuse_search` was at `1e-4`.
- Explicit epoch-3 checkpoints were used. The protected M1 YAML/checkpoint hashes remained unchanged.
- Exact metrics, config/checkpoint hashes, manifests, and evaluation contract are recorded in `reports/dbf_igf_replacement_short_20261009.md` and the asset registry.

## Evidence quality

The source/config/report findings are static code or artifact facts. The tracking metrics are an existing single-seed proxy observation. No model execution or new metric computation was performed for this audit bundle.
