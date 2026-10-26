# Obsolete experiment routes cleanup (2026-10-13)

Removed executable configurations, runners, diagnostic tools, and model-layer integrations for the stopped DBF, LCGA, and MCCA routes. Inactive strip-score and fusion-signal admission utility scripts that referenced retired interfaces were also removed.

Historical reports, manifests, predictions, output directories, asset records, and source evidence were retained. Pre-cleanup source archive:

`experiments/snapshots/cleanup_pre_obsolete_routes_20261013_sources.tar.gz`

SHA256: `e6e7b46c998f8878ea6708ebe0f68ed4b1fa0541b95f6648bf53fcdf63bc1686`

The active model path is now the legacy M1 path plus explicitly gated retained modules. Optimizer construction has no DBF/LCGA/post-fusion branches; training entrypoints have no MCCA-only or LCGA-only modes; MCCA rank loss and retired post-search/fusion compatibility paths are removed.

Python 3.8 compilation passed for modified configuration, model, tracker, actor, optimizer, and training-entrypoint files. The protected M1 YAML loaded and built as `TBSITrack` through the legacy path.

Protected hashes are unchanged:

- M1 YAML: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`
- M1 epoch-15 checkpoint: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`

