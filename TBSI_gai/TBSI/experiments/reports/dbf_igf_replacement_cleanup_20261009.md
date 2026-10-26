# DBF Legacy Cleanup 2026-10-09

Cleanup was requested before the post-backbone replacement experiment. The
protected M1 checkpoint/config, 4090D reproduction assets, old DBF diagnostic
report, and pre-change snapshot were retained.

The following four old location-9 short-run checkpoints were removed. Their
hashes are recorded here so deletion is auditable:

| Experiment | Checkpoint SHA256 |
|---|---|
| `DBF_SHORT_R3_M1_CONT_seed42` | `2d038a0ffac59e4cb71db47baf19103a2eec579d0b5de8c16801e30ff2174146` |
| `DBF_SHORT_R3_VANILLA_seed42` | `e5755792f18ff2457b8b4bc887503ee149dfd82504d43db56d401ead2045ad1e` |
| `DBF_SHORT_R3_ADDITIVE_seed42` | `df2c720d93abdf4dba5a262e1e49ab44ec72eb9bc8102d80cce8a0d2a22da77d` |
| `DBF_SHORT_R3_DIFFERENTIAL_seed42` | `1be476170554d30782e5450eb04d586acd2b91c6749e6d56959c3d365c2094e9` |

Removed output includes the old DBF short-run checkpoint directories, DBF
short tracking/result-plot directories, and obsolete location-9 DBF launch,
queue, and analysis helpers. `DifferentialBridge` and the new post-backbone
replacement implementation remain.
