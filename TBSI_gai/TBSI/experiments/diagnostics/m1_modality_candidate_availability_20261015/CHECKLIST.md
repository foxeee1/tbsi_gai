- [x] Freeze hypothesis, parent, dataset members, one-factor intervention, metrics, and decision threshold.
- [x] Confirm cached RGB, TIR, fused features and same-state GT/capture assets exist.
- [x] Verify hashes, sequence/frame alignment, output path, and resource availability.
- [x] Run the frozen candidate audit.
- [x] Check baseline and feature recomposition parity; confirm complete finite outputs.
- [x] Produce aggregate and per-sequence comparison tables.
- [x] Update report, experiment registry Markdown, and 23-column CSV.

## Result

Final retry2 complete. M1 fused mean IoU `0.643824`; RGB-only `0.564839` (`-0.078985`); TIR-only `0.457479` (`-0.186345`); best-of-three oracle `0.687779` (`+0.043955`, sequence CI `[+0.020567,+0.052689]`). Candidate routing is supported only as a next OOF selector experiment; no full tracking/test yet.
