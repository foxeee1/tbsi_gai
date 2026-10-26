# Spatial Token Bridge Amplitude Diagnostic Checklist

## Before run

- [x] Recover E10 bridge checkpoint and fixed-state parent outputs.
- [x] Freeze the single intervention: post-bridge message multiplier only.
- [x] Save pre-change tool/model/config bundle and hash.
- [x] Preregister arms, parity checks, measurements, and decision rules.
- [x] Confirm protected M1 hashes and available resource slots immediately before launch.
- [x] Persist event and sequence manifests and hashes for the first attempt.
- [x] Record first-attempt parser failure before inference; corrected retry groups pre/onset rows by sequence.

## Run and analysis

- [ ] Complete retry off plus target/shifted at 1x, 32x, 128x, all 46 states.
- [ ] Verify scale-1 paired parity against the E10 replay.
- [ ] Compare state and sequence means; CENTER/query rank and replay IoU; onset phases.
- [ ] Update report, Markdown registry, and CSV registry.

## Stop gates

- [x] Scale-1 parity passed exactly; all seven arms completed all 46 states.
- [x] Do not infer benchmark performance or proceed to full training from this run; amplitude-only explanation failed the preregistered gate.
