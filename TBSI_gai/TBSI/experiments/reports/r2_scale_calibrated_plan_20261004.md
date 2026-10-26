# R2 scale-calibrated frequency relation experiment

## Run contract

R0 mechanism audit showed that the learned HF-Relation bias was too small to produce a measurable frozen inference change. R2 changes only the relation-bias scale: each cross-modal frequency relation row is standardized and multiplied by the native cross-modal MHSA-logit row standard deviation. It does not add a reliability gate, Value injection, message residual, target mask, second attention, or new fusion module.

The serial runner is `tools/run_r2_scale_calibrated_serial.sh`. It first performs R2 P0, then audits the existing R0 epoch-15 checkpoint on the fixed 24-sequence/48-event development set with `R0_NORMAL`, calibrated alpha values `0/0.05/0.10/0.20`, and fixed shuffle/sign-flip controls. Full R2 training is permitted only when at least one nonzero alpha has median attention L1 > `1e-3`, median attention KL > `1e-5`, and median message relative delta > `1e-4`.

If the gate passes, the runner starts exactly one full R2 run: seed 42, 15 epochs, same LasHeR train protocol, SOT initialization, serial FP32 full-244 test, and standard analysis. If it fails, no training starts.

## Validation completed

- R2 P0 passed: `[2,12,128,128]`, finite, same-modal blocks zero, zero-alpha parity, nonzero alpha gradients.
- R2 TBSI-E model build passed: location 9 uses `ScaleCalibratedFreqRelationBias`; locations 3 and 6 are disabled.
- Protected canonical baseline YAML is unchanged.
- Current resources: 1 x RTX4090D, approximately 14 GiB free disk.

## Current status

`READY-FOR-SCALE-AUDIT`; no R2 training, checkpoint, or tracking results have been created. The next command is the serial runner below; it owns the gate and will stop safely before training if the scale intervention is not observable.
