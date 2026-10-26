# Proxy checkpoint cleanup (2026-10-07)

The following non-canonical proxy/G0 checkpoint weights were removed to
release storage. Logs, reports, metrics JSON, configs, manifests, and failure
audits were retained.

- `output/experiments/G0_CPLAIN_DIAGNOSTIC_seed42/checkpoints/TBSITrack_ep0001.pth.tar`
- `output/experiments/G0_FFM_CDIAG_seed42/checkpoints/TBSITrack_ep0001.pth.tar`
- `output/experiments/G0_FFM_CDIAG_seed3407/checkpoints/TBSITrack_ep0001.pth.tar`
- `output/experiments/G0_FFM_CSPATIAL_seed42/checkpoints/TBSITrack_ep0001.pth.tar`
- `output/experiments/G0_CSCSU_DIAGNOSTIC_seed42/checkpoints/TBSITrack_ep0001.pth.tar`
- `output/experiments/g0_cscsu_20261005/gamma_checkpoints/cscsu_gamma0.00.pth.tar`
- `output/experiments/g0_cscsu_20261005/gamma_checkpoints/cscsu_gamma0.25.pth.tar`
- `output/experiments/g0_cscsu_20261005/gamma_checkpoints/cscsu_gamma0.50.pth.tar`
- `output/experiments/g0_cscsu_20261005/gamma_checkpoints/cscsu_gamma1.00.pth.tar`

Approximate reclaimed space: 4.18 GiB. This cleanup did not touch the
canonical baseline, M1 epoch-15 reference checkpoint, full matched experiment
weights, frozen RPP manifests, or the TAM temporal trace.

The deleted checkpoint files are not needed for the recorded conclusions;
their reports and metrics remain available. Re-running those closed-loop
experiments would require retraining.
