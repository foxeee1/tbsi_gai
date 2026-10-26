# Deprecated IGF proxy checkpoint cleanup (2026-10-17)

The user-authorized cleanup removed only the three IGF pre-fusion proxy checkpoints, whose SHA256 values were recorded before deletion:

- `M1_IGF_PREFUSION_PROXY_20261017/checkpoints/TBSITrack_ep0001.pth.tar`: `844657b764bcf4a1adc49faa90f794316c89da9eb2dd358cfb38e3fffbed76f6`
- `M1_IGF_PREFUSION_PROXY_20261017/checkpoints/TBSITrack_ep0002.pth.tar`: `bd4afa9a6af3529a016347e445f05d81fdca2d16582a3eced0424b14b6ab65ba`
- `M1_IGF_PREFUSION_PROXY_20261017/checkpoints/TBSITrack_ep0003.pth.tar`: `4508f457fe3d583918e9462713452f9d7465b4b77f8b6a347454ad24cb5ea4f3`

The IGF config, logs, tracking results, and prior report remain for audit. Protected M1, the 4090D equivalent reproduction weights, the pretrained weight, and the E2 parent/expanded checkpoints were retained.
