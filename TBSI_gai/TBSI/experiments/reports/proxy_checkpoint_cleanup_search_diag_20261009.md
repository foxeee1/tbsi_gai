# Search-stage diagnostic space cleanup (2026-10-09)

User authorized deletion of abandoned proxy experiment weights before the
M1 search-stage diagnosis. The five checkpoint files below were checked against
their existing asset/report SHA256 values, then removed by explicit paths.
Training logs, configs, manifests, prediction files, metrics, comparison JSON,
and source snapshots remain. Original experiment records retain their historical
checkpoint hashes; the files are now intentionally absent.

| Checkpoint under `output/experiments/` | SHA256 | Size (bytes) |
|---|---|---:|
| `DBF_IGF_REPLACE_SHORT_LEGACY_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | `0fe4171003c27c628628ae09c663cbb7a0b2f40cb4e96e394f487173b64bdfa5` | 1,321,826,467 |
| `DBF_IGF_REPLACE_SHORT_MEAN_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | `02615162c14efda0f9e4b09d610b737db7a4dffaa128` | 1,236,870,963 |
| `DBF_IGF_REPLACE_SHORT_VANILLA_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | `673851ea2238491fd80c654ac1197673499fd32e189f827b777273ecfcb527bc` | 1,257,416,675 |
| `DBF_IGF_REPLACE_SHORT_DIFFERENTIAL_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | `f50886904730e54843957bfaa51ad56ec0b1994147f4ed8d13ea41c0154f04d8` | 1,260,570,059 |
| `DBF_G1_VANILLA_LOADER4_BENCH_seed42/checkpoints/TBSITrack_ep0001.pth.tar` | `321c6af42c906430ea855d59a5b40c12bec1bded2e4ff71efa1e9ce8de7563d4` | 1,342,354,131 |

Total file size removed: 6,419,038,295 bytes (5.978 GiB). `df -h` showed
approximately 5.1 GiB available beforehand and 12 GiB afterward. The protected
M1 checkpoint, both current 4090D reproduction checkpoints, and the provisional
G1 worker-4 epoch-3 checkpoint were not touched. Re-running the removed proxy
weights requires retraining; their already archived evaluations remain readable.
