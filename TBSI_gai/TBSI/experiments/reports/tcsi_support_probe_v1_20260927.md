# TCSI-P1 精确支持审计

日期：2026-09-27  
状态：`DIAGNOSTIC` / `P1_FAIL_STOP`

## 协议

- 固定 `single_modality_blur_v2` manifest 的 24 条序列，每条前 2 个事件，共 48 个事件。
- 只做 offset 0 inference；层 `3/6/9`；方向 `RGB2F/TIR2F`；K=`8/16/32`。
- 保存 softmax 后、value 聚合前的真实 attention Top-K values/indices。
- 两个方向都是 search-to-template，key 都是同一 `16x16=256` search grid，因此共同坐标条件成立。
- 当前 `Attention_st` 实际输出 `[B,Q,K]`，没有显式 head reshape；本轮按 `H=1` 记录，不能声称多 head 证据。

## 主要结果

| K | layer | shared ratio | RGB fg enrichment | TIR fg enrichment | RGB comp enrichment | TIR comp enrichment |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 0.0822 | 1.035 | 0.242 | 1.112 | 0.245 |
| 8 | 6 | 0.0433 | 2.779 | 0.697 | 2.851 | 0.672 |
| 8 | 9 | 0.0295 | 1.777 | 0.040 | 1.818 | 0.039 |
| 16 | 3 | 0.2016 | 0.980 | 0.478 | 1.161 | 0.457 |
| 16 | 6 | 0.1864 | 2.223 | 0.874 | 2.495 | 0.780 |
| 16 | 9 | 0.0854 | 1.527 | 0.198 | 1.600 | 0.152 |
| 32 | 3 | 0.4663 | 1.014 | 0.817 | 0.805 | 0.297 |
| 32 | 6 | 0.4905 | 1.649 | 0.893 | 2.086 | 0.415 |
| 32 | 9 | 0.2885 | 1.241 | 0.559 | 1.166 | 0.248 |

## 准入判断

- 共同坐标空间：通过。
- shared ratio 非退化：通过，K=16/32 多数层存在可测重叠。
- 两层目标富集 `>1.2`：失败；RGB 某些层富集，但 TIR 整体未达到，不能称为双模态目标支持。
- 互补支持前景富集：失败；RGB 个别层较高，但 TIR complementary enrichment 普遍低于 1。
- 相对同规模随机支持稳定胜出：失败。
- 训练准入：`P1_FAIL_STOP`，不实现或训练 TBSI-SCI。

结论是当前 attention support 存在方向和层级不对称，但没有形成稳定的共享目标证据与双模态互补证据。继续叠加 TPTU、entmax 或额外 gate 没有证据基础。

## 证据

- 脚本：[tcsi_support_probe_v1.py](../../tools/tcsi_support_probe_v1.py)
- 分析脚本：[analyze_tcsi_support_v1.py](../../tools/analyze_tcsi_support_v1.py)
- raw：`output/experiments/tcsi_support_probe_v1/batch/raw_support_indices.npz`
- summary：`output/experiments/tcsi_support_probe_v1/batch/support_summary.csv`
- 分析：`output/experiments/tcsi_support_probe_v1/batch/analysis/support_analysis.json`
- manifest SHA256：`45c13f5721a181a1e88a109a42acba83b589cbc9680b4a00551d095295264683`
- canonical YAML SHA256：`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
- checkpoint SHA256：`df7390978514b4a8e97924a20097cb932314cdf00d6feed47d1dc338b04d7689`
