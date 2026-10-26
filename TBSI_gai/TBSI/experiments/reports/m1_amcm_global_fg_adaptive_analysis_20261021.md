# M1 Global/Foreground/Adaptive Matching 机制对照分析

## 对照对象

本分析比较同一 M1 parent、同一 3 epoch proxy 训练合同和同一 25-event 闭环评测下的三组 AMCM：

1. foreground-only：仅 foreground weighted matching；
2. mix50：`0.5 * global + 0.5 * foreground`；
3. adaptive：按 foreground attention entropy 逐 token 混合 global/foreground。

## 性能层结论

| 组别 | failure delta | control delta | all delta |
|---|---:|---:|---:|
| foreground-only | -0.03017 | +0.00349 | -0.02344 |
| mix50 | -0.00111 | +0.00703 | +0.00052 |
| adaptive | +0.00048 | +0.00280 | +0.00095 |

foreground-only 的主要极端退化是 `basketball849 -0.5033` 和 `rightcar-chongT -0.3811`。mix50 将前者改善到 `-0.0268`，说明保留 global template evidence 能有效抑制前景 mask 错配造成的错误 residual。adaptive 将 `basketball849` 进一步变为 `-0.0036`，但 `rightcar-chongT` 仍为 `-0.3811`，没有解决第二类错误峰。

## 自适应机制是否真正生效

逐序列 delta 的 Pearson 相关：

- foreground-only vs mix50：`0.7006`；
- mix50 vs adaptive：`0.9949`；
- foreground-only vs adaptive：`0.6767`。

mix50 与 adaptive 几乎给出相同的序列排序。三个训练输出的末步日志也显示 residual/message norm、RGB/TIR gate probability 和 AMCM gradient norm 几乎一致。例如 adaptive 与 mix50 的 message norm 约为 RGB `0.070`、TIR `0.138`，delta norm 约为 RGB `0.037`、TIR `0.070`，gate probability 约 `0.502/0.498`。

因此当前 entropy 权重虽然改变了 matching map，但经过零初始化 residual、空间中心化、cross-modal route 和很小的注入幅度后，没有形成可测量的下游差异。当前收益主要来自 global fallback，而不是 entropy-based target confidence。

## 序列级归因

### 已被修复的 foreground 过激错误

`basketball849`：foreground-only 的大幅退化在 mix50 和 adaptive 中基本消失。这支持 foreground mask 与 search/template 外观不一致时，全局模板证据可以提供稳定回退。

### 仍无法解决的错误峰

`rightcar-chongT`：三组结果均约 `-0.381`。这说明问题不是 foreground/global 比例或 entropy 回退，而可能是：

- 错误峰在 AMCM 输入之前已形成；
- template 与当前 search 的身份匹配已失效；
- residual 对 CENTER 选峰的有效注入过小；
- 或闭环状态/crop 已进入错误反馈链。

只在 AMCM 内继续改变 matching 算子，预计无法解决该类序列。

### 仍有明显收益但无法稳定泛化的序列

`middrillmaster` 和 `rightgirltakingcup` 在 mix50/adaptive 中明显改善，说明 global fallback 在部分遮挡或背景混淆场景有效。但收益集中于少数序列，不能作为稳定方法增益。

## 机制判定

**已支持**：foreground-only 过度依赖前景 mask；global fallback 能抑制一类极端退化。

**尚未支持**：entropy-based uncertainty 能稳定估计目标可信度；adaptive mixture 能解决目标/背景排序问题。

**已定位的边界**：当前 AMCM residual 的有效注入很小，日志中 message 只约为输入特征范数的千分之一量级；因此 matching map 的细微变化很难穿过 TBSI 和 CENTER 形成稳定选峰变化。

## 下一步建议

停止继续搜索 global/foreground 比例、entropy 温度和阈值。只做一次固定 state/crop/frame 诊断，针对 `rightcar-chongT`、`basketball849`、`middrillmaster` 记录：

1. global/foreground/adaptive 在 GT 与 hard-background 的 matching peak；
2. AMCM residual 的 GT/background 方向和幅度；
3. `tbsi_fuse_search` 前后 feature margin；
4. CENTER GT rank 和 peak margin。

若 `rightcar-chongT` 在 matching 前就没有目标优势，应停止旁路 matching 线，转向模板身份/状态恢复；若 matching 有优势但 CENTER 不变，才有理由研究更强但仍可逆的 fusion-to-readout 注入。

