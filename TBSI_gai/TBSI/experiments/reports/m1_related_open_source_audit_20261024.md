# M1 相近工作与公开源码审计

日期：2026-10-24  
研究 parent：`TBSI_E_MHSA_FULL_4090D_seed42`（M1，受保护）  
审计目的：确认 Early-conditioned MSHA gate 的相近工作、可复用源码和可声明的区分点。

## 1. 结论摘要

当前最接近的已公开 RGB-T 工作是：

1. **BTMTrack (2025)**：双模板桥接、时序/模态候选消除，思想上最接近“目标条件地筛选并写入模板消息”。截至本审计没有找到可核验的作者官方 GitHub 源码，因此只能作为论文邻近工作，不能写成已复现的源码基线。
2. **CSTNet (2024)**：有完整 MIT 源码，显式进行跨模态、通道和空间融合，是目前最适合做接口对照和移植实验的公开实现。
3. **CFBT (2024)**：跨时空增强、互补融合和双流 adapter；截至本审计没有找到可核验的官方公开源码。
4. **USTrack (2023)**：官方仓库存在，但当前 commit 只有 README，README 承诺未来发布代码，不能作为可运行源码。

跨领域参考实现：

- **NLformer**：局部一阶窗口注意力 + 二阶通道协方差 + Woodbury complementary branch。
- **WeightFormer**：由当前输入动态生成线性层/深度卷积残差权重。
- **SDANet**：动态稀疏注意力、hard top-k、STE 和局部分支；当前频域路线按顶层规则暂停，不作为近期候选。

因此，下一条可区分且逻辑清晰的路线应写成：

> 在 M1 的 crop 内目标仍存在但发生干扰或模态冲突的场景中，MSHA 对模板到搜索区消息缺少目标条件的空间选择，导致背景证据与目标证据被同样写入。本文设计一个保留搜索空间对应的目标条件写入算子，控制每个 search position 的跨模态消息写入，而不是只输出全局 scalar gate。

这与当前 Early gate 的区别是结构性的：当前 gate 使用 `max` 压缩搜索区并输出 `3 -> 32 -> 2` 的方向级幅度；新路线保留 `H x W` 对应关系，并将门控作用于 search-side message/write。

## 2. 相近工作对照

| 工作 | 年份 | 源码状态 | 核心操作 | 模板更新/空间对应 | 与当前方法的关系 | 可区分点 |
|---|---:|---|---|---|---|---|
| BTMTrack | 2025 | 未找到作者官方可核验仓库 | TMCE 候选消除；TDTB 双模板桥接 | 目标 token 筛选；保留候选结构 | 最近邻，涉及目标条件和双模板 | 聚焦 M1 的 crop 内干扰，做单一 search-side 写入算子，不复制 TMCE/TDTB 全系统 |
| CFBT | 2024 | 未找到作者官方可核验仓库 | CSTAF、CSTCF、DSTA | 跨时空/模态 adapter | 说明显式交互比 scalar gate 更常见 | 不做完整双流 adapter，限定为 MSHA 消息写入选择 |
| USTrack | 2023 | 仓库仅 README | 统一单阶段 RGB-T Transformer | 无可复用实现证据 | 只能作论文邻近参考 | 不把未公开源码当作基线 |
| CSTNet | 2024 | MIT，源码完整 | CFM -> GIM -> SFM，通道/空间显式融合 | patch 空间保留，残差回写 | 最适合接口对照 | 当前方法以目标条件 write mask 为中心，不复刻三段融合 |
| NLformer | 2026 | MIT，源码完整 | 局部窗口、一阶/二阶协方差、互补分支 | 保留空间结构 | 可提供比 scalar gate 更丰富的交互算子 | 需要目标条件 adapter；不直接移植整套去噪网络 |
| WeightFormer | 2026 | MIT，源码完整 | 输入条件动态生成权重 | 条件化算子但无目标对应 | 可作为动态融合后备 | 需要加入 RGB/TIR 模态互补和模板条件，变量较多 |
| SDANet | 2026 | 当前目录有源码 | 动态稀疏、top-k、STE、局部分支 | 稀疏选择 | 与候选消除方向相关 | 频域/STE 混杂，当前规则暂停，不作为第一候选 |

## 3. 已搭建的外部源码

所有仓库均位于独立只读参考目录，未接入 TBSI 的 import 路径，也没有覆盖 M1 文件。

### CSTNet

- URL：<https://github.com/LiYunfengLYF/CSTNet>
- commit：`0adfb336c9424907b0895c5e94ac875a4e9a2f1a`
- license：MIT（见 `external_repos/CSTNet/LICENSE`）
- 关键入口：
  - `external_repos/CSTNet/lib/models/cstnet/rgbt_fusion.py`
  - `external_repos/CSTNet/lib/models/cstnet/cfm.py`
  - `external_repos/CSTNet/lib/models/cstnet/sfm.py`
- `rgbt_layer.forward` 的信息流是：拆分 template/search token -> `token2patch` -> CFM 跨模态融合 -> GIM 交互 -> SFM 空间融合 -> 残差加回 -> `patch2token`。
- 对 TBSI 的映射：可借鉴 patch 空间保留和残差回写；不能直接替换 MSHA，因为 token 数、骨干特征和训练接口不同。

### NLformer

- URL：<https://github.com/Ryann2001/NLformer_Lightweight_Denoising>
- commit：`e20911303c1529218fc55c607ca19b9e338ea44f`
- license：MIT（见 `external_repos/NLformer_Lightweight_Denoising/LICENSE`）
- 关键入口：`external_repos/NLformer_Lightweight_Denoising/model/nlformer_arch.py` 的 `Attention_block`。
- 信息流：局部窗口 attention；通道协方差 attention；Woodbury complementary branch；三路拼接后投影。
- 对 TBSI 的映射：可以只提取“局部空间 + 模态/通道二阶互补”这一算子思想，并增加 template-conditioned adapter；不能把去噪任务的完整 encoder-decoder 作为 RGB-T tracker 模块直接使用。

### WeightFormer

- URL：<https://github.com/LeapLabTHU/WeightFormer>
- commit：`95d5359ec69bdca0e8b4267e30af9e06fca36037`
- license：MIT（见 `external_repos/WeightFormer/LICENSE`）
- 关键入口：`external_repos/WeightFormer/model/h_wfm.py` 的 `DynLinear`、`DynCnn`。
- 信息流：池化当前 token/feature map，生成低秩动态权重残差，与静态权重相加后完成线性层或 depth-wise convolution。
- 对 TBSI 的映射：适合研究场景条件化算子，但必须额外引入模板条件和 RGB/TIR 互补约束；如果与空间 write mask 同时引入，会无法区分增益来源。

### SDANet

- URL：仓库已保存于 `external_repos/SDANet`；commit `8527d9da191d29b11b6e8737bd9f00f5c412de0a`
- 关键入口：`external_repos/SDANet/basicsr/archs/SDAnet_arch.py` 的 `DSCA`。
- 信息流：动态稀疏因子 -> hard top-k/STE mask -> global sparse attention，并行 local depth-wise branch。
- 当前处理：不启动该路线。顶层实验规则已暂停频域改进线，且该实现同时引入稀疏、STE 和频域分支，无法作为近期单变量验证。

### USTrack

- URL：<https://github.com/xiajianqiang/USTrack>
- commit：`c9fd13bac89d7ec8e12546efb285691b2e8d99d9`
- 当前仓库只有 README；README 写明“code will be released”，没有可运行的模型源码。

## 4. 与当前 Early-conditioned MSHA gate 的边界

当前 gate 已验证的结构是：对每个 early template token 计算 RGB/search 最大 cosine、TIR/search 最大 cosine、RGB/TIR template 一致性，经 `3 -> 32 -> 2` MLP 得到双方向 gate，只缩放后续 cross-modal template Value。其主要限制是：

1. `max` 抹掉搜索区位置；
2. 没有显式目标 token 身份；
3. 只改变消息幅度，不改变消息内容或空间对应；
4. gate 均值接近恒等，当前 fixed-state 尚无 CENTER 排序改善证据。

因此论文或报告中不能把当前方法描述成“首次做目标条件融合”，因为 BTMTrack、CSTNet 等已经覆盖了目标筛选或显式空间/模态融合。可声明的差异应限制为：

- 问题边界：M1 中 crop 内目标仍存在、但背景/模态冲突导致闭环写入漂移；
- 算子位置：MSHA template-to-search message 的 search-side write control；
- 设计约束：空间保留、identity initialization、同一 crop 下 target/shifted/decoy 固定状态诊断；
- 目标：减少错误消息写入，而不是重新设计完整 tracker。

## 5. 推荐的移植顺序

1. **第一候选：目标条件空间写入控制（BTMTrack-inspired，但不复制其系统）**
   - 每个 search position 生成 write mask/weight；
   - 条件来自 early template token 与 search token 的局部相似度和 RGB/TIR 一致性；
   - identity 初始化；
   - 先做 fixed-state target vs shifted/decoy，再做单变量 quick train。
2. **第二候选：CSTNet-style local/channel spatial fusion**
   - 仅移植 CFM/GIM/SFM 的最小空间交互子集；
   - 保留 MSHA 主路径，作为旁路 residual；
   - 不和第一候选同时改，避免无法归因。
3. **第三候选：NLformer 的局部 + 二阶互补算子**
   - 只移植 `Attention_block` 的局部/协方差思想；
   - 加模板条件 adapter 后再测；
   - 计算量和稳定性风险高于前两者。

WeightFormer 作为后备动态算子，SDANet 暂缓。

## 6. 合规与复现实验说明

- 外部仓库保留各自 LICENSE 和 commit；若复制 MIT 代码到 TBSI，必须保留版权与许可文本，并在新文件头部注明来源。
- 当前没有复制外部代码到 TBSI；上述目录仅用于审计和接口对照。
- BTMTrack、CFBT 未找到官方公开源码，不应声称“已搭建其源码”。如后续找到作者发布仓库，应新增独立目录和 commit 记录，不覆盖本报告。
- 任何移植实验必须使用新的实验名、config 和 output 目录，不得修改 M1 的配置、checkpoint、结果或自动回写逻辑。

## 7. 张量级信息流对照

### 7.1 当前 M1 原生 MSHA

对一个 MSHA 层，输入先按 token 轴拼接 RGB/TIR 的 template 与 search。以 template-to-search 路径为例：

```text
z_rgb [B,64,C] + z_tir [B,64,C] + x_rgb [B,256,C] + x_tir [B,256,C]
        |
        +-- q = search tokens [B,256,C]
        +-- k,v = template tokens [B,64,C]
        |
        +-- logits = q k^T / sqrt(d)        [B,256,64]
        +-- softmax/entmax(logits)           [B,256,64]
        +-- message = attention @ v          [B,256,C]
        +-- linear projection                [B,256,C]
        +-- residual + FFN
```

这里的空间对应主要隐含在 `q k^T` 的相似度中。标准 MSHA 没有显式的 RGB/TIR 融合特征生成器；两种模态只是共同进入 token 集合，随后由 attention 权重决定消息来源。

### 7.2 当前 Early-conditioned MSHA gate

你的 Early gate 在主 MSHA 之前另取 early template/search 特征，仅用于生成条件：

```text
early_rgb_t [B,64,C], early_tir_t [B,64,C]
early_rgb_s [B,256,C], early_tir_s [B,256,C]
        |
        +-- RGB cosine: [B,64,256] -- max(search) --> [B,64]
        +-- TIR cosine: [B,64,256] -- max(search) --> [B,64]
        +-- RGB/TIR template agreement             --> [B,64]
        |
        +-- per-token [3] -> MLP(3,32,2) -> gate_delta [B,64,2]
        +-- exp(gate_delta) -> cross_value_gate
        |
main MSHA: a_rt @ (v_tir * gate_tir)
           a_tr @ (v_rgb * gate_rgb)
```

因此它改变的是：

- `Value` 消息的幅度；
- RGB->TIR、TIR->RGB 两个方向的消息强度。

它没有改变的是：

- `q`、`k` 以及 attention logits；
- `argmax/排序` 所依据的 attention 几何；
- 每个 search position 的独立写入权重；
- template token 的内容本身；
- 模板更新状态。

这正是 fixed-state 中“消息范数有变化但 CENTER 排序不变”的结构性原因。

### 7.3 CSTNet

CSTNet 的 `rgbt_layer` 信息流为：

```text
RGB/TIR template 和 search token
        |
        +-- token2patch -> [B,C,H,W]
        |
        +-- CFM(x_rgb,x_tir)
        |     concat channel -> Linear(2C,C)
        |     -> LSA spatial relation
        |     -> SE channel reweight
        |     -> shared fusion residual 加回 RGB/TIR
        |
        +-- GIM(x_rgb,x_tir)
        |     patch2token -> concat token sequence
        |     -> shared MLP + LayerNorm
        |     -> split 回 RGB/TIR
        |
        +-- SFM(x_rgb,x_tir)
        |     LPU local positional update
        |     -> cross channel attention
        |     -> x_rgb + x_tir
        |     -> CFN spatial/channel projection
        |
        +-- fusion residual 加回原始 RGB/TIR
        +-- patch2token -> 后续 tracking head
```

与当前 gate 的根本差别：CSTNet 在融合特征形成阶段同时改变通道混合、局部空间关系和跨模态交互；你的 gate 在主 attention 已产生消息后只缩放 Value。CSTNet 的融合输出是新的 `C` 维特征，Early gate 的输出仍是原消息乘以一个接近 1 的标量。

### 7.4 NLformer

NLformer 不是 RGB-T tracker，但它展示了比单一幅度 gate 更丰富的算子组合：

```text
x [B,C,H,W]
  |
  +-- first-order local window branch
  |     q,k,v -> overlapping window attention -> out_s
  |
  +-- second-order channel covariance branch
  |     q_c,k_c,v_c -> covariance(q_c) -> out_c
  |
  +-- complementary branch
  |     phi(x) -> U -> Woodbury inverse -> out_phi
  |
  +-- concat(out_s, out_c, out_phi) -> 1x1 projection -> residual
```

它的价值不在于直接复制，而在于说明“融合”可以同时改变局部空间关系、通道统计和互补残差。当前 gate 只有一个 evidence MLP，没有形成这三类新的特征分支。

### 7.5 WeightFormer

WeightFormer 的核心不是显式 RGB/TIR 交互，而是让算子参数随输入变化：

```text
x token/map
  |
  +-- pooled x -> weight generator
  |     -> low-rank dynamic residual weight
  |
  +-- static weight + dynamic residual weight
  +-- dynamic Linear 或 depth-wise Conv
  +-- output feature
```

你的 gate 是“输入条件的消息系数”；WeightFormer 是“输入条件的算子参数”。后者能改变特征变换方向，但没有目标 token、模态互补或 tracking crop 语义，直接移植会引入较多新变量。

### 7.6 BTMTrack/CFBT 的论文级信息流

从论文公开描述可确认的抽象流程是：

```text
双模板/历史模板 + 当前 RGB/TIR search
        |
        +-- temporal-modal candidate elimination
        |     筛除低相关或冲突 token
        |
        +-- dual-template bridging
        |     在保留目标候选的条件下进行模板桥接和跨模态交互
        |
        +-- tracking prediction / template update
```

但由于没有官方可核验源码，不能进一步声称其具体 tensor shape、loss 或更新时机。它与当前路线最接近的地方是“先判断哪些证据值得传播”；区别在于你目前只做了全局 max evidence 后的 Value 幅度调制，还没有实现候选 token 的显式保留或 search-position 写入控制。

## 8. 一句话比较

```text
别人多数在改变：融合对象、空间/通道交互、候选 token 或动态算子本身
你的 Early gate 当前改变：已经生成的跨模态 Value 消息的方向级幅度
```

因此，当前方法不是理论错误，而是作用位置偏后、自由度偏小。若要与公开工作拉开距离，最清晰的升级是：保留 M1 的 MSHA 主体，只新增一个目标条件的 `search-position write operator`，让每个 search token 决定跨模态消息写入多少；这样既承接现有 gate 的证据输入，又真正改变融合特征，而不是只改变消息范数。
