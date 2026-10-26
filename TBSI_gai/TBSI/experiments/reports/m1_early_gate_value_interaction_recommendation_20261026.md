# Early gate 后续算子建议：先修复归因，再检验接收条件 Value 残差

本记录仅复核源码、论文草稿与已有结果，不修改模型，不运行训练、跟踪测试或模型前向。新设计尚未实现、尚未验证。GPUHours=0。

## 1. 研究锚点与核心判断

研究 parent 为 Early-conditioned MSHA gate full15，checkpoint SHA256 `e38d18fe50c2895e75e4b9392bf22d9323bb8e6ad31ccbe39ad8c038e3fc1811`；受保护 M1 仍为方法参照。论文草稿 `dcvr_manuscript_draft_20261026.tex` 的主边界是模板 MSHA 的跨模态 Value 作用路径。建议保留该边界和 Early gate 先导方法，将后续创新收窄为“接收模板 token 条件的 Value 内容交互”，不继续叠加模态质量 MLP、方向预算、熵和多种 margin。

这不是已经成立的贡献。当前最明确的证据是实现/归因合同存在缺口，而不是已证实需要一种更复杂的新算子。先完成下述最小判别，才决定是否实现候选。

## 2. 对上一轮 FCSV-R 结论的更正

### 2.1 真实运行与方案不一致

文档要求 v1、gate-bypass native、替代 correct、替代 shifted；实际 runner 只有 NATIVE/correct/shifted 三臂，没有关闭 `TBSI_TEST_EARLY_GATE_STRENGTH`，加载配置又启用了 Early gate。因此实际 NATIVE 是保留 Early gate 的 parent，候选是 Early gate 乘 FCSV-R，不是替代 gate 的独立路由。该运行可以描述一个叠加干预，但不能验证原先设计的替代方案。manifest 的 control 文案还误留了 query-probe 对照，缺少 FCSV alpha/temperature 显式字段；不能以该 manifest 声称全部设计合同已验证。

### 2.2 alpha0 精确相同不代表连续扰动合同通过

实现中 `q=softmax(log(pi)+alpha*z)`，`g=q/clamp(pi)`。对 pi=0 的 token，alpha 非零时 g=0；代码仅在 alpha 恰为 0 时将所有 g 硬设为 1。

因此在背景 token 上 `g(0)=1`，但 `lim(alpha→0, alpha≠0)g(alpha)=0`。alpha0 的精确逐框 parity 是真实结果，却没有证明一个平滑、微小的 Value 重分配。alpha=0.1 同时包含“删除零前景来源”和“前景内部轻微重分配”两个因素。两臂近似相同可能被共同的背景删除主导，目前缺 mask-only 对照，不能确认。

修复用的非创新基线应在 pi>0 上用 `g_k=exp(alpha*z_k)/sum_l(pi_l*exp(alpha*z_l))`，pi=0 上用 g=1；空前景显式报错。这样背景来源不被静默删除，alpha→0 连续恢复恒等。此项是合同修复，不能作为论文贡献。

### 2.3 接收输出变化不是来源消息分解

`query_feature_deltas.csv` 和 boundary CSV 按输出 query 的首帧 mask 分层；测量的是接收 token 的变化，并未把 cross message 按来源 Value 的前景/背景分解。日志中的 hard_bg 是非前景权重补集，不是已定位的最强 hard-negative。`r≈0.025` 是28事件上“接收前景/背景变化比”与 IoU 差的描述相关，既不是来源效用测量，也不能推出不存在可学习的非线性效用关系。

因此撤回“证明是无方向扰动、不能通过训练补足、已否定整条 Value 交互路线”的表述。应写为：**当前实现未显示 correct 优于 shifted；背景删除与关系重分配尚未区分；来源机制尚未测量。暂不支持当前算子进入昂贵训练，但不能据此否定 Value 内容建模。**

### 2.4 数值观察仍保留

28-event mean IoU native/correct/shifted=`0.543864/0.548393/0.548425`。correct-shifted 配对平均差为 `-0.0000319998`。去掉 `whiteshoesleftbottle-gai/frame22` 单事件后，correct-native 平均差为 `-0.00176757`。这些数值不支持广泛改善；固定状态 IoU 不能换算为 LasHeR AUC 涨点。

## 3. 有证据的设计限制与待检验问题

已有事实：Early gate 3→32→2 输入在 search 轴取 max；部分序列方向均值近似固定，但结果正负不一。gate-only full244 单 seed 均值差约 -0.213 pp；full15 有小幅历史正点估计，但逐序列 provenance 未闭合。不能把 gate-only 与 full15 混成同一候选，也不能把任何一个写成稳健收益。

结构限制：对同一来源 k，源权重 g_k 被全部接收 query 共享。它能调节“这个来源总体传多少”，不能表达“这个来源对哪个接收 token 更有用”。FCSV-R 又将 C 的接收维先平均成 `s=pi_receiver^T C`，进一步丢掉接收者差异。

待检验假设：在具有模态局部不一致和多个竞争位置的 crop 内，接收 token 需要不同的跨模态 Value 组合；统一来源缩放会丢失部分对应内容。这个假设尚不是 M1 失败首因，也没有证据指定某一属性标签普遍受益。必须在独立 train/dev 场景核实，不以 test 序列名选机制。

## 4. 主候选：接收条件的去均值 Value 内容残差

这是对 Early gate 路径的一次结构增量，不替换整套 attention。保留 v1；只在首个 MSHA 位置检验一次新的内容交互。第一版不加质量预测器、方向预算和第二分支。首层的选择用于限制干预与归因，未被证明是最优位置。

### 4.1 从相同早期特征保留候选支持

记早期模板/搜索表示为 E_t/E_s，模板数 M=64、搜索数 N=256。两模态共享 crop 与增强定义：

`P_m[k,j] = softmax_j(cos(E_t^m[k],E_s^m[j])/tau)`，初始固定 tau=0.07，不作温度搜索。

对接收模态 n 与来源模态 m：`C_{n<-m}=P_n P_m^T`。本候选保留 C 的每一行，不先做接收前景均值池化。C 只是共同候选支持，不等于可靠性或目标正确性。

### 4.2 每个接收 token 的来源分布

`pi_m[k]=w_m[k]/sum_k w_m[k]`，来源前景为空显式报错。

`z[i,k]=tanh(log(C[i,k]+eps)-sum_l pi_m[l]*log(C[i,l]+eps))`。

`T[i,k]=pi_m[k]*exp(eta*z[i,k])/sum_l(pi_m[l]*exp(eta*z[i,l]))`，首版 eta=1 固定，禁止首轮扫多个 eta。

T 是接收 token i 对来源模板 token k 的分布；pi=0 的来源在新增残差中权重为0，但原生消息不删除。反向用 C 转置和对应来源 pi。若证据显示 C 各行几乎相同/接近均匀，或正确与错位对照缺乏分辨力，应在此停止，不继续做内容变换。

### 4.3 只补接收相关的内容差异

每个 head 使用当前 MSHA 的原生 Value。令 `U_m[k]=g_v1,m[k]*V_m[k]`，g_v1 方向按来源模态解读，不能交换索引。

`bar_U_m=sum_k pi_m[k]*U_m[k]`。

`D_{n<-m}[i]=sum_k(T[i,k]-pi_m[k])*U_m[k]`。

D 是“接收特异内容相对共同前景内容的差”，不是又一个 g_k V_k：同一来源可以对不同接收者产生不同贡献，也可以改变消息向量方向。`sum_k(T[i,k]-pi_m[k])=0`，因此当全部来源 Value 相同或 T 每行等于 pi 时，残差严格为0。该条件只消去共同加性内容，不保证去掉真实背景或守恒消息范数；不能称为退化检测器。

### 4.4 在原 cross message 后做有界修正

父模型 cross message 为 `M0=A_native,cross U_m`，self message 保留父路径。原生 cross-attention mass 为 `a_i=sum_k A_native,cross[i,k]`。

令 `R_i=w_n[i]*a_i*D_i`，使用连续有界缩放：

`Rcap_i=R_i*[rho*||M0_i||/(||R_i||+rho*||M0_i||+eps)]`。

`Mnew_i=M0_i+tanh(gamma)*Rcap_i`。

每个 head/接收 token 独立算范数，rho 首版固定0.1，gamma 为双方向共享单个参数、初始0。此定义使新增消息范数小于 rho 倍父 cross message，避免旧 alpha 无法约束实际消息的缺陷；但不能保证经过输出投影或后续网络后仍是10%变化。M0=0 时 residual=0 是刻意保守边界。

gamma=0 时沿父运算路径直接返回，零幅度与小正/负幅度都必须检查。单 batch 检查 gamma 梯度是否有限非零；后续 backbone 的关系路径在 gamma 更新后才起效。不加入零初始化投影叠加另一重零乘因子。

新的信息流：早期模板/搜索支持 → 每个接收模板 token 的来源分布 → 去共同前景分量的 Value 内容残差 → 原 cross-message 有界修正 → 原投影/模板残差 → 原后续网络。

复杂度来自 M×M 的接收—来源交互，而不是更大MLP。新增可学习参数首版只有一个gamma，但整个主干联合训练仍能调整P/T所依赖的表示；不能据参数少断言学不到东西，也不能预先保证任务损失能驱动这个关系。若复用已有P，M=64、N=256、D=768时，构造C约1.05M MAC，双向T乘Value约6.29M MAC，另有softmax、去均值、范数和逐元素运算；这是首个MSHA位置每样本静态估算，未计P计算、不代表实测FPS。

### 4.5 论文表述边界

同一被干预层的原生 Q/K 和 softmax attention 没有改写；新增残差相当于一个有正负项的有效 cross-Value 混合权重。因此只能写“保留原生 QK 匹配并增加有界 Value 内容修正”，不能写“总交互权重不变”或仍把所有有效权重叫概率。后续层输入已变，后续 Q/K 当然可间接变化。

它仍在跨模态模板 Value 路径内；无需把论文改成搜索帧修复或模板缓存更新。MSHA 的每次模板特征交互也不能等同于在线模板更新。与现稿 v2 相比，方法节须把方向/token 预算公式改为接收条件内容残差；标题中的协同 Value 交互比通用不确定性估计更贴切。

## 5. 条件备选与禁止的扩展

若实测发现 correct/shifted 的共同误响应主要落于 GT 外竞争背景，才考虑把 C 的证据换成前景相对背景支持（同一内容残差不变）。先用初始模板前景/上下文构造搜索响应的背景参考，检查它能否区分 GT 支持和 hard-negative；测试时不得使用当前帧 GT。这会引入第二假设，不与主候选首轮同时实施。

若 C 根本无候选对应信息，优先研究匹配证据的可学习性/监督路径，暂不增加 transport rank 或 MLP。历史随机 query-conditioned probe 无提升不能排除训练后 pairwise 模块；历史已训练 Value-reconstruction 同 checkpoint 开关影响很小，则是其已测合同下的有限负证据，不能略过。

不建议把更大 MLP、更多熵/margin、head级 scalar gate、QKV前输入重构、搜索帧分支、模板更新同时组合。FCCR²/QKV前重构只在用户决定改变 Value-only 论文边界时再议。频域线保持停用。

## 6. 与相关工作的真实区别和重合风险

本轮没有重新检索/获取外部论文源码；以下按用户提供的文献对照限定，不声称已完成2025–2026文献核验或查重。

| 方法簇 | 已知重合 | 候选需要证实的区别 |
|---|---|---|
| TUMFNet / UATrack | 模板—搜索证据、质量/不确定性、token或模态加权 | 不宣称新不确定性理论；研究原模板MSHA内部的接收相关Value内容差分 |
| QSTNet / MRTTrack | token选择、目标相关证据 | 新增控制的对象是模板来源Value的组合；search仅提供证据，不写入新search特征 |
| CGATrack | source/receiver协同、双向传递；最接近 | 不以“来源可靠+接收需求”作新颖性；核对其是否已有同位置接收条件、去共同分量的Value残差 |
| CAFormer及attention校正 | 交互关系修正 | 原生QK不替换，但有效Value混合系数发生变化；不伪称与attention重权完全无关 |
| 历史FCCR²/随机pairwise/Value reconstruction | 共支持、接收条件、内容残差各自已有先例 | 必须用有效对照证明本候选的接收条件与去均值共同必要，不能靠名称区分 |

跨领域依据是残差适配、条件聚合和共同分量中心化这三种通用原则；不能把它们本身写成原创。潜在贡献只能是：在已有模板MSHA的跨模态Value路径，用候选支持表达接收相关内容，并通过去共同分量/有界修正降低不必要扰动，且实验证明这一组合必要。需引用具体来源，不作“首次/最新/优于”保证。

安防监控属于应用背景，低照度/热干扰是待验证场景。添加场景前缀不会产生方法区别；如果最接近工作已有等价算子和作用点，应换问题或停止主张新颖性。

## 7. 最快且可判别的实验安排（本轮仅设计）

1. **一次合同辨析，先不训练。** 在同 checkpoint/同 crop 缓存张量上比较现有 Early gate、mask-only、连续修复后的FCSV correct/shifted，分开检查v1保留/旁路身份。记录真实来源消息 `A_cross(U*w)` 与 `A_cross(U*(1-w))` 的向量及范数/夹角；两向量可能相消，范数不具可加性。检查零幅度及趋近零连续性、C逐行差异、环境开关与真实layer mapping。此步只区分共性合同障碍，不以不训练IoU无涨点否定一个可学习候选。
2. **主候选机制辨析。** 仅在C有真实候选对应证据时，比较v1、接收条件残差、行平均残差、接收行打乱、search轴错位；后两项要检查残差范数差异，必要时仅作诊断的范数匹配。C、T、D对准目标与干扰物的差别先要可测。不用GT驱动推理算子；GT只用于离线读数。
3. **匹配学习验证。** 合同、梯度、mask train/test一致和小样本可拟合通过后，允许一组parent/候选短训检验可学习性。不把随机或未训练probe不涨点设成一票否决。默认quick3只有12次optimizer更新、quick7是反复使用开发集；完成快速重复性/增强后输入hash核验后，其结果仍只属探索。无充分学习且无增益时，结论是“当前预算未检出”，不能写成模块无效；不因此反复扩大架构。
4. **论文证据。** 在更大预先固定dev cohort复核后，再以同一SOT预训练起点、训练日程、实际steps和完整LasHeR评测比较M1、重训Early gate、主候选。学习到的残差在同checkpoint关闭后效应应衰减；正确vs打乱、去均值vs不去均值分别解释机制。报告多seed/至少重复control的不确定性、逐序列尾部伤害与效率。RGBT234未下载未测，保留NA。

阶段2若证据矩阵无对应分辨率，禁止扩展新模块；阶段3若连续两轮只有无判别信息的负结果，回查输入/梯度/监督/状态合同。只有步骤4才支持最终性能贡献。所有新运行使用独立身份、先保存代码快照并遵守进程上限。

## 8. 证据与版本

- git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；dirty worktree，只做源码和文档审阅。
- 当前`tbsi_layer.py` SHA256 `d93f487fc06b16f511ea423bc3bd1f27f2a67a769256f9db71512178a986c4c9`。
- fixed-state runner SHA256 `23d534bf59211cf97ad20b328cb473e14ab5e1f0ceffd46c202f2b31c495e4a8`。
- 论文草稿 SHA256 `4cdae70eebb14bc70d4e33e6704ee0668d377d9431072fbd4d81700dafe6c194`。
- FCSV fixed28 `center_results.csv`、`branch_stats.csv`、`query_feature_deltas.csv`、`query_boundary_deltas.csv`与原manifest保持原样；本次不改原始输出。
- 参照报告：`m1_dcvr_v1_v2_value_source_reallocation_20261026.md`、`m1_early_gate_optimization_analysis_20261026.md`、`m1_early_gate_sequence_scene_attribution_20261026.md`、`m1_query_conditioned_cross_value_fixed28_20261024.md`、`m1_early_template_value_reconstruction_fixed46_20261023.md`、`m1_dcvr_full15_attribution_operator_design_20261026.md`、`m1_early_gate_control_repeatability_20261026.md`。

当前结论：`AUDIT-CORRECTED / SOURCE-MECHANISM-INCONCLUSIVE / SINGLE-CANDIDATE-PROPOSED / NO-TRAINING`。

## FCSV 连续性与 mask-only 对照结果

为区分“背景硬删除”和“共支持重分配”，在同一 Early gate full15 checkpoint、同一 fixed28 状态上新增三臂：`FCSV_R_MASK_ONLY`、`FCSV_R_CONTINUOUS_CORRECT`、`FCSV_R_CONTINUOUS_SHIFTED`。连续版本对正前景先验使用 `exp(alpha*z)` 的 simplex 归一化，零前景 token 保留 gate=1；mask-only 只将零前景来源置零。alpha 和 temperature 仍为 `0.1/0.07`，不训练。

| arm | mean fixed-state IoU | 相对 NATIVE | 胜/负/平 |
|---|---:|---:|---:|
| NATIVE | 0.543571 | 0 | - |
| MASK_ONLY | 0.548271 | +0.004699 | 14/13/1 |
| CONTINUOUS_CORRECT | 0.543578 | +0.0000066 | 9/18/1 |
| CONTINUOUS_SHIFTED | 0.543578 | +0.0000066 | 14/13/1 |

mask-only 几乎复现此前 correct 的平均变化和单事件大幅提升；连续 correct/shifted 的 L1 IoU 变化仅 `0.00329/0.00233`，最大单事件变化小于 `0.00061`。因此旧 FCSV-R 结果主要由零前景来源的硬删除造成，而不是由正确共支持产生的来源重分配。该结果同时说明连续版本在当前固定 `alpha=0.1` 下扰动过小，不能据此判断接收条件内容交互是否无效。

**决策：** FCSV-R 不进入训练，也不再做 alpha/temperature 搜索。保留其作为失败归因资产：`mask_only` 是背景截断效应，continuous correct/shifted 是共支持效应。下一步只做接收条件 Value 内容残差的 zero-init、梯度和同状态固定预检；若接收行打乱与正确关系无法区分，再停止该交互线。
