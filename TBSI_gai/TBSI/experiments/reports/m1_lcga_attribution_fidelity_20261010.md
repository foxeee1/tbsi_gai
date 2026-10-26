# M1 LCGA-style A1 即时归因与源码忠实度审计（2026-10-10）

## 结论

停止当前实现的 full training/full test 是对的，但 **A0/A1 不是论文 LCGA 的失败证据**。当前 A1 是缺失曲率引导的 LCGA-style 双路窗口适配器；它没有通过原 30 条代理筛选的稳健收益与速度门槛。同权重 alpha 干预显示它并非完全无效：少数序列确有显著即时贡献，但跨序列效应不稳定。此时不训练新版本；忠实实现若继续立项，必须作为新的方法版本、使用另一个预先冻结的 dev 清单。

## 同 checkpoint 快速对照

固定 `M1_LCGA_SEARCH_PROXY_seed42` epoch-3 权重 SHA256 `584cc2ce405e360fc65ebc671805c919ffaa31fcea9c9ae557942e3180943696`，在原训练侧留出 30 条序列上分别运行正常 alpha 和将 `alpha_rgb/alpha_tir` 临时置零。两模式都执行同一 LCGA-style 计算，仅改变残差注入；两遍各 16,701 个跟踪帧，单进程 FP32。正常重放的整数预测框对原 A1 归档逐帧一致，且正常模式的逐序列 canonical AUC/PR/NPR/OP50/OP75 与既有 A1 指标差小于 `1e-5`。alpha-zero 模式的注入残差逐帧严格为零。

| Canonical 30 序列均值 | 正常 A1 | 同权重 alpha-zero | 正常-置零 |
|---|---:|---:|---:|
| AUC | 71.6902 | 69.6217 | +2.0686 |
| PR | 87.5710 | 84.8906 | +2.6804 |
| NPR | 84.1569 | 81.6690 | +2.4879 |
| OP50 | 85.7012 | 83.0794 | +2.6219 |
| OP75 | 70.3173 | 68.5296 | +1.7877 |

AUC 15 胜、14 负、1 平；中位差 `+0.022`，按序列 bootstrap 的均值差 95% 区间 `[-0.258,+5.462]`。最大即时增益为 `redgirl1497` +43.907、`leftbasketball` +15.267、`rightgirlplayingphone` +9.088；去掉这三条，余下 27 条平均 `-0.230`。主要损失为 `boysback` -4.644、`3whitemen` -2.036、`leftcup` -1.607。正常/置零共 2,019/16,701 帧响应峰 token 不同；该计数包括闭环轨迹分叉后的连锁变化，不可解释为 2,019 次独立即时 head 决策。

实际注入量按 `||alpha*(LCGA(x)-x)||_2/(||x||_2+eps)` 逐帧测得。正常帧定义为有效 GT 且 A1 IoU>=0.5；失配起点为有效 GT 的连续至少 30 帧 IoU<0.2 区段首帧，前驱为它的前一帧。正常帧 12,163 个，前驱/起点各 24 个。RGB/TIR 注入比中位数：正常 `3.913e-4/3.005e-4`，前驱 `3.908e-4/2.991e-4`，起点 `3.908e-4/2.988e-4`；没有在失配前驱/起点出现明显放大。小残差仍可能跨越响应 argmax 边界，并经闭环放大，不能由系数或范数单独判定“没有作用”。

CUDA event 测得正常模式 RGB/TIR 分支中位 `0.875/0.822 ms`，合计约 `1.70 ms/frame`；alpha-zero 仍执行分支，中位 `0.881/0.827 ms`。带 hook、同步与逐帧记录的总 tracker 速度为正常 `59.45` FPS、置零 `59.83` FPS，两者接近；这不是不执行模块时的速度。前次无 hook 的 A0/A1 分别为 `73.04/57.47` FPS，约 -21.3%，两种计时口径不能直接相减推导精确模块开销。

上述是 **A1 已联合训练权重中的即时干预**，不是重新训练的公平 A0 对照，也不能将收益唯一归于局部特征而排除 CENTER head 适配。原 A0/A1 均值 +1.19 且 13 胜/16 负/1 平的结论仍成立；alpha 干预进一步说明 `redgirl1497` 的大增益不是单纯由“模块完全未注入、仅联合续训随机漂移”解释。

## 官方源码与当前实现

作者仓库 `github.com/mwaleedaslam/CGA` 固定 commit `8941ace6065fdb9bddb7b56e6c00152f67defd78`，审计文件 `basicsr/archs/cga_arch.py` SHA256 `dd189242247a1edf9439f810a61acdbc595e7900546d7b3aef05a2d9d1ed2ad9`。用户粘贴版 SHA256 `298412eebfebe650ef26680bd0b6f4f6b962add3c0be6f5913fedd9d327c9bcb`；AST 对照表明 `img2windows`、`windows2img`、`DynamicPosBias`、`WindowAttention` 完全相同，`LCGA.__init__`、`calculate_mask`、`forward` 也完全相同，类说明文字不同。故粘贴版核心计算与该官方提交同语义，但原 F 盘实体文件并未在远程读取。

| 机制 | 官方 LCGA | 当前 A1 LCGA-style |
|---|---|---|
| 曲率 | depthwise 3x3 `curvature_conv`、归一化曲率、logit 调制及标准/曲率 attention 混合 | 无曲率路径 |
| 矩形窗口 | 通道对半，`2x4` 与 `4x2` 两方向后 concat | 两支均处理全通道 `2x4`，结果平均 |
| shift | 由 `idx/rs_id` 决定；shift 时使用边界 mask；当前给出的 `idx=0,rs_id=0` 不 shift | 第二支固定 `torch.roll`，没有边界 mask |
| value 局部卷积 | depthwise 3x3 `get_v` 与 attention 输出相加 | 无该分支 |
| 位置偏置 | 官方分层 `DynamicPosBias` | 简化 MLP 偏置 |

当前模块的固定 roll 且无 mask 允许循环移位后的跨边界邻接；这改变了算子定义。官方曲率 alpha 初始为零，移植后仍须做梯度/更新和机制检查，不能假定接入即有效。论文的原始任务为遥感超分辨率，迁移到 TBSI 跟踪还需单独验证。

## 下一步边界

当前 LCGA-style 版本停止，不以其结果否定论文 LCGA。若决定验证官方机制，先移植经上述提交核对的独立新模块，完成 T0-T4、同状态固定 crop 的中心误差/响应峰/目标背景 margin 检查，再考虑一个新的 matched 小规模训练对照。未来评测清单已在查看性能前按 `SHA256("LCGA-FAITHFUL-DEV-20261010:"+序列名)` 排序，从 LasHeR `TBSI_gai/data/lasher/trainingset` 中排除旧训练 176 条、旧 dev 30 条和 3 条坏注释序列后，在 771 个可选成员中取前 30 条并按名称保存；manifest 为 `experiments/proxy/lcga_faithful_dev30_20261010.txt`，SHA256 `4739caca72bb3f8d2ec6dce71a7e1abb9be4c7cac6a8ee354e584a6378bab6d2`。该清单目前未训练、未评测、未用于版本选择。若未来改变训练成员或评价口径，必须重新跑直接 parent 对照。

## 证据与命令

最终诊断源码快照 `output/experiments/M1_LCGA_ATTRIBUTION_20261010/provenance/diagnostic_source_final_v2.tar.gz` SHA256 `c2c1924e2a114d441515b76a269c86f0d04a6aa0681286b8e6d8d261cff1dc3d`；启动前和中间快照保留在同一 provenance 目录。Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`。输入 dev manifest SHA256 `37ad5c642c246d49dbdf7d05bcaab8551bff0a0c75aaa92aa57b5570779c33f2`。逐帧完整记录 `output/experiments/M1_LCGA_ATTRIBUTION_20261010/full30.json` SHA256 `b95791aaadb24a8dbe2ac7da2a1f5796e75fa2868e54846eade627fef5bea58c`，canonical 配对分析 `canonical_summary.json` SHA256 `e5c1430948208e7e9ba292c06ecd46818f6c61e0872653f1fe49544801a6f50b`。6 个无效 GT 帧的非有限中心误差被规范为 JSON `null`，未改框和指标；严格 JSON 解析及 canonical parity 复核通过。首条序列 smoke 输出保留；早期导入/hook 失败未产生预测产物。GPUHours 未独立计时，记 NA。未更改 M1/A0/A1 配置、权重或归档预测。运行后仅修正当前 LCGA-style 源码的说明文字，不改其计算。

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/diagnose_lcga_alpha_20261010.py --output output/experiments/M1_LCGA_ATTRIBUTION_20261010/full30.json --max-sequences 30
TBSI_DBF_G1_EVAL_LIST=/root/autodl-tmp/TBSI_gai/TBSI/experiments/proxy/dbf_g1_20261009/eval_train_sequences.txt /root/autodl-tmp/conda_envs/tbsi/bin/python tools/analyze_lcga_alpha_20261010.py --input output/experiments/M1_LCGA_ATTRIBUTION_20261010/full30.json --output output/experiments/M1_LCGA_ATTRIBUTION_20261010/canonical_summary.json
```
