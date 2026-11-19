#!/usr/bin/env python3
"""Paired CPU analysis for the fixed-source Early gate screen."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from lib.test.analysis.extract_results import calc_seq_err_robust

RUNS = Path('/autodl-fs/data/early_gate_closed_loop/dual_message_p1_v2_20261027/runs')
FIXED = Path('/autodl-fs/data/early_gate_fixed_control_20261028/runs/FIXED')
EXP = ROOT / 'experiments/early_gate_closed_loop_20261026/early_gate_fixed_control_20261028'
GROUPS = ROOT / 'experiments/early_gate_closed_loop_20261026/dual_message_p1_20261027/frozen_groups.json'
MANIFEST = ROOT / 'experiments/early_gate_closed_loop_20261026/dual_message_p1_20261027/data_manifest.json'
TEST_ROOT = ROOT / 'data/lasher/testingset'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def measure(pred_path, sequence, expected_frames):
    pred = np.loadtxt(str(pred_path), ndmin=2).astype(np.float64)
    gt = np.loadtxt(str(TEST_ROOT / sequence / 'init.txt'), delimiter=',', ndmin=2).astype(np.float64)
    if len(pred) != expected_frames or pred.shape != gt.shape or not np.isfinite(pred).all():
        raise ValueError('Invalid prediction shape/value for ' + sequence)
    iou, center, norm, valid = calc_seq_err_robust(torch.from_numpy(pred), torch.from_numpy(gt), 'lasher')
    iou, center, norm, valid = iou.numpy(), center.numpy(), norm.numpy(), valid.numpy()
    if (not np.isfinite(iou).all() or np.isnan(center).any() or
            not np.isfinite(norm).all() or np.isinf(center[valid]).any()):
        raise ValueError('Nonfinite metric for ' + sequence)
    # Independently verify the overlap implementation used by the repository metric.
    p = pred.copy()
    p[0] = gt[0]
    lo = np.maximum(p[:, :2], gt[:, :2])
    hi = np.minimum(p[:, :2] + p[:, 2:], gt[:, :2] + gt[:, 2:])
    inter = np.maximum(hi - lo, 0).prod(1)
    direct = inter / np.maximum(p[:, 2:].prod(1) + gt[:, 2:].prod(1) - inter, 1e-12)
    if not np.allclose(iou[valid], direct[valid], atol=1e-9, rtol=1e-9):
        raise ValueError('Independent IoU check failed for ' + sequence)
    thresholds = np.arange(0., 1.00001, .05)
    values = {
        'auc': float(100 * np.mean([(iou > t).mean() for t in thresholds])),
        'precision': float(100 * (center <= 20).mean()),
        'norm_precision': float(100 * (norm <= .2).mean()),
        'op50': float(100 * (iou > .5).mean()),
        'op75': float(100 * (iou > .75).mean()),
        'failure_fraction_iou_lt_01': float((iou < .1).mean()),
        'valid_gt_frames': int(valid.sum()),
    }
    return values, iou


def paired_stats(rows, group):
    selected = [r for r in rows if group is None or r['group'] == group]
    delta = np.asarray([r['auc_delta'] for r in selected], dtype=np.float64)
    rng = np.random.RandomState(426)
    boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1)
    ordered = np.sort(delta)
    metrics = {}
    for key in ('auc', 'precision', 'norm_precision', 'op50', 'op75'):
        metrics[key] = {
            'dynamic': float(np.mean([r['dynamic_' + key] for r in selected])),
            'fixed': float(np.mean([r['fixed_' + key] for r in selected])),
            'delta_fixed_minus_dynamic': float(np.mean([r[key + '_delta'] for r in selected])),
        }
    return {
        'n': len(selected),
        'mean_auc_delta_fixed_minus_dynamic_pp': float(delta.mean()),
        'median_auc_delta_pp': float(np.median(delta)),
        'trimmed10_auc_delta_pp': float(ordered[int(.1*len(delta)):len(delta)-int(.1*len(delta))].mean()),
        'sequence_bootstrap_90pct_pp': [float(x) for x in np.quantile(boot, [.05, .95])],
        'wins_ties_losses': [int((delta > 1e-6).sum()), int((abs(delta) <= 1e-6).sum()), int((delta < -1e-6).sum())],
        'losses_over_5pp': int((delta < -5).sum()),
        'new_severe_failure_frames': int(sum(r['new_severe_failure_frames'] for r in selected)),
        'recovered_frames': int(sum(r['recovered_frames'] for r in selected)),
        'metrics': metrics,
        'worst5': sorted([{'sequence': r['sequence'], 'delta_auc_pp': r['auc_delta']} for r in selected],
                         key=lambda x: x['delta_auc_pp'])[:5],
        'best5': sorted([{'sequence': r['sequence'], 'delta_auc_pp': r['auc_delta']} for r in selected],
                        key=lambda x: x['delta_auc_pp'], reverse=True)[:5],
    }


def learned_static_gains(checkpoint):
    state = torch.load(str(checkpoint), map_location='cpu')['net']
    result = []
    for index in range(3):
        prefix = 'backbone.tbsi_layers.%d.v2_template_fusion.' % index
        logits = state[prefix + 'fixed_source_gate.source_logits'].float()
        scale = state[prefix + 'early_gate_scale'].float()
        gain = torch.exp(torch.tanh(scale) * torch.tanh(logits)).clamp(.25, 4.)
        result.append({'fusion_index': index, 'source_logits': logits.tolist(),
                       'scale': float(scale), 'gains_rgb_source_then_tir_source': gain.tolist()})
    return result


def validate_pairing():
    dynamic = RUNS / 'P'
    fixed = FIXED
    dynamic_init = json.loads((dynamic/'initial_state.json').read_text())
    fixed_init = json.loads((fixed/'initial_state.json').read_text())
    dynamic_hash = dynamic_init.get('paired_common_state_sha256') or dynamic_init['shared_state_sha256']
    fixed_hash = fixed_init.get('paired_common_state_sha256') or fixed_init['matched_parent_state_sha256']
    if dynamic_hash != fixed_hash:
        raise ValueError('Common public-pretrained initialization differs')
    mismatches = 0
    batches = 0
    with (dynamic/'batches.jsonl').open() as left, (fixed/'batches.jsonl').open() as right:
        import itertools
        for index, (a, b) in enumerate(itertools.zip_longest(left, right)):
            if a is None or b is None:
                raise ValueError('Training input trace length differs at ' + str(index))
            if json.loads(a) != json.loads(b):
                mismatches += 1
            batches += 1
    if mismatches:
        raise ValueError('Augmented training inputs differ in %d microbatches' % mismatches)
    dtrain = json.loads((dynamic/'completed.json').read_text())
    ftrain = json.loads((fixed/'completed.json').read_text())
    if dtrain['successful_updates'] != ftrain['successful_updates'] or ftrain['successful_updates'] != 468:
        raise ValueError('Successful training update count differs')
    return {
        'common_initial_state_sha256': dynamic_hash,
        'dynamic_state_sha256': dynamic_init['shared_state_sha256'],
        'fixed_full_state_sha256': fixed_init['shared_state_sha256'],
        'exact_augmented_microbatch_records_match': True,
        'microbatches': batches,
        'samples': batches * 16,
        'successful_updates': [dtrain['successful_updates'], ftrain['successful_updates']],
        'skipped_attempts_replayed': [dtrain['skipped_attempts_replayed'], ftrain['skipped_attempts_replayed']],
        'train_sequence_coverage': [dtrain['unique_sequences'], ftrain['unique_sequences']],
    }


def main():
    pairing = validate_pairing()
    groups = json.loads(GROUPS.read_text())['groups']
    membership = {name: group for group, names in groups.items() for name in names}
    test_manifest = json.loads(MANIFEST.read_text())['test_sequences']
    frames = {r['sequence']: int(r['frames']) for r in test_manifest}
    names = [x.strip() for x in (ROOT / 'experiments/early_gate_closed_loop_20261026/dual_message_p1_20261027/screen96_max2000.txt').read_text().splitlines() if x.strip()]
    if len(names) != 96 or len(set(names)) != 96:
        raise ValueError('Frozen panel identity mismatch')
    rows = []
    for name in names:
        if name not in frames or name not in membership:
            raise ValueError('Missing frozen sequence metadata: ' + name)
        dynamic, dyn_iou = measure(RUNS / 'P/eval96' / (name + '.txt'), name, frames[name])
        fixed, fix_iou = measure(FIXED / 'eval96' / (name + '.txt'), name, frames[name])
        row = {'sequence': name, 'group': membership[name], 'frames': frames[name]}
        for key in dynamic:
            row['dynamic_' + key] = dynamic[key]
            row['fixed_' + key] = fixed[key]
            row[key + '_delta'] = fixed[key] - dynamic[key]
        row['auc_delta'] = fixed['auc'] - dynamic['auc']
        row['new_severe_failure_frames'] = int(((dyn_iou >= .5) & (fix_iou < .1)).sum())
        row['recovered_frames'] = int(((dyn_iou < .1) & (fix_iou >= .5)).sum())
        rows.append(row)

    comparison = {
        'all': paired_stats(rows, None),
        'low_target': paired_stats(rows, 'low_target'),
        'middle': paired_stats(rows, 'middle'),
        'normal_protection': paired_stats(rows, 'normal_protection'),
    }
    updates = [json.loads(line) for line in (FIXED / 'updates.jsonl').read_text().splitlines()]
    completed = json.loads((FIXED / 'completed.json').read_text())
    eval_completed = json.loads((FIXED / 'eval96/completed.json').read_text())
    eval_times = np.concatenate([np.loadtxt(str(path), ndmin=1)
                                 for path in sorted((FIXED/'eval96').glob('*_time.txt'))])
    dynamic_times = np.concatenate([np.loadtxt(str(path), ndmin=1)
                                    for path in sorted((RUNS/'P/eval96').glob('*_time.txt'))])
    if len(eval_times) != sum(frames[n] for n in names) or not np.isfinite(eval_times).all():
        raise ValueError('Evaluation timing/frame count mismatch')
    report = {
        'question': 'Does per-token early evidence conditioning improve Value-source gain beyond static per-source calibration?',
        'scope': 'single seed; 468 updates; frozen development screen96; <=2000 frames/sequence; not independent confirmation',
        'comparison': 'fixed minus dynamic Early gate; paired common pre-optimizer state, all 3744 augmented microbatches, fixed evaluator identities',
        'pairing': {**pairing, 'updates': len(updates),
            'successful_updates': sum(bool(r['updated']) for r in updates),
            'skipped_updates': sum(not r['updated'] for r in updates),
            'training_samples': completed['samples'], 'train_sequences': completed['unique_sequences'],
            'evaluation_sequences': len(names), 'evaluation_frames': sum(frames[n] for n in names)},
        'results': comparison,
        'fixed_source_gains_by_fusion_layer': learned_static_gains(FIXED/'checkpoint_u0468.pth.tar'),
        'runtime': {
            'eval_wall_seconds': eval_completed.get('seconds'),
            'peak_gpu_memory_bytes': eval_completed.get('peak_gpu_memory_bytes'),
            'fps': float(1 / eval_times.mean()),
            'mean_frame_time_ms': float(eval_times.mean() * 1000),
            'dynamic_fps': float(1 / dynamic_times.mean()),
            'dynamic_mean_frame_time_ms': float(dynamic_times.mean() * 1000),
            'dynamic_eval_wall_seconds': json.loads((RUNS/'P/eval96/completed.json').read_text()).get('seconds'),
        },
        'limits': [
            'Fixed arm has six active source-logit values total versus three per-layer dynamic predictors; lower capacity, not equal-capacity control.',
            'screen96 is a reused development panel and excludes sequences over 2000 frames.',
            'Sequence bootstrap quantifies this checkpoint panel only, not training-seed uncertainty.',
            'Normal-protection stratum is defined by high dynamic-parent AUC and is not a verified no-degradation cohort.',
            'No full-training or novel-scene claim is supported by this run.'
        ],
        'source_hashes': {
            str(ROOT/'lib/models/layers/fixed_source_value_gate.py'): digest(ROOT/'lib/models/layers/fixed_source_value_gate.py'),
            str(ROOT/'lib/models/layers/tbsi_layer.py'): digest(ROOT/'lib/models/layers/tbsi_layer.py'),
            str(ROOT/'experiments/tbsi_track/EG_EARLY_GATE_FIXED_SOT_QUICK_S42_20261028.yaml'): digest(ROOT/'experiments/tbsi_track/EG_EARLY_GATE_FIXED_SOT_QUICK_S42_20261028.yaml'),
            str(FIXED/'checkpoint_u0468.pth.tar'): digest(FIXED/'checkpoint_u0468.pth.tar'),
        },
        'per_sequence': rows,
    }
    out = EXP / 'analysis_u0468'
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [
        EXP/'source_before/code_before.tar.gz',
        ROOT/'lib/config/tbsi_track/config.py',
        ROOT/'lib/models/layers/fixed_source_value_gate.py',
        ROOT/'lib/models/layers/tbsi_layer.py',
        ROOT/'lib/models/tbsi_track/tbsi_track.py',
        ROOT/'lib/models/tbsi_track/vit_tbsi_care.py',
        ROOT/'tools/run_dual_message_p1.py',
        ROOT/'tools/check_early_gate_fixed_contract_20261028.py',
        ROOT/'tools/record_early_gate_fixed_20261028.py',
        ROOT/'tools/analyze_early_gate_fixed_20261028.py',
        ROOT/'tools/finalize_early_gate_fixed_20261028.py',
        ROOT/'experiments/tbsi_track/EG_DUALMSG_V2_P_QUICK_SOT_S42_20261027.yaml',
        ROOT/'experiments/tbsi_track/EG_EARLY_GATE_FIXED_SOT_QUICK_S42_20261028.yaml',
        EXP/'PROTOCOL.md', GROUPS, MANIFEST,
    ]
    source_manifest = {str(path): digest(path) for path in source_paths}
    source_manifest[str(ROOT/'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar')] = digest(
        ROOT/'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar')
    (EXP/'source_manifest.json').write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2)+'\n')
    report['source_hashes'] = source_manifest
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    fields = list(rows[0])
    with (out/'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    all_result = comparison['all']
    static_gains = [gain for layer in report['fixed_source_gains_by_fusion_layer']
                    for gain in layer['gains_rgb_source_then_tir_source']]
    report_lines = [
        '# Early gate 动态证据与固定来源幅度对比', '',
        '日期：2026-10-28。唯一问题：原 Early gate 的逐模板 token 证据条件化幅度，是否优于每层每来源固定学习幅度。', '',
        '## 实验合同', '',
        '- 动态臂复用 fresh public-SOT P checkpoint u0468；固定臂从相同训练前公共初始化快照开始，fresh AdamW/StepLR，468 成功更新。',
        '- 两臂共有初态哈希相同；3744 个增强后 microbatch 全字段相同；均覆盖 977 条训练序列、59904 个 pairs。',
        '- 测试为冻结 screen96、54666 帧，FP32、threads=0、模板更新关闭；这是重复使用的开发面板，不是独立确认。',
        '- 固定臂每个融合层只有两个活跃 source logits；动态臂每层有 3→32→2 MLP（194 个参数）。固定臂是低容量比较，不是等容量对照。', '',
        '## 结果', '',
        '| 分组 | n | 动态 AUC | 固定 AUC | 固定−动态 AUC (pp) | 90% 序列区间 | 胜/平/负（固定） | 固定>5pp损失 |',
        '|---|---:|---:|---:|---:|---|---:|---:|',
    ]
    for label in ('all', 'low_target', 'middle', 'normal_protection'):
        item = comparison[label]
        m = item['metrics']['auc']
        lo, hi = item['sequence_bootstrap_90pct_pp']
        wins, ties, losses = item['wins_ties_losses']
        report_lines.append('| %s | %d | %.4f | %.4f | %+.4f | [%+.4f, %+.4f] | %d/%d/%d | %d |' %
                            (label, item['n'], m['dynamic'], m['fixed'],
                             item['mean_auc_delta_fixed_minus_dynamic_pp'], lo, hi,
                             wins, ties, losses, item['losses_over_5pp']))
    report_lines += [
        '', '总体其他指标（动态 → 固定）：PR %.4f → %.4f（Δ%+.4f），NPR %.4f → %.4f（Δ%+.4f），OP50 %.4f → %.4f（Δ%+.4f），OP75 %.4f → %.4f（Δ%+.4f）。' % (
            all_result['metrics']['precision']['dynamic'], all_result['metrics']['precision']['fixed'], all_result['metrics']['precision']['delta_fixed_minus_dynamic'],
            all_result['metrics']['norm_precision']['dynamic'], all_result['metrics']['norm_precision']['fixed'], all_result['metrics']['norm_precision']['delta_fixed_minus_dynamic'],
            all_result['metrics']['op50']['dynamic'], all_result['metrics']['op50']['fixed'], all_result['metrics']['op50']['delta_fixed_minus_dynamic'],
            all_result['metrics']['op75']['dynamic'], all_result['metrics']['op75']['fixed'], all_result['metrics']['op75']['delta_fixed_minus_dynamic']),
        '', '替换成固定幅度后，相对动态臂新增严重失败帧 %d，恢复帧 %d；这些帧计数是轨迹差异读数，不是失败原因归因。' % (
            all_result['new_severe_failure_frames'], all_result['recovered_frames']),
        '三层固定增益范围为 %.6f–%.6f，均接近单位增益。动态幅度证据的总体 AUC 优势为 +%.4fpp；低 AUC 组方向一致但序列区间跨零，差异最大在 middle 组。normal protection 组相差约 %.2fpp，接近持平。' % (
            min(static_gains), max(static_gains), -all_result['mean_auc_delta_fixed_minus_dynamic_pp'],
            comparison['normal_protection']['mean_auc_delta_fixed_minus_dynamic_pp']),
        '', '## 判断边界', '',
        '结果支持“动态 Early gate 在当前训练合同与开发面板上优于固定来源校准”，不充分证明涨幅来自证据内容而非动态 MLP 的有效容量。低组区间跨零，不能宣称只改善退化困难序列。单 seed 与重复使用的 screen96 也不能确认 full 方向或稳健性。当前保留动态 Early gate 为性能参照；不晋级 full，不用这组结果宣称机制因果或 EI 投稿条件已满足。下一项若继续，应使用同容量、证据盲的控制来区分输入条件化与预测器容量；这需要新的配对训练，不从本轮结果自动启动。',
        '', '第一次离线分析因把 LasHeR 无效 GT 帧的预期 `Inf` 中心误差当成异常而停止；修正后按仓库 `calc_seq_err_robust` 的 valid 标记计算并对有效帧独立复核 IoU。模型运行与原始预测未改变。',
        '', '逐序列明细：`analysis_u0468/per_sequence.csv`；完整读数、序列 bootstrap、失败/恢复计数及系数：`analysis_u0468/summary.json`。',
        '源文件哈希：`source_manifest.json`。checkpoint：`/autodl-fs/data/early_gate_fixed_control_20261028/runs/FIXED/checkpoint_u0468.pth.tar`。',
    ]
    (EXP/'REPORT.md').write_text('\n'.join(report_lines)+'\n')
    print(json.dumps({'pairing': report['pairing'], 'results': comparison,
                      'fixed_gains': report['fixed_source_gains_by_fusion_layer'],
                      'runtime': report['runtime']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
