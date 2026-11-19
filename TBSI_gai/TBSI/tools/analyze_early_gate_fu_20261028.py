#!/usr/bin/env python3
"""Paired offline analysis of FU against its public-SOT Early-gate parent."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust

PARENT = Path('/autodl-fs/data/early_gate_closed_loop/dual_message_p1_v2_20261027/runs/P')
FU = Path(os.environ.get('TBSI_FU_RUN_DIR', '/autodl-fs/data/early_gate_fu_20261028/runs/FU'))
EXP = Path(os.environ.get('TBSI_FU_EXP_DIR', str(
    ROOT / 'experiments/early_gate_closed_loop_20261026/fine_grained_unified_20261028')))
MANIFEST = ROOT / 'experiments/early_gate_closed_loop_20261026/dual_message_p1_20261027/data_manifest.json'
GROUPS = ROOT / 'experiments/early_gate_closed_loop_20261026/dual_message_p1_20261027/frozen_groups.json'
PANEL = ROOT / 'experiments/early_gate_closed_loop_20261026/dual_message_p1_20261027/screen96_max2000.txt'
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
    if pred.shape != gt.shape or len(pred) != expected_frames or not np.isfinite(pred).all():
        raise ValueError('Prediction frame/shape/value mismatch: ' + sequence)
    iou, center, norm, valid = calc_seq_err_robust(
        torch.from_numpy(pred), torch.from_numpy(gt), 'lasher')
    iou, center, norm, valid = iou.numpy(), center.numpy(), norm.numpy(), valid.numpy()
    if (not np.isfinite(iou).all() or np.isnan(center).any() or
            not np.isfinite(norm).all() or np.isinf(center[valid]).any()):
        raise ValueError('Unexpected nonfinite metric: ' + sequence)
    lo = np.maximum(pred[:, :2], gt[:, :2])
    hi = np.minimum(pred[:, :2] + pred[:, 2:], gt[:, :2] + gt[:, 2:])
    inter = np.maximum(hi - lo, 0).prod(1)
    direct = inter / np.maximum(pred[:, 2:].prod(1) + gt[:, 2:].prod(1) - inter, 1e-12)
    if not np.allclose(iou[valid], direct[valid], atol=1e-9, rtol=1e-9):
        raise ValueError('Independent IoU check failed: ' + sequence)
    thresholds = np.arange(0.0, 1.00001, 0.05)
    return {
        'auc': float(100 * np.mean([(iou > t).mean() for t in thresholds])),
        'precision': float(100 * (center <= 20).mean()),
        'norm_precision': float(100 * (norm <= 0.2).mean()),
        'op50': float(100 * (iou > 0.5).mean()),
        'op75': float(100 * (iou > 0.75).mean()),
        'valid_gt_frames': int(valid.sum()),
    }, iou


def summarize(rows, group=None):
    selected = [row for row in rows if group is None or row['group'] == group]
    delta = np.asarray([row['auc_delta'] for row in selected], dtype=np.float64)
    rng = np.random.RandomState(428)
    bootstrap = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1)
    ordered = np.sort(delta)
    result = {
        'n': len(selected),
        'mean_auc_delta_fu_minus_parent_pp': float(delta.mean()),
        'median_auc_delta_pp': float(np.median(delta)),
        'trimmed10_auc_delta_pp': float(ordered[int(.1 * len(delta)):len(delta) - int(.1 * len(delta))].mean()),
        'sequence_bootstrap_90pct_pp': [float(x) for x in np.quantile(bootstrap, [.05, .95])],
        'wins_ties_losses': [int((delta > 1e-6).sum()), int((np.abs(delta) <= 1e-6).sum()), int((delta < -1e-6).sum())],
        'losses_over_5pp': int((delta < -5).sum()),
        'new_severe_failure_frames': int(sum(row['new_severe_failure_frames'] for row in selected)),
        'recovered_frames': int(sum(row['recovered_frames'] for row in selected)),
        'metrics': {},
        'worst5': sorted([{'sequence': r['sequence'], 'delta_auc_pp': r['auc_delta']} for r in selected],
                         key=lambda item: item['delta_auc_pp'])[:5],
        'best5': sorted([{'sequence': r['sequence'], 'delta_auc_pp': r['auc_delta']} for r in selected],
                        key=lambda item: item['delta_auc_pp'], reverse=True)[:5],
    }
    for key in ('auc', 'precision', 'norm_precision', 'op50', 'op75'):
        result['metrics'][key] = {
            'parent': float(np.mean([r['parent_' + key] for r in selected])),
            'fu': float(np.mean([r['fu_' + key] for r in selected])),
            'delta_fu_minus_parent': float(np.mean([r[key + '_delta'] for r in selected])),
        }
    return result


def main():
    groups = json.loads(GROUPS.read_text())['groups']
    membership = {name: group for group, names in groups.items() for name in names}
    metadata = json.loads(MANIFEST.read_text())['test_sequences']
    frames = {item['sequence']: int(item['frames']) for item in metadata}
    names = [line.strip() for line in PANEL.read_text().splitlines() if line.strip()]
    if len(names) != 96 or len(set(names)) != 96:
        raise ValueError('Frozen screen96 identity is invalid')

    rows = []
    for name in names:
        if name not in frames or name not in membership:
            raise ValueError('Missing frozen sequence metadata: ' + name)
        parent, parent_iou = measure(PARENT / 'eval96' / (name + '.txt'), name, frames[name])
        fu, fu_iou = measure(FU / 'eval96' / (name + '.txt'), name, frames[name])
        row = {'sequence': name, 'group': membership[name], 'frames': frames[name]}
        for key in parent:
            row['parent_' + key] = parent[key]
            row['fu_' + key] = fu[key]
            row[key + '_delta'] = fu[key] - parent[key]
        row['auc_delta'] = fu['auc'] - parent['auc']
        row['new_severe_failure_frames'] = int(((parent_iou >= .5) & (fu_iou < .1)).sum())
        row['recovered_frames'] = int(((parent_iou < .1) & (fu_iou >= .5)).sum())
        rows.append(row)

    results = {'all': summarize(rows)}
    for group in ('low_target', 'middle', 'normal_protection'):
        results[group] = summarize(rows, group)
    training = json.loads((FU / 'completed.json').read_text())
    eval_info = json.loads((FU / 'eval96' / 'completed.json').read_text())
    parent_train = json.loads((PARENT / 'completed.json').read_text())
    parent_eval = json.loads((PARENT / 'eval96' / 'completed.json').read_text())
    parent_init = json.loads((PARENT / 'initial_state.json').read_text())
    fu_init = json.loads((FU / 'initial_state.json').read_text())
    p_hashes = (PARENT / 'batches.jsonl').read_bytes().splitlines()
    f_hashes = (FU / 'batches.jsonl').read_bytes().splitlines()
    if p_hashes != f_hashes:
        raise ValueError('Augmented input traces differ')
    if parent_init['shared_state_sha256'] != fu_init['matched_parent_state_sha256']:
        raise ValueError('Common public-SOT state mismatch')
    if training['successful_updates'] != parent_train['successful_updates'] or training['successful_updates'] != 468:
        raise ValueError('Training update counts differ')
    if eval_info['sequences'] != len(names) or eval_info['frames'] != sum(frames[x] for x in names):
        raise ValueError('Evaluation coverage differs from the frozen panel')

    checkpoint = torch.load(str(FU / 'checkpoint_u0468.pth.tar'), map_location='cpu')['net']
    learned = []
    for index in range(3):
        prefix = 'backbone.tbsi_layers.%d.v2_template_fusion.fu_allocator.' % index
        learned.append({
            'fusion_index': index,
            'relation_scale': float(checkpoint[prefix + 'relation_scale']),
            'rejection_bias': float(checkpoint[prefix + 'rejection_bias']),
            'initial_rejection_prior': float(.5 * torch.sigmoid(torch.tensor(-6.0))),
            'effective_relation_bound': float(.5 * torch.tanh(checkpoint[prefix + 'relation_scale']).abs()),
            'effective_rejection_upper_bound': 0.5,
        })
    del checkpoint

    report = {
        'question': 'Does fine-grained target/background-referenced receiver-source allocation improve the Early-gate parent?',
        'scope': 'single seed; 468 updates; frozen reused screen96_max2000; exploratory only',
        'pairing': {
            'common_public_sot_sha256': parent_init['shared_state_sha256'],
            'candidate_matched_common_sha256': fu_init['matched_parent_state_sha256'],
            'common_state_equal': True,
            'manifest_sha256_parent': json.loads((PARENT / 'started.json').read_text())['manifest_sha256'],
            'manifest_sha256_fu': json.loads((FU / 'started.json').read_text())['manifest_sha256'],
            'exact_augmented_batch_records_equal': True,
            'microbatches': len(p_hashes),
            'parent_updates': parent_train['successful_updates'],
            'fu_updates': training['successful_updates'],
            'fu_skipped_attempts': training['skipped_attempts_replayed'],
            'train_sequences': training['unique_sequences'],
            'samples': training['samples'],
            'evaluation_sequences': eval_info['sequences'],
            'evaluation_frames': eval_info['frames'],
        },
        'results': results,
        'learned_allocator_scales': learned,
        'runtime': {
            'parent_wall_fps': parent_eval['throughput_fps'],
            'fu_wall_fps': eval_info['throughput_fps'],
            'parent_per_frame_p50_ms': parent_eval['per_frame_latency_ms_p50'],
            'fu_per_frame_p50_ms': eval_info['per_frame_latency_ms_p50'],
            'parent_per_frame_p95_ms': parent_eval['per_frame_latency_ms_p95'],
            'fu_per_frame_p95_ms': eval_info['per_frame_latency_ms_p95'],
            'parent_peak_memory_bytes': parent_eval['peak_gpu_memory_bytes'],
            'fu_peak_memory_bytes': eval_info['peak_gpu_memory_bytes'],
        },
        'limits': [
            'The parent and FU were trained as paired single-stage public-SOT runs, but this is one seed.',
            'The 96-sequence panel is reused development data, not an independent holdout.',
            'FU jointly changes evidence and the interaction operator; this run does not isolate their separate contributions.',
            'Bootstrap intervals quantify sequence variation for these checkpoints, not training-seed uncertainty.',
            'Attention coefficient mass preservation does not imply message-vector norm preservation or tracking utility.',
        ],
        'per_sequence': rows,
    }
    out = EXP / 'analysis_u0468_screen96'
    out.mkdir(parents=True, exist_ok=True)
    files = [
        EXP / ('source_before/source_before_retry1.tar.gz'
               if 'RETRY1' in str(FU).upper() else 'source_before/source_before.tar.gz'),
        ROOT / 'lib/config/tbsi_track/config.py',
        ROOT / 'lib/models/layers/fine_grained_unified_allocator.py',
        ROOT / 'lib/models/layers/tbsi_layer.py',
        ROOT / 'lib/models/tbsi_track/tbsi_track.py',
        ROOT / 'lib/models/tbsi_track/vit_tbsi_care.py',
        ROOT / 'tools/run_dual_message_p1.py',
        ROOT / 'tools/run_early_gate_fu_20261028.py',
        ROOT / 'tools/record_early_gate_fu_20261028.py',
        ROOT / 'tools/analyze_early_gate_fu_20261028.py',
        ROOT / 'experiments/tbsi_track/EG_EARLY_GATE_FU_QUICK_SOT_S42_20261028_RETRY1.yaml'
        if 'RETRY1' in str(FU).upper() else
        ROOT / 'experiments/tbsi_track/EG_EARLY_GATE_FU_QUICK_SOT_S42_20261028.yaml',
        (EXP.parent / 'PROTOCOL.md' if 'RETRY1' in str(FU).upper() else EXP / 'PROTOCOL.md'),
        *([EXP / 'AMENDMENT.md'] if 'RETRY1' in str(FU).upper() else []),
        MANIFEST, GROUPS, PANEL,
        ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar',
        FU / 'checkpoint_u0468.pth.tar', FU / 'eval96/identity.json',
    ]
    source_manifest = {str(path): digest(path) for path in files}
    report['source_hashes'] = source_manifest
    (EXP / 'source_manifest.json').write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (out / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric = results['all']['metrics']
    lines = [
        '# FU: Early gate 细粒度统一分配快训结果', '',
        '日期：2026-10-28。唯一问题：目标／背景参照内容与接收者—来源关系修正，是否改善原 Early gate 的跨模态消息分配。', '',
        '## 配对合同', '',
        '- P 与 FU 均从公共 SOT 初始化单阶段训练，seed 42、468 次成功更新、59,904 samples、977 条序列覆盖；FU 未从训练后 P checkpoint 续训。',
        '- 两臂公共初态 SHA256 一致，训练 manifest 一致，3,744 条增强后 microbatch 记录逐条完全匹配；FU 无跳步/AMP 重试。',
        '- 测试为冻结 screen96_max2000：96 条、54,666 帧，FP32、threads=0、模板更新关闭。', '',
        '## 结果', '',
        '| 分组 | n | P AUC | FU AUC | FU−P (pp) | 90%序列区间 | 胜/平/负（FU） | >5pp损失 |',
        '|---|---:|---:|---:|---:|---|---:|---:|',
    ]
    for group in ('all', 'low_target', 'middle', 'normal_protection'):
        item = results[group]
        auc = item['metrics']['auc']
        lo, hi = item['sequence_bootstrap_90pct_pp']
        wins, ties, losses = item['wins_ties_losses']
        lines.append('| %s | %d | %.4f | %.4f | %+.4f | [%+.4f, %+.4f] | %d/%d/%d | %d |' % (
            group, item['n'], auc['parent'], auc['fu'], item['mean_auc_delta_fu_minus_parent_pp'],
            lo, hi, wins, ties, losses, item['losses_over_5pp']))
    lines += [
        '', '总体指标（P → FU）：PR %.4f → %.4f（Δ%+.4f）；NPR %.4f → %.4f（Δ%+.4f）；OP50 %.4f → %.4f（Δ%+.4f）；OP75 %.4f → %.4f（Δ%+.4f）。' % (
            metric['precision']['parent'], metric['precision']['fu'], metric['precision']['delta_fu_minus_parent'],
            metric['norm_precision']['parent'], metric['norm_precision']['fu'], metric['norm_precision']['delta_fu_minus_parent'],
            metric['op50']['parent'], metric['op50']['fu'], metric['op50']['delta_fu_minus_parent'],
            metric['op75']['parent'], metric['op75']['fu'], metric['op75']['delta_fu_minus_parent']),
        'FU 相对 P 新严重失败帧 %d，恢复帧 %d。三个融合层的关系系数与拒收上界见完整 JSON；系数变化不等于关系判断带来任务收益。' % (
            results['all']['new_severe_failure_frames'], results['all']['recovered_frames']),
        '运行性能：P wall throughput %.2f FPS、FU %.2f FPS；逐帧 p50 %.3f→%.3f ms，p95 %.3f→%.3f ms；峰值显存 %.1f→%.1f MiB。' % (
            report['runtime']['parent_wall_fps'], report['runtime']['fu_wall_fps'],
            report['runtime']['parent_per_frame_p50_ms'], report['runtime']['fu_per_frame_p50_ms'],
            report['runtime']['parent_per_frame_p95_ms'], report['runtime']['fu_per_frame_p95_ms'],
            report['runtime']['parent_peak_memory_bytes'] / 1048576,
            report['runtime']['fu_peak_memory_bytes'] / 1048576),
        '', '## 结论边界', '',
        '本轮把证据与交互形式一起改变，只能判断 FU 组合相对 Early gate parent 的当前快训效用，不能单独归因于内容证据、关系修正或拒收分支。screen96 是重复使用的开发面板，单 seed 区间不是训练 seed 不确定性；不能据此宣布 full 方向或论文创新性成立。',
        '', '逐序列结果：`analysis_u0468_screen96/per_sequence.csv`；完整统计与来源哈希：`analysis_u0468_screen96/summary.json`、`source_manifest.json`。',
    ]
    (EXP / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'pairing': report['pairing'], 'results': results,
                      'learned_allocator_scales': learned, 'runtime': report['runtime']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    from lib.test.analysis.extract_results import calc_seq_err_robust
    torch.set_num_threads(1)
    main()
