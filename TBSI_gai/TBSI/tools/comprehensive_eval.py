"""
综合评测：全指标 + 全19属性分析 + 多实验对比

从 eval_data.pkl 中提取指标，用 LasHeR 属性标注按19种挑战分类评估。

用法:
  python tools/comprehensive_eval.py 完美基线小数据 基线+时序令牌 --compare
"""

import argparse
import json
import os.path as osp
import pickle
import numpy as np
import torch

# ===== LasHeR 19 属性定义 (Attributes_order.txt) =====
ATTR_NAMES = [
    'NO',   # 1.  Normal Occlusion
    'PO',   # 2.  Partial Occlusion
    'TO',   # 3.  Total Occlusion
    'HO',   # 4.  Heavy Occlusion
    'MB',   # 5.  Motion Blur
    'LI',   # 6.  Low Illumination
    'HI',   # 7.  High Illumination
    'AIV',  # 8.  Appearance Illumination Variation
    'LR',   # 9.  Low Resolution
    'DEF',  # 10. Deformation
    'BC',   # 11. Background Clutter
    'SA',   # 12. Scale Variation
    'CM',   # 13. Camera Motion
    'TC',   # 14. Thermal Cross
    'FL',   # 15. Fast Motion
    'OV',   # 16. Out of View
    'FM',   # 17. Fast Motion
    'SV',   # 18. Scale Variation
    'ARV',  # 19. Aspect Ratio Variation
]

# 中文名
ATTR_CN = {
    'NO':'正常遮挡','PO':'部分遮挡','TO':'完全遮挡','HO':'严重遮挡',
    'MB':'运动模糊','LI':'低光照','HI':'高光照','AIV':'表观变化',
    'LR':'低分辨率','DEF':'形变','BC':'背景杂波','SA':'尺度变化',
    'CM':'相机运动','TC':'热交叉','FL':'快速运动','OV':'出视野',
    'FM':'快速运动','SV':'尺度变化','ARV':'宽高比变化',
}

ATTR_ATT_DIR = '/tmp/lasher_attr/AttriSeqsTxt'


def load_attributes(seq_names: list) -> dict:
    """加载每序列的19维属性标注。返回 {seq_name: [19 ints]}"""
    attrs = {}
    for s in seq_names:
        fp = osp.join(ATTR_ATT_DIR, f'{s}.txt')
        if osp.exists(fp):
            with open(fp) as f:
                raw = f.read().strip().split(',')
                attrs[s] = [int(x) for x in raw]
        else:
            attrs[s] = None
    return attrs


def compute_metrics(eval_data: dict) -> dict:
    overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    seq_names = eval_data['sequences']
    valid = torch.tensor(eval_data['valid_sequence']).bool()

    seq_list = [seq_names[i] for i, v in enumerate(valid) if v]
    seq_attrs = load_attributes(seq_list)

    # 全局指标
    v_overlap = overlap[valid, 0]
    v_center = center[valid, 0]
    v_center_norm = center_norm[valid, 0]

    thresh = eval_data['threshold_set_overlap']
    op50_idx = min(range(len(thresh)), key=lambda i: abs(thresh[i] - 0.50))
    op75_idx = min(range(len(thresh)), key=lambda i: abs(thresh[i] - 0.75))
    center_thresh = eval_data['threshold_set_center']
    prec_idx = min(range(len(center_thresh)), key=lambda i: abs(center_thresh[i] - 20))
    norm_thresh = eval_data['threshold_set_center_norm']
    nprec_idx = min(range(len(norm_thresh)), key=lambda i: abs(norm_thresh[i] - 0.20))

    auc = v_overlap.mean(dim=0).mean().item() * 100
    op50 = v_overlap[:, op50_idx].mean().item() * 100
    op75 = v_overlap[:, op75_idx].mean().item() * 100
    precision = v_center[:, prec_idx].mean().item() * 100
    norm_precision = v_center_norm[:, nprec_idx].mean().item() * 100

    results = {
        'AUC': round(auc, 2), 'OP50': round(op50, 2), 'OP75': round(op75, 2),
        'Precision': round(precision, 2), 'Norm_Precision': round(norm_precision, 2),
    }

    # ===== 按19属性分场景 =====
    attr_results = {}
    for attr_name in ATTR_NAMES:
        # 找出包含此属性的序列
        idx_tensor = torch.zeros(len(seq_list), dtype=torch.bool)
        count = 0
        for i, s in enumerate(seq_list):
            a = seq_attrs.get(s)
            if a is not None:
                attr_idx = ATTR_NAMES.index(attr_name)
                if attr_idx < len(a) and a[attr_idx] == 1:
                    idx_tensor[i] = True
                    count += 1
        if count == 0:
            attr_results[attr_name] = {'AUC': 'N/A', 'count': 0}
            continue

        a_overlap = v_overlap[idx_tensor]
        a_center = v_center[idx_tensor]
        a_center_norm = v_center_norm[idx_tensor]

        a_auc = a_overlap.mean(dim=0).mean().item() * 100
        a_prec = a_center[:, prec_idx].mean().item() * 100
        a_nprec = a_center_norm[:, nprec_idx].mean().item() * 100

        attr_results[attr_name] = {
            'AUC': round(a_auc, 2), 'Precision': round(a_prec, 2),
            'Norm_Precision': round(a_nprec, 2), 'count': count,
        }

    results['per_attribute'] = attr_results
    results['_seq_attrs'] = {s: seq_attrs.get(s) for s in seq_list}
    return results


def load_experiment(exp_name, base_dir='output/experiments'):
    paths = [
        osp.join(base_dir, exp_name, 'test_analysis'),
        osp.join(base_dir, exp_name, 'test_results'),
        osp.join('output/test/result_plots', exp_name),
        osp.join('output/test/result_plots', 'mini_lasher_test'),
    ]
    for p in paths:
        ep = osp.join(p, 'eval_data.pkl')
        if osp.exists(ep):
            with open(ep, 'rb') as f:
                data = pickle.load(f)
            metrics = compute_metrics(data)
            metrics['_source'] = ep
            return metrics
    return None


def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation')
    parser.add_argument('experiments', nargs='+')
    parser.add_argument('--base_dir', default='output/experiments')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    all_results = {}
    for exp_name in args.experiments:
        m = load_experiment(exp_name, args.base_dir)
        if m is None:
            print(f'  ❌ {exp_name}: not found')
            continue
        all_results[exp_name] = m

    exp_names = list(all_results.keys())

    print(f"\n{'='*70}")
    print(f"  综合评测报告 (19 属性)")
    print(f"{'='*70}\n")

    # 全局指标
    print(f"  全局指标对比:")
    h = f"  {'实验':<22}"
    for _ in exp_names:
        h += f" {'AUC':>6} {'OP50':>6} {'OP75':>6} {'PR':>6} {'NPR':>6}"
    print(h)
    print(f"  {'-'*22}{'-'*32*len(exp_names)}")
    for exp in exp_names:
        m = all_results[exp]
        row = f"  {exp:<22}"
        row += f" {m['AUC']:>6} {m['OP50']:>6} {m['OP75']:>6} {m['Precision']:>6} {m['Norm_Precision']:>6}"
        print(row)
        if exp != exp_names[-1]:
            print(f"  {'↓'*22}")

    # 按19属性对比
    print(f"\n  19 属性 AUC 对比:")
    h = f"  {'属性(中文)':<16}{'英文':>4}"
    for _ in exp_names:
        h += f" {'AUC':>7} {'PR':>7} {'NPR':>7}"
    h += f" {'n':>4}"
    print(h)
    print(f"  {'-'*16}{'-'*4}{'-'*24*len(exp_names)}{'-'*4}")

    for attr in ATTR_NAMES:
        row = f"  {ATTR_CN.get(attr, attr):<16}{attr:>4}"
        # 对比基线(第一个实验)和最后一个实验
        base_m = all_results[exp_names[0]].get('per_attribute', {}).get(attr, {})
        last_m = all_results[exp_names[-1]].get('per_attribute', {}).get(attr, {})
        base_auc = base_m.get('AUC', 'N/A')
        last_auc = last_m.get('AUC', 'N/A')
        count = last_m.get('count', 0)

        for exp in exp_names:
            attr_m = all_results[exp].get('per_attribute', {}).get(attr, {})
            auc = attr_m.get('AUC', 'N/A')
            prec = attr_m.get('Precision', 'N/A')
            nprec = attr_m.get('Norm_Precision', 'N/A')
            row += f" {str(auc):>7} {str(prec):>7} {str(nprec):>7}"
        row += f" {count:>3}"
        # 标记下降
        if isinstance(base_auc, (int, float)) and isinstance(last_auc, (int, float)):
            delta = last_auc - base_auc
            if delta < -0.5:
                row += ' 🔴'
            elif delta > 0.5:
                row += ' 🟢'
        print(row)

    # JSON摘要
    print(f"\n{'='*70}")
    print("  JSON 摘要:")
    summary = {}
    for exp in all_results:
        m = all_results[exp]
        s = {'AUC': m.get('AUC'), 'Precision': m.get('Precision'),
             'Norm_Precision': m.get('Norm_Precision')}
        s['per_attribute'] = {}
        for attr in ATTR_NAMES:
            if attr in m.get('per_attribute', {}):
                s['per_attribute'][attr] = m['per_attribute'][attr].get('AUC', 'N/A')
        summary[exp] = s
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  已保存: {args.output}")


if __name__ == '__main__':
    main()
