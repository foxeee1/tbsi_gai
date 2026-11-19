#!/usr/bin/env python3
"""Independent P1 measurement/reporting, with no legacy audit gate imports."""
import argparse
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

RUN = ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3'
METRICS = ['auc', 'precision', 'norm_precision', 'op50', 'op75']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def metrics(pred, gt):
    if pred.shape != gt.shape or not np.isfinite(pred).all():
        raise ValueError('Prediction length/finite-value contract failed')
    iou, center, norm, valid = calc_seq_err_robust(torch.tensor(pred, dtype=torch.float64),
                  torch.tensor(gt, dtype=torch.float64), 'lasher')
    iou, center, norm, valid = iou.numpy(), center.numpy(), norm.numpy(), valid.numpy()
    thresholds = np.arange(0., 1.00001, .05)
    success = np.array([(iou > t).mean() for t in thresholds])
    pred = pred.copy(); pred[0] = gt[0]
    if (pred[:, 2:] > 0).all() and (gt[:, 2:] > 0).all():
        # Independent direct intersection calculation validates the readout.
        lo = np.maximum(pred[:, :2], gt[:, :2])
        hi = np.minimum(pred[:, :2] + pred[:, 2:], gt[:, :2] + gt[:, 2:])
        intersection = np.maximum(hi - lo, 0).prod(1)
        direct_iou = intersection / (pred[:, 2:].prod(1) + gt[:, 2:].prod(1) - intersection)
        if not np.allclose(iou, direct_iou, atol=1.e-9, rtol=1.e-9):
            raise ValueError('Independent IoU cross-check failed')
    failed = iou < .1
    longest, current = 0, 0
    for item in failed:
        current = current + 1 if item else 0; longest = max(longest, current)
    result = {'auc': float(100 * success.mean()), 'op50': float(100 * (iou > .5).mean()),
              'op75': float(100 * (iou > .75).mean()), 'precision': float(100 * (center <= 20).mean()),
              'norm_precision': float(100 * (norm <= .20).mean()),
              'failure_fraction': float(failed.mean()), 'longest_failure_frames': longest}
    return result, iou


def read_result(folder, name, frames):
    pred = np.loadtxt(str(Path(folder) / (name + '.txt')), ndmin=2)
    gt = np.loadtxt(str(ROOT / 'data/lasher/testingset' / name / 'init.txt'), delimiter=',', ndmin=2)
    if len(gt) != frames:
        raise ValueError('Manifest GT count changed: ' + name)
    return metrics(pred, gt)


def freeze(args):
    target = RUN / 'frozen_groups.json'
    if target.exists():
        raise RuntimeError('Refusing to redefine frozen outcome groups')
    data = json.loads((RUN / 'data_manifest.json').read_text())
    frames = {r['sequence']: r['frames'] for r in data['test_sequences']}
    sampling_path = RUN / 'screen_sampling_max2000.json'
    if not sampling_path.exists():
        sampling_path = RUN / 'screen_sampling.json'
    sampling = json.loads(sampling_path.read_text())
    rows = []
    for sample in sampling['members']:
        name = sample['sequence']
        m, iou = read_result(args.reference, name, frames[name])
        rows.append({'sequence': name, **m, 'result_sha256': digest(Path(args.reference) / (name + '.txt'))})
    ranked = sorted(rows, key=lambda r: (r['auc'], r['sequence']))
    groups = {'low_target': [r['sequence'] for r in ranked[:32]],
              'middle': [r['sequence'] for r in ranked[32:64]],
              'normal_protection': [r['sequence'] for r in ranked[64:]]}
    dump(target, {'source': str(args.reference), 'reference_identity_sha256': digest(Path(args.reference) / 'identity.json'),
                 'definition': 'fresh fixed-parent AUC tertiles within frozen96; no candidate results used',
                 'groups': groups, 'reference_metrics': rows})
    for label, names in groups.items():
        (RUN / (label + '.txt')).write_text('\n'.join(sorted(names)) + '\n')
    print(json.dumps({k: {'n': len(v), 'parent_auc': float(np.mean([r['auc'] for r in rows if r['sequence'] in v]))}
                      for k, v in groups.items()}, indent=2))


def validate_pair(control, candidate, output):
    c0, c1 = [json.loads((Path(p) / 'initial_state.json').read_text()) for p in [control, candidate]]
    c0_pair = c0.get('paired_common_state_sha256') or c0['shared_state_sha256']
    c1_pair = c1.get('paired_common_state_sha256') or c1['shared_state_sha256']
    common_state_basis = 'recorded_pair_hash'
    common_state_hash = c0_pair
    if c0_pair != c1_pair:
        # Older control runs recorded the full pre-optimizer state, while a
        # candidate made from that snapshot records the hash of the compatible
        # tensor intersection. Recompute that intersection rather than treating
        # different bookkeeping granularity as a training confound.
        try:
            snapshot_meta = c1.get('shared_init_snapshot') or {}
            source_identity = snapshot_meta.get('source_identity', {})
            source_snapshot = source_identity.get('source_snapshot')
            if source_snapshot and Path(source_snapshot).exists():
                source = torch.load(source_snapshot, map_location='cpu')
                source_state = source.get('state', source)
                keys = c1.get('matched_parent_keys', [])
                intersection = {k: source_state[k] for k in keys if k in source_state}
                intersection_hash = state_hash(intersection)
                if (intersection_hash == c1_pair and
                        source.get('source_shared_state_sha256') == c0.get('shared_state_sha256')):
                    common_state_basis = 'control_full_state_candidate_common_tensor_intersection'
                    common_state_hash = intersection_hash
                else:
                    raise ValueError('intersection/source hash mismatch')
            else:
                raise ValueError('candidate source snapshot unavailable')
        except Exception as exc:
            raise ValueError('Initial common states differ: ' + str(exc))
    mismatches, batches, samples = [], 0, 0
    # Stream traces to avoid retaining input tensors or huge expanded metadata.
    import itertools
    with (Path(control) / 'batches.jsonl').open() as f0, (Path(candidate) / 'batches.jsonl').open() as f1:
        for index, (l0, l1) in enumerate(itertools.zip_longest(f0, f1)):
            if l0 is None or l1 is None:
                mismatches.append({'batch': index, 'reason': 'trace lengths differ'}); break
            r0, r1 = json.loads(l0), json.loads(l1)
            if r0 != r1:
                mismatches.append({'batch': index, 'reason': 'augmented tensor or sample metadata mismatch'})
            batches += 1; samples += len(r0['samples'])
    done = [json.loads((Path(p) / 'completed.json').read_text()) for p in [control, candidate]]
    payload = {'shared_state_equal': True, 'paired_common_state_sha256': common_state_hash,
               'common_state_basis': common_state_basis,
               'control_recorded_state_hash': c0_pair,
               'candidate_recorded_state_hash': c1_pair,
               'full_state_hashes_equal': c0['shared_state_sha256'] == c1['shared_state_sha256'],
               'augmented_input_equal': not mismatches,
               'batches': batches, 'samples': samples, 'mismatches': mismatches[:30],
               'successful_updates': [r['successful_updates'] for r in done],
               'coverage': [r['unique_sequences'] for r in done],
               'skip_retries': [r['skipped_attempts_replayed'] for r in done]}
    dump(output, payload)
    if mismatches or done[0]['successful_updates'] != done[1]['successful_updates']:
        raise ValueError('Comparison confounded; repair pairing before utility ranking')
    return payload


def summarize(rows, group, seed=426):
    selected = [r for r in rows if group is None or r['sequence'] in group]
    if not selected:
        return {'n': 0, 'status': 'NA'}
    d = np.array([r['auc_delta'] for r in selected])
    rng = np.random.RandomState(seed)
    # This bootstrap is about sequences, not independent training seeds.
    bootstrap = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
    trimmed = np.sort(d)[int(.1 * len(d)):len(d)-int(.1 * len(d))]
    gain = d[d > 0]
    stats = {'n': len(selected), 'auc_delta': float(d.mean()), 'median': float(np.median(d)),
          'trimmed_mean_10pct': float(trimmed.mean()), 'mean_without_top3_gains': float(np.sort(d)[:-3].mean()) if len(d)>3 else None,
          'sequence_bootstrap_90pct': np.quantile(bootstrap, [.05, .95]).tolist(),
          'sequence_bootstrap_95pct': np.quantile(bootstrap, [.025, .975]).tolist(),
          'wins': int((d > 1.e-6).sum()), 'ties': int((abs(d) <= 1.e-6).sum()), 'losses': int((d < -1.e-6).sum()),
          'losses_over_5pp': int((d < -5).sum()), 'losses_over_10pp': int((d < -10).sum()),
          'worst_decile_mean': float(np.sort(d)[:max(1, int(np.ceil(.1 * len(d))))].mean()),
          'top3_share_of_positive_mass': float(np.sort(gain)[-3:].sum()/gain.sum()) if len(gain) else None,
          'failure_fraction_delta': float(np.mean([r['failure_fraction_delta'] for r in selected]))}
    stats['metrics'] = {k: {'control': float(np.mean([r[k+'_control'] for r in selected])),
                      'candidate': float(np.mean([r[k+'_candidate'] for r in selected])),
                      'delta': float(np.mean([r[k+'_delta'] for r in selected]))} for k in METRICS}
    return stats


def compare(args):
    groups = json.loads((RUN / 'frozen_groups.json').read_text())
    manifest = json.loads((RUN / 'data_manifest.json').read_text())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.train_control and args.train_candidate:
        validate_pair(args.train_control, args.train_candidate, out / 'pairing.json')
    names = set(Path(args.list).read_text().splitlines()) if args.list else None
    membership = {name: label for label, seqs in groups['groups'].items() for name in seqs}
    prior = {r['sequence']: r for r in groups['reference_metrics']}
    rows, errors = [], []
    for item in manifest['test_sequences']:
        name = item['sequence']
        if names is not None and name not in names:
            continue
        c, ciou = read_result(args.control, name, item['frames'])
        d, diou = read_result(args.candidate, name, item['frames'])
        row = {'sequence': name, 'frames': item['frames'], 'group': membership.get(name, 'screen_external_confirmation'),
               'attributes': json.dumps(item['attributes'], sort_keys=True)}
        for k in c:
            row[k + '_control'] = c[k]; row[k + '_candidate'] = d[k]; row[k + '_delta'] = d[k] - c[k]
        # The frozen groups are defined from the fresh control checkpoint.
        # Keep that reference explicit; it is not the historical full15 run.
        row['frozen_control_auc'] = prior[name]['auc'] if name in prior else None
        row['recovered_from_control_failure_frames'] = int(((ciou < .1) & (diou >= .5)).sum())
        row['new_failure_from_control_good_frames'] = int(((ciou >= .5) & (diou < .1)).sum())
        for q in range(4):
            lo, hi = q*len(ciou)//4, (q+1)*len(ciou)//4
            row['iou_delta_quarter%d' % (q+1)] = float((diou-ciou)[lo:hi].mean())
        rows.append(row)
    with (out / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'all_evaluated': summarize(rows, None),
               **{label: summarize(rows, members) for label, members in groups['groups'].items()},
               'screen_external_confirmation': summarize(rows, {r['sequence'] for r in rows if r['sequence'] not in prior}),
               'scope': {'sequences': len(rows), 'frames': sum(r['frames'] for r in rows),
                         'full244': len(rows) == 244, 'inference_precision': 'FP32',
                         'limited_inference': '96 fixed development sequences with <=2000 frames each; length-based replacements invalidate the original inclusion probabilities. No unbiased 237/244-sequence estimate or quick/full direction calibration is claimed.'},
               'measurement_validation': 'Robust canonical errors independently checked against direct intersection for positive-size GT/predictions',
               'evidence_axes': {'execution': 'comparable if pairing.json valid', 'learning': 'see actual successful update traces',
                                 'utility': 'preliminary only', 'claim': 'single-seed paired observations'},
               'quick_full_calibration': 'UNVALIDATED; no matched historical transfer pairs with closed provenance',
               'not_a_final_promotion': 'Requires fixed-endpoint broad confirmation, time-point agreement, seed3407 and learned-checkpoint attribution per PROTOCOL.md'}
    summary['group_reference'] = {
        'source': 'fresh Early gate control checkpoint at the same public-pretrain 468-update protocol',
        'historical_full15_not_used_to_define_groups': True,
        'n': len(prior),
    }
    a, l, n = summary['all_evaluated'], summary['low_target'], summary['normal_protection']
    if n.get('n', 0):
        checks = {'overall_positive': a['auc_delta'] > 0, 'low_target_positive': l['auc_delta'] > 0,
                  'not_top3_only': a['mean_without_top3_gains'] > 0,
                  'normal_mean_within_tolerance': n['auc_delta'] >= -.25,
                  'normal_uncertainty_within_tolerance': n['sequence_bootstrap_90pct'][0] >= -1.,
                  'normal_large_loss_fraction': n['losses_over_5pp']/n['n'] <= .05}
        summary['screen_followup_checks'] = checks
        summary['next_action'] = 'SUPPORTS_BROADER_CONFIRMATION_NOT_FULL_PROMOTION' if all(checks.values()) else 'INCONCLUSIVE_OR_HARM_UNDER_CURRENT_BUDGET; inspect subgroup/progress; no automatic module rejection'
    dump(out / 'summary.json', summary)
    lines = ['# P1配对读数（探索性）', '', '| 分组 | n | ΔAUC(pp) | 90%序列区间 | >5pp损失 |', '|---|---:|---:|---|---:|']
    for key in ['all_evaluated', 'low_target', 'middle', 'normal_protection', 'screen_external_confirmation']:
        v = summary[key]
        if v['n']:
            lines.append('| %s | %d | %+.4f | [%+.4f, %+.4f] | %d |' % (key, v['n'], v['auc_delta'], *v['sequence_bootstrap_90pct'], v['losses_over_5pp']))
    lines += ['', '分组参考：本轮公共预训练 Early gate control 的468-update结果；历史full15未用于冻结分组。',
              '', '序列区间不替代seed方差；quick→full方向关系尚未标定。本文只报告实际范围与进度，不宣称论文稳健提升。',
              '', '下一步：'+summary.get('next_action','see PROTOCOL.md')]
    (out / 'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['freeze', 'compare'])
    p.add_argument('--reference'); p.add_argument('--control'); p.add_argument('--candidate')
    p.add_argument('--list'); p.add_argument('--output', required=True)
    p.add_argument('--train-control'); p.add_argument('--train-candidate')
    p.add_argument('--run-root')
    args = p.parse_args()
    if args.run_root:
        global RUN
        RUN = Path(args.run_root)
    torch.set_num_threads(1)
    freeze(args) if args.mode == 'freeze' else compare(args)


if __name__ == '__main__':
    main()
