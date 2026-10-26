#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np

METRICS = ('auc', 'precision', 'norm_precision', 'op50', 'op75')
ARMS = ('m1', 'fixed_rgb', 'fixed_tir', 's1', 's2', 's3', 's1s2', 's1s2s3')
FLOW = ('gt_rank_hann', 'peak_distance_tokens', 'score_entropy', 'rgb_norm',
        'tir_norm', 'fused_norm', 'branch_cosine', 'iou')

def rows(path):
    with (path / 'per_sequence.csv').open(newline='') as f:
        return {r['sequence']: r for r in csv.DictReader(f)}

def info(path):
    with (path / 'information_flow.csv').open(newline='') as f:
        return list(csv.DictReader(f))

def boot(values):
    rng = np.random.RandomState(20261012)
    x = np.asarray(values, dtype=float)
    return [float(v) for v in np.percentile(rng.choice(x, (5000, len(x)), replace=True).mean(1), [2.5, 97.5])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    root, out = Path(args.root_dir).resolve(), Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing nonempty output directory')
    paths = {'m1': root/'matched_m1_full62'}
    for arm in ARMS[1:]:
        paths[arm] = root/(arm+'_full62')
    paths['s1s2s3'] = root/'s1s2s3_full62_retry'
    data = {arm: rows(path) for arm, path in paths.items()}
    expected = sorted(data['m1'])
    if len(expected) != 62: raise RuntimeError('expected 62 baseline sequences')
    for arm in ARMS:
        if sorted(data[arm]) != expected: raise RuntimeError('sequence mismatch '+arm)
        if len(info(paths[arm])) != 1974: raise RuntimeError('info mismatch '+arm)
        summary = json.loads((paths[arm]/'summary.json').read_text())
        if summary['status'] != 'COMPLETE': raise RuntimeError('incomplete '+arm)
    table = []
    for arm in ARMS:
        item = {'arm': arm, 'n_sequences': 62, 'n_info_samples': 1974}
        for metric in METRICS:
            vals = np.array([float(data[arm][s][metric]) for s in expected])
            item[metric] = float(vals.mean())
            if arm != 'm1':
                delta = vals - np.array([float(data['m1'][s][metric]) for s in expected])
                item[metric+'_delta'] = float(delta.mean())
                item[metric+'_delta_ci95'] = boot(delta)
                if metric == 'auc':
                    item['auc_wins'] = int((delta > 0).sum())
                    item['auc_losses'] = int((delta < 0).sum())
                    item['auc_ties'] = int((delta == 0).sum())
        item['fps_sequence_macro'] = float(np.mean([float(data[arm][s]['fps']) for s in expected]))
        inf = info(paths[arm])
        for metric in FLOW:
            item[metric+'_sequence_macro'] = float(np.mean([float(r[metric]) for r in inf]))
        table.append(item)
    result = {'id':'M1-SINGLE-SIGNAL-CONTROLS-20261012',
              'scope':'62 archived-M1 AUC<40 sequences; matched current-code reruns',
              'table':table,
              'limitations':['archive selection is a diagnostic subgroup',
                             'current-code reruns are not archived frame parity',
                             'constant controls are required to interpret signal gains']}
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    fields=[]
    for k,v in table[0].items():
        if not isinstance(v,list): fields.append(k)
    with (out/'eight_rows.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows({k:r.get(k,'') for k in fields} for r in table)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
