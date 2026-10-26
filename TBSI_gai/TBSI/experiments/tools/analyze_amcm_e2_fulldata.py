import csv
import json
import os
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
base = os.path.join(ROOT, 'output/test/tracking_results/tbsi_track')
run = 'M1_AMCM_E2_FULLDATA_10EP_M1MATCH_20261018'
ref_csv = os.path.join(ROOT, 'output/experiments/compare_m1_s1_e2_repr_full244_20261018.csv')
eval_pkl = os.path.join(ROOT, 'output/test/result_plots/lasher_test/eval_data.pkl')
out_dir = os.path.join(ROOT, 'output/experiments', run, 'analysis')
os.makedirs(out_dir, exist_ok=True)

with open(ref_csv, newline='') as f:
    ref = {r['sequence']: r for r in csv.DictReader(f)}
with open(eval_pkl, 'rb') as f:
    ev = pickle.load(f)
names = ev['sequences']
ov = np.asarray(ev['ave_success_rate_plot_overlap'], dtype=float)[:, 0, :]
pc = np.asarray(ev['ave_success_rate_plot_center'], dtype=float)[:, 0, :]
pn = np.asarray(ev['ave_success_rate_plot_center_norm'], dtype=float)[:, 0, :]
rows = []
for i, n in enumerate(names):
    if n not in ref:
        continue
    auc = float(ov[i].mean() * 100)
    row = {'sequence': n, 'E2_FULL_AUC': auc,
           'E2_FULL_OP50': float(ov[i, 10] * 100),
           'E2_FULL_OP75': float(ov[i, 15] * 100),
           'E2_FULL_PR': float(pc[i, 20] * 100),
           'E2_FULL_NPR': float(pn[i, 20] * 100)}
    for key in ('M1', 'S1', 'E2_10EP', 'REPR'):
        row[key] = float(ref[n][key])
        row['FULL_minus_' + key] = auc - row[key]
    rows.append(row)
rows.sort(key=lambda r: r['FULL_minus_M1'])
fields = list(rows[0])
with open(os.path.join(out_dir, 'per_sequence_comparison.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def stats(key):
    x = np.asarray([r[key] for r in rows])
    return {'mean_delta': float(x.mean()), 'median_delta': float(np.median(x)),
            'wins_gt_0.05': int((x > .05).sum()), 'losses_lt_-0.05': int((x < -.05).sum()),
            'ties': int((np.abs(x) <= .05).sum()),
            'top10': [{'sequence': r['sequence'], 'delta': r[key], 'full_auc': r['E2_FULL_AUC']} for r in rows[-10:][::-1]],
            'bottom10': [{'sequence': r['sequence'], 'delta': r[key], 'full_auc': r['E2_FULL_AUC']} for r in rows[:10]]}

summary = {
    'n_sequences': len(rows),
    'official_full_metrics': {'AUC': 56.09, 'OP50': 68.18, 'OP75': 47.06, 'Precision': 69.99, 'NormPrecision': 66.15},
    'per_sequence_mean': {k: float(np.mean([r[k] for r in rows])) for k in ('E2_FULL_AUC', 'E2_FULL_OP50', 'E2_FULL_OP75', 'E2_FULL_PR', 'E2_FULL_NPR')},
    'vs_M1': stats('FULL_minus_M1'), 'vs_S1': stats('FULL_minus_S1'),
    'vs_E2_proxy': stats('FULL_minus_E2_10EP'), 'vs_repr': stats('FULL_minus_REPR')}
with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
