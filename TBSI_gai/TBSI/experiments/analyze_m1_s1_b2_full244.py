import csv
import json
import os
import pickle
import numpy as np

root = os.path.dirname(os.path.dirname(__file__))
official = os.path.join(root, 'output/experiments/M1_S1_FULL244_CONFIRM_20261014/per_sequence_official.csv')
eval_pkl = os.path.join(root, 'output/test/result_plots/lasher_test/eval_data.pkl')
out_dir = os.path.join(root, 'output/experiments/M1_TARGET_COMPLEMENTARY_ADAPTIVE_PROXY_20261016/full244_analysis')
os.makedirs(out_dir, exist_ok=True)

with open(official, newline='') as f:
    ref = {r['sequence']: r for r in csv.DictReader(f)}
with open(eval_pkl, 'rb') as f:
    ev = pickle.load(f)
names = ev['sequences']
ov = np.asarray(ev['ave_success_rate_plot_overlap'], dtype=float)[:, 0, :]
pc = np.asarray(ev['ave_success_rate_plot_center'], dtype=float)[:, 0, :]
pn = np.asarray(ev['ave_success_rate_plot_center_norm'], dtype=float)[:, 0, :]
io = np.asarray(ev['avg_overlap_all'], dtype=float)[:, 0]
b2 = {}
for i, n in enumerate(names):
    b2[n] = dict(B2_AUC=float(ov[i].mean()*100), B2_PR=float(pc[i, 20]*100),
                 B2_NPR=float(pn[i, 20]*100), B2_OP50=float(ov[i, 10]*100),
                 B2_OP75=float(ov[i, 15]*100), B2_IoU=float(io[i]*100))

rows = []
for n in names:
    r = ref[n].copy(); r.update(b2[n]); r['B2_minus_M1_AUC'] = r['B2_AUC'] - float(r['M1_AUC'])
    r['B2_minus_S1_AUC'] = r['B2_AUC'] - float(r['S1_AUC'])
    for m in ['PR','NPR','OP50','OP75']:
        r['B2_minus_M1_'+m] = r['B2_'+m] - float(r['M1_'+m])
        r['B2_minus_S1_'+m] = r['B2_'+m] - float(r['S1_'+m])
    rows.append(r)
fields = list(rows[0].keys())
csv_out = os.path.join(out_dir, 'per_sequence_m1_s1_b2.csv')
with open(csv_out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def stats(prefix):
    vals = np.array([float(r[prefix]) for r in rows])
    return dict(mean=float(vals.mean()), median=float(np.median(vals)), wins=int((vals>0.05).sum()),
                losses=int((vals<-0.05).sum()), ties=int((np.abs(vals)<=0.05).sum()),
                best=sorted([(float(r[prefix]), r['sequence']) for r in rows], reverse=True)[:10],
                worst=sorted([(float(r[prefix]), r['sequence']) for r in rows])[:10])

summary = {
    'B2': {k: float(np.mean([float(r['B2_'+k]) for r in rows])) for k in ['AUC','PR','NPR','OP50','OP75','IoU']},
    'M1': {k: float(np.mean([float(r['M1_'+k]) for r in rows])) for k in ['AUC','PR','NPR','OP50','OP75']},
    'S1': {k: float(np.mean([float(r['S1_'+k]) for r in rows])) for k in ['AUC','PR','NPR','OP50','OP75']},
    'B2_vs_M1_AUC': stats('B2_minus_M1_AUC'), 'B2_vs_S1_AUC': stats('B2_minus_S1_AUC')}
with open(os.path.join(out_dir, 'summary.json'), 'w') as f: json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
