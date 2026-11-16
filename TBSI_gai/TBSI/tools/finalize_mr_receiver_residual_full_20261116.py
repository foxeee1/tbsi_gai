#!/usr/bin/env python3
"""Compute matched full244 metrics and update the two full-run asset rows."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from run_dual_message_p1 import sha, write_json
from analyze_receiver_message_residual_quick_20261115 import metrics, METRICS
from lib.test.evaluation import get_dataset

RUN = ROOT / 'output/experiments/MR-RECEIVER-RESIDUAL-FULL15-S42-20261116'


def main():
    dataset = {s.name: s for s in get_dataset('lasher_test')}
    rows = []
    for name in sorted(dataset):
        c = metrics(RUN / 'eval_control', dataset[name]); x = metrics(RUN / 'eval_candidate', dataset[name])
        row = {'sequence': name, 'frames': c['frames']}
        for k in METRICS:
            row['control_' + k] = c[k]; row['candidate_' + k] = x[k]; row['delta_' + k] = x[k] - c[k]
        rows.append(row)
    deltas = np.asarray([r['delta_auc'] for r in rows], dtype=float)
    wins = int((deltas > 1e-9).sum()); ties = int((abs(deltas) <= 1e-9).sum()); losses = int((deltas < -1e-9).sum())
    order = np.sort(deltas); k = max(1, int(np.ceil(len(order) * .1)))
    rng = np.random.default_rng(20261116); boot = rng.integers(0, len(deltas), size=(5000, len(deltas)))
    summary = dict(sequence_count=len(rows), frames=sum(r['frames'] for r in rows),
        control={k: float(np.mean([r['control_' + k] for r in rows])) for k in METRICS},
        candidate={k: float(np.mean([r['candidate_' + k] for r in rows])) for k in METRICS},
        delta={k: float(np.mean([r['delta_' + k] for r in rows])) for k in METRICS},
        auc_wins_ties_losses=[wins, ties, losses], auc_median=float(np.median(deltas)),
        practical_wins=int((deltas > .25).sum()), practical_ties=int((abs(deltas) <= .25).sum()),
        practical_losses=int((deltas < -.25).sum()), practical_netwins=int((deltas > .25).sum() - (deltas < -.25).sum()),
        sequence_bootstrap95=np.quantile(deltas[boot].mean(1), [.025, .975]).tolist(),
        worst_decile_mean=float(order[:k].mean()), mean_without_top3=float(order[:-3].mean()),
        gain_gt5=int((deltas > 5).sum()), loss_gt5=int((deltas < -5).sum()),
        top3_positive_fraction=float(np.maximum(order[-3:], 0).sum() / max(np.maximum(order, 0).sum(), 1e-12)),
        checkpoint_sha256={'candidate': sha(RUN / 'candidate_retry1/checkpoint_u7020.pth.tar'), 'control': sha(RUN / 'control/checkpoint_u7020.pth.tar')},
        pairing=json.loads((RUN / 'pairing.json').read_text()))
    dest = RUN / 'analysis'; dest.mkdir(exist_ok=True)
    with (dest / 'per_sequence.csv').open('w', newline='') as f:
        fields = list(rows[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    write_json(dest / 'summary.json', summary)
    report = ['# MR Receiver Message Residual Full15 / Full244', '',
        'Candidate: receiver-conditioned message residual; control: same full contract with residual disabled.',
        'Both arms use public SOT fresh initialization, seed42, 15 epochs and 7020 successful updates. Candidate/control shared pre-optimizer state and train input order were checked before ranking.', '',
        '| Metric | Control | Candidate | Candidate-control |', '|---|---:|---:|---:|']
    for k in METRICS: report.append(f'| {k} | {summary["control"][k]:.4f} | {summary["candidate"][k]:.4f} | {summary["delta"][k]:+.4f} |')
    report += ['', f'AUC wins/ties/losses: {wins}/{ties}/{losses}; practical ±0.25pp: {summary["practical_wins"]}/{summary["practical_ties"]}/{summary["practical_losses"]}.',
        f'Sequence bootstrap 95%: {summary["sequence_bootstrap95"]}; worst decile mean: {summary["worst_decile_mean"]:+.4f}pp; mean without top3: {summary["mean_without_top3"]:+.4f}pp.',
        'This is a single seed and one full budget. It tests the selected mechanism against a matched control; it does not establish cross-seed robustness.']
    (dest / 'REPORT.md').write_text('\n'.join(report) + '\n')
    # Update the existing full arm records without changing their identities.
    path = ROOT / 'experiments/experiment_asset_registry.csv'
    with path.open(newline='') as f: reader = csv.DictReader(f); fields = reader.fieldnames; assets = list(reader)
    elapsed = {}
    for arm in ['candidate', 'control']:
        completed = json.loads((RUN / ('eval_candidate' if arm == 'candidate' else 'eval_control') / 'completed.json').read_text())
        elapsed[arm] = completed
    for r in assets:
        if r['ID'] == 'MR-RECEIVER-RESIDUAL-FULL15-S42-20261116-CANDIDATE':
            r.update(AUC=str(summary['candidate']['auc']), PR=str(summary['candidate']['precision']), NPR=str(summary['candidate']['npr']), OP50=str(summary['candidate']['op50']), OP75=str(summary['candidate']['op75']), FPS=str(elapsed['candidate']['throughput_fps']), Status='FULL15-FULL244-COMPLETE / EXPLORATORY', Evidence=str(dest/'REPORT.md'), Checkpoint=str(RUN/'candidate_retry1/checkpoint_u7020.pth.tar')+' SHA256 '+summary['checkpoint_sha256']['candidate'], Conclusion='Full244 matched evaluation complete; candidate minus control '+str(summary['delta']['auc'])+'pp AUC; WTL '+str([wins,ties,losses])+'. See full report; single seed.')
        if r['ID'] == 'MR-RECEIVER-RESIDUAL-FULL15-S42-20261116-CONTROL':
            r.update(AUC=str(summary['control']['auc']), PR=str(summary['control']['precision']), NPR=str(summary['control']['npr']), OP50=str(summary['control']['op50']), OP75=str(summary['control']['op75']), FPS=str(elapsed['control']['throughput_fps']), Status='FULL15-FULL244-COMPLETE / CONTROL', Evidence=str(dest/'REPORT.md'), Checkpoint=str(RUN/'control/checkpoint_u7020.pth.tar')+' SHA256 '+summary['checkpoint_sha256']['control'], Conclusion='Matched full244 control complete; candidate comparison is recorded in full report.')
    with path.open('w', newline='') as f: w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(assets)
    md = ROOT / 'experiments/实验资产数据库.md'; text = md.read_text()
    for ident in ['MR-RECEIVER-RESIDUAL-FULL15-S42-20261116-CANDIDATE', 'MR-RECEIVER-RESIDUAL-FULL15-S42-20261116-CONTROL']:
        r = next(x for x in assets if x['ID'] == ident); begin, end = '<!-- '+ident+' BEGIN -->', '<!-- '+ident+' END -->'; i = text.index(begin); j = text.index(end, i) + len(end); block = begin+'\n### '+ident+'\n\n'+''.join('- **'+k+'**：'+r[k]+'\n' for k in fields)+end; text = text[:i]+block+text[j:]
    md.write_text(text); write_json(RUN / 'full_result.json', summary); print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
