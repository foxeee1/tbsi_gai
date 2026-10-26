#!/usr/bin/env python3
from __future__ import print_function
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('TBSI_PYTHON', '/root/autodl-tmp/conda_envs/tbsi/bin/python')
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
M1_RESULTS = ROOT / 'output/test/tracking_results/tbsi_track' / (CONFIG + '_9904')
OLD_RESULTS = ROOT / 'output/test/tracking_results/tbsi_track' / (CONFIG + '_9915')
MANIFEST = ROOT / 'experiments/manifests/m1_normal20_lowauc62_union_20261013.txt'
LOW = set((ROOT / 'output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/matched_m1_full62/sequence_list.txt').read_text().split())
NORMAL = set((ROOT / 'experiments/manifests/m1_normal20_control_20261012.txt').read_text().split())
ARMS = [
    ('S1_INDEPENDENT', 1021, 'm1_readout_independent_20261014.json'),
    ('S2_RELIABILITY', 1022, 'm1_readout_reliability_20261014.json'),
    ('S3_HYSTERESIS', 1023, 'm1_readout_hysteresis_20261014.json'),
]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.

def metrics(result_dir, seq):
    pred = np.loadtxt(result_dir / (seq + '.txt'), delimiter='\t', ndmin=2).astype(float)
    gt = np.loadtxt(ROOT / 'data/lasher/testingset' / seq / 'init.txt', delimiter=',', ndmin=2).astype(float)
    n = min(len(pred), len(gt)); pred, gt = pred[:n], gt[:n]
    ov = np.asarray([iou(pred[i], gt[i]) for i in range(n)])
    pc = np.column_stack((pred[:,0]+pred[:,2]/2, pred[:,1]+pred[:,3]/2))
    gc = np.column_stack((gt[:,0]+gt[:,2]/2, gt[:,1]+gt[:,3]/2))
    d = np.linalg.norm(pc-gc, axis=1); diag = np.sqrt(gt[:,2]*gt[:,3])
    return {'AUC':float(np.mean([(ov >= t).mean() for t in np.linspace(0,1,101)])*100),
            'PR':float((d<=20).mean()*100), 'NPR':float((d/np.maximum(diag,1e-9)<=.2).mean()*100),
            'OP50':float((ov>=.5).mean()*100), 'OP75':float((ov>=.75).mean()*100), 'frames':int(n)}

def summarize(rows):
    out = {'n':len(rows)}
    for key in ('AUC','PR','NPR','OP50','OP75'):
        ds = np.asarray([r['delta_'+key] for r in rows])
        out[key] = {'delta':float(ds.mean()), 'wins':int((ds>0).sum()),
                    'losses':int((ds<0).sum()), 'ties':int((ds==0).sum())}
    return out

def main():
    out = ROOT / 'output/experiments/M1_READOUT_STRUCTURES_PROXY_20261014'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output: {}'.format(out))
    seqs = [x.strip() for x in MANIFEST.read_text().splitlines() if x.strip()]
    if len(seqs) != 82 or len(set(seqs)) != 82 or not LOW.issubset(seqs) or not NORMAL.issubset(seqs):
        raise RuntimeError('fixed union manifest validation failed')
    cfg = ROOT / 'experiments/tbsi_track' / (CONFIG + '.yaml')
    ckpt = ROOT / 'output/experiments' / CONFIG / 'checkpoints/TBSITrack_ep0015.pth.tar'
    if sha(cfg) != 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e':
        raise RuntimeError('protected M1 config hash mismatch')
    if sha(ckpt) != '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef':
        raise RuntimeError('protected M1 checkpoint hash mismatch')
    if not M1_RESULTS.is_dir() or not OLD_RESULTS.is_dir():
        raise RuntimeError('existing parent results are missing')
    out.mkdir(parents=True)
    (out/'sequence_list.txt').write_text('\n'.join(seqs)+'\n')
    contract = {'id':'M1-READOUT-STRUCTURES-PROXY-20261014','parent':CONFIG,
                'parent_runid':9904,'old_predictor_runid':9915,'manifest_sha256':sha(MANIFEST),
                'config_sha256':sha(cfg),'checkpoint_sha256':sha(ckpt),'cohort':82,
                'low_auc_count':len(LOW),'normal_count':len(NORMAL),'seed':42,
                'precision':'FP32','serial':True,'arms':ARMS}
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    base = {s:metrics(M1_RESULTS,s) for s in seqs}
    old = {s:metrics(OLD_RESULTS,s) for s in seqs}
    results = {'M1':{},'OLD_BOUNDED_PREDICTOR':{}}
    for label, rows in [('M1',base),('OLD_BOUNDED_PREDICTOR',old)]:
        paired=[]
        for s in seqs:
            row={'sequence':s,'cohort':'lowauc62' if s in LOW else 'normal20'}
            row.update({'M1_'+k:v for k,v in base[s].items() if k != 'frames'})
            row.update({label+'_'+k:v for k,v in rows[s].items() if k != 'frames'})
            row.update({'delta_'+k:rows[s][k]-base[s][k] for k in ('AUC','PR','NPR','OP50','OP75')})
            paired.append(row)
        results[label]={'per_sequence':paired,'summary':{}}
        for cohort in ('lowauc62','normal20','union82'):
            sub=paired if cohort=='union82' else [r for r in paired if r['cohort']==cohort]
            results[label]['summary'][cohort]=summarize(sub)
    envbase=os.environ.copy()
    envbase.update({'TBSI_RPP_TEST_LIST':str(out/'sequence_list.txt'),'TBSI_INFER_FP16':'0',
                    'TBSI_CHECKPOINT_OVERRIDE':str(ckpt),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                    'PYTHONUNBUFFERED':'1'})
    for label, runid, policy_name in ARMS:
        result_dir=ROOT/'output/test/tracking_results/tbsi_track'/(CONFIG+'_'+str(runid))
        if result_dir.exists(): raise RuntimeError('refusing existing result dir {}'.format(result_dir))
        policy=ROOT/'experiments/artifacts'/policy_name
        env=envbase.copy(); env['TBSI_READOUT_POLICY']=str(policy)
        env['TBSI_BOUNDED_READOUT_PREDICTOR']=str(ROOT/'experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json')
        env['TBSI_READOUT_AUDIT_DIR']=str(out/'action_audit'/label)
        cmd=[PYTHON,'-u','tracking/test.py','tbsi_track',CONFIG,'--runid',str(runid),
             '--dataset_name','rpp_lasher_test','--threads','0','--num_gpus','1']
        log=out/(label+'.log')
        print('START {} {}'.format(label,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),flush=True)
        started=time.time()
        with log.open('w') as stream:
            subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if not result_dir.is_dir(): raise RuntimeError('missing results {}'.format(result_dir))
        rows=[]
        for s in seqs:
            m=metrics(result_dir,s)
            row={'sequence':s,'cohort':'lowauc62' if s in LOW else 'normal20',
                 **{'M1_'+k:v for k,v in base[s].items() if k!='frames'},
                 **{label+'_'+k:v for k,v in m.items() if k!='frames'},
                 **{'delta_'+k:m[k]-base[s][k] for k in ('AUC','PR','NPR','OP50','OP75')},
                 'frames':m['frames']}
            rows.append(row)
        results[label]={'runid':runid,'result_dir':str(result_dir),'seconds':time.time()-started,
                        'per_sequence':rows,'summary':{}}
        for cohort in ('lowauc62','normal20','union82'):
            sub=rows if cohort=='union82' else [r for r in rows if r['cohort']==cohort]
            results[label]['summary'][cohort]=summarize(sub)
        print('FINISH {} {}'.format(label,json.dumps(results[label]['summary'],sort_keys=True)),flush=True)
        (out/'summary_partial.json').write_text(json.dumps(results,indent=2)+'\n')
    all_rows=[]
    by_label={k:{r['sequence']:r for r in v['per_sequence']} for k,v in results.items()}
    for s in seqs:
        row={'sequence':s,'cohort':'lowauc62' if s in LOW else 'normal20'}
        for name in results:
            r=by_label[name][s]
            row[name+'_AUC']=r.get(name+'_AUC',r.get('AUC'))
            for metric in ('AUC','PR','NPR','OP50','OP75'):
                row[name+'_delta_'+metric]=r['delta_'+metric]
        all_rows.append(row)
    with (out/'per_sequence_all_arms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    results['action_audit']={}
    for label,_,_ in ARMS:
        files=list((out/'action_audit'/label).glob('*/actions.jsonl'))
        counts={'noop':0,'offset':0,'size':0,'joint':0,'frames':0}
        for p in files:
            for line in p.read_text().splitlines():
                a=json.loads(line)['action']; counts[a]=counts.get(a,0)+1; counts['frames']+=1
        results['action_audit'][label]={'sequences':len(files),**counts}
    results['decision']={'status':'PROXY-COMPLETE / ANALYSIS-PENDING','note':'same fixed 82-sequence cohort; proxy only'}
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('COMPLETE {}'.format(out),flush=True)

if __name__=='__main__': main()
