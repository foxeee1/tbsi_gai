#!/usr/bin/env python3
"""Read-only trajectory/attribute audit. No model changes or optimizer steps."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import csv
import gzip
import hashlib
import html
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT/'output/experiments/MSHA-MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115'
OUT = RUN/'analysis/box_attribute_audit_20261115'
DATA = ROOT.parent/'data/lasher'
LIST = ROOT/'experiments/early_gate_closed_loop_20261026/p1_sot_v3/screen96_max2000.txt'
HISTORY = ROOT/'output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/analysis/two_seed_failure_20261113/per_sequence_two_seed.csv'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def write_csv(p, rows):
    with Path(p).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def box_stats(p,g):
    valid=(g[:,2:]>0).all(1);safe=g.copy();safe[~valid,2:]=1
    inter=np.maximum(0,np.minimum(p[:,:2]+p[:,2:],g[:,:2]+g[:,2:])-np.maximum(p[:,:2],g[:,:2])).prod(1)
    iou=inter/np.maximum(p[:,2:].prod(1)+g[:,2:].prod(1)-inter,1e-12)
    err=((p[:,:2]+(p[:,2:]-1)/2)-(g[:,:2]+(g[:,2:]-1)/2))/safe[:,2:]
    cn=np.linalg.norm(err,axis=1)
    center=np.linalg.norm(err*g[:,2:],axis=1)
    area=np.abs(np.log(np.maximum(p[:,2:].prod(1),1e-9)/safe[:,2:].prod(1)))
    aspect=np.abs(np.log(np.maximum(p[:,2]/p[:,3],1e-9)/(safe[:,2]/safe[:,3])))
    iou[~valid]=-1;cn[~valid]=np.inf;center[~valid]=np.inf
    auc=(iou[:,None]>np.arange(21)[None,:]/20).mean(1)*100
    return dict(iou=iou,cn=cn,center=center,area=area,aspect=aspect,auc=auc)

def crop_stats(prev,gt):
    side=np.ceil(np.sqrt(prev[:,2:].prod(1))*4.)
    tl=np.rint(prev[:,:2]+prev[:,2:]/2-side[:,None]/2)
    gc=gt[:,:2]+gt[:,2:]/2
    inside=((gc>=tl)&(gc<tl+side[:,None])).all(1)
    inter=np.maximum(0,np.minimum(gt[:,:2]+gt[:,2:],tl+side[:,None])-np.maximum(gt[:,:2],tl)).prod(1)
    return inside,inter/np.maximum(gt[:,2:].prod(1),1e-12),tl,side

def runs(mask, minimum=10):
    edges=np.diff(np.r_[False,mask,False].astype(int)); ss=np.flatnonzero(edges==1);ee=np.flatnonzero(edges==-1)
    return [(int(s),int(e)) for s,e in zip(ss,ee) if e-s>=minimum]

def summarize(rows):
    v=np.array([r['delta_auc'] for r in rows]);s=np.sort(v); n=len(v)
    rng=np.random.default_rng(20261115)
    ci=np.quantile(v[rng.integers(0,n,size=(5000,n))].mean(1),[.025,.975])
    k=max(1,math.ceil(n*.1));trim=int(n*.1)
    return dict(n=n,control_auc=float(np.mean([r['control_auc'] for r in rows])),candidate_auc=float(np.mean([r['candidate_auc'] for r in rows])),
        delta_auc=float(v.mean()),median=float(np.median(v)),wins=int((v>1e-9).sum()),ties=int((abs(v)<=1e-9).sum()),losses=int((v< -1e-9).sum()),
        practical_wins_gt_025=int((v>.25).sum()),practical_ties_le_025=int((abs(v)<=.25).sum()),practical_losses_lt_m025=int((v<-.25).sum()),
        sequence_bootstrap95=ci.tolist(),trimmed10=float(s[trim:n-trim].mean()),
        mean_without_top3=float(s[:n-3].mean()) if n>3 else None,worst_ceil10pct=float(s[:k].mean()),
        loss_gt5=int((v< -5).sum()),gain_gt5=int((v>5).sum()),
        positive_gain_top3_fraction=float(np.maximum(s[-3:],0).sum()/max(np.maximum(s,0).sum(),1e-12)),
        top3_over_net_fraction=float(s[-3:].sum()/v.sum()) if abs(v.sum())>1e-9 else None)

def frames(seq):
    def key(p):
        import re
        return int(re.match(r'^[^0-9]*(\d+)',p.stem).group(1)),p.name
    return [sorted((DATA/'testingset'/seq/m).glob('*.jpg'),key=key) for m in ['visible','infrared']]

def panel(seq,row,cache):
    gt,preds,stats,ev=cache
    paths=frames(seq)
    fig=plt.figure(figsize=(12,7.8)); gs=fig.add_gridspec(3,3,height_ratios=[1,1,.8])
    colors=['#ff982e','#18c7eb'];cols=[tuple(int(c[i:i+2],16) for i in (5,3,1)) for c in colors]
    for m in range(2):
        for j,e in enumerate(ev):
            f=e['frame'];im=cv2.imread(str(paths[m][f])); assert im is not None
            # Full image is the evidence; zoom crop is not used to hide lost boxes.
            for b,col in [(gt[f],(40,220,40)),(preds[0][f],cols[0]),(preds[1][f],cols[1])]:
                x,y,w,h=b;cv2.rectangle(im,(round(x),round(y)),(round(x+w),round(y+h)),col,2)
            ax=fig.add_subplot(gs[m,j]);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.axis('off')
            ax.set_title('%s f%d %s\nIoU %.3f / %.3f'%('RGB' if m==0 else 'TIR',f+1,e['kind'],stats[0]['iou'][f],stats[1]['iou'][f]),fontsize=9)
    ax=fig.add_subplot(gs[2,:]); x=np.arange(1,len(gt)+1)
    for a in range(2):ax.plot(x,stats[a]['iou'],color=colors[a],lw=.85,label=['Control','Candidate'][a])
    for e in ev:ax.axvline(e['frame']+1,color='#777',ls=':',lw=.7)
    ax.set(ylim=(-.02,1.02),xlabel='Frame (1 based)',ylabel='IoU');ax.legend(loc='lower left',ncol=2,fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('%s | delta AUC %+.3f pp | %s\nGreen: GT / Orange: control / Cyan: candidate'%(seq,row['delta_auc'],row['attributes']),fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'sequences'/(seq+'.png'),dpi=105);plt.close(fig)

def main():
    start=time.time();cv2.setNumThreads(0)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'sequences').mkdir(exist_ok=True);(OUT/'source').mkdir(exist_ok=True)
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    names=[s.strip() for s in LIST.read_text().splitlines() if s.strip()]
    old={r['sequence']:r for r in csv.DictReader((RUN/'analysis/per_sequence.csv').open())}
    hist={r['sequence']:r for r in csv.DictReader(HISTORY.open())}
    attrfile=DATA/'Attributes&Order.zip'
    with zipfile.ZipFile(attrfile) as z:
        codes=[s.strip() for s in z.read('Attributes_order.txt').decode().strip().split(',')]
        attrs={s:dict(zip(codes,map(int,z.read('AttriSeqsTxt/'+s+'.txt').decode().strip().split(',')))) for s in names}
    # This contract is written before calculating any new trajectory/attribute outputs.
    write_json(OUT/'CONTRACT.json',dict(scope='post-hoc development analysis, not independent validation; no training',
        list_sha256=sha(LIST),attributes_sha256=sha(attrfile),history_sha256=sha(HISTORY),
        selection='all 96 evaluated sequences; no exclusions or silent truncation',
        events='first >=5-frame IoU gap >=0.3 in either sign; max abs 15-frame rolling IoU gap; 80% sequence position; dedup then fill 25/50/75%',
        sustained_failure='IoU <0.1 for >=10 consecutive frames, excludes initialization',
        decomposition='C=control IoU, N=candidate IoU: rescue C<.1 N>=.5; harm N<.1 C>=.5; both_good C,N>=.5; both_bad C,N<.1; rest remaining',
        crop='exact sample_target ceil(4*sqrt(previous width*height)), rounded top-left; GT-center inside and coverage from respective prior prediction; geometric, not physical visibility',
        bootstrap='5000 sequence resamples; seed20261115; post-hoc19 overlapping attributes; no seed or multiplicity inference',
        practical_tolerance='0.25pp reported as descriptive sensitivity only; exact wins also reported',
        mechanism_followup='same trained checkpoint, same frame and control-previous-state crop; off/bias-only/shuffled-receiver are distribution interventions, not matched retraining'))
    rows=[];events=[];decomp=[];frame_file=gzip.open(OUT/'per_frame.csv.gz','wt',newline='');writer=None
    for idx,name in enumerate(names):
        g=np.loadtxt(DATA/'testingset'/name/'init.txt',delimiter=',',ndmin=2)
        pp=[np.loadtxt(RUN/a/(name+'.txt'),ndmin=2) for a in ['eval_control','eval_candidate']]
        assert all(p.shape==g.shape and np.isfinite(p).all() for p in pp),(name,'invalid length')
        valid=(g[:,2:]>0).all(1);valid_noinit=valid.copy();valid_noinit[0]=False
        for p in pp:p[0]=g[0]
        ss=[box_stats(p,g) for p in pp]; io=[s['iou'] for s in ss];d=io[1]-io[0]
        # Confirm the independent vectorized calculation reproduces existing official sequence metrics.
        assert max(abs(ss[a]['auc'].mean()-float(old[name][['control_auc','candidate_auc'][a]])) for a in range(2))<1e-8
        crops=[crop_stats(np.vstack([p[0],p[:-1]]),g) for p in pp]
        r=dict(old[name]);r={k:(v if k=='sequence' else float(v)) for k,v in r.items()};r['frames']=len(g)
        r['attributes']=' '.join(c for c in codes if attrs[name][c]);r.update({'attr_'+c:attrs[name][c] for c in codes})
        r['invalid_gt_frames']=int((~valid).sum())
        r['history_group']=hist[name]['cross_seed_group'];r['history_s42_mr_minus_b0']=float(hist[name]['seed42_delta_auc']);r['history_s43_mr_minus_b0']=float(hist[name]['seed43_delta_auc'])
        for a in range(2):
            label=['control','candidate'][a];s=ss[a];fail=runs((s['iou'][1:]<.1)&valid[1:])
            r[label+'_mean_iou']=float(s['iou'][valid_noinit].mean());r[label+'_median_iou']=float(np.median(s['iou'][valid_noinit]))
            r[label+'_center_norm_median']=float(np.median(s['cn'][valid_noinit]));r[label+'_abs_log_area_median']=float(np.median(s['area'][valid_noinit]));r[label+'_abs_log_aspect_median']=float(np.median(s['aspect'][valid_noinit]))
            r[label+'_long_failure_frames']=sum(e-b for b,e in fail);r[label+'_first_long_failure_frame']=fail[0][0]+2 if fail else -1
            r[label+'_gt_center_outside_crop_fraction']=float((~crops[a][0][valid_noinit]).mean());r[label+'_gt_coverage_lt50_fraction']=float((crops[a][1][valid_noinit]<.5).mean())
        r['iou_delta_median_noninit']=float(np.median(d[1:]));r['better_frames_gt005']=int((d[1:]>.05).sum());r['worse_frames_lt_m005']=int((d[1:]<-.05).sum())
        categories={'rescue':(io[0]<.1)&(io[1]>=.5),'harm':(io[1]<.1)&(io[0]>=.5),'both_good':(io[0]>=.5)&(io[1]>=.5),'both_bad':(io[0]<.1)&(io[1]<.1)}
        categories['rest']=~np.logical_or.reduce(list(categories.values()))
        for cat,mask in categories.items():
            mask &= valid_noinit;contribution=float((ss[1]['auc']-ss[0]['auc'])[mask].sum()/len(g))
            decomp.append(dict(sequence=name,category=cat,frames=int(mask.sum()),delta_auc_contribution_pp=contribution))
            r[cat+'_frames']=int(mask.sum());r[cat+'_auc_contribution_pp']=contribution
        # Event selection only from observed paired trajectories, explicitly retrospective.
        rr=sorted(runs(d[1:]>.3,5)+runs(d[1:]<-.3,5));ev=[]
        if rr:ev.append(dict(frame=rr[0][0]+1,kind='first_gap'))
        roll=np.convolve(d,np.ones(15)/15,mode='same');peak=int(np.argmax(abs(roll[1:]))+1)
        ev.append(dict(frame=peak,kind='peak_15f'));ev.append(dict(frame=max(1,int(len(g)*.8)),kind='late80'))
        ev=list({e['frame']:e for e in ev}.values())
        for frac in [.25,.5,.75]:
            if len(ev)<3 and int(len(g)*frac) not in [e['frame'] for e in ev]:ev.append(dict(frame=max(1,int(len(g)*frac)),kind='uniform'))
        ev=[e for e in ev if valid[e['frame']]][:3]
        for f in np.flatnonzero(valid_noinit):
            if len(ev)==3:break
            if f not in [e['frame'] for e in ev]:ev.append(dict(frame=int(f),kind='valid_fill'))
        for e in ev:
            f=e['frame'];events.append(dict(sequence=name,frame=f,display_frame=f+1,kind=e['kind'],crop_state=pp[0][f-1].tolist(),control_iou=float(io[0][f]),candidate_iou=float(io[1][f])))
        rows.append(r)
        for f in range(len(g)):
            fr=dict(sequence=name,frame=f+1,valid_gt=int(valid[f]),gt_x=g[f,0],gt_y=g[f,1],gt_w=g[f,2],gt_h=g[f,3])
            for a in range(2):
                label=['control','candidate'][a]
                for k,j in [('x',0),('y',1),('w',2),('h',3)]:fr[label+'_'+k]=pp[a][f,j]
                for k in ['iou','cn','area','aspect','auc']:fr[label+'_'+k]=ss[a][k][f]
                fr[label+'_gt_center_inside_crop']=int(crops[a][0][f]);fr[label+'_gt_crop_coverage']=crops[a][1][f]
            if writer is None:writer=csv.DictWriter(frame_file,fieldnames=list(fr));writer.writeheader()
            writer.writerow(fr)
        panel(name,r,(g,pp,ss,ev))
        if (idx+1)%12==0:print(json.dumps(dict(done=idx+1,total=96,seconds=round(time.time()-start,1))),flush=True)
    frame_file.close();write_csv(OUT/'per_sequence.csv',rows);write_csv(OUT/'frame_decomposition.csv',decomp);write_json(OUT/'events.json',events)
    groups={};attr_rows=[]
    for c in codes:
        ins=[r for r in rows if r['attr_'+c]];out=[r for r in rows if not r['attr_'+c]]
        if ins:
            s=summarize(ins);s['attribute']=c;s['complement_delta_auc']=summarize(out)['delta_auc'] if out else None
            s['member_sequences']=[r['sequence'] for r in ins];groups[c]=s
            ar={k:v for k,v in s.items() if k not in ['member_sequences','sequence_bootstrap95']}
            ar.update(ci95_low=s['sequence_bootstrap95'][0],ci95_high=s['sequence_bootstrap95'][1]);attr_rows.append(ar)
    history_groups={k:summarize([r for r in rows if r['history_group']==k]) for k in sorted(set(r['history_group'] for r in rows))}
    # TO/AIV union is motivated by earlier work, but still not an independent subgroup confirmation here.
    groups['TO_or_AIV']=summarize([r for r in rows if r['attr_TO'] or r['attr_AIV']])
    groups['neither_TO_nor_AIV']=summarize([r for r in rows if not(r['attr_TO'] or r['attr_AIV'])])
    summary=dict(overall=summarize(rows),attributes=groups,history_groups=history_groups,
        decomposition={cat:dict(total_frames=sum(r['frames'] for r in decomp if r['category']==cat),delta_auc_contribution_pp=sum(r['delta_auc_contribution_pp'] for r in decomp if r['category']==cat)/len(rows)) for cat in categories},
        frame_weighted_delta_auc=float(np.average([r['delta_auc'] for r in rows],weights=[r['frames'] for r in rows])),
        provenance=dict(public_state='paired common hash verified below',
            not_historical_mr_g1='quick uses dynamic Value gate; historical MR-only g=1 uses fixed gate; do not call this strict historical MR baseline reproduction'),seconds=time.time()-start)
    aa=[json.loads((RUN/a/'initial_state.json').read_text()) for a in ['control_final3','candidate_final']]
    pairing=dict(common_init_equal=aa[0]['paired_common_state_sha256']==aa[1]['paired_common_state_sha256'],batches=0,input_mismatches=0,sample_mismatches=0)
    bs=[(RUN/a/'batches.jsonl').read_text().splitlines() for a in ['control_final3','candidate_final']];assert len(bs[0])==len(bs[1])
    for a,b in zip(*bs):
        x,y=json.loads(a),json.loads(b);pairing['batches']+=1;pairing['input_mismatches']+=x['input_sha256']!=y['input_sha256'];pairing['sample_mismatches']+=x['samples']!=y['samples']
    summary['pairing']=pairing;write_json(OUT/'summary.json',summary);write_csv(OUT/'attributes.csv',attr_rows)
    fig,axs=plt.subplots(1,2,figsize=(13,4));order=sorted(rows,key=lambda r:r['delta_auc']); vals=[r['delta_auc'] for r in order]
    axs[0].bar(np.arange(len(vals)),vals,color=['#e16b5a' if v<0 else '#279a89' for v in vals]);axs[0].axhline(0,color='k',lw=.5);axs[0].set(xlabel='Sequence sorted by delta',ylabel='AUC delta (pp)',title='All 96 sequences')
    atr=sorted(attr_rows,key=lambda r:r['delta_auc']); axs[1].barh([r['attribute']+' n='+str(r['n']) for r in atr],[r['delta_auc'] for r in atr],color='#5692c4');axs[1].axvline(0,color='k',lw=.5);axs[1].set(xlabel='AUC delta (pp)',title='Overlapping official sequence attributes (post hoc)');fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
    # HTML uses local files, search/filter/sort and complete per-sequence PNGs.
    items=[]
    for r in rows:
        items.append('<article data-name="%s" data-attrs="%s" data-delta="%.8f"><h3>%s | %+.3f pp</h3><p>%s | historical: %s</p><a href="sequences/%s.png"><img loading="lazy" src="sequences/%s.png"></a></article>'%(html.escape(r['sequence'],quote=True),r['attributes'],r['delta_auc'],html.escape(r['sequence']),r['delta_auc'],r['attributes'],r['history_group'],html.escape(r['sequence'],quote=True),html.escape(r['sequence'],quote=True)))
    page='''<!doctype html><meta charset="utf-8"><title>96 sequence box audit</title><style>body{font:16px sans-serif;max-width:1300px;margin:20px auto;background:#fafafa}header{position:sticky;top:0;background:white;padding:12px;z-index:2}img{width:100%}article{margin:25px 0;background:white;border:1px solid #ddd;padding:8px}input,select{padding:8px}p{color:#555}</style><header><b>96 序列框对照</b> | 绿 GT，橙 control，青 candidate。属性为序列级，事件由预测差异回顾性选择。<br><input id="q" placeholder="序列或属性代码"><select id="order"><option value="name">序列名</option><option value="gain">收益优先</option><option value="loss">退化优先</option></select><span id="count"></span></header><main>'''+''.join(items)+'''</main><script>const q=document.querySelector('#q'),o=document.querySelector('#order'),m=document.querySelector('main'),a=[...document.querySelectorAll('article')];function update(){let n=0;a.sort((x,y)=>o.value==='name'?x.dataset.name.localeCompare(y.dataset.name):(o.value==='gain'?-1:1)*(+x.dataset.delta-+y.dataset.delta));for(const el of a){m.appendChild(el);let show=(el.dataset.name+' '+el.dataset.attrs).toLowerCase().includes(q.value.toLowerCase());el.hidden=!show;n+=show;}document.querySelector('#count').textContent=' '+n+' / 96';}q.oninput=update;o.onchange=update;update();</script>'''
    (OUT/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in summary.items() if k!='attributes'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
