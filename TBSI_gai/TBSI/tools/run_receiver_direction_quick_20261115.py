#!/usr/bin/env python3
"""Isolated residual-direction supervision quick; historical exact parent reuse."""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_dual_message_p1 import train,evaluate,sha,write_json
RUN=ROOT/'output/experiments/MSHA-MR-RESIDUAL-TARGET-DIRECTION-QUICK-S42-20261115'
PARENT=ROOT/'output/experiments/MSHA-MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115'
CFG=ROOT/'experiments/tbsi_track/MR-RESIDUAL-TARGET-DIRECTION-QUICK-S42-RETRY1-20261115.yaml'
MANIFEST=ROOT/'experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json'
LIST=MANIFEST.parent/'screen96_max2000.txt'

def record(output,cfg,seed,arm,status,conclusion,checkpoint='NA',gpu_hours='NA'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity='MR-RESIDUAL-TARGET-DIRECTION-S42-20261115-'+output.name.upper()
    row=dict(ID=identity,Date='20261115',Parent='MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-CANDIDATE; matched historical residual/no-loss arm',
        CommitOrSnapshot=str(RUN/'source/retry1_launch_snapshot.tar.gz'),Config=str(cfg)+' SHA256 '+sha(cfg),Seed=str(seed),
        Dataset='LasHeR train977,59904 pairs at468updates; fixed screen96_max2000 96seq/54666frames; paths/hashes in PROTOCOL.json',
        EvalScope='Public SOT fresh single-stage; batch16 accum8,AMP,workers4; same common init and input-order contract; train-only normalized target/background utility gradient teaches only residual parameters',
        GPU='1x NVIDIA GeForce RTX 4090 D 24 GiB',HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8; seed42; exclusive worker4 task; prefetch2,persistentfalse',
        GPUHours=str(gpu_hours),Checkpoint=str(checkpoint),
        AssumptionOrHypothesis='Adding one residual-only target-direction objective can learn locally useful message direction; operator,input,base task loss,data,steps and optimizer fixed.',
        Conclusion=conclusion,Status=status,Evidence=str(output),
        Notes='Loss weight0.1,margin0.1; teacher detached first-order gradient of raw GT-center vs strongest background log response. No inference GT or new model parameters. No full promotion; original quick parent is dynamic gate+MR,not historical MRg1.')
    if output.name.startswith('contract'):
        row['Dataset']='LasHeR real training prefix:128 pairs,8 microbatches,1 successful update; no tracker evaluation'
        row['EvalScope']='Training-contract diagnostic only; finite auxiliary gradients into residual readouts,zero unrelated gradients; real optimizer step and disabled parity; no checkpoint saved'
    csvpath=ROOT/'experiments/experiment_asset_registry.csv';mdpath=ROOT/'experiments/实验资产数据库.md'
    with csvpath.open(newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
    full={k:row.get(k,'NA') for k in fields};matches=[i for i,r in enumerate(old) if r['ID']==identity]
    assert len(matches)<=1
    if matches:old[matches[0]]=full
    else:old.append(full)
    tmp=csvpath.with_suffix('.direction.tmp')
    with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(old)
    tmp.replace(csvpath)
    start='<!-- '+identity+' BEGIN -->';end='<!-- '+identity+' END -->'
    block=start+'\n### '+identity+'\n\n'+''.join('- **'+k+'**：'+full[k]+'\n' for k in fields)+end
    text=mdpath.read_text()
    if start in text:
        a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+block+text[b:]
    else:text+='\n\n'+block+'\n'
    mdpath.write_text(text)
    write_json(output/'run_record.json',full)

def install_trace(args):
    import torch
    import lib.train.actors as actors
    original=actors.TBSITrackActor
    class TracedActor(original):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.trace_calls=0
        def compute_losses(self,pred_dict,gt_dict,return_status=True):
            loss,stats=super().compute_losses(pred_dict,gt_dict,return_status=True)
            self.trace_calls+=1
            if self.trace_calls==1:
                teacher=self.receiver_target_direction
                assert teacher is not None
                aux=teacher.last_loss
                residual=[(n,p) for n,p in self.net.named_parameters() if 'receiver_message_residual' in n]
                readouts=[(n,p) for n,p in residual if 'readout' in n]
                others=[p for n,p in self.net.named_parameters() if 'receiver_message_residual' not in n and p.requires_grad]
                ag=torch.autograd.grad(aux,[p for n,p in residual],retain_graph=True,allow_unused=True)
                other_grad=torch.autograd.grad(aux,others,retain_graph=True,allow_unused=True)
                task=loss-self.cfg.TRAIN.RECEIVER_TARGET_DIRECTION_WEIGHT*aux
                tg=torch.autograd.grad(task,[p for n,p in readouts],retain_graph=True,allow_unused=True)
                all_finite=all(g is None or torch.isfinite(g).all() for g in ag)
                auxnorm=float(torch.stack([g.float().square().sum() for g in ag if g is not None]).sum().sqrt())
                tasknorm=float(torch.stack([g.float().square().sum() for g in tg if g is not None]).sum().sqrt())
                contract=dict(real_training_batch=True,auxiliary_gradients_finite=bool(all_finite),auxiliary_gradient_norm=auxnorm,
                    task_readout_gradient_norm=tasknorm,weighted_aux_to_task_readout_ratio=.1*auxnorm/max(tasknorm,1e-30),
                    auxiliary_unrelated_parameter_gradients=sum(g is not None for g in other_grad),
                    gradient_norms={n:float(g.norm()) if g is not None else None for (n,p),g in zip(residual,ag)},
                    initial_residual_readout_zero=all(torch.count_nonzero(p)==0 for n,p in readouts),
                    initial_auxiliary_value=float(aux.detach()),teacher_stats=stats,
                    original_tracking_forward_unchanged='capture hooks return original output; no host model forward code modified',
                    no_parameter_step_before_check=True)
                assert contract['auxiliary_gradients_finite'] and auxnorm>0 and contract['auxiliary_unrelated_parameter_gradients']==0
                assert contract['initial_residual_readout_zero'] and abs(contract['initial_auxiliary_value'])<1e-10
                write_json(Path(args.output)/'direction_contract.json',contract)
                print(json.dumps({'direction_contract':contract}),flush=True)
            if self.trace_calls<=2 or self.trace_calls%32==0:
                with (Path(args.output)/'direction_trace.jsonl').open('a') as f:f.write(json.dumps(dict(microbatch_call=self.trace_calls,**stats),allow_nan=False)+'\n')
            return (loss,stats) if return_status else loss
    actors.TBSITrackActor=TracedActor

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['contract','train','evaluate'])
    parser.add_argument('--output',required=True);parser.add_argument('--updates',type=int,default=468)
    args=parser.parse_args();Path(args.output).mkdir(parents=True,exist_ok=True);write_json(Path(args.output)/'process.json',dict(pid=os.getpid(),mode=args.mode));args.config=str(CFG);args.seed=42;args.arm='residual_target_direction';args.manifest=str(MANIFEST)
    args.milestones=str(args.updates);args.diagnostic_only=args.mode=='contract';args.checkpoint_weights_only=True
    args.shared_init_snapshot=str(PARENT/'shared_init_final.pth.tar');args.export_shared_init=None
    args.expected_shared_state_sha256=json.loads((PARENT/'candidate_final/initial_state.json').read_text())['shared_state_sha256']
    args.checkpoint=str(RUN/'candidate_retry1/checkpoint_u0468.pth.tar');args.list=str(LIST)
    # Block project execution/output on the archival filesystem.
    assert str(ROOT.resolve()).startswith('/root/autodl-tmp/') and str(Path(args.output).resolve()).startswith('/root/autodl-tmp/')
    try:
        if args.mode=='evaluate':evaluate(args)
        else:
            install_trace(args);train(args,recorder=record)
    except Exception as exc:
        import traceback
        write_json(Path(args.output)/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        record(args.output,args.config,42,args.arm,'FAILED / NO-UTILITY-CONCLUSION',type(exc).__name__+': '+str(exc))
        raise

if __name__=='__main__':main()
