"""Synthetic MR integration/latency diagnostic. No dataset, loss or optimizer.

Run from the project disk. Explicit checkpoint is a read-only speed reference,
never a new candidate's training initialization. Outputs use a distinct run dir.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.models.tbsi_track.tbsi_track import build_tbsi_track
from lib.models.layers.tbsi_layer import V2TemplateFusion
from lib.models.layers.receiver_source_mass_reallocator import ReceiverSourceMassReallocator
from lib.models.layers.low_rank_joint_mass_reallocator import LowRankJointMassReallocator


def measure(fn, iterations, warmup=30, rounds=5):
    rows = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(rounds):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            start.record()
            wall = time.perf_counter()
            for _ in range(iterations):
                fn()
            end.record()
            torch.cuda.synchronize()
            rows.append({'cuda_ms': start.elapsed_time(end) / iterations,
                         'wall_ms': (time.perf_counter() - wall) * 1000 / iterations,
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    wall_ms = statistics.median(r['wall_ms'] for r in rows)
    return {'rounds': rows, 'median_wall_ms': wall_ms,
            'synthetic_forward_per_second': 1000 / wall_ms,
            'median_cuda_ms': statistics.median(r['cuda_ms'] for r in rows)}


def state_hash(state):
    out = hashlib.sha256()
    for k, v in sorted(state.items()):
        out.update(k.encode()); out.update(v.detach().cpu().contiguous().numpy().tobytes())
    return out.hexdigest()


def check_contracts(device):
    torch.manual_seed(401)
    b, h, n, d = 1, 12, 64, 768
    a = torch.randn(b, h, n, n * 2, device=device).softmax(-1)
    p, s = a[..., n:], a[..., :n].sum(-1, keepdim=True)
    module = LowRankJointMassReallocator().to(device).eval()
    cross0, scale0 = module.allocate(p, s, torch.zeros(b, n, n, device=device))
    pos, pos_self = module.allocate(p, s, torch.full((b,n,n), 2., device=device))
    neg, neg_self = module.allocate(p, s, torch.full((b,n,n), -2., device=device))
    zero, zero_scale = module.allocate(torch.zeros_like(p), torch.ones_like(s), torch.full((b,n,n), 4., device=device))
    error = (cross0-p).abs().max().item()
    if error > 1e-6 or (scale0-1).abs().max().item() > 1e-6:
        raise RuntimeError('zero correction does not preserve joint probabilities')
    if not (neg.sum(-1) < p.sum(-1)).all() or not (pos.sum(-1) > p.sum(-1)).all():
        raise RuntimeError('constant row correction cannot change cross budget')
    if zero.count_nonzero().item() or not torch.equal(zero_scale,torch.ones_like(zero_scale)):
        raise RuntimeError('zero cross support is not preserved')
    # Compare against a full joint softmax for nonconstant scores; reuse the
    # parent's dropout mask, rather than normalizing post-dropout rows.
    delta=torch.randn(b,n,n,device=device).clamp(-4,4)
    dropout=torch.nn.Dropout(0.2).train()
    torch.manual_seed(301); dropped=dropout(a)
    cr, ss=module.allocate(p,s,delta,dropped[...,n:])
    expected=torch.cat((a[...,:n],p*delta[:,None].exp()),dim=-1)
    expected=expected/expected.sum(-1,keepdim=True)
    mask=torch.where(a>0,dropped/a,torch.zeros_like(a))
    actual=torch.cat((dropped[...,:n]*ss,cr),dim=-1)
    drop_err=(actual-expected*mask).abs().max().item()
    if drop_err>1e-6: raise RuntimeError('pre-dropout budget allocation mismatch')
    return {'zero_probability_max_abs':error,'dropout_reference_max_abs':drop_err,
            'negative_row_lowers_cross_budget':True,'positive_row_raises_cross_budget':True,
            'zero_cross_support_preserved':True,'training':False,'backward':False,'optimizer_steps':0}


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--output',required=True)
    args.add_argument('--reference-config',required=True)
    args.add_argument('--reference-checkpoint',required=True)
    args.add_argument('--before-output',required=True)
    args.add_argument('--iterations',type=int,default=100)
    args.add_argument('--reverse',action='store_true')
    opt=args.parse_args()
    out=Path(opt.output).resolve()
    if '/autodl-fs' in str(ROOT) or '/autodl-fs' in str(out) or '/autodl-fs' in str(Path.cwd().resolve()):
        raise RuntimeError('project disk boundary')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'benchmark.json').exists():raise RuntimeError('refusing to overwrite completed benchmark')
    # Caller additionally checks host processes. Refuse another CUDA workload.
    import subprocess
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if running: raise RuntimeError('another CUDA process is active: '+running)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device='cuda'
    update_config_from_file(opt.reference_config)
    base=copy.deepcopy(cfg)
    # Reconstruct the archived runner's actual factory overrides; YAML alone
    # names a historical relation branch, not the MR-only executed path.
    base.MODEL.BACKBONE.CROSS_MASS_PRESERVING_RELATION=False
    base.MODEL.BACKBONE.MASS_REDIST_ENABLE=True
    base.MODEL.BACKBONE.MASS_REDIST_MODE='pair'
    base.MODEL.INIT_CHECKPOINT=''
    weights=torch.load(opt.reference_checkpoint,map_location='cpu')['net']
    reference=torch.load(opt.before_output,map_location='cpu')
    z=reference['z'].to(device);x=reference['x'].to(device)
    z=list(z.split(3,dim=1));x=list(x.split(3,dim=1))
    result={'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(),'threads':1,'precision':'FP32','TF32':False,
              'cudnn_benchmark':False,'input':'fixed Gaussian synthetic tensors; template128/search256; B1',
              'dataset':'NA','training':False,'optimizer_steps':0},
            'checkpoint':str(Path(opt.reference_checkpoint).resolve()),
            'config':str(Path(opt.reference_config).resolve()),
            'config_overrides':{'CROSS_MASS_PRESERVING_RELATION':False,'MASS_REDIST_ENABLE':True,'g':1},
            'contracts':check_contracts(device),'model':{},'fusion':{},'allocator':{}}
    # One input set, deterministic order. Alternate measurement order in a
    # subsequent pass to expose clock/order artifacts without extra variants.
    identity_outputs={}
    modes=['pair','lr_joint','lr_conserved','b0']
    if opt.reverse:modes=list(reversed(modes))
    for mode in modes:
        local=copy.deepcopy(base)
        if mode=='b0':
            local.MODEL.BACKBONE.MASS_REDIST_ENABLE=False
            local.MODEL.BACKBONE.EARLY_CONDITIONED_CROSS_GATE=False
        else:local.MODEL.BACKBONE.MASS_REDIST_MODE=mode
        torch.manual_seed(42);model=build_tbsi_track(local,training=False)
        target=model.state_dict()
        shared={k:v for k,v in weights.items() if k in target and target[k].shape==v.shape
                and not (mode.startswith('lr') and 'mass_reallocator_' in k)}
        missing,unexpected=model.load_state_dict(shared,strict=False)
        if mode=='pair' and (missing or unexpected):raise RuntimeError('parent state does not load exactly')
        if mode.startswith('lr') and any('mass_reallocator_' not in k for k in missing):
            raise RuntimeError('unexpected non-candidate state missing: '+str(missing))
        for m in model.modules():
            if isinstance(m,V2TemplateFusion) and hasattr(m,'early_gate_scale'):
                m.early_gate_scale.data.zero_();m.early_gate_scale.requires_grad_(False)
        loaded_state=model.state_dict()
        copied_hash=state_hash({k:loaded_state[k] for k in shared})
        expected_hash=state_hash(shared)
        if copied_hash!=expected_hash:raise RuntimeError('shared state changed on loading')
        model=model.to(device).eval()
        fusions=[m for m in model.modules() if isinstance(m,V2TemplateFusion)]
        captured={}
        def hook(m, inputs, kwargs):
            if not captured:
                captured['args']=inputs
                captured['kwargs']=kwargs
        handle=fusions[0].register_forward_pre_hook(hook,with_kwargs=True)
        with torch.inference_mode():prediction=model(z,x)
        handle.remove()
        identity_outputs[mode]=prediction['pred_boxes'].detach().cpu()
        details={'parameter_count':sum(p.numel() for p in model.parameters()),
                 'missing_keys':missing,'shared_tensor_count':len(shared),
                 'shared_state_sha256':copied_hash,'attn_dropout':[m.attn_drop.p for m in fusions]}
        if mode=='pair':
            errors={k:(prediction[k].cpu()-v).abs().max().item() for k,v in reference['out'].items()}
            if any(v!=0 for v in errors.values()):raise RuntimeError('disabled path differs from pre-edit reference: '+str(errors))
            details['pre_edit_output_max_abs']=errors
        if mode.startswith('lr'):
            with torch.inference_mode():
                old=fusions[0].search_context_gain
                ev=torch.randn(1,64,3,device=device)
                context=(torch.randn(1,64,768,device=device),torch.randn(1,64,768,device=device),
                         torch.randn(1,256,768,device=device),torch.randn(1,256,768,device=device))
                _,original=old(ev,context,return_context=True)
                lean=old.context_only(ev,context)
                err=(original-lean).abs().max().item()
                if err>2e-6:raise RuntimeError('batched context exceeded numerical tolerance')
                details['context_only_max_abs']=err
            # Stress nonzero path for timing too; zero-init must not become a
            # shortcut. No optimization, real data, learned-quality comparison.
            for m in model.modules():
                if isinstance(m,LowRankJointMassReallocator):
                    torch.nn.init.normal_(m.receiver.weight,std=0.02)
            with torch.inference_mode():changed=model(z,x)
            if not torch.isfinite(changed['pred_boxes']).all():raise RuntimeError('nonfinite active model')
            details['synthetic_nonzero_vs_identity_box_max_abs']=(changed['pred_boxes']-prediction['pred_boxes']).abs().max().item()
        print('MEASURE model',mode,flush=True)
        details.update(measure(lambda:model(z,x),opt.iterations))
        result['model'][mode]=details
        print('MEASURE fusion',mode,flush=True)
        result['fusion'][mode]=measure(lambda:fusions[0](*captured['args'],**captured['kwargs']),max(opt.iterations,150))
        # release large tensors before next model
        del model,fusions,prediction,loaded_state,target,shared,captured
        if mode.startswith('lr'):del changed,original,lean,old,context
        gc.collect();torch.cuda.empty_cache()
    result['identity_lr_joint_vs_b0_boxes_max_abs']=(identity_outputs['lr_joint']-identity_outputs['b0']).abs().max().item()
    if result['identity_lr_joint_vs_b0_boxes_max_abs']>1e-5:raise RuntimeError('zero candidate differs from common B0')
    # Direct allocator shape, B1 is actual tracker use. B16 only a tensor
    # kernel scaling diagnostic, not a training throughput measurement.
    for b in [1,16]:
        h,n,d=12,64,768
        p=torch.randn(b,h,n,n*2,device=device).softmax(-1)
        self_p,cross=p[...,:n],p[...,n:]
        v=torch.randn(b,h,n,d//h,device=device)
        recv=(self_p@v).transpose(1,2).reshape(b,n,d)
        ctx=torch.randn(b,n,64,device=device);ev=torch.randn(b,n,3,device=device)
        for mode in ['pair','lr_joint','lr_conserved']:
            m=(ReceiverSourceMassReallocator() if mode=='pair' else LowRankJointMassReallocator(allocation='joint' if mode=='lr_joint' else 'conserved')).to(device).eval()
            if mode=='pair':fn=lambda:m(cross,recv,v,ctx,ev)
            else:
                torch.nn.init.normal_(m.receiver.weight,std=.02)
                fn=lambda:m(cross,self_p.sum(-1,keepdim=True),recv,v,ctx,ev)
            result['allocator'][mode+'_B'+str(b)]=measure(fn,250)
            del m,fn
        del p,self_p,cross,v,recv,ctx,ev
        gc.collect();torch.cuda.empty_cache()
    # Save after all diagnostics; no accuracy, sequence or tracking FPS claims.
    (out/'benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{m:round(v['median_wall_ms'],4) for m,v in result[k].items()} for k in ['model','fusion','allocator']},indent=2),flush=True)


if __name__=='__main__':
    main()
