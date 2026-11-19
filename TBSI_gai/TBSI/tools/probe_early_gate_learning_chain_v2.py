"""P0: independent real-task learning/propagation probe; no efficacy admission test.

Uses only live forward tensors. Does not import/call any historical audit tool.
Parent is frozen to isolate the new path; this is not matched joint training.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.models.tbsi_track import build_tbsi_track
from lib.models.layers.early_conditioned_value_direction import install_value_direction
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap

KEYS = ('pred_boxes', 'score_map', 'size_map', 'offset_map')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def reset_rng():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def get_samples(args, config, output):
    names = set(x.strip().split(',')[0] for x in
                (args.dataset.parent / 'trainingsetList.txt').read_text().splitlines() if x.strip())
    names = sorted(names, key=lambda n: hashlib.sha256(('EVDR-v2-p0:' + n).encode()).hexdigest())[:6]
    templates, searches, targets, rows = [[], []], [[], []], [], []
    for index, name in enumerate(names):
        folder = args.dataset / name
        boxes = np.loadtxt(str(folder / 'init.txt'), delimiter=',').reshape(-1, 4)
        frame = min(20, len(boxes) - 1)
        if frame < 1 or not np.isfinite(boxes[[0, frame - 1, frame]]).all() or (boxes[[0, frame - 1, frame], 2:] <= 0).any():
            raise ValueError('invalid preselected annotation: ' + name)
        zbox = torch.tensor(boxes[0], dtype=torch.float32)
        xbox = torch.tensor(boxes[frame], dtype=torch.float32)
        # Designed state, not an observed tracker state. Avoid GT-centered current crop.
        state = torch.tensor(boxes[frame - 1], dtype=torch.float32)
        sign = 1 if index % 2 == 0 else -1
        state[:2] += sign * torch.tensor([.35, -.20]) * state[2:]
        files = []
        for modality, mi in (('visible', 0), ('infrared', 1)):
            def frame_key(p):
                match = re.match(r'^[^0-9]*(\d+)', p.stem)
                if match is None: raise ValueError('non-numeric frame filename')
                return int(match.group(1)), p.name
            images = sorted((p for p in (folder / modality).iterdir()
                             if p.suffix.lower() in ('.jpg', '.jpeg', '.png')), key=frame_key)
            if len(images) != len(boxes): raise ValueError('frame/GT length mismatch: ' + name)
            mean = torch.tensor(config.DATA.MEAN[mi*3:mi*3+3]).view(3, 1, 1)
            std = torch.tensor(config.DATA.STD[mi*3:mi*3+3]).view(3, 1, 1)
            for destination, fi, box, size, factor in (
                (templates, 0, zbox, 128, 2.), (searches, frame, state, 256, 4.)):
                image = cv2.imread(str(images[fi]))
                if image is None: raise ValueError('unreadable image: ' + str(images[fi]))
                crop, resize, _ = sample_target(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), box, factor, size)
                tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255.
                destination[mi].append((tensor - mean) / std)
                files.append({'path': str(images[fi].resolve()), 'sha256': sha(images[fi])})
                if destination is searches and mi == 0:
                    targets.append(transform_image_to_crop(xbox, state, resize, torch.tensor([size, size]), normalize=True))
        rows.append(dict(sequence=name, role='fit' if index < 3 else 'readout',
                         template_index=0, search_index=frame,
                         designed_state_xywh=state.tolist(), gt_xywh=xbox.tolist(), annotation_sha256=sha(folder/'init.txt'), files=files))
    z, x, gt = [torch.stack(t).cuda() for t in templates], [torch.stack(t).cuda() for t in searches], torch.stack(targets).cuda()
    manifest = dict(rule='first six unique official train names sorted by SHA256(EVDR-v2-p0:+name); first three fit, last three readout; frame=min(20,last); no outcome replacement',
                    dataset=str(args.dataset.resolve()), official_list_sha256=sha(args.dataset.parent/'trainingsetList.txt'),
                    state='previous GT shifted +/-0.35 width, -/+0.20 height; designed diagnostic state, NOT actual tracker state',
                    augmentation='none; deterministic paired modality crops; no online template updates', samples=rows,
                    tensor_sha256={k:tsha(t) for k,t in zip(('z_rgb','z_tir','x_rgb','x_tir','gt'),z+x+[gt])})
    path=output/'sample_manifest.json'; path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return z,x,gt,dict(path=str(path),sha256=sha(path),samples=rows)


class LiveTrace:
    """Read-only hooks; route tensors derive from actual captured operator outputs."""
    def __init__(self, model):
        self.handles=[]; self.state={}; self.capture={}
        for name, module in model.named_modules():
            if module.__class__.__name__ != 'V2TemplateFusion': continue
            self.state[name]={'module':module, 'op':[]}
            def qkv_hook(m, inputs, out, key=name): self.state[key]['qkv']=out.detach()
            def attn_hook(m, inputs, out, key=name): self.state[key]['attn']=out.detach()
            def gate_hook(m, inputs, out, key=name): self.state[key]['gate']=out.detach()
            def op_hook(m, inputs, out, key=name):
                self.state[key]['op'].append((inputs[0].detach(),out.detach()))
            def fusion_hook(m, inputs, out, key=name):
                s=self.state[key]; raw=s['qkv']; b,length,_=raw.shape; n=length//2
                v=raw.reshape(b,length,3,m.num_heads,m.head_dim).permute(2,0,3,1,4)[2]
                a=s['attn']; gate=torch.exp(torch.tanh(m.early_gate_scale)*torch.tanh(s['gate'])).clamp(.25,4.)
                vr,vt=v[:,:,:n],v[:,:,n:]
                cr,ct=(s['op'][0][1],s['op'][1][1]) if len(s['op'])==2 else (vr,vt)
                routes={'RR':a[:,:,:n,:n]@vr,'RT':a[:,:,:n,n:]@(ct*gate[:,None,:,1:2]),
                        'TT':a[:,:,n:,n:]@vt,'TR':a[:,:,n:,:n]@(cr*gate[:,None,:,0:1])}
                for k,t in routes.items(): self.capture[key+'/'+k]=t.float().cpu()
                self.capture[key+'/V_rgb']=cr.float().cpu(); self.capture[key+'/V_tir']=ct.float().cpu()
                self.capture[key+'/source_V_rgb']=vr.float().cpu(); self.capture[key+'/source_V_tir']=vt.float().cpu()
                self.capture[key+'/fusion']=torch.cat(out,1).detach().float().cpu()
                s['op']=[]
            self.handles += [module.qkv.register_forward_hook(qkv_hook),module.attn_drop.register_forward_hook(attn_hook),module.early_gate.register_forward_hook(gate_hook),module.value_direction.register_forward_hook(op_hook),module.register_forward_hook(fusion_hook)]
        for index in model.backbone.tbsi_loc:
            next_index=index+1
            def block_hook(m, inputs, out, key=next_index):
                # Shared ViT block executes once per modality. First64 tokens are template.
                suffix='rgb' if 'next_block_%d/rgb'%key not in self.capture else 'tir'
                self.capture['next_block_%d/'%key+suffix]=out[:,64:].detach().float().cpu()
            self.handles.append(model.backbone.blocks[next_index].register_forward_hook(block_hook))
    def close(self):
        for h in self.handles:h.remove()


def load_operator_state(model, state):
    # A partial model.load_state_dict without BN metadata may synthesize and load
    # missing num_batches_tracked=0. Load only the explicitly owned operator.
    used=set()
    for name,module in model.named_modules():
        if module.__class__.__name__!='EarlyConditionedValueDirection':continue
        prefix=name+'.'
        local={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
        module.load_state_dict(local,strict=True)
        used.update(prefix+key for key in local)
    if used!=set(state):raise ValueError('operator state contains unexpected/missing owners')


def readout(model,z,x,trace=False):
    tracer=LiveTrace(model) if trace else None
    try:
        with torch.no_grad(): out=model(template=z,search=x)
        result={k:out[k].detach().float().cpu() for k in KEYS}
        if tracer: result.update(tracer.capture)
        return result
    finally:
        if tracer:tracer.close()


def differences(left,right):
    return {key:dict(max_abs=float((right[key]-left[key]).abs().max()),l2=float((right[key]-left[key]).norm())) for key in left if key in right}


def run(args, report):
    torch.set_num_threads(1); cv2.setNumThreads(0); reset_rng()
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    report['environment']=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),gpu_count=1,threads=1,workers=0,worker_reason='bounded six-sample direct-load diagnostic; not production joint training')
    report['parent']=dict(config=str(args.config),config_sha256=sha(args.config),checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint))
    if report['parent']['checkpoint_sha256']!='e38d18fe50c2895e75e4b9392bf22d9323bb8e6ad31ccbe39ad8c038e3fc1811': raise ValueError('wrong explicit parent')
    update_config_from_file(str(args.config)); model=build_tbsi_track(copy.deepcopy(cfg),training=False)
    checkpoint=torch.load(str(args.checkpoint),map_location='cpu'); model.load_state_dict(checkpoint['net'],strict=True); del checkpoint
    model.cuda().eval()
    for p in model.parameters():p.requires_grad_(False)
    parent_hash={k:tsha(t) for k,t in model.state_dict().items()}
    z,x,gt,report['data']=get_samples(args,cfg,args.output)
    native=readout(model,z,x); report['installed_layers']=install_value_direction(model,rank=8,max_step=.1)
    initial=readout(model,z,x)
    report['measurement']={'initial_parent_delta':differences(native,initial),'initial_repeat_delta':differences(initial,readout(model,z,x))}
    init_state={k:t.cpu().clone() for k,t in model.state_dict().items() if '.value_direction.' in k}
    if args.recover_from:
        saved_init=torch.load(args.recover_from/'common_operator_init.pth',map_location='cpu')
        if saved_init['parent_sha256']!=report['parent']['checkpoint_sha256']:raise ValueError('recovery parent mismatch')
        init_state=saved_init['net']
    torch.save({'net':init_state,'parent_sha256':report['parent']['checkpoint_sha256']},args.output/'common_operator_init.pth')
    params=[p for p in model.parameters() if p.requires_grad]
    heatmap=generate_heatmap(gt.unsqueeze(0),256,16)[0].unsqueeze(1)
    focal=FocalLoss()
    def loss_of(out, sl):
        boxes=out['pred_boxes'].reshape(-1,4)
        target=gt[sl]; giou,iou=giou_loss(box_cxcywh_to_xyxy(boxes),box_xywh_to_xyxy(target).clamp(0,1))
        center=torch.cat((target[:,:2]+target[:,2:]/2,target[:,2:]),-1)
        l1=F.l1_loss(boxes,center); loc=focal(out['score_map'],heatmap[sl])
        return 2*giou+5*l1+loc,dict(giou=float(giou.detach()),l1=float(l1.detach()),focal=float(loc.detach()),mean_decoded_iou=float(iou.detach().mean()))
    report['arms']={}
    for mode in ('radial','direction'):
        load_operator_state(model,init_state)
        for m in model.modules():
            if hasattr(m,'value_direction_enabled'):m.value_direction_enabled=True;m.value_direction_mode=mode
        reset_rng(); optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
        scaler=torch.cuda.amp.GradScaler(enabled=True)
        successes=[]
        handle=optimizer.register_step_post_hook(lambda *unused:successes.append(True))
        trace=[]
        fit_z=[t[:3] for t in z]; fit_x=[t[:3] for t in x]
        if args.recover_from:
            source=args.recover_from/(mode+'_diagnostic.pth')
            saved=torch.load(source,map_location='cpu')['net']
            if set(saved)!=set(init_state):raise ValueError('recovery keys differ')
            load_operator_state(model,saved)
            rows=[json.loads(line) for line in args.recovery_log.read_text().splitlines() if line.startswith('{"arm":')]
            matches=[row for row in rows if row['arm']==mode]
            if len(matches)!=1:raise ValueError('cannot recover original successful-step count')
            original_steps=int(matches[0]['successful_updates'])
            report.setdefault('recovery',{})[mode]=dict(checkpoint=str(source),sha256=sha(source),stdout=str(args.recovery_log),stdout_sha256=sha(args.recovery_log),original_successful_updates=original_steps,original_attempts=args.steps,original_full_gradient_trace='NA; serialization failure; do not reconstruct',new_optimizer_updates=0)
        else:
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=True):out=model(template=fit_z,search=fit_x)
                # Same production GIoU/L1/focal targets; compute scalar losses FP32.
                out={k:v.float() if torch.is_tensor(v) else v for k,v in out.items()}
                loss,parts=loss_of(out,slice(0,3))
                if not torch.isfinite(loss):raise RuntimeError('nonfinite task loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                grad={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad}
                scale_before=scaler.get_scale();before=len(successes)
                torch.nn.utils.clip_grad_norm_(params,.1);scaler.step(optimizer);scaler.update()
                trace.append(dict(attempt=step+1,successful_update=len(successes),step_executed=len(successes)>before,loss=float(loss.detach()),components=parts,gradient_norms=grad,amp_scale_before=scale_before,amp_scale_after=scaler.get_scale()))
        handle.remove()
        plain=readout(model,z,x); measured=readout(model,z,x,trace=True);repeat=readout(model,z,x,trace=True)
        for m in model.modules():
            if hasattr(m,'value_direction_enabled'):m.value_direction_enabled=False
        off=readout(model,z,x,trace=True)
        for m in model.modules():
            if hasattr(m,'value_direction_enabled'):m.value_direction_enabled=True
        learned={k:t.detach().cpu().clone() for k,t in model.state_dict().items() if '.value_direction.' in k}
        # Actual disk save/load of the newly learned module state.
        checkpoint_path=args.output/(mode+'_diagnostic.pth');torch.save({'net':learned,'scope':'frozen-parent P0, not joint trained tracker'},checkpoint_path)
        load_operator_state(model,torch.load(checkpoint_path,map_location='cpu')['net'])
        reload_out=readout(model,z,x)
        with torch.no_grad():
            full_out=model(template=z,search=x)
            per_sample=[]
            for i in range(6):
                one={k:v[i:i+1] for k,v in full_out.items() if k in KEYS}
                _,parts=loss_of(one,slice(i,i+1));per_sample.append(parts)
        geometry={}
        for source in ('V_rgb','V_tir'):
            for key in measured:
                if key.endswith('/'+source):
                    v=measured[key.rsplit('/',1)[0]+'/source_'+source];w=measured[key];norm=v.norm(dim=-1)
                    geometry[key]=dict(max_absolute_norm_delta=float((w.norm(dim=-1)-norm).abs().max()),max_relative_vector_delta=float(((w-v).norm(dim=-1)/norm.clamp_min(1e-8)).max()),max_angle_degrees=float(torch.atan2((w-(w*v).sum(-1,keepdim=True)*v/v.square().sum(-1,keepdim=True).clamp_min(1e-16)).norm(dim=-1),(w*v).sum(-1)/norm.clamp_min(1e-8)).abs().max()*180/math.pi))
        optimizer.zero_grad(set_to_none=True)
        task_out=model(template=fit_z,search=fit_x)
        task_loss,task_parts=loss_of(task_out,slice(0,3))
        task_loss.backward()
        measured_grads={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad}
        current_learning=dict(loss=float(task_loss.detach()),components=task_parts,output_finite=all(bool(torch.isfinite(task_out[k]).all()) for k in KEYS),gradient_norms=measured_grads,gradient_finite=all(v is None or math.isfinite(v) for v in measured_grads.values()),nonzero_gradient_tensors=sum(v is not None and math.isfinite(v) and v>0 for v in measured_grads.values()),optimizer_updates_in_measurement=0)
        report['arms'][mode]=dict(attempted_steps=args.steps,successful_steps=original_steps if args.recover_from else len(successes),trace=trace,
            changed_parameter_tensors=sum(not torch.equal(init_state[k],v) for k,v in learned.items()),
            measurement_hook_delta=differences(plain,measured),measurement_repeat_delta=differences(measured,repeat),
            saved_reload_delta=differences(plain,reload_out),off_vs_original_parent=differences(native,off),
            active_path_changes=differences(off,measured),source_geometry=geometry,per_sample_readouts=per_sample,current_learning=current_learning,
            fit_samples=3,readout_samples=3,readout_scope='fixed crop decoded boxes without Hann/window/tracker feedback; no tracking/AUC claim')
        print(json.dumps({'arm':mode,'successful_updates':report['arms'][mode]['successful_steps'],'changed_parameter_tensors':report['arms'][mode]['changed_parameter_tensors'],'hook_max_abs':max(v['max_abs'] for v in report['arms'][mode]['measurement_hook_delta'].values())}),flush=True)
    report['parent_state_drift_keys']=[k for k,v in parent_hash.items() if tsha(model.state_dict()[k])!=v]
    report['parent_state_unchanged']=not report['parent_state_drift_keys']
    report['scope']='P0 frozen-parent learning connectivity; separate radial/direction fit diagnostics, no efficacy ranking'
    report['status']='COMPLETE-P0-OBSERVATIONS'


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=16)
    p.add_argument('--recover-from',type=Path);p.add_argument('--recovery-log',type=Path)
    args=p.parse_args()
    if bool(args.recover_from)!=bool(args.recovery_log):raise ValueError('recovery requires checkpoint directory AND original stdout')
    if not 1<=args.steps<=32:raise ValueError('P0 budget is 1..32 updates per arm')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'id':'EG-EVDR-P0-READOUT-RECOVERY-V2-20261026' if args.recover_from else 'EG-EVDR-P0-LEARNING-CHAIN-V2-20261026','implementation':'independent tool, no historical audit imports','command':sys.argv,'script_sha256':sha(__file__),'result_metrics':{k:'NA' for k in ('AUC','PR','NPR','OP50','OP75','FPS')}}
    start=time.monotonic()
    try:run(args,report)
    except Exception as error:
        report.update(status='EXECUTION-ERROR-NOT-METHOD-REJECTION',error=repr(error));raise
    finally:
        report['wall_seconds']=time.monotonic()-start;report['gpu_hours_wall_upper_bound']=report['wall_seconds']/3600
        nonfinite=[]
        def clean(value,path='root'):
            if isinstance(value,float) and not math.isfinite(value):
                nonfinite.append(path);return {'nonfinite':str(value)}
            if isinstance(value,dict):return {k:clean(v,path+'.'+k) for k,v in value.items()}
            if isinstance(value,list):return [clean(v,path+'.'+str(i)) for i,v in enumerate(value)]
            return value
        serializable=clean(report);serializable['nonfinite_diagnostic_paths']=nonfinite
        (args.output/'observations.json').write_text(json.dumps(serializable,indent=2,ensure_ascii=False,allow_nan=False)+chr(10))
        print(json.dumps({k:v for k,v in report.items() if k not in ('arms','data','measurement')},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
