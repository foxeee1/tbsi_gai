#!/usr/bin/env python3
"""Focused integration checks for the single EVDR receiver-transport change."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from run_early_gate_p1_v2 import setup, config, state_hash, environment, sha
import torch
from lib.models.layers.early_conditioned_value_direction import EarlyConditionedValueDirection
from lib.models.layers.early_conditioned_value_direction_aligned import AttentionAlignedValueDirection

OUT = ROOT / 'experiments/early_gate_closed_loop_20261026/evdr_scope_v2_20261026'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def algebra():
    setup(42)
    old = EarlyConditionedValueDirection(48, 3, 4)
    new = AttentionAlignedValueDirection(48, 3, 4)
    new.load_state_dict(old.state_dict(), strict=True)
    v = torch.randn(2,3,5,16); src = torch.randn(2,5,48); rec = torch.randn(2,5,48)
    evidence = torch.randn(2,5,3); eye = torch.eye(5)[None,None].repeat(2,3,1,1)
    require(torch.equal(new(v,src,rec,evidence,eye),v), 'Zero initialization not identity')
    with torch.no_grad(): old.up.normal_(std=.05)
    new.load_state_dict(old.state_dict(),strict=True)
    reference = old(v,src,rec,evidence)
    diagonal = new(v,src,rec,evidence,eye)
    require(torch.equal(reference,diagonal), 'Identity receiver map must reproduce v1 exactly in FP32')
    permutation = torch.tensor([2,4,0,1,3])
    a = torch.rand(2,3,5,5)
    y = new(v,src,rec,evidence,a)
    yp = new(v,src,rec[:,permutation],evidence,a[:,:,permutation,:])
    require(torch.allclose(y,yp,rtol=1.e-6,atol=1.e-6), 'Receiver/attention row permutation changed semantics')
    # Column scaling must not change receiver distribution.
    ys = new(v,src,rec,evidence,a*torch.rand(2,3,1,5).add(.1))
    require(torch.allclose(y,ys,rtol=1.e-6,atol=1.e-6), 'Wrong source-column normalization')
    norm_error = float((y.norm(dim=-1)-v.norm(dim=-1)).abs().max())
    require(norm_error < 2.e-6, 'Value source norm not preserved')
    require(torch.equal(new(v,src,rec,evidence,a,enabled=False),v), 'Disabled path not exact identity')
    zero = new(torch.zeros_like(v),src,rec,evidence,torch.zeros_like(a))
    require(torch.equal(zero,torch.zeros_like(v)), 'Zero Values/empty cross mass unstable')
    # A nontrivial learned path transports gradients to the original attention
    # map; it does not freeze or replace the QK computation.
    a.requires_grad_(); new(v,src,rec,evidence,a).square().mean().backward()
    # Squared norm is deliberately invariant; use a directional task readout.
    new.zero_grad(); a.grad = None
    weight = torch.randn_like(v)
    (new(v,src,rec,evidence,a)*weight).mean().backward()
    require(torch.isfinite(a.grad).all() and a.grad.abs().max()>0, 'Transport path attention gradient absent')
    restored = copy.deepcopy(new)
    restored.load_state_dict(new.state_dict(),strict=True)
    require(torch.equal(new(v,src,rec,evidence,a),restored(v,src,rec,evidence,a)), 'State roundtrip differs')
    # Disabled host must retain the exact pre-change learned Early gate math.
    spec = importlib.util.spec_from_file_location('evdr_prechange_host', OUT/'before/tbsi_layer.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from lib.models.layers.tbsi_layer import V2TemplateFusion
    host_old = module.V2TemplateFusion(48,3,early_conditioned_cross_gate=True)
    host_new = V2TemplateFusion(48,3,early_conditioned_cross_gate=True)
    host_new.load_state_dict(host_old.state_dict(),strict=True)
    early_context = (src, rec, torch.randn(2,9,48), torch.randn(2,9,48))
    with torch.no_grad():
        a0 = host_old(src,rec,early_context); a1 = host_new(src,rec,early_context)
    require(all(torch.equal(x,y) for x,y in zip(a0,a1)), 'Disabled host differs from source snapshot')
    return {'zero_init_exact':True,'identity_transport_v1_exact':True,
            'receiver_permutation_invariant':True,'column_scaling_invariant':True,
            'source_norm_maxabs':norm_error,'empty_mass_finite':True,
            'transport_attention_gradient':True,'state_roundtrip_exact':True,
            'disabled_host_prechange_exact':True,
            'new_parameters':sum(p.numel() for p in new.parameters()),
            'same_parameters_as_v1':sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())}


def real_batch():
    from lib.models.tbsi_track import build_tbsi_track
    from lib.train.actors import TBSITrackActor
    from lib.train.admin.settings import Settings
    from lib.train.base_functions import update_settings, get_optimizer_scheduler
    from lib.train.data import processing, transforms as tfm
    from lib.train.data.image_loader import opencv_loader
    from lib.train.dataset.lasher import LasHeR
    from lib.train.data.early_gate_paired_sampling import PairedStratifiedSampler, paired_collate
    from lib.utils import TensorDict
    from lib.utils.box_ops import giou_loss
    from lib.utils.focal_loss import FocalLoss
    setup(42)
    cfg = config(ROOT/'experiments/tbsi_track/EG_EVDR_ALIGNED_P1_SOT_S42_20261026.yaml')
    net = build_tbsi_track(cfg,training=True)
    initial_hash = state_hash({k:v for k,v in net.state_dict().items() if '.value_direction.' not in k})
    control_hash = json.loads((ROOT/'output/experiments/EG_EVDR_P1_SOT_V3_CONTROL_S42_20261026/initial_state.json').read_text())['shared_state_sha256']
    require(initial_hash==control_hash,'Shared initialized state does not match existing control')
    net = net.cuda()
    settings = Settings(); settings.local_rank=-1; update_settings(settings,cfg)
    settings.num_template=cfg.DATA.TEMPLATE.NUMBER;settings.num_search=cfg.DATA.SEARCH.NUMBER
    dataset = LasHeR(str(ROOT/'data/lasher/trainingset'),split='train',image_loader=opencv_loader,
                    sequence_list_file=str(ROOT/'experiments/early_gate_closed_loop_20261026/p1_sot_v3/train_sequences.txt'))
    joint=tfm.Transform(tfm.ToGrayscale(probability=0),tfm.RandomHorizontalFlip(probability=.5))
    transform=tfm.Transform(tfm.ToTensorAndJitter(.2),tfm.RandomHorizontalFlip_Norm(probability=.5),tfm.Normalize(mean=cfg.DATA.MEAN,std=cfg.DATA.STD))
    process=processing.STARKProcessing(search_area_factor=settings.search_area_factor,output_sz=settings.output_sz,
               center_jitter_factor=settings.center_jitter_factor,scale_jitter_factor=settings.scale_jitter_factor,
               mode='sequence',transform=transform,joint_transform=joint,settings=settings)
    manifest=json.loads((ROOT/'experiments/early_gate_closed_loop_20261026/p1_sot_v3/data_manifest.json').read_text())
    sampler=PairedStratifiedSampler(manifest,42,datasets=[dataset],p_datasets=[1],samples_per_epoch=16,
             max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,num_search_frames=settings.num_search,num_template_frames=settings.num_template,
             processing=process,frame_sample_mode=cfg.DATA.SAMPLER_MODE)
    data,metadata=paired_collate([sampler[i] for i in range(16)])
    gpu=TensorDict({k:TensorDict(v).to('cuda') for k,v in data.items()})
    gpu['epoch']=1;gpu['settings']=settings
    actor=TBSITrackActor(net,{'giou':giou_loss,'l1':torch.nn.functional.l1_loss,'focal':FocalLoss()},
                         {'giou':cfg.TRAIN.GIOU_WEIGHT,'l1':cfg.TRAIN.L1_WEIGHT,'focal':1.},settings,cfg)
    net.eval()
    with torch.no_grad():
        on_loss,_=actor(gpu)
        for l in net.backbone.tbsi_layers:l.v2_template_fusion.value_direction_enabled=False
        off_loss,_=actor(gpu)
        for l in net.backbone.tbsi_layers:l.v2_template_fusion.value_direction_enabled=True
    require(torch.equal(on_loss,off_loss),'Real batch zero-init task loss parity differs')
    optimizer,scheduler=get_optimizer_scheduler(net,cfg)
    net.train();scaler=torch.cuda.amp.GradScaler(init_scale=128.)
    rows=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():loss,stats=actor(gpu)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grads={n:float(p.grad.norm()) for n,p in net.named_parameters() if '.value_direction.' in n and p.grad is not None}
        require(all(torch.isfinite(p.grad).all() for n,p in net.named_parameters() if '.value_direction.' in n and p.grad is not None),'Nonfinite operator gradients')
        require(all(v>0 for n,v in grads.items() if n.endswith('.up')),'No up-transform task gradient')
        if step==1:require(all(v>0 for v in grads.values()),'Source/receiver/early condition not reached after up update')
        torch.nn.utils.clip_grad_norm_(net.parameters(),cfg.TRAIN.GRAD_CLIP_NORM)
        scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        require(scaler.get_scale()>=scale,'Smoke step overflowed')
        rows.append({'step':step+1,'loss':float(loss),'iou':float(stats['IoU']),'new_parameter_gradient_norms':grads})
    return {'shared_state_sha256':initial_hash,'matches_existing_control':True,
            'real_gt_samples':metadata,'worker_count':0,'worker_difference_reason':'single-batch diagnostic only; no throughput or training comparison',
            'zero_init_task_loss_exact':True,'amp_successful_steps':2,'rows':rows,
            'no_utility_or_fit_claim':True,'no_smoke_state_reused_for_training':True}


if __name__=='__main__':
    result={'algebra':algebra(),'real_batch':real_batch(),'environment':environment()}
    (OUT/'contract_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='real_batch'},indent=2))
