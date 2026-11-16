#!/usr/bin/env python3
"""Targeted invariants and disabled-path parity against the before snapshot."""
import importlib.util
import io
import json
import sys
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from run_dual_message_p1 import setup, write_json, config
from lib.models.layers.cycle_difference_message_repair import CycleDifferenceMessageRepair
from lib.models.layers.cycle_laplacian_message_repair import CycleLaplacianMessageRepair, _box_coverage
from lib.models.layers.tbsi_layer import V2TemplateFusion
from lib.models.tbsi_track import build_tbsi_track


def main():
    setup(42)
    out=ROOT/'output/experiments/MSHA-MR-CYCLE-DIFFERENCE-QUICK-S42-20261115'
    old_path=ROOT/'output/experiments/MSHA-MR-CBMR-AUDIT-S42-20261115/before/lib/models/layers/tbsi_layer.py'
    spec=importlib.util.spec_from_file_location('lib.models.layers.cbmr_snapshot',old_path)
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    kwargs=dict(dim=768,num_heads=12,qkv_bias=True,early_conditioned_cross_gate=True,
                early_gate_search_context_mode='cross',mass_redist_enable=True)
    before=old.V2TemplateFusion(**kwargs).cuda().eval()
    after=V2TemplateFusion(**kwargs).cuda().eval()
    after.load_state_dict(before.state_dict(),strict=True)
    inputs=[torch.randn(2,64,768,device='cuda') for _ in range(2)]
    early=tuple(torch.randn(2,n,768,device='cuda') for n in (64,64,256,256))
    boxes=torch.tensor([[[.25,.25,.5,.5]]*2]*2,device='cuda')
    with torch.no_grad():
        a=before(*inputs,early_context=early,template_boxes=boxes)
        b=after(*inputs,early_context=early,template_boxes=boxes)
    disabled=all(torch.equal(x,y) for x,y in zip(a,b))
    assert disabled
    # The historical enabled path must also be identical.
    before.cbmr_enabled=True; before.cbmr=CycleLaplacianMessageRepair()
    after.cbmr_enabled=True; after.cbmr=CycleLaplacianMessageRepair()
    with torch.no_grad():
        a=before(*inputs,early_context=early,template_boxes=boxes)
        b=after(*inputs,early_context=early,template_boxes=boxes)
    historical=all(torch.equal(x,y) for x,y in zip(a,b))
    assert historical
    repair=CycleDifferenceMessageRepair()
    msg=[torch.randn(2,3,64,8,requires_grad=True) for _ in range(2)]
    rel=[torch.rand(2,3,64,64,requires_grad=True) for _ in range(2)]
    constant=torch.randn(2,3,1,8).expand(-1,-1,64,-1)
    c=repair(constant,constant,rel[0],rel[1],boxes.cpu())
    constant_error=max(float((x-constant).abs().max()) for x in c)
    assert constant_error < 1e-6
    repaired=repair(*msg,*rel,boxes.cpu())
    foreground=_box_coverage(boxes.cpu()[:,0],8)>1e-6
    fg_identity=all(torch.equal(x.transpose(1,2)[foreground],y.transpose(1,2)[foreground])
                    for x,y in zip(msg,repaired))
    assert fg_identity
    sum(x.square().mean() for x in repaired).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in msg+rel)
    repair.strength=0.
    zero=repair(*msg,*rel,boxes.cpu())
    assert all(torch.equal(x,y) for x,y in zip(zero,msg))
    isolated=CycleDifferenceMessageRepair()(*msg,torch.zeros_like(rel[0]),torch.zeros_like(rel[1]),boxes.cpu())
    assert all(torch.equal(x,y) for x,y in zip(isolated,msg))
    cfg=config(ROOT/'experiments/tbsi_track/MR-CYCLE-DIFFERENCE-QUICK-S42-20261115-candidate.yaml')
    net=build_tbsi_track(cfg,training=False).cuda().eval()
    init=torch.load(str(ROOT/'output/experiments/MSHA-MR-CBMR-QUICK-S42-20261115/contracts/shared_init.pth.tar'),map_location='cpu')
    net.load_state_dict(init['state'],strict=True)
    assert len([m for m in net.modules() if isinstance(m,CycleDifferenceMessageRepair)])==3
    template=[torch.randn(1,3,128,128,device='cuda') for _ in range(2)]
    search=[torch.randn(1,3,256,256,device='cuda') for _ in range(2)]
    with torch.no_grad():
        result=net(template=template,search=search,template_boxes=boxes[:1])
    buffer=io.BytesIO()
    torch.save(net.state_dict(),buffer)
    buffer.seek(0)
    net.load_state_dict(torch.load(buffer,map_location='cuda'),strict=True)
    with torch.no_grad():
        repeat=net(template=template,search=search,template_boxes=boxes[:1])
    keys=['score_map','size_map','offset_map']
    assert all(torch.equal(result[k],repeat[k]) for k in keys)
    record=dict(disabled_before_after_bitwise=disabled,historical_enabled_before_after_bitwise=historical,
        constant_max_abs_error=constant_error,foreground_boundary_identity=fg_identity,
        zero_strength_bitwise=True,isolated_identity=True,message_and_relation_gradients_finite_nonzero=True,
        actual_model_difference_module_count=3,common_init_strict_loaded=True,
        full_model_save_load_bitwise=True,optimizer_steps=0,
        scope='Implementation contracts only; no efficacy claim')
    write_json(out/'contracts/implementation_checks.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
