"""Train-only target-response direction teaching for receiver message residuals.

Only the existing residual predictor receives auxiliary gradients. The teacher
is a detached first-order gradient, not an additional inference input or model.
"""
import math
import torch
from torch.nn import functional as F
from lib.models.layers.receiver_conditioned_message_residual import ReceiverConditionedMessageResidual
from lib.utils.heapmap_utils import generate_heatmap


class ReceiverResidualTargetDirectionLoss:
    def __init__(self, model, margin=0.1):
        if margin < 0:
            raise ValueError('margin must be nonnegative')
        self.margin=float(margin)
        self.modules=[m for m in model.modules() if isinstance(m,ReceiverConditionedMessageResidual)]
        if len(self.modules)!=6:
            raise ValueError('Expected exactly six active receiver residual modules')
        self.captured={};self.last_loss=None
        self.handles=[m.register_forward_hook(self._capture(i)) for i,m in enumerate(self.modules)]

    def _capture(self,index):
        def hook(module,args,out):
            if torch.is_grad_enabled():
                self.captured[index]=(args[0].detach(),args[1].detach(),out)
            return out
        return hook

    def compute(self,score_map,gt_bbox):
        if len(self.captured)!=6:
            raise RuntimeError('Training forward did not capture all six residual outputs')
        records=[self.captured[i] for i in range(6)];self.captured={}
        b,_,h,w=score_map.shape
        boxes=gt_bbox.detach().float()
        # Use the original focal target's exact positive cell, not a new coordinate convention.
        positive=generate_heatmap(boxes[None],h*16,16)[-1].to(score_map.device).eq(1)
        yy,xx=torch.meshgrid((torch.arange(h,device=boxes.device)+0.5)/h,
                             (torch.arange(w,device=boxes.device)+0.5)/w,indexing='ij')
        x,y,bw,bh=boxes.unbind(-1);cx=x+bw/2;cy=y+bh/2
        dilated=((xx[None]>=x[:,None,None]-1/w)&(xx[None]<=x[:,None,None]+bw[:,None,None]+1/w)&
                 (yy[None]>=y[:,None,None]-1/h)&(yy[None]<=y[:,None,None]+bh[:,None,None]+1/h))
        negative=~dilated & ~positive
        valid=(torch.isfinite(boxes).all(-1)&(bw>0)&(bh>0)&(cx>=0)&(cx<1)&(cy>=0)&(cy<1)&
               positive.flatten(1).any(-1)&negative.flatten(1).any(-1))
        logscore=score_map[:,0].float().clamp_min(1e-6).log()
        pos=logscore.masked_fill(~positive,-torch.inf).flatten(1).amax(-1)
        neg=logscore.masked_fill(~negative,-torch.inf).flatten(1).amax(-1)
        q=torch.where(valid,pos-neg,torch.zeros_like(pos))
        if not valid.any():
            # Keep a valid zero-gradient path to only residual parameters.
            zero=sum(m.readout.weight.sum()*0 for m in self.modules)
            self.last_loss=zero
            return zero,{'ReceiverDirection/valid_fraction':0.,'ReceiverDirection/teacher_nonzero_fraction':0.,'Loss/receiver_direction_raw':0.}
        teacher_loss=F.softplus(self.margin-q[valid]).mean()
        outputs=tuple(r[2] for r in records)
        # A positive common scale cancels in the normalized teacher direction.
        # Bound the gradient entering an AMP score tensor, including hard low-score GTs.
        positive_score=score_map[:,0].detach().float().masked_fill(~(positive & valid[:,None,None]),torch.inf).flatten(1).amin(-1)[valid]
        negative_score=score_map[:,0].detach().float().masked_fill(~negative,-torch.inf).flatten(1).amax(-1)[valid]
        minimum_active_score=torch.minimum(positive_score,negative_score).clamp_min(1e-6).amin()
        teacher_scale=min(1024.,float(minimum_active_score)*int(valid.sum())*4.)
        gradients=torch.autograd.grad(teacher_loss*teacher_scale,outputs,retain_graph=True,create_graph=False,allow_unused=False)
        losses=[];cosines=[];norms=[];nonzero=[]
        with torch.cuda.amp.autocast(enabled=False):
            for module,(receiver,raw,_),gradient in zip(self.modules,records,gradients):
                g=-gradient.detach().float()/teacher_scale
                if not torch.isfinite(g).all():raise ValueError('Nonfinite residual-direction teacher')
                gn=g.flatten(1).norm(dim=1)
                teacher=g/gn.clamp_min(1e-20).view(b,1,1,1)
                _,heads,tokens,dh=raw.shape
                message=raw.float().transpose(1,2).reshape(b,tokens,heads*dh)
                # Detached inputs ensure the auxiliary cannot improve the head/backbone directly.
                latent=module.receiver_proj(module.receiver_norm(receiver.float()))*module.message_proj(module.message_norm(message))
                scale=message.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
                delta=module.max_delta*torch.tanh(module.readout(F.gelu(latent)))*scale
                delta=delta.reshape(b,tokens,heads,dh).transpose(1,2)
                rms=raw.float().square().flatten(1).mean(-1).sqrt().clamp_min(1e-6)
                dot=(delta*teacher).flatten(1).sum(-1)
                usable=valid & (gn>1e-20)
                denom=rms*math.sqrt(delta[0].numel())
                if usable.any():
                    losses.append(-(dot[usable]/denom[usable]).mean())
                    cosines.append((dot[usable]/delta.detach().flatten(1).norm(dim=1)[usable].clamp_min(1e-20)).detach().mean())
                norms.append(gn.detach().mean());nonzero.append(usable.float().mean())
        loss=torch.stack(losses).mean() if losses else sum(m.readout.weight.sum()*0 for m in self.modules)
        self.last_loss=loss
        return loss,{'Loss/receiver_direction_raw':float(loss.detach()),
            'ReceiverDirection/valid_fraction':float(valid.float().mean()),
            'ReceiverDirection/teacher_nonzero_fraction':float(torch.stack(nonzero).mean()),
            'ReceiverDirection/teacher_gradient_norm':float(torch.stack(norms).mean()),
            'ReceiverDirection/teacher_backward_scale':teacher_scale,
            'ReceiverDirection/current_teacher_cosine':float(torch.stack(cosines).mean()) if cosines else 0.,
            'ReceiverDirection/raw_target_margin':float(q[valid].detach().mean()),
            'ReceiverDirection/teacher_margin_loss':float(teacher_loss.detach())}
