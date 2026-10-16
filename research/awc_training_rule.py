"""Two-modality reconstruction of the AWC paper's training rule (not its model).

Paper: Qi et al., arXiv:2501.05264v5, CVPR2026. Eq1/4/6/7/8 and Algorithm1.
Local mRI architecture and missing-route policies are explicitly adaptations.
Fisher uses the average of squared *individual* MPJPE gradients at epoch start.
"""
import torch
from torch import nn
from torch.func import functional_call,grad,vmap
from fusion_models import FusionModel

class AWCConcat(FusionModel):
    def __init__(self):super().__init__('concat')
    def forward(self,rgb,points,mask,confidence):
        # Same numerical model as concat, without tensor-dependent validation
        # branches that are unsupported by torch.func.vmap. Validate externally.
        vv,rv=(confidence>=.3).any(1),mask.any(1)
        vf=self.rgb_encoder(rgb)*vv[:,None];_,rf=self.point_encoder(points,mask)
        video=self.video_head(vf).reshape(-1,17,3);radar=self.radar_head(rf).reshape(-1,17,3)
        fused=self.concat_head(torch.cat((vf,rf),-1)).reshape(-1,17,3)
        fused=torch.where(~rv[:,None,None],video,fused);fused=torch.where(~vv[:,None,None],radar,fused)
        return {'video':video,'radar':radar,'fused':fused,'video_valid':vv,'radar_valid':rv,'both':vv&rv}

def mpjpe(pred,target):return (pred-target).norm(dim=-1).mean()

def pearson_sum(y,pred,epsilon=1e-12):
    y,pred=y.flatten(1),pred.flatten(1)
    a=y-y.mean(0);b=pred-pred.mean(0)
    denom=(a.square().sum(0)*b.square().sum(0)).sqrt()
    return ((a*b).sum(0)/denom.clamp_min(epsilon)).sum()

@torch.no_grad()
def shapley_two(model,inputs,target):
    rgb,points,mask,conf=inputs
    vf=model.rgb_encoder(rgb)*(conf>=.3).any(1)[:,None];_,rf=model.point_encoder(points,mask)
    values={}
    for a,b in [(False,False),(True,False),(False,True),(True,True)]:
        pred=model.concat_head(torch.cat([vf if a else torch.zeros_like(vf),rf if b else torch.zeros_like(rf)],-1)).reshape(-1,17,3)
        values[a,b]=pearson_sum(target,pred)
    rgbscore=.5*(values[True,False]-values[False,False]+values[True,True]-values[False,True])
    radscore=.5*(values[False,True]-values[False,False]+values[True,True]-values[True,False])
    return torch.stack([rgbscore,radscore])

def encoder_parameters(model):
    return {n:p for n,p in model.named_parameters() if n.startswith(('rgb_encoder.','point_encoder.'))}

def empirical_fisher(model,inputs,target,batch_size=32):
    """Exact finite-sample Eq8, vectorized individual gradients, no subsampling."""
    params=dict(model.named_parameters());buffers=dict(model.named_buffers())
    encoder=encoder_parameters(model);result={n:torch.zeros_like(p) for n,p in encoder.items()}
    def individual(parameters,buffers,rgb,points,mask,conf,y):
        out=functional_call(model,(parameters,buffers),(rgb[None],points[None],mask[None],conf[None]))
        return mpjpe(out['fused'],y[None])
    gradient=vmap(grad(individual),in_dims=(None,None,0,0,0,0,0))
    for start in range(0,len(target),batch_size):
        batch=[x[start:start+batch_size].to(target.device) for x in inputs]
        grads=gradient(params,buffers,*batch,target[start:start+batch_size])
        for n in result:result[n].add_(grads[n].detach().square().sum(0))
    return {n:p/len(target) for n,p in result.items()}

def awc_regularizer(model,anchor,fisher,scores,alpha_superior=20000.,alpha_inferior=10000.):
    # With two distinct scores, two-cluster K-means is exactly high vs low.
    alpha=torch.where(scores>scores.min(),scores.new_tensor(alpha_superior),scores.new_tensor(alpha_inferior))
    result=scores.new_zeros(())
    for n,p in encoder_parameters(model).items():
        group=0 if n.startswith('rgb_encoder.') else 1
        result=result+.5*alpha[group]*(fisher[n]*(p-anchor[n]).square()).sum()
    return result
