"""Ordinary robust pose objectives, shared by all methods; no novelty claim."""
import torch
from torch.nn import functional as F

def robust_pose_loss(output,target):
    fused=(output['fused']-target).norm(dim=-1).mean()
    terms=[]
    for name in ('video','radar'):
        valid=output[name+'_valid']
        error=(output[name]-target).norm(dim=-1).mean(-1)
        terms.append((error*valid).sum()/valid.sum().clamp_min(1))
    return fused+.25*sum(terms)

def displacement_loss(prediction,target,valid):
    error=F.smooth_l1_loss(prediction,target,reduction='none',beta=.05).mean((-1,-2))
    return (error*valid).sum()/valid.sum().clamp_min(1)
