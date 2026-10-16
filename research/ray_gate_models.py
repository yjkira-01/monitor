"""Small geometric correction selectors and a same-feature free residual control."""
import torch
from torch import nn

class RaySelector(nn.Module):
    def __init__(self,free_residual=False):
        super().__init__();self.free_residual=free_residual
        self.joint=nn.Parameter(torch.randn(17,16)*.02)
        self.network=nn.Sequential(nn.Linear(31,32),nn.ReLU(),nn.Linear(32,3 if free_residual else 1))
        nn.init.normal_(self.network[-1].weight,std=.001)
        nn.init.constant_(self.network[-1].bias,0. if free_residual else -2.)

    def forward(self,base,candidate,features,eligible):
        identity=self.joint[None].expand(len(base),-1,-1)
        value=self.network(torch.cat((features,identity),-1))
        if self.free_residual:
            correction=.1*torch.tanh(value)
            return base+correction*eligible[...,None]
        weight=value.sigmoid()*eligible[...,None]
        return base+weight*(candidate-base)
