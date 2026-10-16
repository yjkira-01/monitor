"""Experimental joint-anchor transport, with same-context temporal controls.

Forward inputs contain observed features and frozen predictions only. Labels
are passed separately to training losses. No method novelty is assumed here.
"""
import numpy as np
import torch
from torch import nn

CONTEXT=9
METHODS=('gru','temporal_mlp','anchor_hold','anchor_transport','anchor_no_motion_loss')

def context_indices(subjects,windows,frames,length=CONTEXT):
    subjects,windows,frames=map(np.asarray,(subjects,windows,frames))
    indices=np.empty((len(frames),length),np.int64)
    valid=np.zeros_like(indices,dtype=bool)
    for i in range(len(frames)):
        for k,offset in enumerate(range(1-length,1)):
            j=i+offset
            good=(j>=0 and subjects[j]==subjects[i] and windows[j]==windows[i] and frames[j]==frames[i]+offset)
            indices[i,k]=j if good else i
            valid[i,k]=good
    return indices,valid

def feature_vector(batch):
    """Identical available inputs for GRU/MLP controls; time is relative only."""
    return torch.cat([batch['vf'],batch['rf'],batch['base'].flatten(-2),
        batch['video'].flatten(-2),batch['radar'].flatten(-2),batch['confidence'],
        batch['vv'][...,None].float(),batch['rv'][...,None].float(),batch['dt'][...,None]],-1)

class TemporalControl(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='gru':
            self.encoder=nn.GRU(429,128,2,batch_first=True)
            self.head=nn.Linear(128,51)
        elif method=='temporal_mlp':
            self.encoder=nn.Sequential(nn.Linear(CONTEXT*429,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
            self.head=nn.Linear(128,51)
        else:raise ValueError(method)
        nn.init.normal_(self.head.weight,std=.001);nn.init.zeros_(self.head.bias)
    def forward(self,batch):
        x=feature_vector(batch)*batch['history_valid'][...,None]
        if self.method=='gru':h=self.encoder(x)[0][:,-1]
        else:h=self.encoder(x.flatten(1))
        residual=.3*torch.tanh(self.head(h).reshape(-1,17,3))
        observed=((batch['vv']|batch['rv'])&batch['history_valid']).any(1)
        output=batch['base'][:,-1]+residual*observed[:,None,None]
        return {'prediction':output}

class AnchorTransport(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method not in METHODS[2:]:raise ValueError(method)
        self.motion=nn.Sequential(nn.Linear(257,128),nn.ReLU(),nn.Linear(128,51))
        self.gate=nn.Sequential(nn.Linear(13,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.normal_(self.motion[-1].weight,std=.001);nn.init.zeros_(self.motion[-1].bias)
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2.)

    def forward(self,batch):
        rf,dt=batch['rf'],batch['dt']
        previous=torch.cat([rf[:,:1],rf[:,:-1]],1)
        velocity=self.motion(torch.cat([previous,rf,dt[...,None]],-1)).reshape(*rf.shape[:2],17,3)
        pair_valid=batch['rv'] & torch.cat([torch.zeros_like(batch['rv'][:,:1]),batch['rv'][:,:-1]],1)
        pair_valid=pair_valid&batch['history_valid']&torch.cat([torch.zeros_like(batch['history_valid'][:,:1]),batch['history_valid'][:,:-1]],1)
        displacement=velocity*dt[...,None,None]*pair_valid[...,None,None]
        if self.method=='anchor_hold':displacement=displacement*0.
        cumulative=displacement.cumsum(1)
        conf=batch['confidence'];visible=(conf>=.3)&batch['history_valid'][...,None]
        positions=torch.arange(conf.shape[1],device=conf.device)[None,:,None].expand_as(conf)
        last=positions.masked_fill(~visible,-1).max(1).values
        have_anchor=last>=0;safe=last.clamp_min(0)
        gather=safe[:,None,:,None].expand(-1,1,17,3)
        anchor=torch.gather(batch['video'],1,gather).squeeze(1)
        anchor_motion=torch.gather(cumulative,1,gather).squeeze(1)
        transported=anchor+cumulative[:,-1]-anchor_motion
        anchor_conf=torch.gather(conf,1,safe[:,None,:]).squeeze(1)
        times=dt.cumsum(1)
        age=times[:,-1,None]-torch.gather(times,1,safe)
        base=batch['base'][:,-1]
        evidence=torch.cat([base,transported,(transported-base).abs(),conf[:,-1,:,None],
            anchor_conf[...,None],age[...,None],batch['rv'][:,-1,None,None].float().expand(-1,17,1)],-1)
        weight=self.gate(evidence).sigmoid().squeeze(-1)
        eligible=have_anchor&(conf[:,-1]<.3)
        weight=weight*eligible
        prediction=base+weight[...,None]*(transported-base)
        return {'prediction':prediction,'displacement':displacement,'velocity':velocity,
                'pair_valid':pair_valid,'gate':weight,'eligible':eligible,
                'transported':transported,'last_anchor':last,'age_seconds':age}

def make_model(method):
    return TemporalControl(method) if method in METHODS[:2] else AnchorTransport(method)
