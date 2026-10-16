"""Larger development experiments with common frozen stems for temporal heads.

Data folds, seeds, budget and methods are saved before fitting. All checkpoints
are last-epoch. Existing complete models may be verified/reused, never silently
overwritten. A two-subject pilot only checks runtime and has no paper standing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
from fusion_models import FusionModel,fusion_loss
from run_fusion_experiments_gpu import normalize_inputs,augmented_batch
from train_mri_point_baselines import point_normalization
from temporal_anchor_models import CONTEXT,METHODS,context_indices,make_model
from analyze_fusion_trials import independent_metrics,strict_load_weights
from robust_action_losses import robust_pose_loss,displacement_loss
from awc_training_rule import AWCConcat,empirical_fisher,shapley_two,awc_regularizer,encoder_parameters,mpjpe

CONDS=['clean','noisy_dark','occlusion']
SCORE=CONDS+['video_missing','radar_missing']
FOLDS=[[5,6,19,20],[1,2,4,7],[8,9,11,13],[14,15,16,17]]
DEV=sorted(sum(FOLDS,[]))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def state_save(model,path):np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})

def collect(cache,subjects):
    arrays=[]
    for s in subjects:
        receipt=json.loads((cache/f'subject{s}.json').read_text())
        path=cache/f'subject{s}.npz';assert sha(path)==receipt['cache_sha256']
        with np.load(path,allow_pickle=False) as f:
            keep=f['target_finite']
            d={k:f[k][keep].copy() for k in ['frame_ids','timestamps_ms','action_ids','window_ids','target_mm','screened','points','point_mask']}
            for c in CONDS:d[c]=f[c+'_features'][keep].copy();d[c+'_poses']=f[c+'_poses'][keep].copy()
            d['subject_ids']=np.full(keep.sum(),s,dtype=np.int64)
            arrays.append(d)
    return {k:np.concatenate([r[k] for r in arrays]) for k in arrays[0]}

def make_inputs(row,stats):
    result={c:normalize_inputs(row[c],row['points'],row['point_mask'],stats) for c in CONDS}
    c=result['clean']
    result['video_missing']=(torch.zeros_like(c[0]),c[1],c[2],torch.zeros_like(c[3]))
    result['radar_missing']=(c[0],torch.zeros_like(c[1]),torch.zeros_like(c[2]),c[3])
    return result

@torch.no_grad()
def encode(model,inputs,stats):
    data={k:[] for k in ['vf','rf','base','video','radar','confidence','vv','rv']}
    captured={}
    hooks=[model.rgb_encoder.register_forward_hook(lambda m,a,v:captured.update(vf=v)),
           model.point_encoder.register_forward_hook(lambda m,a,v:captured.update(rf=v[1]))]
    try:
        for start in range(0,len(inputs[0]),128):
            batch=tuple(v[start:start+128].cuda() for v in inputs);out=model(*batch)
            values={'vf':captured['vf']*out['video_valid'][:,None], 'rf':captured['rf'],
                    'base':out['fused'],'video':out['video'],'radar':out['radar'],
                    'confidence':batch[3],'vv':out['video_valid'],'rv':out['radar_valid']}
            for k,v in values.items():data[k].append(v.cpu())
    finally:
        for h in hooks:h.remove()
    return {k:torch.cat(v) for k,v in data.items()}

def index_context(encoded,row):
    idx,valid=context_indices(row['subject_ids'],row['window_ids'],row['frame_ids'])
    stamps=row['timestamps_ms'][idx].astype(np.float64)/1000
    dt=np.zeros_like(stamps);dt[:,1:]=np.maximum(0,np.diff(stamps,axis=1))
    pair=valid.copy();pair[:,0]=False;pair[:,1:]&=valid[:,:-1]
    dt=np.where(pair,dt,0).astype(np.float32)
    if (dt>.5).any():raise ValueError('Selected within-window timestamp gap > 0.5 seconds')
    return {'encoded':encoded,'idx':torch.tensor(idx),'valid':torch.tensor(valid),'dt':torch.tensor(dt)}

def temporal_batch(context,indices,drop=None):
    idx=context['idx'][indices]
    result={k:v[idx].clone() for k,v in context['encoded'].items()}
    result['history_valid']=context['valid'][indices].clone();result['dt']=context['dt'][indices].clone()
    # Optional whole-history missing routes are already separately encoded.
    return {k:v.cuda() for k,v in result.items()}

def metrics(pred,row):
    m=independent_metrics(pred,row['target_mm'],row['subject_ids'])
    root=pred.reshape(-1,17,3);root=root-root[:,[11,12]].mean(1)[:,None]
    m['hip_recentered_MPJPE_mm']=independent_metrics(root.reshape(-1,51),row['target_mm'],row['subject_ids'])['MPJPE_mm']
    good=row['screened']
    m['screened_frames']=int(good.sum())
    m['screened_MPJPE_mm']=independent_metrics(pred[good],row['target_mm'][good],row['subject_ids'][good])['MPJPE_mm'] if good.any() else None
    m['per_action_MPJPE_mm']={str(a):float(np.linalg.norm((pred-row['target_mm']).reshape(-1,17,3),axis=-1)[row['action_ids']==a].mean()) for a in np.unique(row['action_ids'])}
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT/'artifacts/mri_action_windows_20261016')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--folds',nargs='+',type=int,default=[0])
    p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27]);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--temporal-epochs',type=int,default=80);p.add_argument('--pilot',action='store_true')
    p.add_argument('--temporal-methods',nargs='+',choices=METHODS,default=list(METHODS));args=p.parse_args()
    status=json.loads((args.cache/'status.json').read_text())
    if not args.pilot and status['status']!='complete':raise ValueError('Complete large cache required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    sources={n:sha(ROOT/'research'/n) for n in [Path(__file__).name,'temporal_anchor_models.py','fusion_models.py','run_fusion_experiments_gpu.py','awc_training_rule.py','robust_action_losses.py','run_action_window_research.py']}
    protocol={'schema':1,'cache_protocol_sha256':sha(args.cache/'protocol.json'),'folds':args.folds,'seeds':args.seeds,
        'epochs':args.epochs,'temporal_epochs':args.temporal_epochs,'batch_size':128,
        'base_methods':['rgb_single','radar_single','concat','concat_mse','concat_mpjpe','awc_concat'],'temporal_methods':args.temporal_methods,
        'framewise_optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001},
        'temporal_optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001},
        'base_loss':'RGB/radar/concat: MPJPE + .25 valid branch MPJPE; concat_mse retains original MSE control; concat_mpjpe and AWC have MPJPE task only',
        'target_template':'train-only coordinate median; stored under legacy numeric target_mean_mm key',
        'data_preflight_sha256':sha(ROOT/'artifacts/mri_action_window_preflight_20261016.json'),
        'environment':{'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'fp32':'IEEE; no mixed precision'},
        'paper_rule_baseline':'AWC Eq1/4/6/7/8 reconstruction on local two-modality architecture; MPJPE task loss, full-clean-train per-example Fisher at epoch start for first20 epochs, alphaSuperior20000/inferior10000; concat_mpjpe identical task-loss control',
        'temporal_loss':'MPJPE; anchor_transport adds .25 adjacent displacement Huber(beta=.05m) over radar-valid pairs',
        'stage2_inputs':'same frozen concat stems and branch predictions; no end-to-end fine-tuning',
        'context':CONTEXT,'checkpoint':'last fixed epoch, no validation checkpoint choice',
        'history_modality_drop':'whole nine-frame context; standard missing conditions reencoded with the modality removed',
        'data_augmentation':'base: uniform actual pixel conditions, 10% RGB drop/10% radar drop when hips originally available, sparse point augmentation; temporal: uniform among 5 conditions',
        'pilot':args.pilot,'source_sha256':sources,'test_subjects_used':[]}
    if args.output.exists():
        if json.loads((args.output/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol differs')
    else:
        args.output.mkdir(parents=True);save(args.output/'protocol.json',protocol)
        snap=args.output/'source_snapshot';snap.mkdir()
        for n in sources:(snap/n).write_bytes((ROOT/'research'/n).read_bytes())
    summary={'status':'running','pilot':args.pilot,'runs':[],'test_subjects_used':[]}
    save(args.output/'summary.json',summary)
    for fold in args.folds:
        val=[2] if args.pilot else FOLDS[fold];train=[1] if args.pilot else [s for s in DEV if s not in val]
        assert not(set(train+val)&{3,10,12,18}) and not(set(train)&set(val))
        tr,va=collect(args.cache,train),collect(args.cache,val)
        fdir=args.output/f'fold{fold}';fdir.mkdir(exist_ok=True)
        pm,ps=point_normalization(tr['points'],tr['point_mask'])
        stats={'point_mean':pm,'point_std':ps,'rgb_mean':tr['clean'].mean(0),
            'rgb_std':np.maximum(tr['clean'].std(0),1e-6),'target_mean_mm':np.median(tr['target_mm'],axis=0)}
        np.savez_compressed(fdir/'normalization.npz',**stats)
        tin,vin=make_inputs(tr,stats),make_inputs(va,stats)
        y=torch.tensor((tr['target_mm']-stats['target_mean_mm']).reshape(-1,17,3)/1000,dtype=torch.float32,device='cuda')
        reference=torch.tensor(np.stack([(tr[c+'_poses'][:,11,2]>=.3)&(tr[c+'_poses'][:,12,2]>=.3) for c in CONDS]))
        foldmeta={'train_subjects':train,'validation_subjects':val,'train_frames':len(y),'validation_frames':len(va['target_mm']),
            'cache_sha256':{str(s):sha(args.cache/f'subject{s}.npz') for s in train+val}}
        save(fdir/'data.json',foldmeta)
        for seed in args.seeds:
            sdir=fdir/f'seed{seed}';sdir.mkdir(exist_ok=True);row={'fold':fold,'seed':seed,'base':{},'temporal':{}}
            for method in protocol['base_methods']:
                dest=sdir/method;dest.mkdir(exist_ok=True);done=dest/'training.json'
                torch.manual_seed(seed);model=AWCConcat() if method in ('concat_mpjpe','awc_concat') else FusionModel('concat' if method=='concat_mse' else method)
                model.target_template_m.copy_(torch.tensor(stats['target_mean_mm'].reshape(17,3)/1000,dtype=torch.float32));model.cuda()
                if done.exists():
                    entry=json.loads(done.read_text());assert sha(dest/'weights.npz')==entry['weights_sha256']
                    assert sha(dest/'predictions.npz')==entry['predictions_sha256'];strict_load_weights(model.cpu(),dest/'weights.npz');model.cuda()
                    print(json.dumps({'event':'reuse_complete','fold':fold,'seed':seed,'method':method}),flush=True)
                else:
                    if (dest/'weights.npz').exists():raise ValueError('Incomplete model state requires explicit recovery')
                    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                    order=torch.Generator().manual_seed(seed);aug=torch.Generator().manual_seed(seed+10000)
                    begin=time.perf_counter();history=[]
                    for epoch in range(1,args.epochs+1):
                        model.train();losses=[]
                        if method=='awc_concat' and epoch<=20:
                            anchor={n:v.detach().clone() for n,v in encoder_parameters(model).items()}
                            fisher=empirical_fisher(model,tin['clean'],y)
                            np.savez_compressed(dest/f'fisher_epoch{epoch}.npz',**{n:v.cpu().numpy() for n,v in fisher.items()})
                        for ix in torch.randperm(len(y),generator=order).split(128):
                            batch=tuple(v.cuda() for v in augmented_batch([tin[c] for c in CONDS],ix,aug,reference))
                            opt.zero_grad(set_to_none=True);out=model(*batch)
                            if method in ('concat_mpjpe','awc_concat'):
                                loss=mpjpe(out['fused'],y[ix])
                                if method=='awc_concat' and epoch<=20:
                                    scores=shapley_two(model,batch,y[ix]);loss=loss+awc_regularizer(model,anchor,fisher,scores)
                            elif method=='concat_mse':loss,_=fusion_loss(out,y[ix],'concat')
                            else:loss=robust_pose_loss(out,y[ix])
                            if not torch.isfinite(loss):raise ValueError('Nonfinite base loss')
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.item())
                        if epoch==1 or epoch%20==0 or epoch==args.epochs:
                            h={'epoch':epoch,'loss':float(np.mean(losses))};history.append(h)
                            print(json.dumps({'stage':'base','fold':fold,'seed':seed,'method':method,**h,'seconds':time.perf_counter()-begin}),flush=True)
                    model.eval();saved={k:va[k] for k in ['target_mm','subject_ids','frame_ids','action_ids','window_ids','screened']};result={}
                    with torch.no_grad():
                        for c,inp in vin.items():
                            pred=torch.cat([model(*(v[i:i+128].cuda() for v in inp))['fused'].cpu() for i in range(0,len(va['target_mm']),128)]).numpy().reshape(-1,51)*1000+stats['target_mean_mm']
                            saved[c+'_prediction_mm']=pred;result[c]=metrics(pred,va)
                    state_save(model,dest/'weights.npz');np.savez_compressed(dest/'predictions.npz',**saved)
                    entry={'metrics':result,'history':history,'seconds':time.perf_counter()-begin,'parameters':sum(p.numel() for p in model.parameters()),
                        'weights_sha256':sha(dest/'weights.npz'),'predictions_sha256':sha(dest/'predictions.npz')};save(done,entry)
                row['base'][method]=entry
                if method=='concat':base=model
                else:del model
                save(sdir/'training.json',row)
            base.eval();base.requires_grad_(False)
            train_ctx={c:index_context(encode(base,inp,stats),tr) for c,inp in tin.items()}
            val_ctx={c:index_context(encode(base,inp,stats),va) for c,inp in vin.items()}
            features={}
            for part,contexts in [('train',train_ctx),('validation',val_ctx)]:
                for c,ctx in contexts.items():
                    for k,v in ctx['encoded'].items():features[f'{part}_{c}_{k}']=v.numpy()
            features.update({f'train_{k}':tr[k] for k in ['target_mm','subject_ids','frame_ids','timestamps_ms','window_ids','action_ids','screened']})
            features.update({f'validation_{k}':va[k] for k in ['target_mm','subject_ids','frame_ids','timestamps_ms','window_ids','action_ids','screened']})
            np.savez_compressed(sdir/'frozen_features.npz',**features)
            for method in args.temporal_methods:
                dest=sdir/method;dest.mkdir(exist_ok=True);done=dest/'training.json'
                if done.exists():
                    entry=json.loads(done.read_text());assert sha(dest/'weights.npz')==entry['weights_sha256'];assert sha(dest/'predictions.npz')==entry['predictions_sha256'];row['temporal'][method]=entry;continue
                torch.manual_seed(seed+30000);model=make_model(method).cuda()
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                order=torch.Generator().manual_seed(seed);aug=torch.Generator().manual_seed(seed+10000)
                begin=time.perf_counter();history=[]
                for epoch in range(1,args.temporal_epochs+1):
                    model.train();losses=[];motion_losses=[]
                    for ix in torch.randperm(len(y),generator=order).split(128):
                        choice=int(torch.randint(len(SCORE),(1,),generator=aug));ctx=train_ctx[SCORE[choice]]
                        batch=temporal_batch(ctx,ix);opt.zero_grad(set_to_none=True);out=model(batch)
                        loss=(out['prediction']-y[ix]).norm(dim=-1).mean();ml=loss*0
                        if method=='anchor_transport':
                            yt=y[ctx['idx'][ix]];delta=torch.zeros_like(yt);delta[:,1:]=yt[:,1:]-yt[:,:-1]
                            valid=out['pair_valid']
                            ml=displacement_loss(out['displacement'],delta,valid);loss=loss+.25*ml
                        if not torch.isfinite(loss):raise ValueError('Nonfinite temporal loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.item());motion_losses.append(ml.item())
                    if epoch==1 or epoch%20==0 or epoch==args.temporal_epochs:
                        h={'epoch':epoch,'loss':float(np.mean(losses)),'motion_Huber_m':float(np.mean(motion_losses))};history.append(h)
                        print(json.dumps({'stage':'temporal','fold':fold,'seed':seed,'method':method,**h,'seconds':time.perf_counter()-begin}),flush=True)
                model.eval();result={};saved={k:va[k] for k in ['target_mm','subject_ids','frame_ids','action_ids','window_ids','screened']}
                with torch.no_grad():
                    for c,ctx in val_ctx.items():
                        outputs=[model(temporal_batch(ctx,torch.arange(i,min(i+128,len(va['target_mm']))))) for i in range(0,len(va['target_mm']),128)]
                        pred=torch.cat([v['prediction'].cpu() for v in outputs]).numpy().reshape(-1,51)*1000+stats['target_mean_mm']
                        saved[c+'_prediction_mm']=pred;result[c]=metrics(pred,va)
                        for key in ['gate','eligible','last_anchor','age_seconds','transported','velocity','displacement','pair_valid']:
                            if key in outputs[0]:saved[c+'_'+key]=torch.cat([v[key].cpu() for v in outputs]).numpy()
                state_save(model,dest/'weights.npz');np.savez_compressed(dest/'predictions.npz',**saved)
                entry={'metrics':result,'history':history,'seconds':time.perf_counter()-begin,'parameters':sum(p.numel() for p in model.parameters()),
                       'base_weights_sha256':sha(sdir/'concat/weights.npz'),'frozen_features_sha256':sha(sdir/'frozen_features.npz'),
                       'weights_sha256':sha(dest/'weights.npz'),'predictions_sha256':sha(dest/'predictions.npz')};save(done,entry)
                row['temporal'][method]=entry;save(sdir/'training.json',row)
                print(json.dumps({'event':'temporal_complete','fold':fold,'seed':seed,'method':method,'clean':result['clean']['MPJPE_mm'],
                    'S':float(np.mean([result[c]['MPJPE_mm'] for c in SCORE]))}),flush=True)
            summary['runs'].append(row);save(args.output/'summary.json',summary)
            del base,model;torch.cuda.empty_cache()
    summary['status']='complete';save(args.output/'summary.json',summary)

if __name__=='__main__':main()
