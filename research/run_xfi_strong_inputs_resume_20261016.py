"""Matched-input X-Fi core experiment on the frozen 1536-frame discovery set."""
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
from fusion_models import FusionModel,fusion_loss
from xfi_strong_inputs import XFiStrongCore
from xfi_paper_baseline import source_manifest,COMMIT
from run_fusion_experiments_gpu import load_data,normalize_inputs,augmented_batch,sparse_mask
from train_mri_point_baselines import point_normalization
from analyze_fusion_trials import independent_metrics
from xfi_strong_resume_support import prepare_reuse,is_reused

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--learning-rates',type=float,nargs='+',default=[.0001,.001]);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise ValueError('New experiment directory required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    old=ROOT/'artifacts/fusion_method_research_20261013_expanded_centroid'
    cache=ROOT/'artifacts/fusion_rgb_expanded_20261013'
    conds=['clean','dark','noisy_dark','occlusion']
    split,rows=load_data(cache,conds,'visible_centroid')
    train,val=split['canonical_ids_train'],split['canonical_ids_validation']
    def stack(ids,key):return np.concatenate([rows[s][key] for s in ids])
    with np.load(old/'normalization.npz',allow_pickle=False) as f:stats={k:f[k] for k in f.files}
    assert not (set(train+val)&{3,10,12,18})
    for key,expected in [('rgb_mean',stack(train,'clean').mean(0)),('rgb_std',np.maximum(stack(train,'clean').std(0),1e-8)),('target_mean_mm',stack(train,'target').mean(0))]:
        np.testing.assert_array_equal(stats[key],expected)
    np.savez_compressed(args.output/'normalization.npz',**stats)
    training=[normalize_inputs(stack(train,c),stack(train,'points'),stack(train,'mask'),stats) for c in conds]
    reference=torch.tensor(np.stack([stack(train,'reference_available_'+c) for c in conds]))
    y=torch.tensor((stack(train,'target')-stats['target_mean_mm']).reshape(-1,17,3)/1000,dtype=torch.float32,device='cuda')
    target=stack(val,'target');subject_ids=np.concatenate([np.full(len(rows[s]['target']),s) for s in val])
    values={c:normalize_inputs(stack(val,c),stack(val,'points'),stack(val,'mask'),stats) for c in conds}
    c=values['clean'];values['video_missing']=(torch.zeros_like(c[0]),c[1],c[2],torch.zeros_like(c[3]))
    values['radar_missing']=(c[0],torch.zeros_like(c[1]),torch.zeros_like(c[2]),c[3])
    thin=torch.tensor(sparse_mask(stack(val,'mask')));values['radar_sparse']=(c[0],c[1].masked_fill(~thin[...,None],0),thin,c[3])
    summary={'status':'running','scope':'X-Fi fusion-core adaptation, shared strong predicted inputs, discovery only',
        'train_subjects':train,'validation_subjects':val,'final_test_subjects_used':[],
        'train_frames':len(y),'validation_frames':len(target),'seeds':args.seeds,
        'official_commit':COMMIT,'official_source_sha256':source_manifest(),
        'adaptations':['Original 128-d RGB/point stems and independent fallback heads from strong concat',
            'Each pooled 128-d modality maps to 512 dimensions plus 32 learned token positions',
            'Official X_Fusion core unchanged: 5 slots, RGB0/radar2, 512 dims, 32 tokens, 8 heads, depth4',
            'Original local MSE + .25 branch objectives and exact fallback policies; not official full X-Fi'],
        'training':{'epochs':args.epochs,'batch':64,'optimizer':'AdamW','learning_rates':args.learning_rates,'weight_decay':.0001,
            'conditions':conds,'checkpoint':'last epoch, no validation choice','augmentation':'unchanged augmented_batch from strong concat',
            'floating_point':'IEEE FP32','deterministic':'warn_only for official adaptive pool backward'},
        'old_summary_sha256':sha(old/'run_summary.json'),'cache_sha256':{str(s):sha(cache/f'subject{s}.npz') for s in train+val},
        'sources':{n:sha(ROOT/'research'/n) for n in [Path(__file__).name,'xfi_strong_inputs.py','xfi_paper_baseline.py','fusion_models.py','run_fusion_experiments_gpu.py','xfi_strong_resume_support.py','run_xfi_strong_inputs.py']},'runs':[]}
    snap=args.output/'source_snapshot';snap.mkdir()
    for n in summary['sources']:(snap/n).write_bytes((ROOT/'research'/n).read_bytes())
    summary['resume']=prepare_reuse(args,summary)
    save(args.output/'summary.json',summary)
    for seed in args.seeds:
        for lr in args.learning_rates:
            runid=f'seed{seed}_lr{lr:g}'
            if is_reused(runid,summary):
                print(json.dumps({'event':'reused_complete_no_retraining','run':runid}),flush=True)
                continue
            dest=args.output/runid;dest.mkdir()
            torch.manual_seed(seed);model=XFiStrongCore().cuda()
            model.target_template_m.copy_(torch.tensor(stats['target_mean_mm'].reshape(17,3)/1000,dtype=torch.float32,device='cuda'))
            # Prove modality stems/branch heads match ordinary concat initialization.
            torch.manual_seed(seed);control=FusionModel('concat')
            matching={k:torch.equal(v.cpu(),control.state_dict()[k]) for k,v in model.state_dict().items() if k in control.state_dict() and k!='target_template_m'}
            assert all(matching.values());del control
            optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
            order=torch.Generator().manual_seed(seed);aug=torch.Generator().manual_seed(seed+10000)
            started=time.perf_counter();history=[];torch.cuda.reset_peak_memory_stats()
            for epoch in range(1,args.epochs+1):
                model.train();losses=[]
                for idx in torch.randperm(len(y),generator=order).split(64):
                    batch=tuple(t.cuda() for t in augmented_batch(training,idx,aug,reference))
                    optimizer.zero_grad(set_to_none=True);output=model(*batch)
                    loss,_=fusion_loss(output,y[idx],'concat')
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss.item())
                if epoch==1 or epoch%20==0 or epoch==args.epochs:
                    entry={'epoch':epoch,'loss':float(np.mean(losses))};history.append(entry)
                    print(json.dumps({'run':runid,**entry,'seconds':time.perf_counter()-started}),flush=True)
            model.eval();metrics={};arrays={'target_mm':target,'subject_ids':subject_ids,'frame_ids':stack(val,'frame_ids')}
            with torch.no_grad():
                for condition,inputs in values.items():
                    outputs=[]
                    for ix in range(0,len(target),64):outputs.append(model(*(t[ix:ix+64].cuda() for t in inputs))['fused'].cpu())
                    pred=torch.cat(outputs).numpy().reshape(-1,51)*1000+stats['target_mean_mm']
                    arrays[condition+'_prediction_mm']=pred;metrics[condition]=independent_metrics(pred,target,subject_ids)
            np.savez_compressed(dest/'predictions.npz',**arrays)
            np.savez_compressed(dest/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            row={'id':runid,'seed':seed,'lr':lr,'metrics':metrics,'history':history,'parameters':sum(p.numel() for p in model.parameters()),
                 'seconds':time.perf_counter()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                 'initial_stems_equal_to_concat':all(matching.values()),'weights_sha256':sha(dest/'weights.npz'),'prediction_sha256':sha(dest/'predictions.npz')}
            save(dest/'training.json',row);summary['runs'].append(row);save(args.output/'summary.json',summary)
            print(json.dumps({'event':'complete','run':runid,'clean':metrics['clean']['MPJPE_mm'],'seconds':row['seconds']}),flush=True)
            del model,optimizer;torch.cuda.empty_cache()
    summary['status']='complete';save(args.output/'summary.json',summary)

if __name__=='__main__':main()
