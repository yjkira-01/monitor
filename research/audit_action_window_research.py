"""Saved-weight replay, independent metrics and transport identity audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from fusion_models import FusionModel
from awc_training_rule import AWCConcat
from temporal_anchor_models import make_model
from analyze_fusion_trials import strict_load_weights,independent_metrics
from run_action_window_research import collect,make_inputs,index_context,temporal_batch,encode

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def check(actual,expected,tolerance,label):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError(label+' shape/finite')
    d=float(np.max(abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.
    if d>tolerance:raise ValueError(f'{label}: {d} > {tolerance}')
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--cache',type=Path,default=ROOT/'artifacts/mri_action_windows_20261016');args=p.parse_args()
    proto=json.loads((args.directory/'protocol.json').read_text());summary=json.loads((args.directory/'summary.json').read_text())
    assert summary['status']=='complete' and not proto['test_subjects_used'] and not summary['test_subjects_used']
    for n,h in proto['source_sha256'].items():assert sha(ROOT/'research'/n)==h and sha(args.directory/'source_snapshot'/n)==h
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    checks=[]
    for run in summary['runs']:
        fold,seed=run['fold'],run['seed'];fdir=args.directory/f'fold{fold}';sdir=fdir/f'seed{seed}'
        data=json.loads((fdir/'data.json').read_text());val=data['validation_subjects']
        assert not ((set(data['train_subjects'])|set(val))&{3,10,12,18})
        for s,h in data['cache_sha256'].items():assert sha(args.cache/f'subject{s}.npz')==h
        va=collect(args.cache,val)
        with np.load(fdir/'normalization.npz',allow_pickle=False) as f:stats={k:f[k] for k in f.files}
        inputs=make_inputs(va,stats)
        for method,record in run['base'].items():
            model=AWCConcat() if method in ('concat_mpjpe','awc_concat') else FusionModel('concat' if method=='concat_mse' else method)
            dest=sdir/method;assert sha(dest/'weights.npz')==record['weights_sha256'];assert sha(dest/'predictions.npz')==record['predictions_sha256']
            strict_load_weights(model,dest/'weights.npz');model.cuda().eval()
            with np.load(dest/'predictions.npz',allow_pickle=False) as f:
                check(f['target_mm'],va['target_mm'],0.,'target alignment')
                for cond,inp in inputs.items():
                    with torch.no_grad():
                        replay=torch.cat([model(*(v[i:i+128].cuda() for v in inp))['fused'].cpu() for i in range(0,len(va['target_mm']),128)]).numpy().reshape(-1,51)*1000+stats['target_mean_mm']
                    saved=f[cond+'_prediction_mm'];d=check(replay,saved,.01,'base CUDA replay')
                    metric=independent_metrics(saved,f['target_mm'],f['subject_ids'])
                    md=abs(metric['MPJPE_mm']-record['metrics'][cond]['MPJPE_mm']);assert md<1e-8
                    checks.append({'fold':fold,'seed':seed,'method':method,'condition':cond,'replay_max_mm':d,'metric_difference_mm':md})
            if method=='concat':base=model
        with np.load(sdir/'frozen_features.npz',allow_pickle=False) as f:
            frozen={k:f[k] for k in f.files}
        contexts={}
        for cond,inp in inputs.items():
            fresh=encode(base,inp,stats)
            for key,value in fresh.items():check(value.numpy(),frozen['validation_'+cond+'_'+key],1e-6,'frozen feature replay')
            contexts[cond]=index_context(fresh,va)
            ctx=contexts[cond]
            idx=ctx['idx'].numpy();valid=ctx['valid'].numpy()
            current=np.arange(len(va['frame_ids']))[:,None]
            assert np.all(idx[valid]<=np.broadcast_to(current,idx.shape)[valid])
            assert np.all(va['subject_ids'][idx][valid]==np.broadcast_to(va['subject_ids'][:,None],idx.shape)[valid])
            assert np.all(va['window_ids'][idx][valid]==np.broadcast_to(va['window_ids'][:,None],idx.shape)[valid])
            if cond=='video_missing':assert not fresh['vv'].any() and not fresh['confidence'].any()
            if cond=='radar_missing':assert not fresh['rv'].any()
        for method,record in run['temporal'].items():
            dest=sdir/method;assert sha(dest/'weights.npz')==record['weights_sha256'];assert sha(dest/'predictions.npz')==record['predictions_sha256']
            assert sha(sdir/'concat/weights.npz')==record['base_weights_sha256'];assert sha(sdir/'frozen_features.npz')==record['frozen_features_sha256']
            model=make_model(method);strict_load_weights(model,dest/'weights.npz');model.cuda().eval()
            with np.load(dest/'predictions.npz',allow_pickle=False) as f:
                for cond,ctx in contexts.items():
                    outputs=[]
                    with torch.no_grad():
                        for i in range(0,len(va['target_mm']),128):outputs.append(model(temporal_batch(ctx,torch.arange(i,min(i+128,len(va['target_mm']))))))
                    saved=f[cond+'_prediction_mm']
                    replay=torch.cat([o['prediction'].cpu() for o in outputs]).numpy().reshape(-1,51)*1000+stats['target_mean_mm']
                    d=check(replay,saved,.01,'temporal CUDA replay')
                    metric=independent_metrics(saved,f['target_mm'],f['subject_ids']);md=abs(metric['MPJPE_mm']-record['metrics'][cond]['MPJPE_mm']);assert md<1e-8
                    if method.startswith('anchor'):
                        # Independently reconstruct transport and gate from
                        # NumPy arrays and saved observation-derived anchor IDs.
                        enc=ctx['encoded'];idx=ctx['idx'].numpy();last=f[cond+'_last_anchor'];disp=f[cond+'_displacement']
                        cum=np.cumsum(disp,axis=1);b=np.arange(len(last))[:,None];j=np.arange(17)[None,:];safe=np.maximum(last,0)
                        video=enc['video'].numpy()[idx]
                        transported=video[b,safe,j]+cum[:,-1]-cum[b,safe,j]
                        check(transported,f[cond+'_transported'],1e-6,'NumPy motion transport')
                        base_pred=enc['base'].numpy();gate=f[cond+'_gate']
                        reconstructed=(base_pred+gate[...,None]*(transported-base_pred)).reshape(-1,51)*1000+stats['target_mean_mm']
                        check(reconstructed,saved,.01,'NumPy final correction')
                        confidence=enc['confidence'].numpy()
                        assert np.all(gate[confidence>=.3]==0) and np.all(gate[last<0]==0)
                        if cond=='video_missing':check(saved,base_pred.reshape(-1,51)*1000+stats['target_mean_mm'],0.,'no visual history exact base')
                    checks.append({'fold':fold,'seed':seed,'method':method,'condition':cond,'replay_max_mm':d,'metric_difference_mm':md})
        print(json.dumps({'audited_fold':fold,'seed':seed,'models':len(run['base'])+len(run['temporal'])}),flush=True)
    receipt={'status':'complete','pilot':proto['pilot'],'checks':checks,'models':sum(len(r['base'])+len(r['temporal']) for r in summary['runs']),
             'source_sha256':sha(Path(__file__)),'summary_sha256':sha(args.directory/'summary.json'),'test_subjects_used':[]}
    (args.directory/'audit.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'status':'complete','models':receipt['models'],'max_replay_mm':max(c['replay_max_mm'] for c in checks)}),flush=True)

if __name__=='__main__':main()
