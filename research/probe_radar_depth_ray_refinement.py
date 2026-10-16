"""Train-only radar depth fit and geometric lifting on frozen strong predictions.

This is an ordinary geometric refinement hypothesis, not a novelty claim.
GT root depth is used ONLY in an explicitly labelled oracle control. The
deployable path estimates depth from observed radar statistics and calibration.
"""
import hashlib,io,json,os,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from audit_mri import load_labels
from fusion_models import FusionModel
from run_fusion_experiments_gpu import load_data,normalize_inputs
from analyze_fusion_trials import strict_load_weights,independent_metrics

def depth_features(points,mask):
    result=[]
    for p,m in zip(points,mask):
        observed=p[m]
        if len(observed):
            q=np.quantile(observed[:,:3],[.2,.5,.8],axis=0).reshape(-1)
            result.append(np.r_[q,observed[:,3:].mean(0),observed[:,3:].std(0),len(observed)/196])
        else:result.append(np.zeros(14))
    return np.asarray(result)

def irls_fit(x,y,steps=25):
    centre=np.median(x,axis=0);scale=np.maximum(np.median(abs(x-centre),axis=0)*1.4826,1e-6)
    a=np.column_stack([(x-centre)/scale,np.ones(len(x))]);w=np.zeros(a.shape[1]);w[-1]=np.median(y)
    penalty=np.eye(a.shape[1]);penalty[-1,-1]=0
    for _ in range(steps):
        residual=a@w-y;weights=np.minimum(1.,.15/np.maximum(abs(residual),1e-8))
        w=np.linalg.solve(a.T@(weights[:,None]*a)+penalty,a.T@(weights*y))
    return {'centre':centre,'scale':scale,'weights':w}

def depth_predict(model,x):
    return np.clip(np.column_stack([(x-model['centre'])/model['scale'],np.ones(len(x))])@model['weights'],.5,10.)

def ray_refine(pred,poses,K,R,depth):
    pred=pred.reshape(-1,17,3)
    pred=pred-pred[:,[11,12]].mean(1)[:,None]
    camera=(pred*np.array([-1.,-1.,1.]))@R.T
    rays=(poses[...,:2]-np.array([K[0,2],K[1,2]]))/np.array([K[0,0],K[1,1]])
    absolute=camera.copy();absolute[...,2]+=depth[:,None]*1000
    absolute[...,:2]=rays*absolute[...,2:]
    lifted=absolute-absolute[:,[11,12]].mean(1)[:,None]
    visible=poses[...,2]>=.3
    valid=visible&visible[:,11,None]&visible[:,12,None]
    camera=np.where(valid[...,None],lifted,camera)
    return (camera@R*np.array([-1.,-1.,1.])).reshape(-1,51),valid

def main():
    output=ROOT/'artifacts/radar_depth_ray_probe_20261016'
    if output.exists():raise ValueError('New experiment directory required')
    output.mkdir();cache=ROOT/'artifacts/fusion_rgb_expanded_20261013'
    split,rows=load_data(cache,['clean','dark','noisy_dark','occlusion'],'visible_centroid')
    train,val=split['canonical_ids_train'],split['canonical_ids_validation'];extra={}
    with zipfile.ZipFile(ROOT/'dataset_release.zip') as z:
        for s,row in rows.items():
            labels=load_labels(z.read(f'dataset_release/aligned_data/pose_labels/subject{s}_all_labels.cpl'))
            camera=labels['camera_matrix'];gt=labels['refined_gt_kps'][row['frame_ids']].transpose(0,2,1)
            xyz=(gt*np.array([-1000.,-1000.,1000.]))@camera['cam1_R'].T+camera['cam1_T']
            depth=xyz[:,[11,12],2].mean(1)/1000
            with np.load(cache/f'subject{s}.npz',allow_pickle=False) as f:poses={c:f[c+'_poses'].copy() for c in ['clean','dark','noisy_dark','occlusion']}
            extra[s]={'depth':depth,'features':depth_features(row['points'],row['mask']),'K':camera['cam1_intrinsic'],'R':camera['cam1_R'],'poses':poses}
    x=np.concatenate([extra[s]['features'] for s in train]);y=np.concatenate([extra[s]['depth'] for s in train])
    assert np.isfinite(y).all();fit=irls_fit(x,y);constant=float(np.median(y))
    np.savez_compressed(output/'train_only_depth_fit.npz',**fit)
    old=ROOT/'artifacts/fusion_method_research_20261013_expanded_centroid'
    with np.load(old/'normalization.npz',allow_pickle=False) as f:stats={k:f[k] for k in f.files}
    subjects=np.concatenate([np.full(len(rows[s]['target']),s) for s in val]);target=np.concatenate([rows[s]['target'] for s in val])
    report={'status':'complete','scope':'Exploratory geometric operator, same frozen 1536-frame strong models; GT-depth control is undeployable. No final-test access.',
        'train_subjects':train,'validation_subjects':val,'test_subjects_used':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'depth_fit':'train-only IRLS Huber .15m/ridge1, 14 observed point statistics; clip[.5,10]m; no GT crop or root input at inference',
        'constant_train_depth_m':constant,'depth_diagnostics':{},'models':{}}
    for s in val:
        estimated=depth_predict(fit,extra[s]['features']);extra[s]['estimated']=estimated
        report['depth_diagnostics'][str(s)]={'radar_depth_MAE_mm':float(abs(estimated-extra[s]['depth']).mean()*1000),
            'constant_depth_MAE_mm':float(abs(constant-extra[s]['depth']).mean()*1000),
            'target_depth_percentiles_m':np.percentile(extra[s]['depth'],[0,50,100]).tolist()}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    for seed in [7,17,27]:
        model=FusionModel('concat');strict_load_weights(model,old/f'seed{seed}/concat_weights.npz');model.cuda().eval();report['models'][str(seed)]={}
        for condition in ['clean','dark','noisy_dark','occlusion','video_missing','radar_missing']:
            values={k:[] for k in ['base','recentered','constant_depth','radar_depth','GT_depth_oracle']};support=[]
            for s in val:
                row,e=rows[s],extra[s];c='clean' if condition.endswith('missing') else condition
                inp=list(normalize_inputs(row[c],row['points'],row['mask'],stats))
                if condition=='video_missing':inp[0].zero_();inp[3].zero_()
                if condition=='radar_missing':inp[1].zero_();inp[2].zero_()
                with torch.no_grad():base=model(*(v.cuda() for v in inp))['fused'].cpu().numpy().reshape(-1,51)*1000+stats['target_mean_mm']
                centred=base.reshape(-1,17,3);centred=(centred-centred[:,[11,12]].mean(1)[:,None]).reshape(-1,51)
                values['base'].append(base);values['recentered'].append(centred)
                pose=e['poses'][c].copy()
                if condition=='video_missing':pose[...,2]=0
                for name,depth in [('constant_depth',np.full(len(base),constant)),('radar_depth',e['estimated']),('GT_depth_oracle',e['depth'])]:
                    # If radar is absent, the deployable range branch must
                    # explicitly use the train-only constant-depth fallback.
                    if name=='radar_depth' and condition=='radar_missing':depth=np.full(len(base),constant)
                    refined,valid=ray_refine(base,pose,e['K'],e['R'],depth)
                    values[name].append(refined)
                    if name=='radar_depth':support.extend(valid.any(1).tolist())
            metrics={name:independent_metrics(np.concatenate(v),target,subjects) for name,v in values.items()}
            gains={name:metrics['recentered']['MPJPE_mm']-v['MPJPE_mm'] for name,v in metrics.items()}
            report['models'][str(seed)][condition]={'metrics':metrics,'gain_vs_recentered_mm':gains,'refinable_frames':sum(support)}
            np.savez_compressed(output/f'seed{seed}_{condition}.npz',target_mm=target,subject_ids=subjects,**{k:np.concatenate(v) for k,v in values.items()})
            print(json.dumps({'seed':seed,'condition':condition,'gain_vs_recentered_mm':gains}),flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
