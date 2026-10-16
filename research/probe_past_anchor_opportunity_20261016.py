"""GT-only upper bounds on held-out development sequences; no fitting.

Compare strictly past observed joint holding, also allowing correction of current detections, with GT displacement transport and optimal
segment weights, on a frozen strong model. Oracles are diagnostic, undeployable.
"""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from fusion_models import FusionModel
from run_fusion_experiments_gpu import normalize_inputs
from temporal_anchor_models import context_indices
from analyze_fusion_trials import strict_load_weights,independent_metrics

def oracle_segment(base,candidate,target,eligible):
    delta=candidate-base
    denominator=(delta*delta).sum(-1)
    numerator=((target-base)*delta).sum(-1)
    weight=np.clip(np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-12),0,1)*eligible
    return base+weight[...,None]*delta

def main():
    cache=ROOT/'artifacts/mri_action_windows_resumed_20261016';out=ROOT/'artifacts/past_anchor_opportunity_20261016'
    if out.exists():raise ValueError('New experiment directory required')
    out.mkdir();subjects=[5,6]
    rows=[]
    for s in subjects:
        with np.load(cache/f'subject{s}.npz',allow_pickle=False) as f:
            d={k:f[k] for k in ['points','point_mask','frame_ids','window_ids','target_mm','screened']}
            for c in ['clean','noisy_dark','occlusion']:d[c]=f[c+'_features']
            d['subject_ids']=np.full(len(d['frame_ids']),s);rows.append(d)
    data={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    idx,valid=context_indices(data['subject_ids'],data['window_ids'],data['frame_ids'])
    old=ROOT/'artifacts/fusion_method_research_20261013_expanded_centroid'
    with np.load(old/'normalization.npz',allow_pickle=False) as f:stats={k:f[k] for k in f.files}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    target=data['target_mm'].reshape(-1,17,3)
    results=[]
    for seed in [7,17,27]:
        model=FusionModel('concat');weightsha=strict_load_weights(model,old/f'seed{seed}/concat_weights.npz');model.cuda().eval()
        for c in ['clean','noisy_dark','occlusion']:
            inputs=normalize_inputs(data[c],data['points'],data['point_mask'],stats)
            with torch.no_grad():outputs=[model(*(v[i:i+128].cuda() for v in inputs)) for i in range(0,len(target),128)]
            pred={k:torch.cat([o[k].cpu() for o in outputs]).numpy()*1000+stats['target_mean_mm'].reshape(17,3) for k in ['fused','video']}
            conf=data[c][:,34:];visible=(conf[idx]>=.3)&valid[...,None]
            visible[:,-1]=False
            last=np.where(visible,np.arange(9)[None,:,None],-1).max(1)
            eligible=(last>=0);safe=last.clip(0);b=np.arange(len(target))[:,None];j=np.arange(17)[None,:]
            frame=idx[b,safe]
            held=pred['video'][frame,j]
            transported=held+target-target[frame,j]
            variants={'base':pred['fused'],
                'hold':np.where(eligible[...,None],held,pred['fused']),
                'GT_motion':np.where(eligible[...,None],transported,pred['fused']),
                'hold_oracle_gate':oracle_segment(pred['fused'],held,target,eligible),
                'GT_motion_oracle_gate':oracle_segment(pred['fused'],transported,target,eligible)}
            record={'seed':seed,'condition':c,'weights_sha256':weightsha,'eligible_joint_fraction':float(eligible.mean()),'populations':{}}
            # Stronger screening diagnostic: end frame and every used anchor
            # pass the predeclared physical test; no metric-dependent selection.
            strict=data['screened']&np.all(~eligible|data['screened'][frame],axis=1)
            for group,keep in [('all',np.ones(len(target),bool)),('screened_target_and_anchor',strict)]:
                metrics={name:independent_metrics(p[keep].reshape(-1,51),data['target_mm'][keep],data['subject_ids'][keep]) for name,p in variants.items()}
                record['populations'][group]={'frames':int(keep.sum()),'metrics':metrics,
                    'gain_mm':{name:metrics['base']['MPJPE_mm']-m['MPJPE_mm'] for name,m in metrics.items()}}
            results.append(record)
    receipt={'status':'complete','scope':'No training. Two already held-out development subjects of frozen old strong concat; GT motion and gate are undeployable diagnostics, not proposed model results.',
             'subjects':subjects,'test_subjects_used':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results}
    (out/'report.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for r in results:print(json.dumps({'seed':r['seed'],'condition':r['condition'],'eligible':r['eligible_joint_fraction'],'screened':r['populations']['screened_target_and_anchor']['gain_mm']}))

if __name__=='__main__':main()
