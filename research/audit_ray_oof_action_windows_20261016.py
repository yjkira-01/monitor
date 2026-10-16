"""OOF source separation, raw-feature replays, NumPy selector and metric audit."""
import hashlib,json,os,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from audit_mri import load_labels
from fusion_models import FusionModel
from ray_gate_models import RaySelector
from run_ray_oof_action_windows_20261016 import payload,features,PIXEL,CONDS,TRAIN,VAL,FOLDS
from ray_action_window_data import load_action_window_rows
from probe_radar_depth_ray_refinement import depth_features,irls_fit
from run_fusion_experiments_gpu import load_data
from analyze_fusion_trials import strict_load_weights,independent_metrics
DEST=ROOT/'artifacts/ray_oof_action_windows_20261016'
OLD=ROOT/'artifacts/action_window_main_20261016'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def compare(a,b,tol,label):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape,label
    if a.dtype.kind=='b' or b.dtype.kind=='b':assert np.array_equal(a,b),label;return 0.
    assert np.isfinite(a).all() and np.isfinite(b).all(),label+' finite'
    d=float(np.max(abs(a.astype(float)-b.astype(float)))) if a.size else 0.
    if d>tol:raise ValueError(f'{label}: {d}>{tol}')
    return d

def main():
    proto=json.loads((DEST/'protocol.json').read_text());summary=json.loads((DEST/'summary.json').read_text())
    assert summary['status']=='complete' and len(summary['runs'])==3 and not summary['test_subjects_used']
    for n,h in proto['source_sha256'].items():assert sha(ROOT/'research'/n)==h and sha(DEST/'source_snapshot'/n)==h
    assert sha(OLD/'summary.json')==proto['old_summary_sha256']
    old=json.loads((OLD/'summary.json').read_text());cache=ROOT/'artifacts/mri_action_windows_resumed_20261016'
    for s,h in json.loads((OLD/'fold0/data.json').read_text())['cache_sha256'].items():assert sha(cache/f'subject{s}.npz')==h
    _,rows=load_action_window_rows(cache);extra={}
    with zipfile.ZipFile(ROOT/'dataset_release.zip') as z:
        for s,row in rows.items():
            labels=load_labels(z.read(f'dataset_release/aligned_data/pose_labels/subject{s}_all_labels.cpl'));camera=labels['camera_matrix']
            with np.load(cache/f'subject{s}.npz',allow_pickle=False) as f:poses={c:f[c+'_poses'].copy() for c in PIXEL}
            gt=labels['refined_gt_kps'][row['frame_ids']].transpose(0,2,1)
            cam=gt*np.array([-1000.,-1000.,1000.])@camera['cam1_R'].T+camera['cam1_T']
            extra[s]={'features':depth_features(row['points'],row['mask']),'poses':poses,'K':camera['cam1_intrinsic'],'R':camera['cam1_R'],
                      'depth_gt':cam[:,[11,12],2].mean(1)/1000}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    checks=[]
    for run in summary['runs']:
        seed=run['seed'];sd=DEST/f'seed{seed}';heldout=[];oof_parts=[]
        for r in run['OOF']:
            assert not(set(r['train_subjects'])&set(r['heldout_subjects']))
            assert set(r['train_subjects'])|set(r['heldout_subjects'])==set(TRAIN)
            assert not((set(r['train_subjects'])|set(r['heldout_subjects']))&set(VAL+[3,10,12,18]))
            heldout+=r['heldout_subjects'];fd=sd/f"oof{r['fold']}"
            assert sha(fd/'weights.npz')==r['weights_sha256'] and sha(fd/'heldout_payload.npz')==r['payload_sha256']
            fit=load(fd/'depth_fit.npz')
            refit=irls_fit(np.concatenate([extra[s]['features'] for s in r['train_subjects']]),np.concatenate([extra[s]['depth_gt'] for s in r['train_subjects']]))
            for k in fit:compare(refit[k],fit[k],1e-10,'Depth fit uses OOF training subjects only')
            model=FusionModel('concat');strict_load_weights(model,fd/'weights.npz');model.cuda().eval()
            fresh=payload(model,load(fd/'normalization.npz'),rows,extra,r['heldout_subjects'],load(fd/'depth_fit.npz'),r['constant_depth_m'])
            saved=load(fd/'heldout_payload.npz')
            d=max(compare(fresh[k],saved[k],1e-5,'OOF observation replay '+k) for k in saved)
            oof_parts.append(saved);checks.append({'seed':seed,'OOF':r['fold'],'max_payload_difference_m':d})
        assert sorted(heldout)==sorted(TRAIN) and len(heldout)==len(set(heldout))
        combined=load(sd/'oof_payload.npz')
        for k in combined:compare(np.concatenate([r[k] for r in oof_parts]),combined[k],0,'OOF concatenation')
        # GT poisoning must have no influence on the features/eligibility sent
        # to selectors. The forward signature itself has no target argument.
        original=features(combined,'radar_candidate');poison=dict(combined);poison['target']=np.full_like(combined['target'],1e10)
        torch.testing.assert_close(original,features(poison,'radar_candidate'),atol=0,rtol=0)
        full=FusionModel('concat');assert strict_load_weights(full,OLD/f'fold0/seed{seed}/concat/weights.npz')==run['base_weights_sha256'];full.cuda().eval()
        validation=load(sd/'validation_payload.npz')
        constant=float(np.median(np.concatenate([extra[s]['depth_gt'] for s in TRAIN])))
        fit=load(sd/'full_depth_fit.npz')
        refit=irls_fit(np.concatenate([extra[s]['features'] for s in TRAIN]),np.concatenate([extra[s]['depth_gt'] for s in TRAIN]))
        for k in fit:compare(refit[k],fit[k],1e-10,'Full depth fit uses training subjects only')
        fresh=payload(full,load(OLD/'fold0/normalization.npz'),rows,extra,VAL,load(sd/'full_depth_fit.npz'),constant)
        for k in validation:compare(fresh[k],validation[k],1e-5,'Validation observation replay '+k)
        for method,r in run['models'].items():
            d=sd/method;assert sha(d/'weights.npz')==r['weights_sha256'] and sha(d/'predictions.npz')==r['predictions_sha256']
            source='constant_candidate' if method in ('constant_ray_gate','free_constant_residual') else 'radar_candidate'
            if method=='free_no_ray_residual':
                validation=dict(validation);validation['radar_candidate']=validation['base'].copy()
            inputs=features(validation,source);selector=RaySelector(method.startswith('free_'))
            strict_load_weights(selector,d/'weights.npz');selector.cuda().eval();replay=[]
            with torch.no_grad():
                for start in range(0,len(inputs),256):
                    sl=slice(start,start+256)
                    out=selector(torch.tensor(validation['base'][sl],dtype=torch.float32,device='cuda'),
                        torch.tensor(validation[source][sl],dtype=torch.float32,device='cuda'),inputs[sl].cuda(),
                        torch.tensor(validation['eligible'][sl],dtype=torch.bool,device='cuda'))
                    replay.append(out.cpu().numpy()*1000)
            saved=load(d/'predictions.npz')['prediction_mm']
            max_replay=compare(np.concatenate(replay).reshape(-1,51),saved,.01,'Selector CUDA replay')
            # Independent NumPy implementation of the tiny learned head.
            w=load(d/'weights.npz');identity=np.broadcast_to(w['joint'],(len(inputs),17,16))
            x=np.concatenate([inputs.numpy(),identity],axis=-1).astype(np.float64)
            h=np.maximum(x@w['network.0.weight'].T+w['network.0.bias'],0)
            logits=h@w['network.2.weight'].T+w['network.2.bias']
            eligible=validation['eligible'][...,None]
            if method.startswith('free_'):pred=validation['base']+.1*np.tanh(logits)*eligible
            else:pred=validation['base']+(1/(1+np.exp(-logits)))*eligible*(validation[source]-validation['base'])
            max_numpy=compare(pred.reshape(-1,51)*1000,saved,.01,'NumPy selector replay')
            metric_diff=0.
            for ci,c in enumerate(CONDS):
                keep=validation['condition_ids']==ci
                metric=independent_metrics(saved[keep],validation['target'][keep].reshape(-1,51)*1000,validation['subject_ids'][keep])
                metric_diff=max(metric_diff,abs(metric['MPJPE_mm']-r['metrics'][c]['MPJPE_mm']))
            assert metric_diff<1e-8
            checks.append({'seed':seed,'method':method,'CUDA_replay_max_mm':max_replay,'NumPy_replay_max_mm':max_numpy,'metric_difference_mm':metric_diff})
            print(json.dumps({'audited_seed':seed,'method':method}),flush=True)
    receipt={'status':'complete','checks':checks,'test_subjects_used':[],'models':15,
        'source_sha256':sha(Path(__file__)),'summary_sha256':sha(DEST/'summary.json'),
        'scope':'All nine OOF expert observation reconstructions, disjoint subject provenance, unchanged original full-base weights, all fifteen learned selectors CUDA+NumPy replay and metrics; no calibrated confidence claim'}
    (DEST/'audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (DEST/'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':'complete','models':15}))

if __name__=='__main__':main()
