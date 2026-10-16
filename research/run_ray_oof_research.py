"""OOF geometric fusion prototype, with mandatory constant/free controls.

OOF predictions for the 12 training subjects come from three models trained on
the other eight, and depth fits on those same eight. A gate never learns labels
from an example used to fit its base model. Evaluation uses the frozen full-12
strong models. This is exploratory and the OOF/deployment distribution differs.
"""
import hashlib,json,os,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from audit_mri import load_labels
from fusion_models import FusionModel,fusion_loss
from ray_gate_models import RaySelector
from probe_radar_depth_ray_refinement import depth_features,irls_fit,depth_predict,ray_refine
from run_fusion_experiments_gpu import load_data,normalize_inputs,augmented_batch
from train_mri_point_baselines import point_normalization
from analyze_fusion_trials import strict_load_weights,independent_metrics

CONDS=['clean','dark','noisy_dark','occlusion','video_missing','radar_missing']
PIXEL=CONDS[:4]
TRAIN=[1,2,4,7,8,9,11,13,14,15,16,17]
VAL=[5,6,19,20]
FOLDS=[[1,2,4,7],[8,9,11,13],[14,15,16,17]]
DEST=ROOT/'artifacts/ray_oof_research_20261016'
OLD=ROOT/'artifacts/fusion_method_research_20261013_expanded_centroid'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def save_state(model,path):np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})

def payload(model,stats,rows,extra,subjects,fit,constant):
    arrays={k:[] for k in ['base','radar_candidate','constant_candidate','confidence','depth','radar_present','eligible',
                         'target','subject_ids','frame_ids','condition_ids']}
    model.eval()
    for ci,condition in enumerate(CONDS):
        for s in subjects:
            row,e=rows[s],extra[s];c='clean' if condition.endswith('missing') else condition
            inp=list(normalize_inputs(row[c],row['points'],row['mask'],stats))
            pose=e['poses'][c].copy();present=np.ones(len(pose),bool)
            if condition=='video_missing':inp[0].zero_();inp[3].zero_();pose[:]=0
            if condition=='radar_missing':inp[1].zero_();inp[2].zero_();present[:]=False
            with torch.no_grad():pred=model(*(v.cuda() for v in inp))['fused'].cpu().numpy().reshape(-1,51)*1000+stats['target_mean_mm']
            base=pred.reshape(-1,17,3);base=base-base[:,[11,12]].mean(1)[:,None]
            depths=depth_predict(fit,e['features']) if present.any() else np.full(len(pose),constant)
            radar,eligible=ray_refine(pred,pose,e['K'],e['R'],depths)
            fixed,_=ray_refine(pred,pose,e['K'],e['R'],np.full(len(pose),constant))
            values={'base':base/1000,'radar_candidate':radar.reshape(-1,17,3)/1000,
                'constant_candidate':fixed.reshape(-1,17,3)/1000,'confidence':pose[...,2],
                'depth':depths,'radar_present':present,'eligible':eligible,'target':row['target'].reshape(-1,17,3)/1000,
                'subject_ids':np.full(len(pose),s),'frame_ids':row['frame_ids'],'condition_ids':np.full(len(pose),ci)}
            for k,v in values.items():arrays[k].append(v)
    return {k:np.concatenate(v) for k,v in arrays.items()}

def features(data,source):
    base,candidate=data['base'],data[source]
    conf=data['confidence'];hips=np.repeat(conf[:,None,[11,12]],17,axis=1)
    delta=candidate-base
    # 15 observed features per joint; no target appears here.
    x=np.concatenate([base,candidate,abs(delta),conf[...,None],hips,
        np.repeat(data['radar_present'][:,None,None],17,axis=1).astype(float),
        np.linalg.norm(delta,axis=-1)[...,None],np.repeat(data['depth'][:,None,None],17,axis=1)],axis=-1)
    assert x.shape[-1]==15
    return torch.tensor(x,dtype=torch.float32)

def main():
    if DEST.exists():raise ValueError('New output required')
    DEST.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    cache=ROOT/'artifacts/fusion_rgb_expanded_20261013'
    split,rows=load_data(cache,PIXEL,'visible_centroid');assert split['canonical_ids_train']==TRAIN and split['canonical_ids_validation']==VAL
    extra={}
    with zipfile.ZipFile(ROOT/'dataset_release.zip') as z:
        for s,row in rows.items():
            labels=load_labels(z.read(f'dataset_release/aligned_data/pose_labels/subject{s}_all_labels.cpl'))
            camera=labels['camera_matrix'];gt=labels['refined_gt_kps'][row['frame_ids']].transpose(0,2,1)
            cam=gt*np.array([-1000.,-1000.,1000.])@camera['cam1_R'].T+camera['cam1_T']
            with np.load(cache/f'subject{s}.npz',allow_pickle=False) as f:poses={c:f[c+'_poses'].copy() for c in PIXEL}
            extra[s]={'depth_gt':cam[:,[11,12],2].mean(1)/1000,'features':depth_features(row['points'],row['mask']),
                'poses':poses,'K':camera['cam1_intrinsic'],'R':camera['cam1_R']}
    sources={n:sha(ROOT/'research'/n) for n in [Path(__file__).name,'ray_gate_models.py','probe_radar_depth_ray_refinement.py','fusion_models.py','run_fusion_experiments_gpu.py']}
    snap=DEST/'source_snapshot';snap.mkdir()
    for n in sources:(snap/n).write_bytes((ROOT/'research'/n).read_bytes())
    protocol={'date':'2026-10-16','train_subjects':TRAIN,'validation_subjects':VAL,'test_subjects_used':[],
        'OOF_folds':FOLDS,'initialization_seeds':[7,17,27],'base_epochs':80,'gate_epochs':80,
        'OOF_model':'exact local concat architecture/objective/augmentation on other8 of12 subjects, own train-only normalization and depth fit',
        'deployment_model':'frozen full12 concat per seed, unchanged weights; recenter prediction hips for every method/control',
        'gate_inputs':'15 observed values per joint +16 joint embedding; base/candidate XYZ,abs correction,raw confidence,hips confidence,radar present,correction norm,estimated radar depth',
        'methods':['radar_ray_gate','constant_ray_gate','free_radar_residual'],'loss':'MPJPE','optimizer':'AdamW lr.001 weight_decay.0001 batch256 last80',
        'depth_fit':'train-only IRLS Huber.15m ridge1 clip[.5,10]m, 14 radar statistics',
        'limitations':'Known projection/stacking/gating components; novelty not established. OOF8-to-full12 error shift. Old development subjects already extensively explored.',
        'source_sha256':sources,'old_summary_sha256':sha(OLD/'run_summary.json')}
    save(DEST/'protocol.json',protocol);summary={'status':'running','runs':[],'test_subjects_used':[]};save(DEST/'summary.json',summary)
    def stack(ids,key):return np.concatenate([rows[s][key] for s in ids])
    for seed in [7,17,27]:
        sdir=DEST/f'seed{seed}';sdir.mkdir();oof=[];records=[]
        for fi,held in enumerate(FOLDS):
            train=[s for s in TRAIN if s not in held];fd=sdir/f'oof{fi}';fd.mkdir()
            pm,ps=point_normalization(stack(train,'points'),stack(train,'mask'))
            stats={'point_mean':pm,'point_std':ps,'rgb_mean':stack(train,'clean').mean(0),
                'rgb_std':np.maximum(stack(train,'clean').std(0),1e-8),'target_mean_mm':stack(train,'target').mean(0)}
            np.savez_compressed(fd/'normalization.npz',**stats)
            training=[normalize_inputs(stack(train,c),stack(train,'points'),stack(train,'mask'),stats) for c in PIXEL]
            reference=torch.tensor(np.stack([stack(train,'reference_available_'+c) for c in PIXEL]))
            target=torch.tensor((stack(train,'target')-stats['target_mean_mm']).reshape(-1,17,3)/1000,dtype=torch.float32,device='cuda')
            torch.manual_seed(seed);model=FusionModel('concat').cuda();model.target_template_m.copy_(torch.tensor(stats['target_mean_mm'].reshape(17,3)/1000,dtype=torch.float32,device='cuda'))
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            order=torch.Generator().manual_seed(seed);aug=torch.Generator().manual_seed(seed+10000)
            for epoch in range(80):
                model.train()
                for ix in torch.randperm(len(target),generator=order).split(64):
                    batch=tuple(v.cuda() for v in augmented_batch(training,ix,aug,reference));opt.zero_grad(set_to_none=True)
                    loss,_=fusion_loss(model(*batch),target[ix],'concat');assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            fit=irls_fit(np.concatenate([extra[s]['features'] for s in train]),np.concatenate([extra[s]['depth_gt'] for s in train]))
            constant=float(np.median(np.concatenate([extra[s]['depth_gt'] for s in train])))
            data=payload(model,stats,rows,extra,held,fit,constant);np.savez_compressed(fd/'heldout_payload.npz',**data)
            save_state(model,fd/'weights.npz');np.savez_compressed(fd/'depth_fit.npz',**fit)
            record={'fold':fi,'train_subjects':train,'heldout_subjects':held,'training_epochs':80,
                'weights_sha256':sha(fd/'weights.npz'),'payload_sha256':sha(fd/'heldout_payload.npz'),'constant_depth_m':constant}
            save(fd/'receipt.json',record);records.append(record);oof.append(data)
            print(json.dumps({'event':'oof_complete','seed':seed,'fold':fi,'heldout':held}),flush=True)
        data={k:np.concatenate([d[k] for d in oof]) for k in oof[0]};np.savez_compressed(sdir/'oof_payload.npz',**data)
        fullfit=irls_fit(np.concatenate([extra[s]['features'] for s in TRAIN]),np.concatenate([extra[s]['depth_gt'] for s in TRAIN]))
        constant=float(np.median(np.concatenate([extra[s]['depth_gt'] for s in TRAIN])))
        base=FusionModel('concat');base_sha=strict_load_weights(base,OLD/f'seed{seed}/concat_weights.npz');base.cuda().eval()
        with np.load(OLD/'normalization.npz',allow_pickle=False) as f:stats={k:f[k] for k in f.files}
        validation=payload(base,stats,rows,extra,VAL,fullfit,constant);np.savez_compressed(sdir/'validation_payload.npz',**validation)
        np.savez_compressed(sdir/'full_depth_fit.npz',**fullfit)
        run={'seed':seed,'base_weights_sha256':base_sha,'OOF':records,'models':{}}
        for method in protocol['methods']:
            source='constant_candidate' if method=='constant_ray_gate' else 'radar_candidate'
            x=features(data,source);vx=features(validation,source)
            tensors={k:torch.tensor(data[k],dtype=torch.float32 if k!='eligible' else torch.bool) for k in ['base',source,'eligible','target']}
            torch.manual_seed(seed+30000);gate=RaySelector(method=='free_radar_residual').cuda()
            opt=torch.optim.AdamW(gate.parameters(),lr=.001,weight_decay=.0001);order=torch.Generator().manual_seed(seed);history=[]
            for epoch in range(1,81):
                gate.train();losses=[]
                for ix in torch.randperm(len(x),generator=order).split(256):
                    opt.zero_grad(set_to_none=True)
                    pred=gate(tensors['base'][ix].cuda(),tensors[source][ix].cuda(),x[ix].cuda(),tensors['eligible'][ix].cuda())
                    loss=(pred-tensors['target'][ix].cuda()).norm(dim=-1).mean();assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),1.);opt.step();losses.append(loss.item())
                if epoch==1 or epoch%20==0:
                    h={'epoch':epoch,'loss':float(np.mean(losses))};history.append(h);print(json.dumps({'seed':seed,'method':method,**h}),flush=True)
            gate.eval();pred=[]
            with torch.no_grad():
                for start in range(0,len(vx),256):
                    sl=slice(start,start+256)
                    v=gate(torch.tensor(validation['base'][sl],dtype=torch.float32,device='cuda'),
                        torch.tensor(validation[source][sl],dtype=torch.float32,device='cuda'),vx[sl].cuda(),
                        torch.tensor(validation['eligible'][sl],dtype=torch.bool,device='cuda'))
                    pred.append(v.cpu().numpy()*1000)
            pred=np.concatenate(pred).reshape(-1,51);metrics={}
            for ci,c in enumerate(CONDS):
                select=validation['condition_ids']==ci
                metrics[c]=independent_metrics(pred[select],validation['target'][select].reshape(-1,51)*1000,validation['subject_ids'][select])
            dest=sdir/method;dest.mkdir();save_state(gate,dest/'weights.npz');np.savez_compressed(dest/'predictions.npz',prediction_mm=pred)
            record={'metrics':metrics,'history':history,'weights_sha256':sha(dest/'weights.npz'),'predictions_sha256':sha(dest/'predictions.npz'),
                    'parameters':sum(p.numel() for p in gate.parameters())};save(dest/'training.json',record);run['models'][method]=record
            print(json.dumps({'event':'gate_complete','seed':seed,'method':method,'clean_mm':metrics['clean']['MPJPE_mm']}),flush=True)
        summary['runs'].append(run);save(DEST/'summary.json',summary)
    summary['status']='complete';save(DEST/'summary.json',summary)

if __name__=='__main__':main()
