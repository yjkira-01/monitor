"""Exploratory fixed quality/correction-size controls, all settings reported."""
import csv,json
from pathlib import Path
import numpy as np
from analyze_fusion_trials import independent_metrics
ROOT=Path(__file__).resolve().parents[1]
directory=ROOT/'artifacts/radar_depth_ray_probe_20261016'
score=['clean','noisy_dark','occlusion','video_missing','radar_missing']
subjects=[5,6,19,20]
poses={}
for c in ['clean','dark','noisy_dark','occlusion']:
    values=[]
    for s in subjects:
        with np.load(ROOT/f'artifacts/fusion_rgb_expanded_20261013/subject{s}.npz',allow_pickle=False) as f:values.append(f[c+'_poses'])
    poses[c]=np.concatenate(values)
rows=[]
for seed in [7,17,27]:
    variants={}
    for condition in score:
        with np.load(directory/f'seed{seed}_{condition}.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
        base=data['recentered'].reshape(-1,17,3);target=data['target_mm'];ids=data['subject_ids']
        pose=poses['clean' if condition.endswith('missing') else condition].copy()
        if condition=='video_missing':pose[...,2]=0
        for conf in [.7,.8,.9]:
            eligible=(pose[...,2]>=conf)&(pose[:,11,None,2]>=conf)&(pose[:,12,None,2]>=conf)
            for cap in [50.,100.,float('inf')]:
                for source in ['constant_depth','radar_depth']:
                    correction=data[source].reshape(-1,17,3)-base
                    keep=eligible&(np.linalg.norm(correction,axis=-1)<=cap)
                    prediction=base+np.where(keep[...,None],correction,0.)
                    name=f'{source}_conf{conf:g}_cap{cap:g}'
                    metric=independent_metrics(prediction.reshape(-1,51),target,ids)
                    variants.setdefault(name,{})[condition]=metric
    for name,m in variants.items():
        rows.append({'seed':seed,'method':name,'clean_mm':m['clean']['MPJPE_mm'],
            'S_mm':float(np.mean([m[c]['per_validation_subject_MPJPE_mm'][str(s)] for c in score for s in subjects]))})
aggregate={}
for name in sorted({r['method'] for r in rows}):
    r=[x for x in rows if x['method']==name]
    aggregate[name]={'clean_mm':float(np.mean([v['clean_mm'] for v in r])), 'S_mm':float(np.mean([v['S_mm'] for v in r]))}
with (directory/'fixed_quality_controls.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(directory/'fixed_quality_controls.json').write_text(json.dumps({'status':'complete','scope':'All 18 prespecified ordinary quality/correction-size controls are exploratory, no novelty or blind validation',
    'settings':{'confidence':[.7,.8,.9],'correction_cap_mm':[50,100,'unbounded']},'aggregate':aggregate,'per_initialization':rows,'test_subjects_used':[]},indent=2))
print(json.dumps(aggregate))
