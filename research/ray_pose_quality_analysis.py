"""Additional evaluation only: aligned pose, PCK and bone errors, all methods."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'artifacts/ray_oof_research_20261016'
CTRL=ROOT/'artifacts/ray_oof_strict_controls_20261016'
CONDS=['clean','dark','noisy_dark','occlusion','video_missing','radar_missing']
BONES=[(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def procrustes(pred,target):
    mx,my=pred.mean(1,keepdims=True),target.mean(1,keepdims=True)
    x,y=pred-mx,target-my
    nx=np.sqrt((x*x).sum((1,2),keepdims=True));ny=np.sqrt((y*y).sum((1,2),keepdims=True))
    nx,ny=np.maximum(nx,1e-12),np.maximum(ny,1e-12)
    h=(x/nx).swapaxes(1,2)@(y/ny)
    u,s,vt=np.linalg.svd(h)
    sign=np.where(np.linalg.det(u@vt)<0,-1.,1.)
    u[:,:,-1]*=sign[:,None];s[:,-1]*=sign
    rotation=u@vt;scale=s.sum(1)[:,None,None]*ny/nx
    return scale*(x@rotation)+my

def verify_alignment():
    rng=np.random.default_rng(42);target=rng.normal(size=(4,17,3))*200
    c,s=np.cos(.65),np.sin(.65);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pred=target@rotation*1.7+np.array([50.,70.,-40.])
    assert np.max(abs(procrustes(pred,target)-target))<1e-9
    mirror=target*np.array([-1.,1.,1.])
    assert np.linalg.norm(procrustes(mirror,target)-target,axis=-1).mean()>1

def metric(pred,target):
    error=np.linalg.norm(pred-target,axis=-1)
    centred=pred-pred[:,[11,12]].mean(1)[:,None]
    pa=procrustes(pred,target)
    bone=np.stack([abs(np.linalg.norm(pred[:,a]-pred[:,b],axis=-1)-np.linalg.norm(target[:,a]-target[:,b],axis=-1)) for a,b in BONES],axis=-1)
    return {'MPJPE_mm':float(error.mean()),'hip_recentered_MPJPE_mm':float(np.linalg.norm(centred-target,axis=-1).mean()),
            'PA_MPJPE_mm':float(np.linalg.norm(pa-target,axis=-1).mean()),'PCK50_pct':float((error<=50).mean()*100),
            'PCK100_pct':float((error<=100).mean()*100),'bone_length_MAE_mm':float(bone.mean())}

def main():
    verify_alignment()
    assert json.loads((SRC/'audit.json').read_text())['status']=='complete' and json.loads((CTRL/'audit.json').read_text())['status']=='complete'
    records=[]
    for seed in [7,17,27]:
        sd=SRC/f'seed{seed}'
        with np.load(sd/'validation_payload.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
        predictions={'recentered_base':data['base']*1000}
        for method in ['radar_ray_gate','constant_ray_gate','free_radar_residual']:
            with np.load(sd/method/'predictions.npz',allow_pickle=False) as f:predictions[method]=f['prediction_mm'].reshape(-1,17,3)
        for method in ['free_constant_residual','free_no_ray_residual']:
            with np.load(CTRL/f'seed{seed}'/method/'predictions.npz',allow_pickle=False) as f:predictions[method]=f['prediction_mm'].reshape(-1,17,3)
        for method,pred in predictions.items():
            for ci,c in enumerate(CONDS):
                keep=data['condition_ids']==ci;records.append({'seed':seed,'method':method,'condition':c,**metric(pred[keep],data['target'][keep]*1000)})
    clean={}
    for method in predictions:
        rows=[r for r in records if r['method']==method and r['condition']=='clean']
        clean[method]={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('seed','method','condition')}
    result={'status':'complete','scope':'Supplementary post-hoc metrics on the same predictions; no checkpoint selection/fitting. PA-MPJPE is proper-rotation Procrustes with scale/translation, not ordinary MPJPE.',
            'alignment_checks':'Known rigid rotation+scale+translation recovered <1e-9mm; reflections not permitted',
            'clean':clean,'per_initialization_condition':records,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_subjects_used':[]}
    (CTRL/'reporting/pose_quality.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# 额外姿态质量评价','','所有指标只复用已审计预测；PA-MPJPE与普通MPJPE不同，不混作论文原表排名。', '',
        '| 方法 | 髋重中心MPJPE mm | PA-MPJPE mm | PCK50 % | PCK100 % | 骨长MAE mm |','|---|---:|---:|---:|---:|---:|']
    for name,r in clean.items():lines.append(f"| {name} | {r['hip_recentered_MPJPE_mm']:.4f} | {r['PA_MPJPE_mm']:.4f} | {r['PCK50_pct']:.3f} | {r['PCK100_pct']:.3f} | {r['bone_length_MAE_mm']:.4f} |")
    lines+=['','12条固定COCO骨边；PCK按欧氏距离<=50/100mm统计。全部为旧1536帧开发协议、三个初始化的均值，最终测试未评价。骨长一致性不是临床有效性或独立动捕准确度保证。']
    (CTRL/'reporting/pose_quality.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(clean))

if __name__=='__main__':main()
