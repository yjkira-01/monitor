"""Paired subject-level report; useful accuracy is separate from novelty."""
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
from analyze_fusion_trials import independent_metrics
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'artifacts/ray_oof_research_20261016'
SCORE=['clean','noisy_dark','occlusion','video_missing','radar_missing']
CONDS=['clean','dark','noisy_dark','occlusion','video_missing','radar_missing']

def main():
    summary=json.loads((DEST/'summary.json').read_text());audit=json.loads((DEST/'audit.json').read_text())
    assert summary['status']=='complete' and audit['status']=='complete'
    records=[];by_subject={}
    for run in summary['runs']:
        seed=run['seed'];sd=DEST/f'seed{seed}'
        with np.load(sd/'validation_payload.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
        predictions={'recentered_base':data['base'].reshape(-1,51)*1000}
        for name in run['models']:
            with np.load(sd/name/'predictions.npz',allow_pickle=False) as f:predictions[name]=f['prediction_mm']
        for name,pred in predictions.items():
            m={}
            for ci,c in enumerate(CONDS):
                keep=data['condition_ids']==ci
                m[c]=independent_metrics(pred[keep],data['target'][keep].reshape(-1,51)*1000,data['subject_ids'][keep])
            score=float(np.mean([v for c in SCORE for v in m[c]['per_validation_subject_MPJPE_mm'].values()]))
            row={'seed':seed,'method':name,'clean_mm':m['clean']['MPJPE_mm'],'S_mm':score,
                 'p95_clean_mm':m['clean']['frame_error_p95_mm']}
            row.update({c+'_mm':m[c]['MPJPE_mm'] for c in CONDS});records.append(row)
            for s in np.unique(data['subject_ids']):
                by_subject.setdefault(name,{}).setdefault(str(int(s)),{})[str(seed)]={
                    'S_mm':float(np.mean([m[c]['per_validation_subject_MPJPE_mm'][str(s)] for c in SCORE])),
                    'clean_mm':m['clean']['per_validation_subject_MPJPE_mm'][str(s)]}
    aggregate={}
    for name,values in by_subject.items():
        selected=[r for r in records if r['method']==name]
        aggregate[name]={k:{'mean':float(np.mean([r[k] for r in selected])), 'initialization_sd':float(np.std([r[k] for r in selected],ddof=1))} for k in ['clean_mm','S_mm','p95_clean_mm']}
        aggregate[name]['per_subject']={s:{k:float(np.mean([v[k] for v in seeds.values()])) for k in ['clean_mm','S_mm']} for s,seeds in values.items()}
    comparisons={}
    for control in ['recentered_base','constant_ray_gate','free_radar_residual']:
        a,b=aggregate[control],aggregate['radar_ray_gate']
        gain=a['S_mm']['mean']-b['S_mm']['mean'];clean=b['clean_mm']['mean']-a['clean_mm']['mean']
        subject_gain={s:a['per_subject'][s]['S_mm']-b['per_subject'][s]['S_mm'] for s in a['per_subject']}
        rng=np.random.default_rng(20261016);v=np.array(list(subject_gain.values()));boot=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
        seed_gain=[]
        for seed in [7,17,27]:
            ar=next(r for r in records if r['seed']==seed and r['method']==control)
            br=next(r for r in records if r['seed']==seed and r['method']=='radar_ray_gate')
            seed_gain.append(ar['S_mm']-br['S_mm'])
        comparisons[control]={'S_gain_mm':gain,'clean_change_mm':clean,'per_subject_S_gain_mm':subject_gain,
            'per_initialization_S_gain_mm':seed_gain,'exploratory_subject_bootstrap_95pct_mm':np.percentile(boot,[2.5,97.5]).tolist(),
            'resource_gate':bool(gain>=max(1.,.02*a['S_mm']['mean']) and clean<=max(1.,.02*a['clean_mm']['mean'])
                and sum(x>0 for x in subject_gain.values())>=3 and sum(x>0 for x in seed_gain)>=2)}
    result={'status':'complete','aggregate':aggregate,'per_initialization':records,'comparisons_radar_gate':comparisons,
        'test_subjects_used':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'summary_sha256':hashlib.sha256((DEST/'summary.json').read_bytes()).hexdigest(),
        'audit_sha256':hashlib.sha256((DEST/'audit.json').read_bytes()).hexdigest()}
    out=DEST/'reporting';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'per_initialization.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# 留出预测训练的几何融合：三初始化真实结果','',
        '同原1536帧开发集，一个主体划分，三个初始化。12训练人内三折8train/4heldout产生OOF预测及8人拟合的雷达深度，再训练小型校正器；评价使用未改动的full12强concat权重。未知GT只在训练目标使用，不进入推理。', '',
        '| 方法 | clean MPJPE mm | 五条件S mm | clean P95 mm |','|---|---:|---:|---:|']
    for name,a in aggregate.items():lines.append(f"| {name} | {a['clean_mm']['mean']:.6f} ± {a['clean_mm']['initialization_sd']:.6f} | {a['S_mm']['mean']:.6f} ± {a['S_mm']['initialization_sd']:.6f} | {a['p95_clean_mm']['mean']:.4f} |")
    lines+=['','| radar_ray_gate 对照 | S改善mm | clean变化mm | 预设投入门槛 |','|---|---:|---:|---|']
    for control,c in comparisons.items():lines.append(f"| {control} | {c['S_gain_mm']:+.6f} | {c['clean_change_mm']:+.6f} | {'通过' if c['resource_gate'] else '未通过'} |")
    lines+=['', '门槛是开发资源决策，不是统计显著性或可发表性。若仅胜旧普通融合、但不胜常深度与同特征自由残差控制，不能归因为新的雷达几何约束贡献。', '',
        'constant_ray_gate使用相同15个观测输入特征（包括估计雷达距离），但几何候选的根深度固定为训练中位数；因此它是更强的算子对照，不是严格雷达无输入模型。自由残差控制看到同样的物理候选与质量特征，参数近同但可输出自由XYZ修正。', '',
        '已知相机投影、IRLS、OOF堆叠与门控均为既有工具。已有ICEET2023的全向相机＋雷达三维姿态先例，全文尚未取得，不能声称首次以雷达解单目尺度。此报告证明实际误差变化与控制差异，不证明新颖性。', '',
        'OOF8人模型与部署full12模型存在误差分布偏移；开发主体已多次探索，四名最终保留主体未评价。所有方法共用强输入与基模型；几何方法额外使用既有RTMO原始2D位置与静态K/R标定，其作用必须与普通神经融合和同信息控制区分。', '',
        '独立审计验证OOF主体分离、训练内深度拟合、全部专家观测重建、九个校正器CUDA及NumPy重建、指标复算和target污染不影响推理特征。']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'aggregate':{n:{k:a[k] for k in ['clean_mm','S_mm']} for n,a in aggregate.items()},'comparisons':comparisons}))

if __name__=='__main__':main()
