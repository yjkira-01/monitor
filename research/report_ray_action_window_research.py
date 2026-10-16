"""Large-data geometric OOF results beside all audited primary baselines."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from analyze_fusion_trials import independent_metrics
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'artifacts/ray_oof_action_windows_20261016'
MAIN=ROOT/'artifacts/action_window_main_20261016'
CONDS=['clean','noisy_dark','occlusion','video_missing','radar_missing']

def main():
    summary=json.loads((DEST/'summary.json').read_text());audit=json.loads((DEST/'audit.json').read_text())
    base_summary=json.loads((MAIN/'summary.json').read_text());base_audit=json.loads((MAIN/'audit.json').read_text())
    assert summary['status']==audit['status']==base_summary['status']==base_audit['status']=='complete'
    records=[];subject_metrics={}
    for run in summary['runs']:
        seed=run['seed'];sd=DEST/f'seed{seed}'
        with np.load(sd/'validation_payload.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
        predictions={'recentered_concat':data['base'].reshape(-1,51)*1000}
        for name in run['models']:
            with np.load(sd/name/'predictions.npz',allow_pickle=False) as f:predictions[name]=f['prediction_mm']
        for name,pred in predictions.items():
            metrics={}
            for ci,c in enumerate(CONDS):
                keep=data['condition_ids']==ci
                metrics[c]=independent_metrics(pred[keep],data['target'][keep].reshape(-1,51)*1000,data['subject_ids'][keep])
            score=float(np.mean([v for c in CONDS for v in metrics[c]['per_validation_subject_MPJPE_mm'].values()]))
            records.append({'seed':seed,'method':name,'clean_mm':metrics['clean']['MPJPE_mm'],'S_mm':score,'p95_clean_mm':metrics['clean']['frame_error_p95_mm']})
            subject_metrics.setdefault(name,{})[str(seed)]={s:float(np.mean([metrics[c]['per_validation_subject_MPJPE_mm'][s] for c in CONDS])) for s in metrics['clean']['per_validation_subject_MPJPE_mm']}
    for run in base_summary['runs']:
        seed=run['seed']
        for stage in ('base','temporal'):
            for name,row in run[stage].items():
                metrics=row['metrics'];score=float(np.mean([v for c in CONDS for v in metrics[c]['per_validation_subject_MPJPE_mm'].values()]))
                records.append({'seed':seed,'method':name,'clean_mm':metrics['clean']['MPJPE_mm'],'S_mm':score,'p95_clean_mm':metrics['clean']['frame_error_p95_mm']})
                subject_metrics.setdefault(name,{})[str(seed)]={s:float(np.mean([metrics[c]['per_validation_subject_MPJPE_mm'][s] for c in CONDS])) for s in metrics['clean']['per_validation_subject_MPJPE_mm']}
    aggregate={}
    for name in sorted({r['method'] for r in records}):
        rows=[r for r in records if r['method']==name]
        aggregate[name]={k:{'mean':float(np.mean([r[k] for r in rows])),'initialization_sd':float(np.std([r[k] for r in rows],ddof=1))} for k in ['clean_mm','S_mm','p95_clean_mm']}
    candidate='free_radar_residual';comparisons={}
    for name in ['recentered_concat','free_constant_residual','free_no_ray_residual','awc_concat','gru','temporal_mlp']:
        a,b=aggregate[name],aggregate[candidate];gain=a['S_mm']['mean']-b['S_mm']['mean'];clean=b['clean_mm']['mean']-a['clean_mm']['mean']
        subjects=list(subject_metrics[name]['7']);sg={s:float(np.mean([subject_metrics[name][str(seed)][s]-subject_metrics[candidate][str(seed)][s] for seed in [7,17,27]])) for s in subjects}
        seed_gain=[float(np.mean([subject_metrics[name][str(seed)][s]-subject_metrics[candidate][str(seed)][s] for s in subjects])) for seed in [7,17,27]]
        comparisons[name]={'gain_S_mm':gain,'clean_change_mm':clean,'per_subject_S_gain_mm':sg,'per_initialization_S_gain_mm':seed_gain,
            'resource_gate':bool(gain>=max(1.,.02*a['S_mm']['mean']) and clean<=max(1.,.02*a['clean_mm']['mean']) and sum(v>0 for v in sg.values())>=3 and sum(v>0 for v in seed_gain)>=2)}
    result={'status':'complete','scope':'Large action-covered development validation; not final test or novelty proof',
        'train_frames':9216,'validation_frames':3072,'train_subjects':[1,2,4,7,8,9,11,13,14,15,16,17],'validation_subjects':[5,6,19,20],
        'aggregate':aggregate,'comparisons_free_radar':comparisons,'per_initialization':records,'test_subjects_used':[],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out=DEST/'reporting';out.mkdir(exist_ok=True);(out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'per_initialization.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# 完整连续动作协议：几何校正与强基线','',
        '16人×12动作×64帧共12288帧，12训练人9216帧、4开发验证人3072帧。全部选定有限发布标签为主人口，含少量极端错误标签；筛查敏感性见主基线报告。一个主体划分、三个初始化，SD不表示独立划分。', '',
        '| 方法 | clean MPJPE mm | S mm | clean P95 mm |','|---|---:|---:|---:|']
    for name,r in aggregate.items():lines.append(f"| {name} | {r['clean_mm']['mean']:.4f} ± {r['clean_mm']['initialization_sd']:.4f} | {r['S_mm']['mean']:.4f} ± {r['S_mm']['initialization_sd']:.4f} | {r['p95_clean_mm']['mean']:.4f} |")
    lines+=['', '| 自由雷达几何残差的对照 | S改善mm | clean变化mm | 投入门槛 |','|---|---:|---:|---|']
    for name,c in comparisons.items():lines.append(f"| {name} | {c['gain_S_mm']:+.4f} | {c['clean_change_mm']:+.4f} | {'通过' if c['resource_gate'] else '未通过'} |")
    lines+=['', '所有几何校正模型重新使用本协议的训练主体、留出专家和训练内距离模型；未混入旧小数据校正头。AWC为原文训练规则的本地2模态架构重建。所有方法统一稳健主损失/中位数模板，另保留MSE损失控制。', '',
        '各校正头共享相同OOF训练数据、初始化/预算/参数量；常深度及无射线头仍可访问相同估计距离与置信特征。基本方法没有额外OOF训练阶段，因此与普通网络比较是整流程比较，不声称更新次数相等。时序控制有9帧历史，几何头为单帧。', '',
        'S为五条件主体宏平均，clean为正常图像。最终[3,10,12,18]未评价。已知投影、堆叠与残差校准均有先例；真实开发误差下降不自动证明新颖性或发表性。若不胜最强同信息控制，必须保留该负结果，不只与较弱X-Fi适配比较。']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'status':'complete','aggregate':aggregate,'comparisons':comparisons}))

if __name__=='__main__':main()
