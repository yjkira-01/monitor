"""Subject-level paired reporting of audited action-window experiments."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
SCORE=['clean','noisy_dark','occlusion','video_missing','radar_missing']

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    proto=json.loads((args.directory/'protocol.json').read_text());summary=json.loads((args.directory/'summary.json').read_text())
    audit=json.loads((args.directory/'audit.json').read_text())
    assert summary['status']=='complete' and audit['status']=='complete' and not proto['pilot']
    assert not summary['test_subjects_used']
    rows=[];subjects={};counts={};parameters={}
    for run in summary['runs']:
        data=json.loads((args.directory/f"fold{run['fold']}"/'data.json').read_text())
        counts[str(run['fold'])]=data
        for stage in ('base','temporal'):
            for method,r in run[stage].items():
                m=r['metrics'];seed=run['seed'];fold=run['fold'];val=data['validation_subjects']
                score=statistics.mean(m[c]['per_validation_subject_MPJPE_mm'][str(s)] for c in SCORE for s in val)
                row={'fold':fold,'seed':seed,'method':method,'stage':stage,'clean_mm':m['clean']['MPJPE_mm'],'S_mm':score,
                     'screened_clean_mm':m['clean']['screened_MPJPE_mm'],'hip_recentered_clean_mm':m['clean']['hip_recentered_MPJPE_mm'],
                     'p95_clean_mm':m['clean']['frame_error_p95_mm'],'parameters':r['parameters'],'training_seconds':r['seconds']}
                row.update({c+'_mm':m[c]['MPJPE_mm'] for c in SCORE});rows.append(row);parameters[method]=r['parameters']
                subjects.setdefault(method,{})
                for s in val:
                    if str(s) not in subjects[method]:subjects[method][str(s)]={}
                    subjects[method][str(s)][str(seed)]={c:m[c]['per_validation_subject_MPJPE_mm'][str(s)] for c in SCORE}
    methods={}
    seeds=list(map(str,proto['seeds']))
    for method,by_subject in subjects.items():
        subject_score={s:statistics.mean(d[c] for d in scores.values() for c in SCORE) for s,scores in by_subject.items()}
        subject_clean={s:statistics.mean(d['clean'] for d in scores.values()) for s,scores in by_subject.items()}
        seed_score=[statistics.mean(by_subject[s][seed][c] for s in by_subject for c in SCORE) for seed in seeds]
        seed_clean=[statistics.mean(by_subject[s][seed]['clean'] for s in by_subject) for seed in seeds]
        selected=[r for r in rows if r['method']==method]
        methods[method]={'S_mm':statistics.mean(subject_score.values()),'clean_mm':statistics.mean(subject_clean.values()),
            'S_initialization_sd_mm':statistics.stdev(seed_score) if len(seed_score)>1 else None,
            'clean_initialization_sd_mm':statistics.stdev(seed_clean) if len(seed_clean)>1 else None,
            'per_subject_S_mm':subject_score,'per_subject_clean_mm':subject_clean,
            'per_seed_S_mm':dict(zip(seeds,seed_score)),'per_seed_clean_mm':dict(zip(seeds,seed_clean)),
            'screened_clean_mean_mm':statistics.mean(r['screened_clean_mm'] for r in selected),
            'hip_recentered_clean_mean_mm':statistics.mean(r['hip_recentered_clean_mm'] for r in selected),
            'p95_clean_mean_mm':statistics.mean(r['p95_clean_mm'] for r in selected),'parameters':parameters[method]}
    comparisons={}
    candidate='anchor_transport'
    controls=[m for m in methods if m not in ('rgb_single','radar_single',candidate)]
    if candidate in methods:
        cand=methods[candidate]
        for name in controls:
            base=methods[name]
            differences=np.array([base['per_subject_S_mm'][s]-cand['per_subject_S_mm'][s] for s in cand['per_subject_S_mm']])
            rng=np.random.default_rng(20261016)
            boot=differences[rng.integers(0,len(differences),(10000,len(differences)))].mean(1)
            gain=base['S_mm']-cand['S_mm'];clean=cand['clean_mm']-base['clean_mm']
            seed_diff=[base['per_seed_S_mm'][s]-cand['per_seed_S_mm'][s] for s in seeds]
            comparisons[name]={'gain_S_mm':gain,'clean_change_mm':clean,'per_subject_gain_mm':differences.tolist(),
                'positive_subjects':int((differences>0).sum()),'positive_initializations':sum(x>0 for x in seed_diff),
                'exploratory_subject_bootstrap_95pct_mm':np.percentile(boot,[2.5,97.5]).tolist(),
                'resource_gate':bool(gain>=max(1.,.02*base['S_mm']) and clean<=max(1.,.02*base['clean_mm'])
                    and (differences>0).sum()>len(differences)/2 and sum(x>0 for x in seed_diff)>=2)}
    best=min(controls,key=lambda m:methods[m]['S_mm']) if controls else None
    result={'status':'complete','scope':'Development cross-subject experiments, not untouched test or novelty verification',
        'folds':proto['folds'],'initialization_seeds':proto['seeds'],'data_by_fold':counts,'methods':methods,
        'candidate':candidate,'strongest_control_by_development_S':best,'comparisons':comparisons,
        'test_subjects_used':[],'audit_sha256':hashlib.sha256((args.directory/'audit.json').read_bytes()).hexdigest(),
        'summary_sha256':hashlib.sha256((args.directory/'summary.json').read_bytes()).hexdigest(),
        'reporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out=args.directory/'reporting';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'metrics_by_fold_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 连续动作开发实验：真实结果','',
        f"验证折{proto['folds']}；初始化{proto['seeds']}。S为五条件、受试者等权平均；SD只表示初始化波动，不是独立划分误差或置信区间。",'',
        '| 方法 | clean主体宏 / mm | S / mm | screened clean / mm | 参数量 |',
        '|---|---:|---:|---:|---:|']
    for name,v in methods.items():lines.append(f"| {name} | {v['clean_mm']:.4f} | {v['S_mm']:.4f} | {v['screened_clean_mean_mm']:.4f} | {v['parameters']:,} |")
    if candidate in methods:
        lines+=['',f"候选相对开发S最强必要控制`{best}`的S改善为{comparisons[best]['gain_S_mm']:+.4f}mm，clean变化{comparisons[best]['clean_change_mm']:+.4f}mm，预设投入门槛{'通过' if comparisons[best]['resource_gate'] else '未通过'}。",'',
            '投入门槛仅用于选择是否扩大研究，不是显著性或可发表性判据。所有方法和异常标签均保留；不得只挑有利条件/主体/初始化。']
    lines+=['', 'AWC为原文训练规则的2模态本地架构重建，非官方MM-Fi完整网络。X-Fi强输入核心的1536帧结果在单独报告中，不能与本表不同人口/损失的数值差作为算法提升。', '',
        '全部有限发布标签为主人口，包含极少数物理不可能的大偏差；screened为预先固定1m最大四肢骨长/3m根相对半径的敏感性报告。发布3D标签由双相机导出，不是独立动捕。', '',
        '全部方法共用训练坐标中位数模板与稳健姿态损失（另保留MSE控制）；全部时序头共用同一个冻结concat预训练。历史9帧不跨主体/动作窗口；缺失模态在整个历史清除。发布雷达窗口可能含下一相机帧点，不能声称整系统严格因果。', '',
        '最终保留主体[3,10,12,18]未使用。主观测者统计单位为人；开发数据已被反复探索，本报告不能充当最终盲测。']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'methods':{k:{'clean_mm':v['clean_mm'],'S_mm':v['S_mm']} for k,v in methods.items()},'strongest_control':best,
                      'candidate_gate':comparisons.get(best)}))

if __name__=='__main__':main()
