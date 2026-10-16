"""Independent NumPy replay and paired reporting of ordinary ray controls."""
import csv,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
import numpy as np
from run_ray_oof_research import features,CONDS
from analyze_fusion_trials import independent_metrics
SRC=ROOT/'artifacts/ray_oof_research_20261016';DEST=ROOT/'artifacts/ray_oof_strict_controls_20261016'
SCORE=['clean','noisy_dark','occlusion','video_missing','radar_missing']
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def main():
    summary=json.loads((DEST/'summary.json').read_text());assert summary['status']=='complete' and len(summary['runs'])==6
    assert sha(SRC/'summary.json')==summary['source_summary_sha256'] and sha(SRC/'audit.json')==summary['source_audit_sha256']
    for n,h in summary['source_sha256'].items():assert sha(ROOT/'research'/n)==h and sha(DEST/'source_snapshot'/n)==h
    checks=[];rows=[]
    for r in summary['runs']:
        seed,method=r['seed'],r['method'];d=DEST/f'seed{seed}'/method
        assert sha(d/'weights.npz')==r['weights_sha256'] and sha(d/'predictions.npz')==r['predictions_sha256']
        assert sha(SRC/f'seed{seed}/oof_payload.npz')==r['OOF_payload_sha256'] and sha(SRC/f'seed{seed}/validation_payload.npz')==r['validation_payload_sha256']
        data=load(SRC/f'seed{seed}/validation_payload.npz')
        if method=='free_no_ray_residual':data['constant_candidate']=data['base'].copy()
        x=features(data,'constant_candidate').numpy().astype(float);w=load(d/'weights.npz')
        x=np.concatenate([x,np.broadcast_to(w['joint'],(len(x),17,16))],axis=-1)
        hidden=np.maximum(x@w['network.0.weight'].T+w['network.0.bias'],0)
        predicted=(data['base']+.1*np.tanh(hidden@w['network.2.weight'].T+w['network.2.bias'])*data['eligible'][...,None]).reshape(-1,51)*1000
        saved=load(d/'predictions.npz')['prediction_mm'];difference=float(np.max(abs(predicted-saved)));assert difference<=.01
        metrics={}
        for ci,c in enumerate(CONDS):
            keep=data['condition_ids']==ci
            metrics[c]=independent_metrics(saved[keep],data['target'][keep].reshape(-1,51)*1000,data['subject_ids'][keep])
            assert abs(metrics[c]['MPJPE_mm']-r['metrics'][c]['MPJPE_mm'])<1e-8
        rows.append({'seed':seed,'method':method,'clean_mm':metrics['clean']['MPJPE_mm'],
                     'S_mm':float(np.mean([v for c in SCORE for v in metrics[c]['per_validation_subject_MPJPE_mm'].values()]))})
        checks.append({'seed':seed,'method':method,'NumPy_replay_max_mm':difference})
    prior=json.loads((SRC/'reporting/results.json').read_text())['per_initialization']
    rows+=prior
    aggregate={}
    for method in sorted({r['method'] for r in rows}):
        selected=[r for r in rows if r['method']==method]
        aggregate[method]={k:{'mean':float(np.mean([r[k] for r in selected])),'initialization_sd':float(np.std([r[k] for r in selected],ddof=1))} for k in ['clean_mm','S_mm']}
    receipt={'status':'complete','models':6,'checks':checks,'source_sha256':sha(Path(__file__)),'test_subjects_used':[]}
    (DEST/'audit.json').write_text(json.dumps(receipt,indent=2));(DEST/'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    out=DEST/'reporting';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps({'status':'complete','aggregate':aggregate,'per_initialization':rows,'test_subjects_used':[]},indent=2))
    lines=['# 同容量几何控制的完整结果','','| 方法 | clean mm | S mm |','|---|---:|---:|']
    for method,r in aggregate.items():lines.append(f"| {method} | {r['clean_mm']['mean']:.6f} ± {r['clean_mm']['initialization_sd']:.6f} | {r['S_mm']['mean']:.6f} ± {r['S_mm']['initialization_sd']:.6f} |")
    lines+=['','所有自由残差头同初始化、同输入维数、同容量、同OOF帧、同80轮预算。free_constant将几何算子根深度固定，但仍可访问同样的观测雷达距离特征；free_no_ray只移除显式几何候选，保留基预测、置信度和估计距离。不能把它们称为完全去掉雷达的模型。', '',
        '原自由雷达射线模型若只胜无几何头而不胜常深度头，支持几何信息的作用，但不足以支持雷达距离具有大的独立贡献。此处是开发结果，一划分三初始化；不是最终盲测或新颖性证明。']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','aggregate':aggregate}))

if __name__=='__main__':main()
