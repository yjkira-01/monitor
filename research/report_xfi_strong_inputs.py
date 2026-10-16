"""Compare completed, audited strong-input X-Fi core with frozen concat."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
directory=ROOT/'artifacts/xfi_strong_inputs_resumed_20261016'
summary=json.loads((directory/'summary.json').read_text());audit=json.loads((directory/'audit.json').read_text())
assert summary['status']=='complete' and audit['status']=='complete' and audit['checked_models']==3
oldpath=ROOT/'artifacts/fusion_method_research_20261013_expanded_centroid/run_summary.json'
old=json.loads(oldpath.read_text());assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==summary['old_summary_sha256']
score=['clean','noisy_dark','occlusion','video_missing','radar_missing']
rows=[]
for run in summary['runs']:
    reference=next(r for r in old['runs'] if r['seed']==run['seed'])['models']['concat']['metrics']
    for method,metrics in [('concat',reference),('xfi_core_strong',run['metrics'])]:
        row={'seed':run['seed'],'method':method,'clean_MPJPE_mm':metrics['clean']['MPJPE_mm'],
             'S_mm':statistics.mean(metrics[c]['per_validation_subject_MPJPE_mm'][str(s)] for c in score for s in summary['validation_subjects'])}
        row.update({c+'_MPJPE_mm':metrics[c]['MPJPE_mm'] for c in run['metrics']})
        rows.append(row)
result={'status':'complete','scope':summary['scope'],'final_test_subjects_used':[],'per_initialization':rows,'methods':{}}
for method in ('concat','xfi_core_strong'):
    r=[x for x in rows if x['method']==method]
    result['methods'][method]={k:{'mean':statistics.mean(x[k] for x in r),'initialization_sd':statistics.stdev(x[k] for x in r)} for k in ('clean_MPJPE_mm','S_mm')}
base,new=result['methods']['concat'],result['methods']['xfi_core_strong']
result['xfi_minus_concat_mm']={k:new[k]['mean']-base[k]['mean'] for k in base}
out=directory/'reporting';out.mkdir(exist_ok=True)
(out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (out/'per_initialization.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# 同强输入 X-Fi 融合核心对照','',
 '三个初始化、同1536个开发帧、同RTMO预测/雷达输入、归一化、80轮末轮、增强与辅助损失。学习率X-Fi核心为0.0001、旧concat为0.001；架构/参数量不同，不声称训练超参和容量完全相同。',
 '', '| 方法 | clean MPJPE / mm | 五条件主体宏S / mm |','|---|---:|---:|']
for method in ('concat','xfi_core_strong'):
    r=result['methods'][method]
    lines.append(f"| {method} | {r['clean_MPJPE_mm']['mean']:.6f} ± {r['clean_MPJPE_mm']['initialization_sd']:.6f} | {r['S_mm']['mean']:.6f} ± {r['S_mm']['initialization_sd']:.6f} |")
lines+=['',f"X-Fi核心相对concat的误差变化：clean {result['xfi_minus_concat_mm']['clean_MPJPE_mm']:+.6f}mm，S {result['xfi_minus_concat_mm']['S_mm']:+.6f}mm。正值代表误差较高。",'',
 '这是X-Fi原生核心的mRI强输入适配：原始RGB CNN/mmWave骨干改为本地强流程编码器，128维池化特征投影为512维并加32位置token，保留作者原X_Fusion定义的512维/32token/8头/4层/五槽编号；缺失回退与独立辅助头按本地强流程。不能称原论文MM-Fi复现、完整官方网络，或将数值差归因为单一模块。',
 '', 'seed7经SHA验证直接复用；只新训seed17/27。全部三套已通过独立CUDA保存状态重建、NumPy指标复算和缺失回退核验；跨CPU/GPU0.3404mm历史限制仍保留，未声称CPU全链通过。',
 '', '一个主体划分，三个初始化；四个验证主体，不是三个独立数据划分。最终保留主体未评价。新12288帧连续数据协议不同，应另表比较，不能直接与本表作算法提升差分。']
(out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result['methods']));print(json.dumps(result['xfi_minus_concat_mm']))
