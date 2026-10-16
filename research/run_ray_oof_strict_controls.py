"""Same-capacity controls on the saved OOF data, no expert retraining."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from ray_gate_models import RaySelector
from run_ray_oof_research import features,CONDS
from analyze_fusion_trials import independent_metrics
SRC=ROOT/'artifacts/ray_oof_research_20261016'
DEST=ROOT/'artifacts/ray_oof_strict_controls_20261016'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def save(path,v):Path(path).write_text(json.dumps(v,indent=2),encoding='utf-8')

def main():
    assert json.loads((SRC/'audit.json').read_text())['status']=='complete'
    if DEST.exists():raise ValueError('New output required')
    DEST.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    summary={'status':'running','source_summary_sha256':sha(SRC/'summary.json'),'source_audit_sha256':sha(SRC/'audit.json'),
        'methods':['free_constant_residual','free_no_ray_residual'],'seeds':[7,17,27],
        'budget':'Same initializer(seed+30000), 80ep/batch256/order(seed)/AdamW.001 wd.0001/MPJPE as original free_radar_residual',
        'scope':'Both controls retain identical observed scalar radar depth/confidences and base predictions. Only explicit geometric candidate features differ; no final-test access.',
        'source_sha256':{n:sha(ROOT/'research'/n) for n in [Path(__file__).name,'ray_gate_models.py','run_ray_oof_research.py']},'test_subjects_used':[],'runs':[]}
    snap=DEST/'source_snapshot';snap.mkdir()
    for n in summary['source_sha256']:(snap/n).write_bytes((ROOT/'research'/n).read_bytes())
    save(DEST/'summary.json',summary)
    for seed in [7,17,27]:
        sd=DEST/f'seed{seed}';sd.mkdir()
        original=load(SRC/f'seed{seed}/oof_payload.npz');original_val=load(SRC/f'seed{seed}/validation_payload.npz')
        for method in summary['methods']:
            data=dict(original);validation=dict(original_val)
            source='constant_candidate'
            if method=='free_no_ray_residual':
                data[source]=data['base'].copy();validation[source]=validation['base'].copy()
            x=features(data,source);vx=features(validation,source)
            tensors={k:torch.tensor(data[k],dtype=torch.bool if k=='eligible' else torch.float32) for k in ['base',source,'eligible','target']}
            torch.manual_seed(seed+30000);model=RaySelector(True).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            order=torch.Generator().manual_seed(seed);history=[]
            for epoch in range(1,81):
                model.train();losses=[]
                for ix in torch.randperm(len(x),generator=order).split(256):
                    opt.zero_grad(set_to_none=True)
                    prediction=model(tensors['base'][ix].cuda(),tensors[source][ix].cuda(),x[ix].cuda(),tensors['eligible'][ix].cuda())
                    loss=(prediction-tensors['target'][ix].cuda()).norm(dim=-1).mean();assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.item())
                if epoch==1 or epoch%20==0:
                    h={'epoch':epoch,'loss':float(np.mean(losses))};history.append(h);print(json.dumps({'seed':seed,'method':method,**h}),flush=True)
            model.eval();pred=[]
            with torch.no_grad():
                for start in range(0,len(vx),256):
                    sl=slice(start,start+256)
                    v=model(torch.tensor(validation['base'][sl],dtype=torch.float32,device='cuda'),
                        torch.tensor(validation[source][sl],dtype=torch.float32,device='cuda'),vx[sl].cuda(),
                        torch.tensor(validation['eligible'][sl],dtype=torch.bool,device='cuda'))
                    pred.append(v.cpu().numpy()*1000)
            pred=np.concatenate(pred).reshape(-1,51);metrics={}
            for ci,c in enumerate(CONDS):
                keep=validation['condition_ids']==ci
                metrics[c]=independent_metrics(pred[keep],validation['target'][keep].reshape(-1,51)*1000,validation['subject_ids'][keep])
            dest=sd/method;dest.mkdir();np.savez_compressed(dest/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            np.savez_compressed(dest/'predictions.npz',prediction_mm=pred)
            row={'seed':seed,'method':method,'metrics':metrics,'history':history,'weights_sha256':sha(dest/'weights.npz'),
                'predictions_sha256':sha(dest/'predictions.npz'),'OOF_payload_sha256':sha(SRC/f'seed{seed}/oof_payload.npz'),
                'validation_payload_sha256':sha(SRC/f'seed{seed}/validation_payload.npz')}
            save(dest/'training.json',row);summary['runs'].append(row);save(DEST/'summary.json',summary)
            print(json.dumps({'event':'complete','seed':seed,'method':method,'clean_mm':metrics['clean']['MPJPE_mm']}),flush=True)
    summary['status']='complete';save(DEST/'summary.json',summary)

if __name__=='__main__':main()
