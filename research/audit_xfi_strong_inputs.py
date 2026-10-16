"""Independent saved-state CUDA replay and NumPy metrics for X-Fi core runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from xfi_strong_inputs import XFiStrongCore
from run_fusion_experiments_gpu import load_data,normalize_inputs,sparse_mask
from analyze_fusion_trials import strict_load_weights,independent_metrics

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--partial',action='store_true');args=p.parse_args()
    summary=json.loads((args.directory/'summary.json').read_text())
    if not args.partial and summary['status']!='complete':raise ValueError('Incomplete run')
    for n,digest in summary['sources'].items():
        assert sha(ROOT/'research'/n)==digest and sha(args.directory/'source_snapshot'/n)==digest
    assert summary['final_test_subjects_used']==[]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    with np.load(args.directory/'normalization.npz',allow_pickle=False) as f:stats={k:f[k] for k in f.files}
    split,rows=load_data(ROOT/'artifacts/fusion_rgb_expanded_20261013',['clean','dark','noisy_dark','occlusion'],'visible_centroid')
    val=split['canonical_ids_validation']
    def stack(k):return np.concatenate([rows[s][k] for s in val])
    values={c:normalize_inputs(stack(c),stack('points'),stack('mask'),stats) for c in ['clean','dark','noisy_dark','occlusion']}
    c=values['clean'];values['video_missing']=(torch.zeros_like(c[0]),c[1],c[2],torch.zeros_like(c[3]));values['radar_missing']=(c[0],torch.zeros_like(c[1]),torch.zeros_like(c[2]),c[3])
    thin=torch.tensor(sparse_mask(stack('mask')));values['radar_sparse']=(c[0],c[1].masked_fill(~thin[...,None],0),thin,c[3])
    results=[]
    for run in summary['runs']:
        dest=args.directory/run['id'];assert sha(dest/'weights.npz')==run['weights_sha256'];assert sha(dest/'predictions.npz')==run['prediction_sha256']
        model=XFiStrongCore();strict_load_weights(model,dest/'weights.npz');model.cuda().eval()
        with np.load(dest/'predictions.npz',allow_pickle=False) as f:
            target,subjects=f['target_mm'],f['subject_ids'];np.testing.assert_array_equal(target,stack('target'))
            for condition,inp in values.items():
                with torch.no_grad():
                    outputs=[model(*(x[i:i+64].cuda() for x in inp)) for i in range(0,len(target),64)]
                    replay=torch.cat([o['fused'].cpu() for o in outputs]).numpy().reshape(-1,51)*1000+stats['target_mean_mm']
                saved=f[condition+'_prediction_mm'];difference=float(np.max(abs(replay-saved)))
                if difference>.01:raise ValueError(f'Replay > preset .01mm: {difference}')
                m=independent_metrics(saved,target,subjects)
                delta=abs(m['MPJPE_mm']-run['metrics'][condition]['MPJPE_mm']);assert delta<1e-8
                # Both missing routes use the existing learned independent heads.
                if condition in ('video_missing','radar_missing'):
                    for o in outputs:
                        # Removing radar does not make an RTMO-failed image
                        # available. The original shared policy applies video
                        # fallback first, then radar fallback if RGB is absent;
                        # both absent therefore uses the learned zero-radar head.
                        expected=(o['radar'] if condition=='video_missing' else
                            torch.where(o['video_valid'][:,None,None],o['video'],o['radar']))
                        torch.testing.assert_close(o['fused'],expected,atol=0,rtol=0)
                results.append({'run':run['id'],'condition':condition,'replay_max_abs_mm':difference,'metric_difference_mm':delta})
        print(json.dumps({'audited':run['id']}),flush=True)
    output={'status':'partial_complete' if args.partial else 'complete','checked_models':len(summary['runs']),
        'checks':results,'test_subjects_used':[],'auditor_sha256':sha(Path(__file__)),
        'cross_backend_verification':'Not attempted. Preset same-CUDA replay tolerance .01mm; original .3404mm CPU limit remains disclosed.'}
    path=args.directory/('audit_partial.json' if args.partial else 'audit.json');path.write_text(json.dumps(output,indent=2))
    print(json.dumps({'status':output['status'],'models':len(summary['runs'])}),flush=True)

if __name__=='__main__':main()
