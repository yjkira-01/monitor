"""Isolated night/partial-body fine-tuning from Stage B, with an explicit comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Low-light long training requires CUDA')
    root = Path.cwd()
    run = root/'artifacts/robust_body_stage_c_20261008'
    run.mkdir(exist_ok=True)
    status_file = run/'status.json'
    if status_file.exists():
        raise FileExistsError('Stage C status already exists; inspect the current run before restarting')
    status = {'state':'running','pid':os.getpid(),'gpu':torch.cuda.get_device_name(),'steps':[]}
    def save():
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8');temporary.replace(status_file)
    script = Path(__file__).with_name('train_video.py')
    common = ['--model','r3d18','--init-checkpoint',str(root/'artifacts/multifall_r3d_v2_stage_b_inference.pt'),
              '--clip-len','16','--size','112','--batch-size','8','--workers','4','--lr','0.00003','--weight-decay','0.001',
              '--cache-clips','--amp','--balanced-sampling','--unfreeze-layer3','--strong-augment','--low-light-augment','--partial-body-augment']
    save()
    try:
        for label, manifest, epochs, patience in [('smoke','multifall_stage_b_smoke.csv',2,0),('full','multifall_stage_b.csv',80,20)]:
            command = [sys.executable,'-u',str(script),'--manifest',str(root/'data/manifests'/manifest),
                       '--epochs',str(epochs),'--patience',str(patience),*common,'--output',str(run/f'r3d_lowlight_{label}.pt')]
            step = {'label':label,'state':'running','command':command,'started_at':time.time()}
            status['steps'].append(step);save()
            subprocess.run(command,check=True)
            step.update(state='complete',finished_at=time.time());save()
        baseline = json.loads((run/'r3d_lowlight_full_baseline.json').read_text())
        candidate = json.loads((run/'r3d_lowlight_full_evaluation.json').read_text())
        measured = {'day':candidate['metrics'],**candidate['metrics']['robustness']}
        deltas = {case:{key:measured[case][key]-old[key] for key in ('val_macro_f1','fall_recall','adl_recall')} for case,old in baseline.items() if case != 'initial_checkpoint_sha256'}
        # Development gate, not proof of real partial-body or night accuracy.
        acceptable = deltas['day']['val_macro_f1'] >= -.02 and deltas['day']['fall_recall'] >= -.03 and all(v['val_macro_f1'] >= -.03 for v in deltas.values())
        improved = sum(v['val_macro_f1'] for k,v in deltas.items() if k != 'day') > 0
        comparison = {'baseline':baseline,'candidate':measured,'deltas':deltas,'eligible_for_review':acceptable and improved,
                      'selected_epoch':candidate['selected_epoch'],'note':'Synthetic development validation; real partial-body/person-crop and night tests remain required.'}
        (run/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
        status['eligible_for_review'] = comparison['eligible_for_review']
        status.update(state='complete',finished_at=time.time());save()
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time());save()
        raise


if __name__ == '__main__':
    main()
