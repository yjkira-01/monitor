"""Isolated 89M backbone experiment; never changes the running v3 task or app weights."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root',required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    run = root/'artifacts/activity_stage_f_internvideo2_base_20261009'
    run.mkdir(exist_ok=True)
    status_path = run/'status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous['state'] != 'failed' or any(s['label']=='training' for s in previous['steps']):
            raise FileExistsError('Inspect the existing job and resume its exact last checkpoint instead of starting a duplicate')
        status_path.rename(run/f'status_setup_failed_{previous["pid"]}.json')
    source = Path(__file__).resolve().parent
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_HOME=str(root/'artifacts/torch_cache'))
    status = {'state':'preparing','pid':os.getpid(),'model':'internvideo2_base','automatic_deployment':False,'steps':[]}

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2))
        temporary.replace(status_path)

    def execute(name,command):
        status['state'] = name
        step = {'label':name,'command':command,'state':'running','started_at':time.time()}
        status['steps'].append(step); save()
        subprocess.run(command,cwd=root,check=True)
        step.update(state='complete',finished_at=time.time()); save()

    save()
    try:
        manifest = root/'data/manifests/moderate_stage_f_20261009.csv'
        original = root/'data/manifests/multiscene_curated_v3_expanded_e.csv'
        expected = 'd6e99ba317e9762e750cf4f0ed9707d22f3bd43a6cedd1dc604f8457b6171a1f'
        if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
            raise ValueError('The frozen eight-source manifest changed')
        if manifest.exists():
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
                raise ValueError('Independent experiment manifest changed')
        else:
            manifest.write_bytes(original.read_bytes())
        with manifest.open(encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        from train_video import validate_splits
        validate_splits(rows)
        counts = dict(Counter(r['split'] for r in rows))
        status.update(manifest_sha256=expected,windows=counts,
                      training_videos=len({r['path'] for r in rows if r['split']=='train'}),
                      effective_batch=16,epochs_limit=120,early_stop_patience=25)
        history = json.loads((root/'artifacts/activity_stage_e_curated_v3_20261009/r3d_multiscene_expanded.json').read_text())['history']
        reference = max(history,key=lambda row:tuple(row['selection_rank']))
        reference_path = run/'r3d_reference.json'
        reference_path.write_text(json.dumps({'manifest_sha256':expected,'model':'r3d18','epoch':reference['epoch'],
                                              'purpose':'frozen development reference; not an independent test',
                                              'metrics':reference},indent=2))
        status['reference_epoch'] = reference['epoch']; save()
        weights = root/'artifacts/internvideo2_base_k400.bin'
        execute('gpu_check',[sys.executable,str(source/'check_moderate_video.py'),'--manifest',str(manifest),
                             '--weights',str(weights),'--output',str(run/'gpu_check.json')])
        command = [sys.executable,'-u',str(source/'train_video.py'),
                   '--manifest',str(manifest),'--model','internvideo2_base','--multi-state',
                   '--pretrained-weights',str(weights),'--reference-metrics',str(reference_path),
                   '--epochs','120','--patience','25','--clip-len','8','--size','224',
                   '--batch-size','2','--accumulate','8','--workers','2','--amp',
                   '--lr','2e-5','--weight-decay','.05','--bounded-sampling','--domain-selection',
                   '--strong-augment','--low-light-augment','--partial-body-augment','--quality-augment',
                   '--risk-loss-weight','.25','--output',str(run/'internvideo2_activity.pt')]
        execute('training',command)
        report = json.loads((run/'internvideo2_activity_evaluation.json').read_text())
        status.update(state='complete',selected_epoch=report['selected_epoch'],
                      development_gate_passed=report['metrics']['regression_gate_passed'],finished_at=time.time())
        save()
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time()); save()
        raise


if __name__=='__main__':
    main()
