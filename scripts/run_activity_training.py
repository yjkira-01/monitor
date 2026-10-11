"""Train verified OmniFall states on the authorized GPU, retaining subject test splits."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Multistate training requires the cloud GPU')
    root = Path.cwd()
    run = root/'artifacts/activity_stage_d_labels_20261008'
    run.mkdir(exist_ok=True)
    status_file = run/'status.json'
    if status_file.exists():raise FileExistsError('Inspect the existing activity run before restarting')
    status = {'state':'running','pid':os.getpid(),'gpu':torch.cuda.get_device_name(),'steps':[]}
    def save():
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8');temporary.replace(status_file)
    def execute(label,command):
        step = {'label':label,'state':'running','command':command,'started_at':time.time()}
        status['steps'].append(step);save()
        subprocess.run(command,check=True)
        step.update(state='complete',finished_at=time.time());save()
    save()
    source = Path(__file__).parent
    try:
        url = 'https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/labels/GMDCSA24.csv'
        label_path = run/'GMDCSA24_labels.csv'
        if label_path.exists():labels = label_path.read_bytes()
        else:
            with urlopen(url,timeout=45) as response:labels = response.read()
            label_path.write_bytes(labels)
        if hashlib.sha256(labels).hexdigest() != 'd4682729276a08be783a2c189c1617f6202f92c87c0b1b74ad88a2e1f634337a':
            raise ValueError('Published GMD activity label snapshot changed; inspect before training')
        status['supplement'] = {'url':url,'sha256':hashlib.sha256(labels).hexdigest(),'bytes':len(labels),
                                'method':'reuse existing original RGB with author temporal labels'};save()
        manifest = root/'data/manifests/multistate_stage_d.csv'
        execute('index',[sys.executable,'-u',str(source/'index_multifall.py'),'--project-root',str(root),'--multi-state',
                         '--datasets','CAUCAFall','UP-Fall','GMDCSA24','--gmd-labels',str(label_path),'--output',str(manifest)])
        with manifest.open(encoding='utf-8-sig') as handle:rows = list(csv.DictReader(handle))
        counts,subset = Counter(),[]
        for row in rows:
            if row['split'] == 'test':continue
            key = (row['dataset'],row['split'],row['label'])
            if counts[key] < 2:subset.append(row);counts[key] += 1
        smoke = root/'data/manifests/multistate_stage_d_smoke.csv'
        with smoke.open('w',encoding='utf-8-sig',newline='') as handle:
            writer = csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(subset)
        status.update(classes=sorted({r['label'] for r in rows}),counts=dict(Counter(f"{r['split']}/{r['label']}" for r in rows)),
                      manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest());save()
        common = ['--model','r3d18','--multi-state','--init-checkpoint',str(root/'artifacts/multifall_r3d_v2_stage_b_inference.pt'),
                  '--clip-len','16','--size','112','--batch-size','8','--workers','4','--lr','0.0001','--weight-decay','0.001',
                  '--cache-clips','--amp','--balanced-sampling','--unfreeze-layer3','--strong-augment','--low-light-augment','--partial-body-augment']
        for label,data,epochs,patience in [('smoke',smoke,2,0),('full',manifest,100,25)]:
            execute(label,[sys.executable,'-u',str(source/'train_video.py'),'--manifest',str(data),'--epochs',str(epochs),
                           '--patience',str(patience),*common,'--output',str(run/f'r3d_activity_{label}.pt')])
        status.update(state='complete',finished_at=time.time());save()
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time());save()
        raise


if __name__ == '__main__':main()
