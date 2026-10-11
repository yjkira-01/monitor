"""Two fixed manifests: start six real sources, then include the larger obstruction/room releases."""
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


def smoke_rows(rows):
    counts,subset=Counter(),[]
    for row in rows:
        if row['split']=='test':continue
        key=(row['dataset'],row['split'],row['label'])
        if counts[key]<2:subset.append(row);counts[key]+=1
    assert {r['label'] for r in subset if r['split']=='train'} == {r['label'] for r in rows if r['split']=='train'}
    return subset


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',default=str(Path.cwd()))
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Cloud GPU required; no CPU long-training fallback')
    root=Path(args.project_root).resolve()
    run=root/'artifacts/activity_stage_e_curated_v3_20261009'
    run.mkdir(exist_ok=True)
    status_file=run/'status.json'
    if status_file.exists():raise FileExistsError('Inspect existing run and resume its exact manifest/checkpoint')
    source=Path(__file__).parent
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_HOME=str(root/'artifacts/torch_cache'))
    status={'state':'preparing','pid':os.getpid(),'gpu':torch.cuda.get_device_name(),'steps':[],'automatic_deployment':False}
    def save():
        temp=status_file.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(status_file)
    def execute(label,command):
        status['state']='training' if label.endswith(('smoke','full')) else 'preparing'
        step={'label':label,'state':'running','command':command,'started_at':time.time()}
        status['steps'].append(step);save()
        subprocess.run(command,cwd=str(root),check=True)
        step.update(state='complete',finished_at=time.time());save()
    def wait_data(keys):
        metadata=json.loads((root/'data/raw/OmniFall/metadata.json').read_text())
        files={r['key']:r for r in metadata['files']}
        deadline=time.monotonic()+8*3600
        status['state']='waiting_data';save()
        for key in keys:
            archive=root/'data/raw/OmniFall'/key
            directory=archive.with_suffix('')
            while not (archive.is_file() and len(list(directory.rglob('mmaction2_annotations*.txt')))==1 and not list(archive.parent.glob(key+'.*.chunk'))):
                if time.monotonic()>deadline:raise TimeoutError('Verified dataset not ready: '+key)
                status['waiting_archive']=key;save();time.sleep(15)
            digest=hashlib.md5()
            with archive.open('rb') as handle:
                while chunk:=handle.read(8*1024**2):digest.update(chunk)
            if digest.hexdigest()!=files[key]['checksum'].removeprefix('md5:'):raise ValueError('Archive checksum changed: '+key)
            status.setdefault('verified_archives',{})[key]={'bytes':archive.stat().st_size,'md5':digest.hexdigest()};save()
        status.pop('waiting_archive',None);save()
    save()
    try:
        labels=run/'GMDCSA24_labels.csv'
        expected='d4682729276a08be783a2c189c1617f6202f92c87c0b1b74ad88a2e1f634337a'
        if hashlib.sha256(labels.read_bytes()).hexdigest()!=expected:raise ValueError('GMD author labels changed')
        for name,expected in {'OOPS_author_labels.csv':'6086dd55d485df47e10459d720af4b95fd46b8ef1aaf853da67caa9e17cb6e7d',
                              'oops_video_mapping.csv':'d35382bc35c03a63f9411b63e75e58f35e5aa0cd2f105da9b73f6cfa17706807'}.items():
            if hashlib.sha256((run/name).read_bytes()).hexdigest()!=expected:raise ValueError('OOPS author snapshot changed')
        base_manifest=root/'data/manifests/multistate_stage_d.csv'
        previous_status=json.loads((root/'artifacts/activity_stage_d_labels_20261008/status.json').read_text())
        if hashlib.sha256(base_manifest.read_bytes()).hexdigest()!=previous_status['manifest_sha256']:
            raise ValueError('Completed D training manifest changed')
        previous=root/'artifacts/activity_stage_d_labels_20261008/r3d_activity_full_inference.pt'
        teacher=run/'gmd_teacher_inference.pt'
        regression=run/'regression/known_fall.mp4'
        assert previous.is_file() and teacher.is_file() and regression.is_file()
        common=['--model','r3d18','--multi-state','--clip-len','16','--size','112','--batch-size','8',
                '--workers','4','--weight-decay','.001','--cache-clips','--amp','--unfreeze-layer3','--strong-augment',
                '--low-light-augment','--partial-body-augment','--quality-augment','--domain-balanced-sampling',
                '--domain-selection','--risk-loss-weight','.3','--teacher-checkpoint',str(teacher),'--teacher-weight','.2']
        stages=[('warmup',['EDF','MCFD','OOPS'],['EDF.zip','MCFD.zip','OOPS_1k.zip'],60,15,'.00003'),
                ('expanded',['LE2I','OCCU'],['LE2I.zip','OCCU.zip'],200,40,'.00002')]
        for stage,datasets,archives,epochs,patience,lr in stages:
            wait_data(archives)
            manifest=root/f'data/manifests/multiscene_curated_v3_{stage}_e.csv'
            command=[sys.executable,'-u',str(source/'index_multifall.py'),'--project-root',str(root),
                     '--multi-state','--datasets',*datasets,'--base-manifest',str(base_manifest),'--output',str(manifest)]
            if stage=='warmup':command.extend(['--oops-labels',str(run/'OOPS_author_labels.csv'),'--oops-mapping',str(run/'oops_video_mapping.csv')])
            execute(stage+'_index',command)
            with manifest.open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            status.setdefault('manifests',{})[stage]={'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 'counts':dict(Counter(f"{r['dataset']}/{r['split']}/{r['label']}" for r in rows)),
                 'windows':len(rows),'training_videos':len({r['path'] for r in rows if r['split']=='train'})};save()
            if stage=='warmup':
                smoke=manifest.with_name('multiscene_curated_v3_smoke_e.csv')
                with smoke.open('w',encoding='utf-8-sig',newline='') as handle:
                    writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(smoke_rows(rows))
                execute(stage+'_smoke',[sys.executable,'-u',str(source/'train_video.py'),'--manifest',str(smoke),
                       '--epochs','2','--patience','0','--lr',lr,'--init-checkpoint',str(previous),*common,'--output',str(run/'r3d_smoke.pt')])
            output=run/f'r3d_multiscene_{stage}.pt'
            execute(stage+'_full',[sys.executable,'-u',str(source/'train_video.py'),'--manifest',str(manifest),
                    '--epochs',str(epochs),'--patience',str(patience),'--lr',lr,'--init-checkpoint',str(previous),*common,'--output',str(output)])
            candidate=output.with_name(output.stem+'_inference.pt')
            regression_report=run/f'{stage}_known_fall_regression.json'
            execute(stage+'_regression',[sys.executable,'-u',str(source/'audit_video_regression.py'),'--video',str(regression),
                                         '--checkpoint',str(candidate),'--output',str(regression_report)])
            evaluation=json.loads(output.with_name(output.stem+'_evaluation.json').read_text())
            reg=json.loads(regression_report.read_text())
            eligible=evaluation['metrics']['regression_gate_passed'] and reg['late_window_detected']
            status.setdefault('candidates',{})[stage]={'selected_epoch':evaluation['selected_epoch'],
                'development_gate_passed':eligible,'deployed':False,
                'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()};save()
            if eligible:previous=candidate
            base_manifest=manifest
        status.update(state='complete',finished_at=time.time());save()
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time());save();raise


if __name__=='__main__':main()
