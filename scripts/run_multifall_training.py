"""Run the authorized cloud curriculum with checked subprocesses and persistent status."""
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

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def wait_pid(pid, marker):
    deadline = time.monotonic()+8*3600
    path = Path(f'/proc/{pid}/cmdline')
    while True:
        try:
            command_line = path.read_bytes()
        except FileNotFoundError:
            return
        if marker.encode() not in command_line:
            return
        if time.monotonic()>deadline:
            raise TimeoutError(f'Previous {marker} job did not finish in 8h')
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['a','b'], required=True)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Cloud training requires CUDA; no silent CPU long training')
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCH_HOME=str(ROOT/'artifacts/torch_cache'))
    status_path = ROOT/f'artifacts/curriculum_stage_{args.stage}.json'
    status = {'pid':os.getpid(), 'stage':args.stage, 'steps':[], 'state':'running'}

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temporary.replace(status_path)

    def run(script, *arguments):
        command = [PYTHON,'-u',f'scripts/{script}',*arguments]
        step = {'command':command,'started_at':time.time(),'state':'running'}
        status['steps'].append(step);save()
        print('RUN '+' '.join(command),flush=True)
        subprocess.run(command, check=True)
        step.update(state='complete',finished_at=time.time());save()

    save()
    try:
        jobs = json.loads((ROOT/'artifacts/multifall_v2_jobs.json').read_text())
        if args.stage=='b':
            status['state']='waiting_for_upfall';save()
            annotation=ROOT/'data/raw/OmniFall/UP_Fall/UP_Fall/mmaction2_annotations.txt'
            deadline=time.monotonic()+4*3600
            while not annotation.exists():
                if time.monotonic()>deadline:
                    raise TimeoutError('UP-Fall download/extraction did not finish in 4h')
                time.sleep(10)
            archive=ROOT/'data/raw/OmniFall/UP_Fall.zip'
            if hashlib.md5(archive.read_bytes()).hexdigest()!='baa972b1d215e04c18574f3a647ebf01':
                raise ValueError('UP-Fall archive checksum mismatch')
            status['state']='waiting_for_stage_a';save()
            # Measured RGB peak is ~1GiB; overlapping the next data cache leaves ample room on 12GiB.
            if 'pose_stage_a_pipeline' in jobs:
                wait_pid(jobs['pose_stage_a_pipeline']['pid'],'scripts/run_multifall_training.py')
            indexed=ROOT/'data/manifests/multifall_stage_b.csv'
            report=ROOT/'artifacts/multifall_stage_b_index.json'
            if not indexed.exists() or not report.exists() or json.loads(report.read_text())['sha256']!=hashlib.sha256(indexed.read_bytes()).hexdigest():
                run('index_multifall.py','--output','data/manifests/multifall_stage_b.csv')
        status['state']='waiting_for_pose_cache';save()
        wait_pid(jobs['pose_cache_stage_a']['pid'],'scripts/cache_video_poses.py')
        manifest=f'data/manifests/multifall_stage_{args.stage}.csv'
        pose_manifest=f'data/manifests/multifall_pose_stage_{args.stage}.csv'
        status.update(state='running',manifest=manifest);save()
        if args.stage=='b':
            with open(manifest,encoding='utf-8-sig') as handle:
                rows=list(csv.DictReader(handle))
            counts=Counter();subset=[]
            for row in rows:
                if row['split']=='test':continue
                key=(row['dataset'],row['split'],row['label'])
                if counts[key]<4:subset.append(row);counts[key]+=1
            smoke='data/manifests/multifall_stage_b_smoke.csv'
            with open(smoke,'w',newline='',encoding='utf-8-sig') as handle:
                writer=csv.DictWriter(handle,list(subset[0]));writer.writeheader();writer.writerows(subset)
            rgb=['--model','r3d18','--batch-size','8','--clip-len','16','--size','112','--workers','4',
                 '--lr','0.0001','--weight-decay','0.001','--cache-clips','--amp','--balanced-sampling','--unfreeze-layer3','--strong-augment']
            run('train_video.py','--manifest',smoke,'--epochs','2',*rgb,'--output','artifacts/multifall_r3d_v2_stage_b_smoke.pt')
            run('train_video.py','--manifest',manifest,'--epochs','150','--patience','30',*rgb,'--output','artifacts/multifall_r3d_v2_stage_b.pt')
        coverage=ROOT/'artifacts'/f'multifall_pose_stage_{args.stage}_coverage.json'
        expected=hashlib.sha256((ROOT/manifest).read_bytes()).hexdigest()
        if not (ROOT/pose_manifest).exists() or not coverage.exists() or json.loads(coverage.read_text())['source_manifest_sha256']!=expected:
            run('cache_video_poses.py','--manifest',manifest,'--output',pose_manifest,'--workers','4')
        pose=['--model','pose_tcn','--manifest',pose_manifest,'--batch-size','64','--clip-len','48','--size','112',
              '--workers','0','--lr','0.001','--weight-decay','0.005','--cache-clips','--amp','--balanced-sampling']
        stem=f'artifacts/multifall_pose_tcn_stage_{args.stage}'
        run('train_video.py','--epochs','2',*pose,'--output',stem+'_smoke.pt')
        run('train_video.py','--epochs','200','--patience','40',*pose,'--output',stem+'.pt')
        status.update(state='complete',finished_at=time.time());save()
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time());save()
        raise


if __name__=='__main__':
    main()
