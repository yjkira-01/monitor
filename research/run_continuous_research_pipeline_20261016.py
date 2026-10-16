"""Foreground local research pipeline; no scheduler, messages or final-test use.

Wait for the existing cache extractor, then serially audit, train and report.
An explicit interrupt terminates the current child process and records a stop.
Each stage fails closed; no unavailable result is marked complete.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'artifacts/continuous_research_pipeline_20261016'
CACHE=ROOT/'artifacts/mri_action_windows_resumed_20261016'
STAGES=[
 ('cache_audit','audit_mri_action_windows.py',[str(CACHE)]),
 ('main_training','run_action_window_formal_20261016.py',['--cache',str(CACHE),'--output',str(ROOT/'artifacts/action_window_main_20261016'),'--folds','0','--seeds','7','17','27','--epochs','80','--temporal-epochs','80']),
 ('main_audit','audit_action_window_research.py',[str(ROOT/'artifacts/action_window_main_20261016'),'--cache',str(CACHE)]),
 ('main_report','report_action_window_research.py',[str(ROOT/'artifacts/action_window_main_20261016')]),
 ('geometry_training','run_ray_oof_action_windows_20261016.py',[]),
 ('geometry_audit','audit_ray_oof_action_windows_20261016.py',[]),
 ('geometry_report','report_ray_action_window_research.py',[])]
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(value):
    tmp=DEST/'status.tmp';tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,DEST/'status.json')

def main():
    if DEST.exists():raise ValueError('New pipeline directory required')
    DEST.mkdir();snapshot=DEST/'source_snapshot';snapshot.mkdir()
    sources={name:sha(ROOT/'research'/name) for _,name,_ in STAGES}
    for name in sources:(snapshot/name).write_bytes((ROOT/'research'/name).read_bytes())
    status={'status':'waiting_for_cache','started_utc':datetime.now(timezone.utc).isoformat(),'stages':[],
        'source_sha256':sources,'test_subjects_used':[],'automatic_restart_after_user_stop':False,
        'scope':'This registered local experiment round, not a publication-complete claim; final test excluded.'}
    save(status);active=None
    try:
        previous=-1
        while True:
            current=json.loads((CACHE/'status.json').read_text())
            count=len(current.get('subjects',[]))
            if count!=previous:
                print(json.dumps({'event':'cache_progress','complete_subjects':count,'expected':16}),flush=True);previous=count
            if current['status']=='complete':break
            if current['status']=='stopped_by_user':raise KeyboardInterrupt
            if time.time()-(CACHE/'status.json').stat().st_mtime>1800:raise RuntimeError('Cache status stale for >30min; no training launched')
            time.sleep(10)
        for key,script,args in STAGES:
            assert sha(ROOT/'research'/script)==sources[script],'Stage source changed after registration'
            status['status']='running';status['current_stage']=key;save(status)
            log=DEST/(key+'.console.log')
            print(json.dumps({'event':'stage_start','stage':key,'log':str(log)}),flush=True)
            with log.open('wb') as stream:
                active=subprocess.Popen([sys.executable,'-u',str(ROOT/'research'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                status['child_pid']=active.pid;save(status)
                code=active.wait();active=None
            if code!=0:raise RuntimeError(f'Stage {key} failed, exit {code}, see {log}')
            status['stages'].append({'name':key,'status':'complete','log_sha256':sha(log)});save(status)
            print(json.dumps({'event':'stage_complete','stage':key}),flush=True)
        status['status']='complete';status['completed_utc']=datetime.now(timezone.utc).isoformat();status.pop('child_pid',None);save(status)
        print(json.dumps({'event':'pipeline_complete','final_test_used':False}),flush=True)
    except KeyboardInterrupt:
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=15)
            except subprocess.TimeoutExpired:active.kill();active.wait()
        status['status']='stopped_by_user';status['stopped_utc']=datetime.now(timezone.utc).isoformat();save(status)
        print(json.dumps({'event':'pipeline_stopped','child_terminated':True}),flush=True)
        raise
    except Exception as error:
        status['status']='failed';status['error']=str(error);save(status);raise

if __name__=='__main__':main()
