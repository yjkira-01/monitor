"""Record explicit user stop; standard library only, no model/data inference."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
now=datetime.now(timezone.utc).isoformat()
for name,filename in [('xfi_strong_inputs_formal_20261016','summary.json'),('mri_action_windows_20261016','status.json')]:
    d=ROOT/'artifacts'/name;p=d/filename;x=json.loads(p.read_text(encoding='utf-8'))
    x['status']='stopped_by_user';x['stopped_unix_utc']=now;x['stop_reason']='User explicitly requested 停了吧; processes interrupted, no automatic resumption'
    save(p,x)
    completed=[]
    if name.startswith('xfi'):
        for r in x['runs']:
            completed.append({'id':r['id'],'epochs_completed':80,'clean_MPJPE_mm':r['metrics']['clean']['MPJPE_mm'],
                              'weights_sha256':sha(d/r['id']/'weights.npz'),'predictions_sha256':sha(d/r['id']/'predictions.npz')})
        incomplete='seed17 stopped after last reported epoch20; no intermediate checkpoint saved; seed27 not started'
    else:
        completed=[{'subject':r['subject'],'frames':r['frames'],'cache_sha256':sha(d/f"subject{r['subject']}.npz")} for r in x['subjects']]
        incomplete='subject4 interrupted after last reported 256/768 frames; no completed subject4 cache; later subjects not started'
    receipt={'status':'stopped_by_user','time_utc':now,'completed':completed,'incomplete':incomplete,
             'final_test_subjects_used':[],'audit_status':'No new formal independent replay audit run before user stop',
             'process_stop':'Both long-running exec sessions returned exit1 after Ctrl+C',
             'automatic_resume':False}
    save(d/'stop_receipt.json',receipt)
    print(json.dumps({'directory':str(d),'completed':len(completed),'status':'stopped_by_user'},ensure_ascii=False))
