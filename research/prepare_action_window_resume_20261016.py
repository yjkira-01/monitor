"""Verify/copy complete cache shards; keep the stopped directory untouched."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'artifacts/mri_action_windows_20261016'
dst=ROOT/'artifacts/mri_action_windows_resumed_20261016'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
receipt=json.loads((src/'stop_receipt.json').read_text(encoding='utf-8'))
protocol=json.loads((src/'protocol.json').read_text(encoding='utf-8'))
assert receipt['status']=='stopped_by_user' and protocol['final_test_subjects_used']==[]
for n,h in protocol['source_sha256'].items():
    assert sha(ROOT/'research'/n)==h and sha(src/'source_snapshot'/n)==h
for row in receipt['completed']:
    assert sha(src/f"subject{row['subject']}.npz")==row['cache_sha256']
if dst.exists():raise ValueError('New resume directory required')
dst.mkdir()
shutil.copy2(src/'protocol.json',dst/'protocol.json')
shutil.copytree(src/'source_snapshot',dst/'source_snapshot')
for row in receipt['completed']:
    for ext in ('npz','json'):
        path=src/f"subject{row['subject']}.{ext}";shutil.copy2(path,dst/path.name)
        assert sha(path)==sha(dst/path.name)
(dst/'resume_receipt.json').write_text(json.dumps({'source_directory':str(src),'source_stop_sha256':sha(src/'stop_receipt.json'),
    'completed_shards_reused':receipt['completed'],'no_reextraction_for_completed_shards':True,'test_subjects_used':[]},indent=2),encoding='utf-8')
print(json.dumps({'status':'prepared','subjects_reused':[r['subject'] for r in receipt['completed']]}))
