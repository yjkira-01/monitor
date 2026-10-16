"""Strict reuse of completed strong-input X-Fi models from a stopped run."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/xfi_strong_inputs_formal_20261016'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare_reuse(args,summary):
    prior=json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    receipt=json.loads((SOURCE/'stop_receipt.json').read_text(encoding='utf-8'))
    assert prior['status']=='stopped_by_user' and receipt['status']=='stopped_by_user'
    for key in ['training','train_subjects','validation_subjects','train_frames','validation_frames','seeds',
                'old_summary_sha256','cache_sha256','official_source_sha256','adaptations']:
        if prior[key]!=summary[key]:raise ValueError('Resume contract differs: '+key)
    for n,h in prior['sources'].items():
        assert sha(ROOT/'research'/n)==h and sha(SOURCE/'source_snapshot'/n)==h
    with (SOURCE/'normalization.npz').open('rb') as a,(args.output/'normalization.npz').open('rb') as b:
        # ZIP metadata may differ when normalization is regenerated, so numeric
        # array equality is checked without inferring identity from file bytes.
        import numpy as np
        with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
            assert set(x.files)==set(y.files)
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
    completed={r['id']:r for r in receipt['completed']}
    for row in prior['runs']:
        checkpoint=SOURCE/row['id'];bound=completed[row['id']]
        assert row==json.loads((checkpoint/'training.json').read_text(encoding='utf-8'))
        for name,key in [('weights.npz','weights_sha256'),('predictions.npz','prediction_sha256')]:
            expected=bound['weights_sha256' if name=='weights.npz' else 'predictions_sha256']
            assert sha(checkpoint/name)==row[key]==expected
        shutil.copytree(checkpoint,args.output/row['id'])
        for name in ('weights.npz','predictions.npz','training.json'):
            assert sha(checkpoint/name)==sha(args.output/row['id']/name)
    summary['runs']=prior['runs'].copy()
    return {'source_directory':str(SOURCE),'source_summary_sha256':sha(SOURCE/'summary.json'),
        'source_stop_receipt_sha256':sha(SOURCE/'stop_receipt.json'),
        'reused_complete_ids':list(completed),'no_retraining_for_completed':True}

def is_reused(runid,summary):
    return runid in summary['resume']['reused_complete_ids']
