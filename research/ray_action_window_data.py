"""Load the exact large development population for geometric OOF experiments."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TRAIN=[1,2,4,7,8,9,11,13,14,15,16,17]
VAL=[5,6,19,20]
PIXEL=['clean','noisy_dark','occlusion']

def load_action_window_rows(cache):
    status=json.loads((cache/'status.json').read_text())
    audit=json.loads((cache/'cache_audit.json').read_text())
    assert status['status']=='complete' and audit['status']=='complete'
    assert not status['test_subjects_used']
    rows={}
    for s in TRAIN+VAL:
        path=cache/f'subject{s}.npz'
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest==json.loads((cache/f'subject{s}.json').read_text())['cache_sha256']
        with np.load(path,allow_pickle=False) as f:
            keep=f['target_finite']
            d={'frame_ids':f['frame_ids'][keep],'points':f['points'][keep],'mask':f['point_mask'][keep],
               'target':f['target_mm'][keep]}
            for c in PIXEL:
                d[c]=f[c+'_features'][keep]
                pose=f[c+'_poses'][keep]
                d['reference_available_'+c]=(pose[:,11,2]>=.3)&(pose[:,12,2]>=.3)
        rows[s]=d
    return {'canonical_ids_train':TRAIN,'canonical_ids_validation':VAL},rows
