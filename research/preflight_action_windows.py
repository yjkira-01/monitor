"""All-development selection/timestamp/target preflight before long extraction."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from audit_mri import load_labels
from cache_mri_action_windows import select_windows,DEV
ROOT=Path(__file__).resolve().parents[1]
rows=[]
with zipfile.ZipFile(ROOT/'dataset_release.zip') as z:
    for s in DEV:
        labels=load_labels(z.read(f'dataset_release/aligned_data/pose_labels/subject{s}_all_labels.cpl'))
        rs,re=labels['radar_avail_frames'];gs,ge=labels['gt_avail_frames']
        ids,actions,windows=select_windows(labels['video_label'],max(rs,gs),min(re,ge))
        ts=np.load(io.BytesIO(z.read(f'dataset_release/raw_data/unixtime/subject{s}_unix.npy')),allow_pickle=False)[ids]
        same=windows[1:]==windows[:-1]
        gaps=np.diff(ts)[same]
        gt=labels['refined_gt_kps'][ids].transpose(0,2,1).astype(float)
        gt=gt-gt[:,[11,12]].mean(1)[:,None]
        finite=np.isfinite(gt).all((1,2));radius=np.linalg.norm(gt,axis=-1).max(1)
        rows.append({'subject':s,'frames':len(ids),'finite_targets':int(finite.sum()),
          'within_window_gap_max_ms':int(gaps.max()),'gaps_above_500ms':int((gaps>500).sum()),
          'root_radius_above_3m':int((radius>3).sum()),'root_radius_max_m':float(radius[finite].max())})
out={'status':'complete','subjects':rows,'test_subjects_used':[]}
(ROOT/'artifacts/mri_action_window_preflight_20261016.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
