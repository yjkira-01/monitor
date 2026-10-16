"""Test source reuse on the frozen strong pipeline, without fitting any model.

Only frames with exact XYZ/Doppler multiset correspondence to the aligned
three-camera-frame window are changed. Deduplication groups by recorded raw
frame ID, preserving multiplicities inside one original radar observation.
"""
import collections
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps/research_cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from audit_mri import load_labels, interval_bounds
from fusion_models import FusionModel
from fusion_pose_representation import visible_centroid_features
from run_fusion_experiments_gpu import normalize_inputs
from analyze_fusion_trials import strict_load_weights, independent_metrics

OUTPUT = ROOT / 'artifacts/source_reuse_probe_20261016'
OLD = ROOT / 'artifacts/fusion_method_research_20261013_expanded_centroid'
CACHE = ROOT / 'artifacts/fusion_rgb_expanded_20261013'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def counter(rows):
    return collections.Counter(map(tuple, rows[:, :4]))

def regroup(feature, aligned_window):
    observed = feature[np.any(feature != 0, axis=1)]
    # CSV and released geometry are exactly comparable on the audited example.
    exact = counter(observed) == counter(aligned_window[:, 2:6])
    if not exact:
        return None, {'exact_window_correspondence': False,
                      'released_rows': len(observed), 'window_rows': len(aligned_window)}
    groups = {}
    for camera in np.unique(aligned_window[:, 8]):
        camera_rows = aligned_window[aligned_window[:, 8] == camera]
        for raw in np.unique(camera_rows[:, 0]):
            rows = camera_rows[camera_rows[:, 0] == raw, 2:6]
            key = int(raw)
            if key in groups and counter(groups[key]) != counter(rows):
                return None, {'exact_window_correspondence': True,
                              'same_raw_id_has_different_rows': True}
            groups[key] = rows
    quotas = sum((counter(rows) for rows in groups.values()), collections.Counter())
    unique_rows = []
    for row in observed:
        key = tuple(row[:4])
        if quotas[key] > 0:
            unique_rows.append(row)
            quotas[key] -= 1
    if any(quotas.values()):
        raise ValueError('Unmatched source multiplicity')
    dedup = np.zeros_like(feature)
    dedup[:len(unique_rows)] = unique_rows
    # Uniformly copying all existing references should be invariant to the
    # mean/max encoder, when copies fit without truncation.
    copies = {}
    for n in (2, 4):
        if len(observed) * n <= 196:
            value = np.zeros_like(feature)
            value[:len(observed) * n] = np.tile(observed, (n, 1))
            copies[str(n)] = value
    ids = sorted(groups)
    reuse = {str(raw): int(len(np.unique(aligned_window[aligned_window[:, 0] == raw, 8]))) for raw in ids}
    return (dedup, copies), {'exact_window_correspondence': True,
         'same_raw_id_has_different_rows': False, 'raw_ids': ids,
         'camera_references_per_raw': reuse, 'released_rows': len(observed),
         'unique_source_rows': len(unique_rows),
         'removed_rows': len(observed) - len(unique_rows)}

def main():
    if OUTPUT.exists():
        raise ValueError('Refusing to overwrite source experiment')
    OUTPUT.mkdir()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    report = json.loads((OLD / 'run_summary.json').read_text())
    train, val = report['train_subjects'], report['validation_subjects']
    if set(train + val) & {3, 10, 12, 18}:
        raise ValueError('Holdout used')
    with np.load(OLD / 'normalization.npz', allow_pickle=False) as f:
        stats = {k: f[k] for k in f.files}
    rows, records = {}, []
    with zipfile.ZipFile(ROOT / 'dataset_release.zip') as archive:
        for s in train + val:
            labels = load_labels(archive.read(f'dataset_release/aligned_data/pose_labels/subject{s}_all_labels.cpl'))
            start, _ = interval_bounds(labels['radar_avail_frames'], len(labels['refined_gt_kps']))
            features = np.load(io.BytesIO(archive.read(f'dataset_release/features/radar/subject{s}_featuremap.npy')), allow_pickle=False).reshape(-1,196,5)
            member = f'dataset_release/aligned_data/radar/singleframe/subject{s}.csv'
            aligned = np.loadtxt(io.BytesIO(archive.read(member)), delimiter=',', skiprows=1)
            by_camera = {int(k): aligned[v] for k,v in _group_indices(aligned[:,8]).items()}
            with np.load(CACHE / f'subject{s}.npz', allow_pickle=False) as c:
                frame_ids = c['frame_ids'].copy()
                focal = np.diag(labels['camera_matrix']['cam1_intrinsic'])[:2]
                rgb = {k: visible_centroid_features(c[k+'_poses'], focal) for k in ('clean','dark','noisy_dark','occlusion')}
            original = features[frame_ids-start].copy()
            dedup, copies = original.copy(), {n: original.copy() for n in ('2','4')}
            eligible = {n: np.zeros(len(original),bool) for n in ('2','4')}
            for i,fid in enumerate(frame_ids):
                window = np.concatenate([by_camera.get(int(fid)+offset, np.zeros((0,9))) for offset in (-1,0,1)])
                value, detail = regroup(original[i], window)
                records.append({'subject': s, 'frame': int(fid), **detail})
                if value is not None:
                    dedup[i] = value[0]
                    for n,p in value[1].items():
                        copies[n][i], eligible[n][i] = p, True
            rows[s] = {'rgb':rgb, 'frame_ids':frame_ids, 'original':original, 'dedup':dedup,
                       'copies':copies,'eligible':eligible}
            np.savez_compressed(OUTPUT / f'subject{s}_inputs.npz', frame_ids=frame_ids,
                                original=original, dedup=dedup)
            selected = [r for r in records if r['subject']==s]
            print(json.dumps({'subject':s,'frames':len(selected),
                  'exact':sum(r['exact_window_correspondence'] for r in selected),
                  'changed':sum(r.get('removed_rows',0)>0 for r in selected)}),flush=True)
    summary = {'status':'complete', 'scope':'Frozen strong-model intervention, no retraining; dedup is an input shift',
               'test_subjects_used':[], 'records':records, 'models':{},
               'source_sha256':sha(Path(__file__)),
               'old_summary_sha256':sha(OLD/'run_summary.json')}
    points = {k: np.concatenate([rows[s][k] for s in val]) for k in ('original','dedup')}
    subjects = np.concatenate([np.full(len(rows[s]['frame_ids']),s) for s in val])
    for seed in report['initialization_seeds']:
        model = FusionModel('concat')
        wsha = strict_load_weights(model, OLD/f'seed{seed}/concat_weights.npz')
        model.cuda().eval()
        result, arrays = {}, {}
        with np.load(OLD/f'seed{seed}/concat_predictions.npz',allow_pickle=False) as f:
            target = f['target_mm'].copy()
            saved_predictions = {k:f[k].copy() for k in f.files}
        for cond in ('clean','dark','noisy_dark','occlusion','video_missing','radar_missing'):
            rgb = np.concatenate([rows[s]['rgb']['clean' if cond.endswith('missing') else cond] for s in val])
            result[cond] = {}
            for variant,p in points.items():
                inputs = list(normalize_inputs(rgb,p,np.any(p!=0,axis=-1),stats))
                if cond=='video_missing':
                    inputs[0].zero_(); inputs[3].zero_()
                if cond=='radar_missing':
                    inputs[1].zero_(); inputs[2].zero_()
                with torch.no_grad():
                    out = model(*(x.cuda() for x in inputs))['fused']
                pred = out.cpu().numpy().reshape(-1,51)*1000 + stats['target_mean_mm']
                if variant=='original':
                    replay = np.max(np.abs(pred-saved_predictions[cond+'_fused_prediction_mm']))
                    if replay>.01: raise ValueError(f'Frozen replay mismatch {replay}')
                    result[cond]['replay_max_abs_mm']=float(replay)
                result[cond][variant]=independent_metrics(pred,target,subjects)
                arrays[cond+'_'+variant]=pred
            result[cond]['dedup_gain_mm'] = result[cond]['original']['MPJPE_mm']-result[cond]['dedup']['MPJPE_mm']
            result[cond]['mean_prediction_shift_mm'] = float(np.linalg.norm((arrays[cond+'_original']-arrays[cond+'_dedup']).reshape(-1,17,3),axis=-1).mean())
        for n in ('2','4'):
            p = np.concatenate([rows[s]['copies'][n] for s in val])
            rgb = np.concatenate([rows[s]['rgb']['clean'] for s in val])
            eligible = np.concatenate([rows[s]['eligible'][n] for s in val])
            inputs = normalize_inputs(rgb,p,np.any(p!=0,axis=-1),stats)
            with torch.no_grad():
                pred=model(*(x.cuda() for x in inputs))['fused'].cpu().numpy().reshape(-1,51)*1000+stats['target_mean_mm']
            shift=np.linalg.norm((pred-arrays['clean_original']).reshape(-1,17,3),axis=-1).mean(1)
            result['uniform_copies_'+n]={'eligible_frames':int(eligible.sum()),'mean_shift_mm':float(shift[eligible].mean()),
                                         'max_shift_mm':float(shift[eligible].max())}
        np.savez_compressed(OUTPUT/f'seed{seed}_predictions.npz',target_mm=target,subject_ids=subjects,**arrays)
        summary['models'][str(seed)]={'weights_sha256':wsha,'conditions':result}
    summary['counts']={'frames':len(records),'exact':sum(r['exact_window_correspondence'] for r in records),
                       'changed':sum(r.get('removed_rows',0)>0 for r in records)}
    (OUTPUT/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'counts':summary['counts'], 'models':{s:{c:d.get('dedup_gain_mm') for c,d in v['conditions'].items()} for s,v in summary['models'].items()}}),flush=True)

def _group_indices(values):
    order=np.argsort(values,kind='stable')
    boundaries=np.flatnonzero(np.diff(values[order]))+1
    return {int(values[g[0]]):g for g in np.split(order,boundaries)}

if __name__=='__main__': main()
