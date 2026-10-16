"""Action-covered continuous RGB/point development cache, no final-test reads.

Select two fixed 32-frame windows per action from annotation intervals, before
looking at any predictions or target errors. Retain physically unusual labels
in the main population; store a screened flag for sensitivity analysis.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
sys.path.insert(0,str(ROOT/'.deps/research_ort_gpu'))
import torch
import cv2
import numpy as np
import onnxruntime as ort
from audit_mri import load_labels,interval_bounds
from cache_mri_rgb import independent_pose,MODEL_SHA,VIDEO_SHA,file_checks
from cache_mri_robust_rgb import pixel_probes,verify_archive_directory,save_json
from fusion_pose_representation import visible_centroid_features

DEV=[1,2,4,5,6,7,8,9,11,13,14,15,16,17,19,20]
ACTIONS=[f'pose_{i}' for i in range(1,11)]+['free_form','walk']
FOLDS=[[5,6,19,20],[1,2,4,7],[8,9,11,13],[14,15,16,17]]
CONDS=['clean','noisy_dark','occlusion']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def select_windows(segments,lo,hi,width=32):
    ids,action,window=[],[],[]
    for a,name in enumerate(ACTIONS):
        left,right=segments[name]
        left,right=max(left,lo),min(right,hi+1)
        if right-left<2*width:raise ValueError(f'Action too short: {name}')
        # Centres at 1/3 and 2/3 of the annotated action, no error selection.
        starts=[max(left,min(right-width,int(left+(right-left)*p)-width//2)) for p in (1/3,2/3)]
        if starts[1]<starts[0]+width:raise ValueError('Overlapping windows')
        for w,start in enumerate(starts):
            ids.extend(range(start,start+width)); action.extend([a]*width); window.extend([a*2+w]*width)
    ids=np.asarray(ids,dtype=np.int64)
    if len(np.unique(ids))!=len(ids) or (np.diff(ids)<=0).any():raise ValueError('Window IDs overlap or unordered')
    return ids,np.asarray(action,dtype=np.int64),np.asarray(window,dtype=np.int64)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve()
    if ROOT not in out.parents:raise ValueError('Workspace output required')
    data_manifest=json.loads((ROOT/'artifacts/research_execution_20261012/mri_archive_manifest.json').read_text())
    video_manifest=json.loads((ROOT/'artifacts/research_execution_20261012/rgb_features/video_manifest.json').read_text())
    verify_archive_directory(ROOT/'dataset_release.zip',data_manifest['index'],data_manifest['bytes'],dataset=True)
    verify_archive_directory(ROOT/'blurred_videos.zip',video_manifest['entries'],14988106753)
    modelpath=ROOT/'desktop/CareMonitor/Assets/Realtime/rtmo.onnx'
    assert sha(modelpath)==MODEL_SHA
    protocol={'date':'2026-10-16','subjects':DEV,'final_test_subjects_used':[],
        'actions':ACTIONS,'fold_validation_subjects':FOLDS,'conditions':CONDS,'window_frames':32,
        'windows_per_action':2,'selection':'32 consecutive frames centred at 1/3 and 2/3 within each clipped half-open annotation interval; selected before target values/predictions',
        'population':'all selected frames with finite GT; main keeps physically unusual labels and missing observations',
        'target':'published dual-camera-derived COCO17 hip-centred 3D metres, converted to mm; not independent mocap',
        'input':'color0 horizontal unmirror; RTMO independent score selection, no GT crops/2D inputs; published radar 196x5 offline window',
        'timing':'recorded integer millisecond timestamps; no use of header FPS for physical time',
        'perturbations':'same source-resolution noisy_dark and occlusion functions as 20261013 cache; no dark supplementary condition',
        'rtmo_sha256':MODEL_SHA,'video_archive_sha256':VIDEO_SHA,'data_archive_sha256':data_manifest['sha256'],
        'software':{'torch':torch.__version__,'ort':ort.__version__,'opencv':cv2.__version__,'numpy':np.__version__},
        'backend':'CUDAExecutionProvider use_tf32=0, CPU shape ops; all conditions recomputed in this backend',
        'source_sha256':{n:sha(ROOT/'research'/n) for n in [Path(__file__).name,'cache_mri_rgb.py','cache_mri_robust_rgb.py','fusion_pose_representation.py']}}
    if out.exists():
        if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Different cache protocol')
    else:
        out.mkdir(parents=True);save_json(out/'protocol.json',protocol)
        snap=out/'source_snapshot';snap.mkdir()
        for n in protocol['source_sha256']:(snap/n).write_bytes((ROOT/'research'/n).read_bytes())
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.log_severity_level=3
    ort.preload_dlls()
    model=ort.InferenceSession(str(modelpath),opts,providers=[('CUDAExecutionProvider',{'use_tf32':'0'}),'CPUExecutionProvider'])
    if model.get_providers()[0]!='CUDAExecutionProvider':raise ValueError('CUDA unavailable')
    cv2.setNumThreads(1)
    report={'status':'running','test_subjects_used':[],'subjects':[],'protocol_sha256':sha(out/'protocol.json'),
            'expected_frames':len(DEV)*len(ACTIONS)*64,'started_unix':time.time()}
    save_json(out/'status.json',report)
    with zipfile.ZipFile(ROOT/'dataset_release.zip') as z:
        for s in DEV:
            path=out/f'subject{s}.npz';receipt=out/f'subject{s}.json'
            if path.exists():
                r=json.loads(receipt.read_text());assert sha(path)==r['cache_sha256'];report['subjects'].append(r);continue
            begin=time.perf_counter()
            labels=load_labels(z.read(f'dataset_release/aligned_data/pose_labels/subject{s}_all_labels.cpl'))
            stamps=np.load(io.BytesIO(z.read(f'dataset_release/raw_data/unixtime/subject{s}_unix.npy')),allow_pickle=False)
            rs,re=interval_bounds(labels['radar_avail_frames'],len(stamps));gs,ge=interval_bounds(labels['gt_avail_frames'],len(stamps))
            ids,action,window=select_windows(labels['video_label'],max(rs,gs),min(re,ge))
            target=labels['refined_gt_kps'][ids].transpose(0,2,1).astype(np.float64)
            target=1000*(target-target[:,[11,12]].mean(1)[:,None,:])
            finite=np.isfinite(target).all((1,2))
            edges=[(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]
            lengths=np.stack([np.linalg.norm(target[:,a]-target[:,b],axis=-1) for a,b in edges],-1)
            screened=finite&(lengths.max(1)<=1000)&(np.linalg.norm(target,axis=-1).max(1)<=3000)
            radar=np.load(io.BytesIO(z.read(f'dataset_release/features/radar/subject{s}_featuremap.npy')),allow_pickle=False)[ids-rs].reshape(-1,196,5)
            if not np.isfinite(radar).all():raise ValueError('Nonfinite radar')
            focal=np.diag(labels['camera_matrix']['cam1_intrinsic'])[:2]
            vpath=ROOT/f'artifacts/research_execution_20261012/rgb_audit/subject{s}_color0.mp4'
            entry=next(e for e in video_manifest['entries'] if e['member'].endswith(f'/subject{s}_color0.mp4'))
            vsha,crc=file_checks(vpath)
            if vpath.stat().st_size!=entry['bytes'] or crc!=entry['crc32']:raise ValueError('Source video differs')
            cap=cv2.VideoCapture(str(vpath));assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(stamps)
            poses={c:[] for c in CONDS};hashes={c:[] for c in CONDS};durations=[]
            try:
                previous=-2
                for i,fid in enumerate(ids):
                    if fid!=previous+1:assert cap.set(cv2.CAP_PROP_POS_FRAMES,int(fid))
                    ok,img=cap.read();assert ok and int(cap.get(cv2.CAP_PROP_POS_FRAMES))==int(fid)+1
                    previous=int(fid);img=cv2.flip(img,1)
                    probes,_=pixel_probes(img,s,int(fid));probes['clean']=img
                    for c in CONDS:
                        pixels=probes[c];hashes[c].append(hashlib.sha256(pixels.tobytes()).hexdigest())
                        t=time.perf_counter();poses[c].append(independent_pose(model,pixels));durations.append(time.perf_counter()-t)
                    if (i+1)%128==0:
                        print(json.dumps({'event':'cache_progress','subject':s,'frames':i+1,'total':len(ids),'seconds':time.perf_counter()-begin}),flush=True)
            finally:cap.release()
            arrays={'frame_ids':ids,'timestamps_ms':stamps[ids],'action_ids':action,'window_ids':window,
                    'target_mm':target.reshape(-1,51),'target_finite':finite,'screened':screened,
                    'points':radar.astype(np.float32),'point_mask':np.any(radar!=0,axis=-1),'focal':focal,
                    'camera_rotation':np.asarray(labels['camera_matrix']['cam1_rot']) if 'cam1_rot' in labels['camera_matrix'] else np.zeros(0)}
            for c in CONDS:
                arrays[c+'_poses']=np.asarray(poses[c]);arrays[c+'_features']=visible_centroid_features(arrays[c+'_poses'],focal)
                arrays[c+'_pixel_sha256']=np.asarray(hashes[c],dtype='U64')
            np.savez_compressed(path,**arrays)
            r={'subject':s,'frames':len(ids),'finite_targets':int(finite.sum()),'screened':int(screened.sum()),
               'missing_radar_frames':int((~arrays['point_mask'].any(1)).sum()),
               'actions':{name:int((action==a).sum()) for a,name in enumerate(ACTIONS)},
               'source_video_sha256':vsha,'source_video_crc32':crc,'cache_sha256':sha(path),
               'seconds':time.perf_counter()-begin,'median_inference_ms':1000*float(np.median(durations)),
               'visible_frames':{c:int((arrays[c+'_poses'][...,2]>=.3).any(1).sum()) for c in CONDS}}
            save_json(receipt,r);report['subjects'].append(r);report['completed_frames']=sum(t['frames'] for t in report['subjects']);save_json(out/'status.json',report)
            print(json.dumps({'event':'subject_complete',**r}),flush=True)
    report['status']='complete';report['finished_unix']=time.time();save_json(out/'status.json',report)

if __name__=='__main__':main()
