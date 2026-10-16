"""Independent cache audit: population, targets, point source and raw pixels."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
sys.path.insert(0,str(ROOT/'.deps/research_ort_gpu'))
import torch
import cv2
import numpy as np
import onnxruntime as ort
from audit_mri import load_labels
from cache_mri_rgb import independent_pose,MODEL_SHA

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(a,b,tol,name):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape,name+' shape'
    if a.dtype.kind in 'USb' or b.dtype.kind in 'USb':assert np.array_equal(a,b),name;return 0.
    assert np.array_equal(np.isfinite(a),np.isfinite(b)),name+' finite support'
    finite=np.isfinite(a)&np.isfinite(b)
    d=float(np.max(abs(a[finite].astype(float)-b[finite].astype(float)))) if finite.any() else 0.
    if d>tol:raise ValueError(f'{name}: {d}>{tol}')
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('--partial',action='store_true');args=p.parse_args()
    proto=json.loads((args.cache/'protocol.json').read_text());status=json.loads((args.cache/'status.json').read_text())
    assert not proto['final_test_subjects_used'] and not status['test_subjects_used']
    subjects=[s for s in proto['subjects'] if (args.cache/f'subject{s}.json').exists()]
    if not args.partial:assert status['status']=='complete' and subjects==proto['subjects']
    assert not(set(subjects)&{3,10,12,18})
    for n,h in proto['source_sha256'].items():assert sha(ROOT/'research'/n)==h and sha(args.cache/'source_snapshot'/n)==h
    modelpath=ROOT/'desktop/CareMonitor/Assets/Realtime/rtmo.onnx';assert sha(modelpath)==proto['rtmo_sha256']==MODEL_SHA
    ort.preload_dlls();opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.log_severity_level=3
    model=ort.InferenceSession(str(modelpath),opts,providers=[('CUDAExecutionProvider',{'use_tf32':'0'}),'CPUExecutionProvider'])
    assert model.get_providers()[0]=='CUDAExecutionProvider';cv2.setNumThreads(1)
    records=[]
    with zipfile.ZipFile(ROOT/'dataset_release.zip') as z:
        for s in subjects:
            receipt=json.loads((args.cache/f'subject{s}.json').read_text());path=args.cache/f'subject{s}.npz';assert sha(path)==receipt['cache_sha256']
            with np.load(path,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
            labels=load_labels(z.read(f'dataset_release/aligned_data/pose_labels/subject{s}_all_labels.cpl'))
            rs,re=labels['radar_avail_frames'];gs,ge=labels['gt_avail_frames'];lo,hi=max(rs,gs),min(re,ge)
            expected=[];actions=[];windows=[]
            for a,action in enumerate(proto['actions']):
                left,right=labels['video_label'][action];left,right=max(left,lo),min(right,hi+1)
                for w,num in enumerate((1,2)):
                    # Protocol uses float proportions 1/3 and 2/3; preserve
                    # their declared evaluation order at integer boundaries.
                    proportion=num/3
                    start=max(left,min(right-32,int(left+(right-left)*proportion)-16))
                    expected.extend(range(start,start+32));actions.extend([a]*32);windows.extend([2*a+w]*32)
            check(arrays['frame_ids'],expected,0,'fixed ID selection');check(arrays['action_ids'],actions,0,'action ID');check(arrays['window_ids'],windows,0,'window ID')
            ids=arrays['frame_ids'];target=labels['refined_gt_kps'][ids].transpose(0,2,1).astype(float)
            target=1000*(target-(target[:,11]+target[:,12])[:,None]/2)
            check(arrays['target_mm'],target.reshape(-1,51),1e-10,'target source')
            check(arrays['target_finite'],np.isfinite(target).all((1,2)),0,'finite flags')
            edges=[(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]
            lengths=np.stack([np.linalg.norm(target[:,a]-target[:,b],axis=-1) for a,b in edges],axis=-1)
            screened=np.isfinite(target).all((1,2))&(lengths.max(1)<=1000)&(np.linalg.norm(target,axis=-1).max(1)<=3000)
            check(arrays['screened'],screened,0,'physical sensitivity flags')
            stamps=np.load(io.BytesIO(z.read(f'dataset_release/raw_data/unixtime/subject{s}_unix.npy')),allow_pickle=False)
            check(arrays['timestamps_ms'],stamps[ids],0,'recorded timestamps')
            radar=np.load(io.BytesIO(z.read(f'dataset_release/features/radar/subject{s}_featuremap.npy')),allow_pickle=False)[ids-rs].reshape(-1,196,5).astype(np.float32)
            check(arrays['points'],radar,0,'point source');check(arrays['point_mask'],np.any(radar!=0,axis=-1),0,'point mask')
            focal=np.diag(labels['camera_matrix']['cam1_intrinsic'])[:2]
            for c in proto['conditions']:
                pose=arrays[c+'_poses'].astype(float);visible=pose[...,2]>=.3
                weights=np.where(visible,np.clip(pose[...,2],0.,1.),0.)
                # The strong representation uses a confidence-weighted visible
                # centroid; recover it independently from its definition.
                centroid=(pose[...,:2]*weights[...,None]).sum(1)/np.maximum(weights.sum(1,keepdims=True),1e-8)
                xy=(pose[...,:2]-centroid[:,None])/focal
                xy=np.where(visible[...,None],xy,0.)
                features=np.concatenate([xy.reshape(-1,34),np.where(visible,pose[...,2],0.)],-1)
                check(arrays[c+'_features'],features,1e-9,'visible centroid features')
            video=ROOT/f'artifacts/research_execution_20261012/rgb_audit/subject{s}_color0.mp4'
            assert sha(video)==receipt['source_video_sha256']
            cap=cv2.VideoCapture(str(video));diffs=[]
            try:
                # Fixed positions cover early/late frames and separate actions.
                for i in (0,31,383,767):
                    fid=int(ids[i]);assert cap.set(cv2.CAP_PROP_POS_FRAMES,fid)
                    ok,img=cap.read();assert ok and int(cap.get(cv2.CAP_PROP_POS_FRAMES))==fid+1
                    img=cv2.flip(img,1)
                    rng=np.random.default_rng(np.random.SeedSequence([20261013,s,fid]))
                    noisy=img.astype(np.float32)*.12
                    noisy+=rng.normal(0.,8.,size=img.shape).astype(np.float32)
                    noisy=np.clip(noisy,0,255).astype(np.uint8)
                    occluded=img.copy();h,w=img.shape[:2];occluded[int(.15*h):int(.90*h),int(.40*w):int(.60*w)]=114
                    for c,pixels in [('clean',img),('noisy_dark',noisy),('occlusion',occluded)]:
                        assert hashlib.sha256(pixels.tobytes()).hexdigest()==str(arrays[c+'_pixel_sha256'][i])
                        actual=independent_pose(model,pixels)
                        diffs.append(check(actual,arrays[c+'_poses'][i],.01,'RTMO CUDA pose replay'))
            finally:cap.release()
            records.append({'subject':s,'frames':len(ids),'finite':int(arrays['target_finite'].sum()),'screened':int(arrays['screened'].sum()),
                            'pixel_replays':12,'pose_replay_max_abs':max(diffs),'cache_sha256':sha(path)})
            print(json.dumps({'audited_subject':s,'frames':len(ids),'max_pose_replay':max(diffs)}),flush=True)
    result={'status':'partial_complete' if args.partial else 'complete','subjects':records,'test_subjects_used':[],
            'scope':'All cached targets/point arrays/timestamps/selection/features; four raw image frames per subject, all three perturbations',
            'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(args.cache/'protocol.json')}
    name='cache_audit_partial.json' if args.partial else 'cache_audit.json'
    (args.cache/name).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'subjects':len(records)}),flush=True)

if __name__=='__main__':main()
