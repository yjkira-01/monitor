"""Workspace-only CUDA RTMO runtime and numerical/throughput probe."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps/research_cuda'))
sys.path.insert(0,str(ROOT/'.deps/research_ort_gpu'))
import torch
import numpy as np
import cv2
import onnxruntime as ort
from cache_mri_rgb import independent_pose, file_checks, MODEL_SHA

def main():
    out=ROOT/'artifacts/rtmo_cuda_probe_20261016'
    if out.exists(): raise ValueError('New output required')
    out.mkdir()
    modelpath=ROOT/'desktop/CareMonitor/Assets/Realtime/rtmo.onnx'
    assert file_checks(modelpath)[0]==MODEL_SHA
    opts=ort.SessionOptions(); opts.intra_op_num_threads=2; opts.inter_op_num_threads=1
    opts.enable_profiling=True; opts.profile_file_prefix=str(out/'profile')
    ort.preload_dlls()
    gpu=ort.InferenceSession(str(modelpath),opts,providers=[('CUDAExecutionProvider',{'use_tf32':'0'}),'CPUExecutionProvider'])
    assert gpu.get_providers()[0]=='CUDAExecutionProvider', gpu.get_providers()
    cpuopts=ort.SessionOptions(); cpuopts.intra_op_num_threads=3; cpuopts.inter_op_num_threads=1
    cpu=ort.InferenceSession(str(modelpath),cpuopts,providers=['CPUExecutionProvider'])
    video=ROOT/'artifacts/research_execution_20261012/rgb_audit/subject1_color0.mp4'
    cap=cv2.VideoCapture(str(video)); images=[]
    for frame in (138,500,1000,1500,2500,3500,4500,6000):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,img=cap.read(); assert ok
        images.append(cv2.flip(img,1))
    cap.release(); cv2.setNumThreads(1)
    independent_pose(gpu,images[0])
    times={'cpu':[],'cuda':[]}; pred={'cpu':[],'cuda':[]}
    for key,model in [('cuda',gpu),('cpu',cpu)]:
        for image in images:
            start=time.perf_counter(); pred[key].append(independent_pose(model,image))
            times[key].append(time.perf_counter()-start)
    for _ in range(12):
        start=time.perf_counter(); independent_pose(gpu,images[0]); times['cuda'].append(time.perf_counter()-start)
    profile=Path(gpu.end_profiling())
    events=json.loads(profile.read_text()); providers={}
    for e in events:
        p=e.get('args',{}).get('provider')
        if p: providers[p]=providers.get(p,0)+1
    a,b=[np.asarray(pred[k]) for k in ('cpu','cuda')]
    report={'status':'complete','ort_version':ort.__version__, 'ort_path':ort.__file__,
        'gpu':torch.cuda.get_device_name(),'model_sha256':MODEL_SHA,
        'provider_kernel_counts':providers, 'milliseconds':{k:1000*float(np.median(v)) for k,v in times.items()},
        'max_xy_difference_pixels':float(np.max(abs(a[...,:2]-b[...,:2]))),
        'max_confidence_difference':float(np.max(abs(a[...,2]-b[...,2]))),
        'visibility_decision_disagreements':int(np.sum((a[...,2]>=.3)!=(b[...,2]>=.3))),
        'scope':'Eight development frames, clean only. Whole-model GPU acceleration; CPU/GPU exact identity not claimed.',
        'test_subjects_used':[], 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    np.savez_compressed(out/'predictions.npz',cpu=a,cuda=b)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
