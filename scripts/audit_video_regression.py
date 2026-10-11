"""Fixed five-second regression windows, never used as training or independent accuracy data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'backend'))
from app.video_model import TrainedVideoModel


def audit(video,checkpoint,output):
    cap=cv2.VideoCapture(str(video))
    count,fps=cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not np.isfinite([count,fps]).all() or count<2 or fps<=0:raise ValueError('Invalid regression video')
    duration=(count-1)/fps
    model=TrainedVideoModel(Path(checkpoint))
    predictions=[]
    for end in np.append(np.arange(5,duration,1),duration):
        start=max(0,float(end)-5)
        indices=np.linspace(round(start*fps),min(round(float(end)*fps),int(count)-1),model.clip_len).round().astype(int)
        frames=[]
        cap=cv2.VideoCapture(str(video))
        try:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
                if not ok:raise ValueError('Regression frame decode failed')
                frames.append(frame)
        finally:cap.release()
        result=model.predict(frames)
        predictions.append({'start_s':start,'end_s':float(end),'fall_score':result['fall_score'],'activity':result['activity']})
    late=[p['fall_score'] for p in predictions if p['end_s'] >= max(5,duration-3)]
    report={'video_sha256':hashlib.sha256(Path(video).read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'scope':'Known fall regression; sliding five-second windows; no training use or accuracy claim',
            'duration_s':duration,'predictions':predictions,'late_window_detected':bool(late and max(late)>=.5)}
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    audit(args.video,args.checkpoint,args.output)
