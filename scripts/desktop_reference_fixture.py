"""Numerical reference for the C# port, with deterministic missing-joint cases."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from yanglao_monitor.pose_temporal import pose_features
from export_desktop_models import VideoExport

torch.set_num_threads(4);torch.manual_seed(81)
poses=torch.rand(1,3,48,33);poses[:,2,::4,[1,12,25]]=0
poses[:,2,17:19,:]=0
fixture={'poses':poses[0].permute(1,2,0).tolist(),'aspect':1.7777778,
         'features':pose_features(poses*torch.tensor([1.7777778,1,1])[None,:,None,None]).flatten().tolist()}
state=torch.load(ROOT/'artifacts/care_video_active_inference.pt',map_location='cpu',weights_only=True)
model=VideoExport(state).eval()
frames=np.array([((np.arange(171*128*3,dtype=np.int64)+t*7)%251).astype(np.uint8).reshape(128,171,3) for t in range(16)])
rgb=frames[:,:,:,[2,1,0]].copy()
with torch.inference_mode():fixture['video_scores']=model(torch.from_numpy(rgb).float()[None])[0].flatten().tolist()
sys.path.insert(0,str(ROOT))
from backend.app.video_person_state import enhance_clip
dark=np.array([(f*(.08+.02*t)).astype(np.uint8) for t,f in enumerate(frames)])
enhanced=enhance_clip(list(dark),trained=True)
with torch.inference_mode():fixture['night_scores']=model(torch.from_numpy(np.array(enhanced)[:,:,:,[2,1,0]].copy()).float()[None])[0].flatten().tolist()
# Audit the migration's float preprocessing against the existing uint8 runtime on a public clip.
from torchvision.models.video import R3D_18_Weights
cap=cv2.VideoCapture(str(ROOT/'data/video_samples/GMDCSA24_Subject1_Fall01.mp4'))
raw=[]
while len(raw)<48:
 ok,frame=cap.read()
 if not ok:break
 raw.append(cv2.resize(frame,(171,128),interpolation=cv2.INTER_AREA))
cap.release();assert len(raw)>=16
arr=np.stack([cv2.cvtColor(raw[i],cv2.COLOR_BGR2RGB) for i in np.linspace(0,len(raw)-1,16).round().astype(int)])
with torch.inference_mode():
 old=model.net(R3D_18_Weights.DEFAULT.transforms()(torch.from_numpy(arr).permute(0,3,1,2)[None])).softmax(1)
 new=model(torch.from_numpy(arr).float()[None])[0]
delta=float((old-new).abs().max());assert delta<.01,delta
fixture['migration_public_clip_score_maxdiff']=delta
out=ROOT/'desktop/CareMonitor/Assets/selfcheck.json';out.write_text(json.dumps(fixture),encoding='utf-8')
print('fixture created; actual public clip score migration delta',delta)
