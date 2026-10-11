"""Check the pretrained detector on public videos and export a separate ONNX candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType

import cv2
import numpy as np
import torch
from torch import nn
from transformers import RfDetrForObjectDetection
from transformers.models.rf_detr.modeling_rf_detr import RfDetrDinov2Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.video_person_state import box_iou, enhance_detection


class PersonDetectorExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        mean = image.new_tensor([.485,.456,.406]).view(1,3,1,1)
        std = image.new_tensor([.229,.224,.225]).view(1,3,1,1)
        output = self.model(pixel_values=(image-mean)/std)
        boxes = output.pred_boxes[0]
        scores = output.logits[0,:,1].sigmoid()
        order = scores.topk(self.model.config.num_queries).indices
        center, extent = boxes[:,:2],boxes[:,2:]/2
        xyxy = torch.cat((center-extent,center+extent),1).clamp(0,1)*320
        return xyxy[order],scores[order],torch.ones_like(scores,dtype=torch.int64)


def people(boxes,scores,labels):
    result=[]
    for box,score,label in zip(boxes,scores,labels):
        if label != 1 or score < .35:
            continue
        box = np.clip(box/320,0,1).tolist()
        if box[2]-box[0]<.02 or box[3]-box[1]<.04 or any(box_iou(p['box'],box)>.45 for p in result):
            continue
        result.append({'box':box,'score':float(score)})
        if len(result)==4:
            break
    return result


def fixed_position_encoding(self, embeddings, height, width):
    # The ONNX input is fixed at the checkpoint's native 576px; interpolation is identical to identity.
    return self.position_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default=str(ROOT/'artifacts/pretrained/rf_detr_medium'))
    parser.add_argument('--video',required=True,help='Public dataset video; never use a private camera recording')
    parser.add_argument('--output',default=str(ROOT/'artifacts/moderate_detector_check.json'))
    parser.add_argument('--export',help='Separate ONNX candidate path, outside the published app assets')
    args=parser.parse_args()
    torch.set_num_threads(4)
    weight=Path(args.model)/'model.safetensors'
    digest=hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest!='e52098adc46969794fbdd16e0548a62b81ba0c0f4b14392676edba50be9a69f6':
        raise ValueError('Official detector weight SHA256 mismatch')
    model=RfDetrForObjectDetection.from_pretrained(args.model,local_files_only=True).eval()
    assert model.config.id2label[1]=='person'
    net=PersonDetectorExport(model).eval()
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Cannot open the public validation video')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cases=[]; sample=None
    try:
        for fraction in (.2,.5,.8):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(max(0,count-1)*fraction))
            ok,frame=cap.read()
            if not ok:
                raise RuntimeError('Cannot decode the validation video')
            for name,view in [('original',frame),('dark',np.clip((frame/255.)**2.2*.25*255,0,255).astype(np.uint8)),('upper_crop',frame[:round(frame.shape[0]*.65)])]:
                enhanced,_=enhance_detection(view)
                rgb=cv2.cvtColor(cv2.resize(enhanced,(576,576),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
                image=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255
                with torch.inference_mode():
                    start=time.perf_counter(); values=net(image)
                    milliseconds=(time.perf_counter()-start)*1000
                arrays=[v.numpy() for v in values]
                assert all(np.isfinite(v).all() for v in arrays)
                detected=people(*arrays)
                if name=='original' and detected:
                    sample=image
                cases.append({'fraction':fraction,'case':name,'inference_ms':milliseconds,'people':detected})
    finally:
        cap.release()
    report={'parameters':sum(p.numel() for p in model.parameters()),'model':'rf_detr_medium',
            'weight_sha256':digest,'torch':torch.__version__,'device':'CPU','cases':cases,
            'note':'Functional public-video checks with synthetic degradation; no box ground truth, no field accuracy claim'}
    Path(args.output).write_text(json.dumps(report,indent=2))
    if args.export:
        import onnxruntime as ort
        if sample is None:
            raise RuntimeError('Export parity needs an actual detected person in a public original frame')
        path=Path(args.export)
        if path.is_file():
            raise FileExistsError('Keep existing exported candidates; use a new output path')
        path.parent.mkdir(parents=True,exist_ok=True)
        for module in model.modules():
            if isinstance(module,RfDetrDinov2Embeddings):
                assert module.patch_size==16 and module.position_embeddings.shape[1]==36*36+1
                module.interpolate_pos_encoding=MethodType(fixed_position_encoding,module)
        torch.onnx.export(net,sample,str(path),input_names=['input'],output_names=['boxes','scores','labels'],opset_version=17,dynamo=False)
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        with torch.inference_mode():
            expected=[v.numpy() for v in net(sample)]
        actual=session.run(None,{'input':sample.numpy()})
        # Sorting may swap nearly tied low-score queries; compare the actual accepted detections.
        expected_people,actual_people=people(*expected),people(*actual)
        assert len(expected_people)==len(actual_people)>0
        for reference in expected_people:
            matching=max(actual_people,key=lambda p:box_iou(p['box'],reference['box']))
            assert box_iou(matching['box'],reference['box'])>.98 and abs(matching['score']-reference['score'])<.01
        report['onnx']={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'bytes':path.stat().st_size,'input':[1,3,576,576],'input_range':'RGB 0..1; normalization included',
                        'boxes':'xyxy in legacy 320 coordinate space','class':'person=1','accepted_detection_parity':True}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
