"""Compare public-frame detector/pose deployment. No training or private frame export."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/action_upgrade_20261010'

def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 3
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])

def enhance(image):
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
    if brightness >= 65:
        return image
    table = (255 * (np.arange(256) / 255) ** max(.35, brightness / 65)).astype(np.uint8)
    return cv2.LUT(image, table)

def rtmo(model, image):
    ratio = min(640 / image.shape[0], 640 / image.shape[1])
    padded = np.full((640, 640, 3), 114, np.uint8)
    resized = cv2.resize(enhance(image), (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    padded[:resized.shape[0], :resized.shape[1]] = resized
    data = padded.transpose(2, 0, 1)[None].astype(np.float32)
    det, poses = model.run(None, {model.get_inputs()[0].name: data})
    boxes, scores = det[0, :, :4] / ratio, det[0, :, 4]
    keep = []
    for i in scores.argsort()[::-1]:
        if scores[i] < .35:
            break
        if any(iou(boxes[i], boxes[j]) > .45 for j in keep):
            continue
        keep.append(int(i))
    result = []
    size = np.array([image.shape[1], image.shape[0]], np.float32)
    for i in keep[:12]:
        p = poses[0, i].copy()
        p[:, :2] /= ratio * size
        p[(p[:, 0] < 0) | (p[:, 0] > 1) | (p[:, 1] < 0) | (p[:, 1] > 1), 2] = 0
        result.append(dict(box=(boxes[i] / np.tile(size, 2)).clip(0, 1).tolist(), score=float(scores[i]), joints=p.tolist()))
    return result

def iou(a, b):
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return inter / max(1e-6, (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter)

def old(detector, pose, image):
    image = enhance(image)
    data = cv2.resize(image, (576, 576))[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    outputs = dict(zip([v.name for v in detector.get_outputs()], detector.run(None, {'input': data})))
    boxes = outputs['boxes'] / 320
    scores = outputs['scores']
    result = []
    for i in np.argsort(scores)[::-1]:
        if outputs['labels'][i] != 1 or scores[i] < .35:
            continue
        box = boxes[i].clip(0, 1)
        if any(iou(box, np.array(p['box'])) > .45 for p in result):
            continue
        cx, cy = (box[:2] + box[2:]) / 2 * [image.shape[1], image.shape[0]]
        width, height = (box[2:] - box[:2]) * [image.shape[1], image.shape[0]] * 1.25
        if width > height * .75:
            height = width / .75
        else:
            width = height * .75
        crop = cv2.warpAffine(image, np.array([[192/width, 0, 96-192*cx/width], [0, 256/height, 128-256*cy/height]]), (192, 256))
        data = ((crop.astype(np.float32)-[123.675, 116.28, 103.53])/[58.395, 57.12, 57.375]).transpose(2, 0, 1)[None].astype(np.float32)
        sx, sy = pose.run(None, {pose.get_inputs()[0].name: data})
        xy = np.stack([sx[0].argmax(-1)/2/192*width+cx-width/2, sy[0].argmax(-1)/2/256*height+cy-height/2], -1) / [image.shape[1], image.shape[0]]
        conf = (sx[0].max(-1)+sy[0].max(-1))/2
        conf[((xy<0)|(xy>1)).any(-1)] = 0
        result.append(dict(box=box.tolist(), score=float(scores[i]), joints=np.column_stack([xy, conf.clip(0,1)]).tolist()))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=Path, default=ROOT/'artifacts/open_source_research_20261009/evaluation_list.json')
    parser.add_argument('--output', type=Path, default=OUT/'person_comparison.json')
    args = parser.parse_args()
    one = session(OUT/'rtmo-m/end2end.onnx')
    detector = session(ROOT/'desktop/CareMonitor/Assets/Realtime/people.onnx')
    pose = session(ROOT/'desktop/CareMonitor/Assets/Realtime/rtmpose.onnx')
    print('RTMO IO:', [(x.name, x.shape) for x in one.get_inputs()+one.get_outputs()], flush=True)
    rows = []
    for item in json.loads(args.list.read_text(encoding='utf-8-sig')):
        cap = cv2.VideoCapture(item['file'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame = 0
        observations = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            index = frame
            frame += 1
            if index % max(1, round(fps/2)):
                continue
            if image.shape[1] > 640:
                image = cv2.resize(image, (640, round(image.shape[0]*640/image.shape[1])))
            watch = time.perf_counter()
            a = old(detector, pose, image)
            old_ms = 1000*(time.perf_counter()-watch)
            watch = time.perf_counter()
            b = rtmo(one, image)
            new_ms = 1000*(time.perf_counter()-watch)
            observations.append(dict(time=index/fps, width=image.shape[1], height=image.shape[0], old=a, rtmo=b, old_ms=old_ms, rtmo_ms=new_ms))
        cap.release()
        rows.append(dict(**item, observations=observations))
        print(Path(item['file']).parent.name, Path(item['file']).name, 'located old/rtmo', sum(bool(o['old']) for o in observations), sum(bool(o['rtmo']) for o in observations), '/', len(observations), flush=True)
        args.output.write_text(json.dumps(dict(state='running', rows=rows)), encoding='utf-8')
    sha = hashlib.sha256((OUT/'rtmo-m/end2end.onnx').read_bytes()).hexdigest()
    args.output.write_text(json.dumps(dict(state='complete', rtmo_sha256=sha, rows=rows, notes='Public GMD development clips; detection presence is not ground-truth box AP or posture accuracy.'), indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
