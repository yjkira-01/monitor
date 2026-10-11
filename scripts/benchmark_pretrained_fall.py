"""Evaluate public YOLO fall weights without training or saving video frames."""
import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_restricted(path):
    classes = []
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(path):
        if not name.startswith(('torch.nn.modules.', 'ultralytics.nn.')):
            raise ValueError(f'Unapproved checkpoint global: {name}')
        module, attr = name.rsplit('.', 1)
        classes.append(getattr(importlib.import_module(module), attr))
    with torch.serialization.safe_globals(classes):
        state = torch.load(path, map_location='cpu', weights_only=True)
    return (state.get('ema') or state['model']).float().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    path = ROOT/'artifacts/open_source_research_20261009/yolo_fall.pt'
    model = load_restricted(path)
    names = model.names
    fall_indices = [int(k) for k, v in names.items() if v.lower() in ('fallen', 'fall', 'falling')]
    if not fall_indices:
        raise ValueError(f'No explicit fall class in {names}')
    report = {'state': 'running', 'model': 'YOLO11 public fall candidate',
              'source': 'https://huggingface.co/melihuzunoglu/human-fall-detection',
              'revision': '97261b03632a9715a27481b2cfdac6a78e26d2d9',
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'license': 'AGPL-3.0',
              'classes': names, 'parameters': sum(p.numel() for p in model.parameters()),
              'device': 'CPU', 'rows': [],
              'notes': 'Public development replay only; Fallen includes static ground lying. '
                       'Prediction: >=2 sampled frames with winning fall class >=0.5. '
                       'No training split information supplied by author; overlap is unknown.'}
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)
    for item in json.loads(args.list.read_text(encoding='utf-8-sig')):
        cap = cv2.VideoCapture(item['file'])
        if not cap.isOpened():
            raise ValueError('Cannot open evaluation input')
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, round(fps/2))
        index = count = high = 0
        peak = 0.
        timings = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if (index-1) % stride:
                continue
            scale = min(640/frame.shape[1], 640/frame.shape[0])
            width, height = round(frame.shape[1]*scale), round(frame.shape[0]*scale)
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            image = np.full((640, 640, 3), 114, np.uint8)
            left, top = (640-width)//2, (640-height)//2
            image[top:top+height, left:left+width] = small
            tensor = torch.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1)[None].float()/255
            start = time.perf_counter()
            with torch.inference_mode():
                output = model(tensor)[0][0]
            timings.append((time.perf_counter()-start)*1000)
            classes = output[4:]
            winners = classes.argmax(0)
            fall = torch.zeros_like(winners, dtype=torch.bool)
            for label in fall_indices:
                fall |= winners == label
            score = float(classes[:, fall].max()) if fall.any() else 0.
            peak = max(peak, score)
            high += score >= .5
            count += 1
        cap.release()
        row = dict(file=item['file'], truth_fall=item['fall'], sampled_frames=count,
                   peak=peak, predicted_fall=high >= 2, high_frames=high,
                   mean_inference_ms=float(np.mean(timings)))
        report['rows'].append(row)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(Path(item['file']).parent.name, Path(item['file']).name, item['fall'],
              high >= 2, round(peak, 4), flush=True)
    rows = report['rows']
    report['confusion'] = {
        'tp': sum(r['truth_fall'] and r['predicted_fall'] for r in rows),
        'fn': sum(r['truth_fall'] and not r['predicted_fall'] for r in rows),
        'fp': sum(not r['truth_fall'] and r['predicted_fall'] for r in rows),
        'tn': sum(not r['truth_fall'] and not r['predicted_fall'] for r in rows),
    }
    report['state'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report['confusion'], flush=True)


if __name__ == '__main__':
    main()
