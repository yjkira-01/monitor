"""Compare dark-video preprocessing on fixed labeled clips; synthetic darkness is not a night test."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
sys.path.insert(0, str(ROOT))
from app.video_model import TrainedVideoModel
from app.video_person_state import enhance_clip
from scripts.audit_video_generalization import sampled_frames
from scripts.train_video import classification_metrics
from yanglao_monitor.video import _read_frames


def main(args):
    with Path(args.manifest).open(encoding='utf-8-sig') as handle:
        rows = [r for r in csv.DictReader(handle) if r['split'] == args.split]
    if not rows:
        raise ValueError('No labeled evaluation clips')
    model = TrainedVideoModel(Path(args.checkpoint))
    model.enhance_low_light = False
    rng = np.random.default_rng(0)
    results = []
    started = time.monotonic()
    for index, row in enumerate(rows):
        if row.get('start_s'):
            tensor = _read_frames(ROOT/row['path'],16,model.read_size,float(row['start_s']),float(row['end_s']))
            original = [cv2.cvtColor(f,cv2.COLOR_RGB2BGR) for f in (tensor.permute(1,2,3,0).numpy()*255).round().astype(np.uint8)]
        else:
            original = [cv2.resize(f, model.read_size, interpolation=cv2.INTER_AREA) for f in sampled_frames(ROOT/row['path'])]
        cases = {'clear': original,
                 'exposure_12pct': [(f*.12).astype(np.uint8) for f in original],
                 'gamma_noise': [(np.clip((f/255.)**2.2*.35+rng.normal(0, .004, f.shape), 0, 1)*255).astype(np.uint8) for f in original]}
        scores = {}
        for name, frames in cases.items():
            brightness = float(np.median([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]))
            gain = min(8., 80/max(brightness, 1.)) if brightness < 65 else 1.
            linear = [np.clip(f.astype(np.float32)*gain, 0, 255).astype(np.uint8) for f in frames]
            predictions = model.predict_many([frames, enhance_clip(frames), linear, enhance_clip(frames,trained=False)])
            scores[name] = dict(zip(('raw', 'gamma', 'gain','adaptive'), [p['fall_score'] for p in predictions]))
            assert all(np.isfinite(v) and 0 <= v <= 1 for v in scores[name].values())
        results.append({'path': row['path'], 'label': row['label'], 'scores': scores})
        if (index+1) % 5 == 0 or index == len(rows)-1:
            print(f'low-light audit {index+1}/{len(rows)}', flush=True)
    labels = [int(r['label'] == 'fall') for r in results]
    metrics = {case: {method: classification_metrics([int(r['scores'][case][method] >= .5) for r in results], labels, 1)
                      for method in ('raw', 'gamma', 'gain','adaptive')} for case in results[0]['scores']}
    report = {'checkpoint': args.checkpoint, 'manifest': args.manifest, 'split': args.split, 'count': len(rows),
              'seconds': time.monotonic()-started, 'metrics': metrics, 'predictions': results,
              'note': 'Fixed weights and 0.5 threshold; synthetic exposure/gamma/noise. Development comparison, not real-night accuracy or an independent blind test.'}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'count': len(rows), 'seconds': report['seconds'], 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/manifests/gmdcsa24.csv')
    parser.add_argument('--split', default='val')
    parser.add_argument('--checkpoint', default='artifacts/gmdcsa24_r3d18_100ep_inference.pt')
    parser.add_argument('--output', default='artifacts/low_light_comparison.json')
    main(parser.parse_args())
