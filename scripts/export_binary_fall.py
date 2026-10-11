"""Export a saved binary checkpoint for native comparison; never starts training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from export_desktop_models import VideoExport, export

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--name', choices=['gmd', 'stage_c'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261009)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert state['classes'] == ['adl', 'fall'] and state['clip_len'] == 16
    model = VideoExport(state).eval()
    assets = ROOT / 'desktop/CareMonitor/Assets/Binary' / args.name
    assets.mkdir(parents=True, exist_ok=True)
    width, height = state['read_size']
    info = export(model, torch.rand(1, 16, height, width, 3)*255,
                  assets/'video.onnx', ['probabilities', 'embedding', 'attention'])
    info.update(classes=state['classes'], revision=args.checkpoint.stem,
                read_size=list(state['read_size']),
                low_light_trained=state.get('low_light_trained', False),
                partial_body_trained=state.get('partial_body_trained', False),
                source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                acceptance='saved_checkpoint_comparison', clinical_validated=False,
                training_metrics=state['metrics'])
    (assets/'models.json').write_text(json.dumps({'video': info}, indent=2), encoding='utf-8')
    frames = np.array([((np.arange(width*height*3, dtype=np.int64)+t*7)%251)
                       .astype(np.uint8).reshape(height, width, 3) for t in range(16)])
    from backend.app.video_person_state import enhance_clip
    dark = np.array([(f*(.08+.02*t)).astype(np.uint8) for t, f in enumerate(frames)])
    night = enhance_clip(list(dark), trained=info['low_light_trained'])
    with torch.inference_mode():
        scores = model(torch.from_numpy(frames[..., [2, 1, 0]].copy()).float()[None])[0]
        night_scores = model(torch.from_numpy(np.array(night)[..., [2, 1, 0]].copy()).float()[None])[0]
    (assets/'selfcheck.json').write_text(json.dumps({
        'video_scores': scores.flatten().tolist(), 'night_scores': night_scores.flatten().tolist(),
        'source_sha256': info['source_sha256'],
    }), encoding='utf-8')
    print(json.dumps({'name': args.name, 'export': info}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
