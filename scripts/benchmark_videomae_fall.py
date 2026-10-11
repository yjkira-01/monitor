"""Research-only pretrained VideoMAE comparison. No training and no saved frames."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPO = 'yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect'
REVISION = '0468eaca8465f45e26c365bafbee4348bceafb0d'


def prepare():
    folder = ROOT/'artifacts/open_source_research_20261009/videomae_fall'
    folder.mkdir(parents=True, exist_ok=True)
    meta = json.load(urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=30))
    (folder/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    for name in ('config.json', 'preprocessor_config.json', 'README.md'):
        if not (folder/name).exists():
            (folder/name).write_bytes(urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', timeout=30).read())
    item = next(s for s in meta['siblings'] if s['rfilename'] == 'model.safetensors')
    size, sha = item['size'], item['lfs']['sha256']
    path = folder/'model.safetensors'
    if not path.exists() or path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        step = 8*1024**2
        def part(start):
            end = min(size, start+step)-1
            target = folder/f'{start}.part'
            if target.exists() and target.stat().st_size == end-start+1:
                return target
            error = None
            for attempt in range(3):
                try:
                    req = Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors?chunk={start}',
                                  headers={'Range': f'bytes={start}-{end}'})
                    with urlopen(req, timeout=35) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Unexpected download byte range')
                        data = response.read()
                    if len(data) != end-start+1:
                        raise ValueError('Incomplete range')
                    target.write_bytes(data)
                    print('download', start+len(data), '/', size, flush=True)
                    return target
                except Exception as exc:
                    error = exc
            raise RuntimeError(error)
        with ThreadPoolExecutor(6) as pool:
            paths = list(pool.map(part, range(0, size, step)))
        temporary = folder/'model.download'
        with temporary.open('wb') as output:
            for piece in paths:
                output.write(piece.read_bytes())
        assert temporary.stat().st_size == size and hashlib.sha256(temporary.read_bytes()).hexdigest() == sha
        temporary.replace(path)
    print('Verified pretrained VideoMAE weights', sha, flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = prepare()
    import cv2
    import numpy as np
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    model = VideoMAEForVideoClassification.from_pretrained(folder, local_files_only=True,
                                                         attn_implementation='sdpa').eval()
    processor = VideoMAEImageProcessor.from_pretrained(folder, local_files_only=True)
    fall = [int(i) for i, name in model.config.id2label.items() if name.lower() == 'falldown']
    assert len(fall) == 1
    report = dict(state='running', source=f'https://huggingface.co/{REPO}', revision=REVISION,
                  classes=model.config.id2label, parameters=sum(p.numel() for p in model.parameters()),
                  device='CPU', license='CC-BY-NC-4.0', rows=[],
                  notes='Seven trained classes collapsed into fall versus other. '
                        'Whole-clip 16-frame sampling at author preprocessing; not streaming event accuracy. '
                        'Author training dataset and overlap are unknown.')
    for item in json.loads(args.list.read_text(encoding='utf-8-sig')):
        cap = cv2.VideoCapture(item['file'])
        assert cap.isOpened()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total-1, 16).round().astype(int)
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            assert ok
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        inputs = processor(frames, return_tensors='pt')
        start = time.perf_counter()
        with torch.inference_mode():
            scores = model(**inputs).logits.softmax(-1)[0]
        row = dict(file=item['file'], truth_fall=item['fall'], fall_score=float(scores[fall[0]]),
                   predicted_fall=bool(scores[fall[0]] >= .5), scores=scores.tolist(),
                   inference_ms=(time.perf_counter()-start)*1000)
        report['rows'].append(row)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(Path(item['file']).parent.name, Path(item['file']).name, row['predicted_fall'],
              round(row['fall_score'], 4), round(row['inference_ms']), flush=True)
    rows = report['rows']
    report['confusion'] = dict(tp=sum(r['truth_fall'] and r['predicted_fall'] for r in rows),
                              fn=sum(r['truth_fall'] and not r['predicted_fall'] for r in rows),
                              fp=sum(not r['truth_fall'] and r['predicted_fall'] for r in rows),
                              tn=sum(not r['truth_fall'] and not r['predicted_fall'] for r in rows))
    report['state'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report['confusion'], flush=True)


if __name__ == '__main__':
    main()
