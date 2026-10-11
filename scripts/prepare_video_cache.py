"""Prepare train/validation RGB caches without using CUDA or touching test data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from yanglao_monitor.clip_cache import disk_cached_dataset
from yanglao_monitor.video import VideoManifestDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.time()
    digest = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    report = {'manifest_sha256': digest, 'cache_dir': args.cache_dir, 'started_at': started, 'splits': {}}
    train_set = VideoManifestDataset(args.manifest, 'train', 8, (224, 224), preserve_aspect=True)
    signature = {'schema': 1, 'manifest_sha256': digest, 'clip_len': 8, 'read_size': [224, 224],
                 'classes': train_set.classes, 'preserve_aspect': True}
    for split in ('val', 'train'):
        dataset = VideoManifestDataset(args.manifest, split, 8, (224, 224),
                                       classes=train_set.classes, preserve_aspect=True)
        before = time.time()
        cached = disk_cached_dataset(dataset, args.cache_dir, {**signature, 'split': split}, args.workers)
        report['splits'][split] = {'windows': len(cached), 'seconds': time.time()-before,
                                  'bytes': cached.tensors[0].numel()}
        Path(args.output).write_text(json.dumps(report, indent=2))
        del cached
    report['complete'] = True
    report['seconds'] = time.time()-started
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
