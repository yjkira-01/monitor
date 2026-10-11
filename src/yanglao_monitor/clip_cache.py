"""Persistent RGB uint8 clip cache on local storage; no train-time augmentation saved."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def disk_cached_dataset(dataset, cache_dir, signature, workers=0):
    signature = json.loads(json.dumps(signature, sort_keys=True))
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = Path(cache_dir) / key
    ready = folder / 'ready.json'
    expected_labels = np.array([label for _, label in dataset.samples], dtype=np.int64)
    expected_shape = (len(dataset), 3, dataset.clip_len, dataset.size[1], dataset.size[0])
    if not ready.exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        # A commit marker prevents an interrupted or concurrent build being mistaken for usable data.
        folder.mkdir()
        loader = iter(DataLoader(dataset, batch_size=1, num_workers=workers))
        first = next(loader)
        clips = np.lib.format.open_memmap(folder / 'clips.npy', mode='w+', dtype=np.uint8, shape=expected_shape)
        labels = np.empty(len(dataset), dtype=np.int64)
        for index, (x, y) in enumerate(itertools.chain([first], loader)):
            if tuple(x.shape) != (1, *expected_shape[1:]) or not torch.isfinite(x).all():
                raise ValueError('Invalid RGB clip shape or nonfinite pixels')
            if x.dtype != torch.uint8 and (x.min() < 0 or x.max() > 1):
                raise ValueError('RGB pixels must be 0..1 or uint8')
            pixels = x if x.dtype == torch.uint8 else (x * 255).round().to(torch.uint8)
            clips[index] = pixels[0].numpy()
            labels[index] = int(y[0])
            if (index + 1) % 200 == 0:
                print(f'disk_caching {index+1}/{len(dataset)}', flush=True)
        if not np.array_equal(labels, expected_labels):
            raise ValueError('Cached labels do not match the frozen dataset')
        clips.flush()
        del clips
        np.save(folder / 'labels.npy', labels)
        temporary = ready.with_suffix('.tmp')
        temporary.write_text(json.dumps({'signature': signature, 'shape': expected_shape}, indent=2))
        temporary.replace(ready)
    metadata = json.loads(ready.read_text())
    if metadata['signature'] != signature or tuple(metadata['shape']) != expected_shape:
        raise ValueError('Clip cache signature/shape mismatch')
    # Copy-on-write permits a torch view while keeping cache files unchanged.
    clips = np.load(folder / 'clips.npy', mmap_mode='c')
    labels = np.load(folder / 'labels.npy', allow_pickle=False)
    if tuple(clips.shape) != expected_shape or clips.dtype != np.uint8:
        raise ValueError('Invalid clip cache tensor')
    if labels.dtype != np.int64 or not np.array_equal(labels, expected_labels):
        raise ValueError('Invalid cached labels')
    print(f'disk_cached {len(dataset)} clips {clips.nbytes/1024**2:.1f} MiB from {folder}', flush=True)
    return TensorDataset(torch.from_numpy(clips), torch.from_numpy(labels))
