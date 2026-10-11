from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from yanglao_monitor.clip_cache import disk_cached_dataset
from scripts.train_video import validate_execution_resume, restore_optimizer_execution


class Clips:
    clip_len = 2
    size = (4, 3)
    samples = [('first', 0), ('second', 1), ('third', 0)]

    def __init__(self):
        self.calls = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.calls += 1
        pixels = (torch.arange(72).reshape(3, 2, 3, 4) + index).to(torch.uint8)
        return pixels.float() / 255, self.samples[index][1]


def test_disk_cache_preserves_pixels_labels_and_reuses_without_decoding(tmp_path):
    original = Clips()
    signature = {'split': 'train', 'manifest_sha256': 'frozen', 'read_size': (4, 3)}
    cache = disk_cached_dataset(original, tmp_path, signature)
    assert original.calls == 3
    for i in range(3):
        x, y = original[i]
        torch.testing.assert_close(cache[i][0].float()/255, x, rtol=0, atol=0)
        assert int(cache[i][1]) == y
    original.calls = 0
    reused = disk_cached_dataset(original, tmp_path, signature)
    assert original.calls == 0
    assert len(reused) == 3
    # Mutating a returned torch view cannot overwrite the persisted source cache.
    reused[0][0].zero_()
    torch.testing.assert_close(disk_cached_dataset(original, tmp_path, signature)[0][0], cache[0][0])


def test_incomplete_or_corrupt_cache_is_rejected(tmp_path):
    source = Clips()
    disk_cached_dataset(source, tmp_path, {'split': 'val'})
    folder = next(tmp_path.iterdir())
    np.save(folder/'labels.npy', np.array([1, 1, 1], dtype=np.int64))
    with pytest.raises(ValueError, match='labels'):
        disk_cached_dataset(source, tmp_path, {'split': 'val'})
    (folder/'ready.json').unlink()
    with pytest.raises(FileExistsError):
        disk_cached_dataset(source, tmp_path, {'split': 'val'})


def test_execution_resume_requires_same_effective_batch_and_explicit_opt_in():
    old = {'batch_size': 2, 'accumulate': 8}
    args = Namespace(model='internvideo2_base', batch_size=16, accumulate=1, performance_resume=False)
    with pytest.raises(ValueError, match='batch_size'):
        validate_execution_resume(old, args)
    args.performance_resume = True
    validate_execution_resume(old, args)
    args.accumulate = 2
    with pytest.raises(ValueError, match='effective batch'):
        validate_execution_resume(old, args)
    args.accumulate = 1
    args.model = 'tiny'
    with pytest.raises(ValueError, match='InternVideo2'):
        validate_execution_resume(old, args)


def test_legacy_exact_resume_defaults_still_work():
    validate_execution_resume({'batch_size': 2}, Namespace(model='tiny', batch_size=2))


def test_optimizer_execution_change_keeps_moments_and_step():
    p = torch.nn.Parameter(torch.tensor([.1, -.2]))
    optimizer = torch.optim.AdamW([p], lr=2e-5)
    p.grad = torch.tensor([.3, -.5])
    optimizer.step()
    state = optimizer.state[p]
    first, second, count = state['exp_avg'].clone(), state['exp_avg_sq'].clone(), state['step'].clone()
    restore_optimizer_execution(optimizer, False)
    torch.testing.assert_close(state['exp_avg'], first, rtol=0, atol=0)
    torch.testing.assert_close(state['exp_avg_sq'], second, rtol=0, atol=0)
    assert torch.equal(state['step'], count)


def test_checkpoint_count_change_needs_performance_resume():
    args = Namespace(model='internvideo2_base', batch_size=2, accumulate=8, checkpoint_blocks=6)
    with pytest.raises(ValueError, match='checkpoint_blocks'):
        validate_execution_resume({'batch_size': 2, 'accumulate': 8}, args)
    args.performance_resume = True
    validate_execution_resume({'batch_size': 2, 'accumulate': 8}, args)
