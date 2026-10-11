from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from yanglao_monitor.checkpoint_io import save_training_checkpoint


def test_save_once_preserves_best_and_full_resume_state(tmp_path):
    last, best, recall = [tmp_path/name for name in ('last.pt', 'best.pt', 'recall.pt')]
    state = {'epoch': 1, 'model': {'weight': torch.tensor([1.])},
             'optimizer': {'step': torch.tensor(4.), 'exp_avg': torch.tensor([.2])},
             'scheduler': {'last_epoch': 1}, 'torch_rng': torch.get_rng_state()}
    with patch('yanglao_monitor.checkpoint_io.torch.save', wraps=torch.save) as save:
        save_training_checkpoint(state, last, [best, recall], tmp_path/'staging')
        assert save.call_count == 1
    assert last.read_bytes() == best.read_bytes() == recall.read_bytes()
    loaded = torch.load(last, weights_only=False)
    assert torch.equal(loaded['optimizer']['exp_avg'], state['optimizer']['exp_avg'])
    assert torch.equal(loaded['torch_rng'], state['torch_rng'])
    save_training_checkpoint({**state, 'epoch': 2}, last, [recall], tmp_path/'staging')
    assert torch.load(best, weights_only=False)['epoch'] == 1
    assert torch.load(recall, weights_only=False)['epoch'] == 2
    assert torch.load(last, weights_only=False)['epoch'] == 2
    assert not list((tmp_path/'staging').iterdir())
    assert not list(tmp_path.glob('*.tmp'))


def test_hardlink_failure_falls_back_to_complete_copy(tmp_path):
    with patch('yanglao_monitor.checkpoint_io.os.link', side_effect=OSError('unsupported')):
        save_training_checkpoint({'epoch': 3}, tmp_path/'last.pt', [tmp_path/'best.pt'])
    assert (tmp_path/'last.pt').read_bytes() == (tmp_path/'best.pt').read_bytes()


def test_preparation_failure_keeps_committed_outputs(tmp_path):
    last, best = tmp_path/'last.pt', tmp_path/'best.pt'
    save_training_checkpoint({'epoch': 1}, last, [best])
    original = last.read_bytes()
    with patch('yanglao_monitor.checkpoint_io.shutil.copyfile', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            save_training_checkpoint({'epoch': 2}, last, [best], tmp_path/'staging')
    assert last.read_bytes() == best.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))
    assert not list((tmp_path/'staging').iterdir())


def test_publication_replaces_last_after_selected_checkpoints(tmp_path):
    last, best = tmp_path/'last.pt', tmp_path/'best.pt'
    real_replace, order = Path.replace, []

    def record(source, target):
        order.append(target)
        return real_replace(source, target)

    with patch.object(Path, 'replace', record):
        save_training_checkpoint({'epoch': 1}, last, [best])
    assert order == [best, last]
