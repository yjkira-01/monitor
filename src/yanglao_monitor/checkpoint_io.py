"""Serialize a complete training checkpoint once, then atomically publish its names."""
import os
from pathlib import Path
import shutil
import tempfile

import torch


def save_training_checkpoint(state, last_path, selected_paths=(), staging_dir=None):
    last_path = Path(last_path)
    selected_paths = [Path(path) for path in selected_paths]
    targets = [*selected_paths, last_path]
    if len(set(targets)) != len(targets):
        raise ValueError('Checkpoint output paths must be distinct')
    if staging_dir:
        Path(staging_dir).mkdir(parents=True, exist_ok=True)
    temporary_paths = []

    def temporary_for(target):
        fd, name = tempfile.mkstemp(prefix=target.name+'.', suffix='.tmp', dir=target.parent)
        os.close(fd)
        path = Path(name)
        temporary_paths.append(path)
        return path

    try:
        last_temporary = temporary_for(last_path)
        if staging_dir:
            with tempfile.TemporaryDirectory(prefix='checkpoint-', dir=staging_dir) as folder:
                local_path = Path(folder)/'state.pt'
                torch.save(state, local_path)
                shutil.copyfile(local_path, last_temporary)
        else:
            torch.save(state, last_temporary)
        with last_temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        prepared = []
        for target in selected_paths:
            temporary = temporary_for(target)
            temporary.unlink()
            try:
                os.link(last_temporary, temporary)
            except OSError:
                shutil.copyfile(last_temporary, temporary)
                with temporary.open('r+b') as stream:
                    os.fsync(stream.fileno())
            prepared.append((temporary, target))
        # Replace last only after every selected checkpoint is complete; never modify a linked inode in place.
        for temporary, target in [*prepared, (last_temporary, last_path)]:
            temporary.replace(target)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
