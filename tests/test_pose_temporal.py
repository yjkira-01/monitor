import csv
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from scripts.download_multifall import extract
from scripts.train_video import prepare_batch, validate_splits, cache_dataset
from yanglao_monitor.pose_temporal import PoseManifestDataset, PoseTemporalNet, pose_features


def test_pose_masks_motion_and_finite_backward():
    torch.set_num_threads(1)
    pose = torch.zeros(2, 3, 32, 33)
    pose[:, 2] = 1
    pose[:, 0, :, 11:13] = .4
    pose[:, 0, :, 23:25] = .4
    pose[:, 1, :, 11:13] = .3
    pose[:, 1, :, 23:25] = .6
    pose[0, 0, 0, 0] = float('nan')
    features = pose_features(pose)
    assert features.shape == (2, 169, 32) and torch.isfinite(features).all()
    assert features[0, 132, 0] == 0  # Invalid joint visibility.
    assert features[:, 66:132].abs().sum() == 0  # Static body has no velocity.
    assert not pose_features(torch.zeros_like(pose)).any()
    moving = pose.clone()
    moving[:, 1, 16:] += .1
    assert pose_features(moving)[:, 165:167].abs().sum() > 0
    model = PoseTemporalNet()
    loss = torch.nn.functional.cross_entropy(model(prepare_batch(pose, augment=True)), torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        pose_features(torch.zeros(1, 3, 16, 17))


def test_pose_cache_keeps_float_coordinates_and_samples_window(tmp_path):
    root = tmp_path
    manifest = root/'data/manifests/pose.csv'
    manifest.parent.mkdir(parents=True)
    pose = np.full((10, 33, 3), .375, np.float32)
    np.savez(root/'pose.npz', times=np.arange(10)/10, poses=pose)
    rows = [dict(path=f'{label}.mp4', split='train', label=label, start_s='.2', end_s='.5', pose_path='pose.npz') for label in ['adl', 'fall']]
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dataset = PoseManifestDataset(manifest, 'train', 8)
    cached = cache_dataset(dataset, 0)
    assert cached[0][0].shape == (3, 8, 33)
    assert cached[0][0].dtype == torch.float32 and torch.equal(cached[0][0], torch.full((3,8,33), .375))


def test_subject_camera_leakage_and_archive_paths(tmp_path):
    rows = [dict(path='camera1.avi', dataset='up', subject='1', parent_id='trial1', split='train'),
            dict(path='camera2.avi', dataset='up', subject='1', parent_id='trial1', split='test')]
    with pytest.raises(ValueError, match='leaked'):
        validate_splits(rows)
    rows[1].update(dataset='other', subject='1', parent_id='trial2')
    validate_splits(rows)
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('../escape', 'unsafe')
    with pytest.raises(ValueError, match='Unsafe'):
        extract(archive, tmp_path/'output')


def test_strong_video_augmentation_preserves_order():
    clip = torch.rand(2, 3, 1, 16, 16).expand(-1, -1, 4, -1, -1)
    output = prepare_batch(clip, augment=True, strong=True)
    assert output.shape == clip.shape and torch.equal(output[:, :, 0], output[:, :, -1])
    assert output.min() >= 0 and output.max() <= 1
