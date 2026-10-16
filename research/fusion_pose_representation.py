"""GT-free alternative for preserving visual information without a hip anchor."""

import numpy as np


def visible_centroid_features(poses, focal):
    poses, focal = np.asarray(poses, dtype=np.float64), np.asarray(focal, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (17, 3) or focal.shape != (2,) or not np.isfinite(poses).all() or not np.isfinite(focal).all() or (focal <= 0).any():
        raise ValueError("Invalid predicted keypoints or camera intrinsic")
    visible = poses[..., 2] >= .3
    weights = np.where(visible, np.clip(poses[..., 2], 0., 1.), 0.)
    anchor = (poses[..., :2] * weights[..., None]).sum(1) / weights.sum(1).clip(1e-8)[:, None]
    xy = (poses[..., :2] - anchor[:, None]) / focal
    xy = np.where(visible[..., None], xy, 0.)
    confidence = np.where(visible, poses[..., 2], 0.)
    return np.concatenate((xy.reshape(len(poses), -1), confidence), -1)
