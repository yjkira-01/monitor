"""GT-free, hip-preserving partial-pose representations before normalization."""

import numpy as np

from fusion_pose_representation import visible_centroid_features

PARTIAL_REPRESENTATIONS = ("hip_centroid_fallback", "hip_confidence_only")


def hip_preserving_fallback_features(poses, focal, hip_features, mode="centroid"):
    """Keep original valid hip features; vary only partial-frame raw XY.

    Both modes retain identical partial raw confidence and availability. In the
    confidence-only mode, raw XY=0; an observed field becomes -mean/std after
    the caller's common hip-clean normalization, rather than normalized zero.
    The caller must retain the original cache's radar-augmentation reference.
    """
    poses, focal = np.asarray(poses, dtype=np.float64), np.asarray(focal, dtype=np.float64)
    hip_features = np.asarray(hip_features)
    if (poses.ndim != 3 or poses.shape[1:] != (17, 3) or focal.shape != (2,) or
            hip_features.shape != (len(poses), 51) or hip_features.dtype.kind != "f" or
            not np.isfinite(poses).all() or not np.isfinite(focal).all() or
            not np.isfinite(hip_features).all() or (focal <= 0).any()):
        raise ValueError("Expected predicted poses, positive camera focal and raw hip-cache features")
    if mode not in ("centroid", "confidence_only"):
        raise ValueError("Unknown partial fallback mode")
    visible = poses[..., 2] >= .3
    hip_valid = (hip_features[:, 34:] >= .3).any(1)
    if not np.array_equal(hip_valid, visible[:, 11] & visible[:, 12]) or np.any(hip_features[~hip_valid] != 0):
        raise ValueError("Hip features must match the original raw hip-anchor missingness")
    partial = ~hip_valid & visible.any(1)
    result = hip_features.copy()
    if mode == "centroid" and partial.any():
        result[partial] = visible_centroid_features(poses[partial], focal)
    elif mode == "confidence_only":
        result[partial, :34] = 0.
        result[partial, 34:] = np.where(visible[partial], poses[partial, :, 2], 0.)
    result[~visible.any(1)] = 0.
    return result


def hip_preserving_centroid_fallback(poses, focal, hip_features):
    return hip_preserving_fallback_features(poses, focal, hip_features, mode="centroid")


def hip_preserving_confidence_fallback(poses, focal, hip_features):
    return hip_preserving_fallback_features(poses, focal, hip_features, mode="confidence_only")
