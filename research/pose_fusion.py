"""Research primitives; coordinates and diagonal variances must share units."""

import numpy as np
import torch


def nearest_timestamp_pairs(video_seconds, radar_seconds, max_gap_seconds):
    """Return video indices, absolute gaps and valid matches for each radar time."""
    video = np.asarray(video_seconds, dtype=np.float64)
    radar = np.asarray(radar_seconds, dtype=np.float64)
    for times in (video, radar):
        if times.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Timestamps must be finite, one-dimensional and strictly increasing")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError("Maximum time gap must be finite and nonnegative")
    if not video.size:
        return np.full(radar.size, -1), np.full(radar.size, np.inf), np.zeros(radar.size, bool)
    right = np.searchsorted(video, radar).clip(0, video.size - 1)
    left = (right - 1).clip(0, video.size - 1)
    index = np.where(abs(video[left] - radar) <= abs(video[right] - radar), left, right)
    gap = abs(video[index] - radar)
    valid = gap <= max_gap_seconds
    return np.where(valid, index, -1), gap, valid


def alignment_moments(scores, means, variances, valid):
    """Marginalize (..., K, J, 3) poses over K candidate correspondences.

    scores/valid have shape (..., K). Unavailable rows have zero weights;
    their returned moments are placeholders, never valid observations.
    """
    if means.ndim < 3 or means.shape[-1] != 3 or min(means.shape[-3:-1]) < 1:
        raise ValueError("Candidate poses must have shape (..., K, J, 3) with K,J >= 1")
    if variances.shape != means.shape or scores.shape != means.shape[:-2] or valid.shape != scores.shape:
        raise ValueError("Pose, variance, score and mask shapes disagree")
    if valid.dtype != torch.bool or not means.is_floating_point():
        raise ValueError("Masks must be boolean and poses floating point")
    if any(x.device != means.device for x in (scores, variances, valid)):
        raise ValueError("All tensors must be on the same device")
    if any(x.dtype != means.dtype for x in (scores, variances)):
        raise ValueError("Scores, poses and variances must use the same dtype")
    if not torch.isfinite(scores[valid]).all() or not torch.isfinite(means[valid]).all():
        raise ValueError("Available candidates must have finite scores and poses")
    if not torch.isfinite(variances[valid]).all() or (variances[valid] <= 0).any():
        raise ValueError("Available candidate variances must be finite and positive")
    available = valid.any(dim=-1)
    masked_scores = scores.masked_fill(~valid, -torch.inf)
    masked_scores = torch.where(available[..., None], masked_scores, torch.zeros_like(scores))
    weights = torch.softmax(masked_scores, dim=-1).masked_fill(~valid, 0)
    safe_means = torch.where(valid[..., None, None], means, 0)
    safe_variances = torch.where(valid[..., None, None], variances, 0)
    w = weights[..., None, None]
    mean = (w * safe_means).sum(dim=-3)
    within = (w * safe_variances).sum(dim=-3)
    between = (w * (safe_means - mean.unsqueeze(-3)).square()).sum(dim=-3)
    return {"weights": weights, "mean": mean, "within": within, "between": between,
            "variance": within + between, "available": available}


def fuse_diagonal_poses(video_mean, video_variance, radar_mean, radar_variance,
                        video_valid, radar_valid):
    """Approximate inverse-variance fusion, with explicit per-joint availability.

    Pose tensors have shape (..., J, 3), masks (..., J). This weighting is
    heuristic under correlated modality errors; no fused posterior is claimed.
    """
    shape = video_mean.shape
    if len(shape) < 2 or shape[-1] != 3 or shape[-2] < 1:
        raise ValueError("Poses must have shape (..., J, 3), J >= 1")
    if any(x.shape != shape for x in (video_variance, radar_mean, radar_variance)):
        raise ValueError("Pose and variance shapes disagree")
    if any(x.shape != shape[:-1] or x.dtype != torch.bool for x in (video_valid, radar_valid)):
        raise ValueError("Availability must be a boolean mask for every joint")
    if not video_mean.is_floating_point():
        raise ValueError("Poses must be floating point")
    if any(x.device != video_mean.device for x in
           (video_variance, radar_mean, radar_variance, video_valid, radar_valid)):
        raise ValueError("All tensors must be on the same device")
    if any(x.dtype != video_mean.dtype for x in (video_variance, radar_mean, radar_variance)):
        raise ValueError("All pose and variance tensors must use the same dtype")
    for mu, var, mask in ((video_mean, video_variance, video_valid),
                          (radar_mean, radar_variance, radar_valid)):
        if not torch.isfinite(mu[mask]).all() or not torch.isfinite(var[mask]).all() or (var[mask] <= 0).any():
            raise ValueError("Available poses must be finite with positive finite variances")
    v = torch.where(video_valid[..., None], video_mean, 0)
    r = torch.where(radar_valid[..., None], radar_mean, 0)
    vv = torch.where(video_valid[..., None], video_variance, 1)
    rv = torch.where(radar_valid[..., None], radar_variance, 1)
    # Normalize before computing the ratio to avoid overflow at extreme scales.
    scale = torch.maximum(vv, rv)
    radar_weight = (vv / scale) / (vv / scale + rv / scale)
    radar_weight = torch.where(~video_valid[..., None], 1, radar_weight)
    radar_weight = torch.where(~radar_valid[..., None], 0, radar_weight)
    available = video_valid | radar_valid
    mean = (1 - radar_weight) * v + radar_weight * r
    return {"mean": mean.masked_fill(~available[..., None], torch.nan),
            "radar_weight": radar_weight.masked_fill(~available[..., None], torch.nan),
            "available": available}
