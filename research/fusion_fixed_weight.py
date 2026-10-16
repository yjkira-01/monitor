"""Ordinary train-only convex late-fusion baselines; no novelty claim."""

import numpy as np


def fit_fixed_weights(video, radar, target, weights):
    """Global and jointwise constrained SSE minimizers, supplied OOF data only."""
    video, radar, target = [np.asarray(x, dtype=np.float64) for x in (video, radar, target)]
    weights = np.asarray(weights, dtype=np.float64)
    if (video.shape != radar.shape or video.shape != target.shape or video.ndim != 3
            or video.shape[1:] != (17, 3) or weights.shape != (len(video),)
            or not all(np.isfinite(x).all() for x in (video, radar, target, weights))
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError("Finite N x 17 x 3 OOF poses and positive sample weights required")
    delta, residual = radar - video, target - video
    numerator = (weights[:, None, None] * delta * residual).sum((0, 2))
    denominator = (weights[:, None, None] * delta ** 2).sum((0, 2))
    joint = np.clip(np.divide(numerator, denominator, out=np.zeros(17), where=denominator > 1e-12), 0., 1.)
    global_weight = float(np.clip(numerator.sum() / denominator.sum(), 0., 1.)) if denominator.sum() > 1e-12 else 0.
    return global_weight, joint


def fixed_fusion(video, radar, video_valid, radar_valid, alpha):
    """Inference accepts predictions/availability only; target cannot enter."""
    video, radar = np.asarray(video), np.asarray(radar)
    video_valid, radar_valid = np.asarray(video_valid), np.asarray(radar_valid)
    alpha = np.asarray(alpha, dtype=np.float64)
    if (video.shape != radar.shape or video.ndim != 3 or video.shape[1:] != (17, 3)
            or video_valid.shape != (len(video),) or radar_valid.shape != (len(video),)
            or video_valid.dtype != np.bool_ or radar_valid.dtype != np.bool_
            or alpha.shape not in ((), (17,)) or not np.isfinite(alpha).all()
            or ((alpha < 0) | (alpha > 1)).any()
            or not np.isfinite(video).all() or not np.isfinite(radar).all()):
        raise ValueError("Invalid predictions, availability or convex weights")
    weight = alpha if alpha.ndim == 0 else alpha[None, :, None]
    fused = video + weight * (radar - video)
    fused = np.where(~radar_valid[:, None, None], video, fused)
    return np.where(~video_valid[:, None, None], radar, fused)


def balanced_fit_weights(subject_ids, condition_ids, both):
    """Equal subject/condition block weights; unavailable examples carry zero."""
    subject_ids, condition_ids, both = [np.asarray(x) for x in (subject_ids, condition_ids, both)]
    if subject_ids.shape != condition_ids.shape or subject_ids.shape != both.shape or both.dtype != np.bool_:
        raise ValueError("Aligned subject/condition/availability vectors required")
    weights = np.zeros(len(both), dtype=np.float64)
    for subject in np.unique(subject_ids):
        for condition in np.unique(condition_ids):
            selected = (subject_ids == subject) & (condition_ids == condition) & both
            if selected.any():
                weights[selected] = 1. / selected.sum()
    if weights.sum() <= 0:
        raise ValueError("No both-available OOF fitting examples")
    return weights / weights.sum()
