"""Native paper blocks and ordinary objective/coordinate controls."""

import numpy as np
import torch

METHODS = ("native_mse", "native_mpjpe", "root_mpjpe")


def native_radar_batch(points):
    """Restore variable-length observations before native zero-padded collate.

    The release's all-zero placeholders are not observations. Keep every nonzero
    five-field row, its order and coordinates; pad to this batch's max length.
    This implements the public X-Fi loader's variable-length padding convention.
    """
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 5 or not np.isfinite(points).all():
        raise ValueError("Finite B x N x 5 published observations required")
    valid = np.any(points != 0, axis=-1)
    count = valid.sum(1)
    padded = np.zeros((len(points), max(1, int(count.max())), 5), dtype=points.dtype)
    for index in range(len(points)):
        padded[index, :count[index]] = points[index, valid[index]]
    return padded, count


def recenter_prediction(pose):
    return pose - ((pose[:, 11] + pose[:, 12]) / 2)[:, None]


def token_prediction(model, rgb_CxT, radar_tokens, rgb_available, radar_available, root=False):
    if rgb_CxT.shape != (len(rgb_CxT), 512, 49) or radar_tokens.shape != (len(rgb_CxT), 32, 512):
        raise ValueError("Native RGB and radar token shapes required")
    prediction = model.training_prior_m[None].expand(len(rgb_CxT), -1, -1).clone()
    for image_flag, point_flag in ((True, True), (True, False), (False, True)):
        selected = (rgb_available == image_flag) & (radar_available == point_flag)
        if selected.any():
            # CxT storage makes transpose reproduce the original ResNet view's
            # strides, including its contiguous Conv1d input after transpose back.
            features = []
            if image_flag:
                features.append(rgb_CxT[selected].transpose(1, 2))
            if point_flag:
                features.append(radar_tokens[selected])
            flag = [image_flag, False, point_flag, False, False]
            projected = model.linear_projector(features, None, flag)
            prediction[selected] = model.X_Fusion_block(projected, flag)
    return recenter_prediction(prediction) if root else prediction


def training_loss(prediction, target_m, method):
    if method not in METHODS:
        raise ValueError("Unknown objective control")
    return (prediction - target_m).square().mean() if method == "native_mse" else (prediction - target_m).norm(dim=-1).mean()
