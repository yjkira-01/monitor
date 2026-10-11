"""不依赖深度模型的姿态判据，供 YOLO Pose 或其它关键点模型复用。"""

from __future__ import annotations

import numpy as np


def posture_fall_score(box_xyxy: np.ndarray | list[float], keypoints_xy: np.ndarray | None = None) -> float:
    """根据人体框和肩/髋关键点返回 0..1 的疑似跌倒分数。"""
    box = np.asarray(box_xyxy, dtype=float).reshape(4)
    width = max(box[2] - box[0], 1.0)
    height = max(box[3] - box[1], 1.0)
    ratio_score = np.clip((width / height - 0.9) / 0.9, 0.0, 1.0)
    score = float(ratio_score)
    if keypoints_xy is not None:
        points = np.asarray(keypoints_xy, dtype=float)
        if points.ndim == 2 and points.shape[0] >= 13 and np.isfinite(points).all():
            shoulder = points[[5, 6]].mean(axis=0)
            hip = points[[11, 12]].mean(axis=0)
            torso_height = abs(float(hip[1] - shoulder[1]))
            torso_width = max(float(np.linalg.norm(points[5] - points[6])), 1.0)
            keypoint_score = np.clip((torso_width / max(torso_height, 1.0) - 0.8) / 1.0, 0.0, 1.0)
            score = max(score, float(keypoint_score))
    return float(np.clip(score, 0.0, 1.0))


class TemporalFallFilter:
    """用连续帧确认告警，避免单帧姿态误报。"""

    def __init__(self, threshold: float = 0.65, patience: int = 3) -> None:
        if patience < 1:
            raise ValueError("patience 必须 >= 1")
        self.threshold = threshold
        self.patience = patience
        self._streak = 0

    def update(self, score: float) -> bool:
        self._streak = self._streak + 1 if score >= self.threshold else 0
        return self._streak >= self.patience
