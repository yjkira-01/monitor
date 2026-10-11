"""端侧视频姿态与跌倒识别。

原始画面只在本地处理；这里输出可供页面展示的结构化状态，不保存视频。
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - 后端运行环境固定带 OpenCV
    cv2 = None

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - 边缘机可按需安装
    mp = None


class VideoPoseAnalyzer:
    """MediaPipe Pose + 可解释几何规则，提供稳定的页面级识别结果。"""

    def __init__(self, sample_every: int = 3, window_size: int = 4, threshold: float = 0.62) -> None:
        self.sample_every = max(1, sample_every)
        self.window_size = max(2, window_size)
        self.threshold = threshold
        self._pose = None
        self._scores: deque[float] = deque(maxlen=self.window_size)
        self._red = False
        self._event_id = 0
        self._frames = 0
        self.enabled = mp is not None and cv2 is not None
        self._pose_api = bool(self.enabled and hasattr(mp, "solutions"))
        self.enabled = cv2 is not None
        self._hog = None
        self._prev_gray = None
        self._last = self._base()

    def _base(self) -> dict[str, Any]:
        return {
            "source": "video",
            "device_id": "CAM_01",
            "analysis_enabled": self.enabled,
            "analysis_backend": "mediapipe_pose" if self._pose_api else "opencv_hog_motion" if self.enabled else "unavailable",
            "pose_detected": False,
            "fall_score": 0.0,
            "confidence": 0.0,
            "action": "normal_activity",
            "is_fall": False,
            "alert_level": None,
            "watch": False,
            "needs_review": False,
            "watch_reason": None,
            "event_id": 0,
            "frames_analyzed": 0,
            "updated_at_ms": 0,
        }

    def _ensure_pose(self) -> None:
        if self._pose_api and self._pose is None:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def _fallback_process(self, frame, now_ms: int) -> dict[str, Any]:
        """OpenCV 内置行人 HOG + 帧差兜底，保证没有旧版 MediaPipe 也能识别。"""
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        motion = 0.0
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            motion = min(1.0, float((diff > 18).mean()) * 5.0)
        self._prev_gray = gray
        if self._hog is None:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = self._hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(boxes) == 0:
            self._last = {**self._last, "pose_detected": False, "confidence": 0.0, "bbox": None, "updated_at_ms": now_ms}
            return dict(self._last)
        best_index = max(range(len(boxes)), key=lambda index: float(boxes[index][2] * boxes[index][3]))
        x, y, width, height = [int(value) for value in boxes[best_index]]
        aspect = width / max(height, 1)
        horizontal = max(0.0, min(1.0, (aspect - 0.85) / 1.15))
        score = round(min(1.0, 0.72 * horizontal + 0.28 * motion), 3)
        confidence = round(min(0.86, 0.52 + float(weights[best_index]) * 0.12), 3)
        return self._update_result(score, confidence, now_ms, (x, y, width, height), "opencv_hog_motion")

    def _update_result(self, score: float, confidence: float, now_ms: int, bbox, backend: str) -> dict[str, Any]:
        self._scores.append(score)
        red = len(self._scores) == self.window_size and all(value >= self.threshold for value in self._scores)
        if red and not self._red:
            self._event_id += 1
        self._red = red
        action = "lying_floor" if red else "falling" if score >= self.threshold else "lying" if score >= 0.5 else "sitting_still" if score >= 0.3 else "standing"
        self._last = {
            **self._last,
            "analysis_enabled": True,
            "analysis_backend": backend,
            "pose_detected": True,
            "fall_score": score,
            "confidence": confidence,
            "action": action,
            "is_fall": red,
            "alert_level": "red" if red else "yellow" if score >= self.threshold else None,
            "watch": not red and score >= self.threshold,
            "needs_review": not red and score >= self.threshold,
            "watch_reason": "人体框接近水平，需要人工确认" if not red and score >= self.threshold else None,
            "event_id": self._event_id,
            "frames_analyzed": self._frames,
            "bbox": bbox,
            "updated_at_ms": now_ms,
        }
        return dict(self._last)

    @staticmethod
    def _mid(points: dict[int, tuple[float, float, float]], first: int, second: int):
        a, b = points.get(first), points.get(second)
        if not a or not b:
            return None
        return tuple((a[i] + b[i]) / 2 for i in range(3))

    @staticmethod
    def _angle_with_vertical(vx: float, vy: float) -> float:
        magnitude = math.hypot(vx, vy) + 1e-9
        cosine = abs(vy) / magnitude
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _score(self, points: dict[int, tuple[float, float, float]]) -> float:
        shoulder = self._mid(points, 11, 12)
        hip = self._mid(points, 23, 24)
        nose = points.get(0)
        if shoulder is None or hip is None:
            return 0.0
        angle = self._angle_with_vertical(shoulder[0] - hip[0], shoulder[1] - hip[1])
        height_ratio = 1.0
        if abs(hip[1] - shoulder[1]) > 1e-6:
            head_y = nose[1] if nose else shoulder[1]
            height_ratio = abs(head_y - hip[1]) / abs(hip[1] - shoulder[1])
        angle_score = min(1.0, angle / 75.0)
        height_score = max(0.0, 1.0 - min(1.0, height_ratio))
        return round(min(1.0, max(0.0, 0.7 * angle_score + 0.3 * height_score)), 3)

    def process(self, frame) -> dict[str, Any]:
        self._frames += 1
        if not self.enabled or self._frames % self.sample_every:
            return dict(self._last)
        now_ms = int(time.time() * 1000)
        if not self._pose_api:
            return self._fallback_process(frame, now_ms)
        self._ensure_pose()
        result = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            self._last = {**self._last, "pose_detected": False, "confidence": 0.0, "updated_at_ms": now_ms}
            return dict(self._last)

        landmarks = result.pose_landmarks.landmark
        points = {i: (lm.x, lm.y, lm.z) for i, lm in enumerate(landmarks)}
        score = self._score(points)
        visibility = sum(float(getattr(lm, "visibility", 0.0)) for lm in landmarks) / max(len(landmarks), 1)
        confidence = round(min(0.99, max(0.5, 0.55 + visibility * 0.4)), 3)
        self._scores.append(score)
        red = len(self._scores) == self.window_size and all(value >= self.threshold for value in self._scores)
        if red and not self._red:
            self._event_id += 1
        self._red = red
        action = "lying_floor" if red else "falling" if score >= self.threshold else "lying" if score >= 0.5 else "sitting_still" if score >= 0.3 else "standing"
        self._last = {
            "source": "video",
            "device_id": "CAM_01",
            "analysis_enabled": True,
            "analysis_backend": "mediapipe_pose",
            "pose_detected": True,
            "fall_score": score,
            "confidence": confidence,
            "action": action,
            "is_fall": red,
            "alert_level": "red" if red else "yellow" if score >= self.threshold else None,
            "watch": not red and score >= self.threshold,
            "needs_review": not red and score >= self.threshold,
            "watch_reason": "姿态接近倒地，需要人工确认" if not red and score >= self.threshold else None,
            "event_id": self._event_id,
            "frames_analyzed": self._frames,
            "updated_at_ms": now_ms,
            "bbox": None,
        }
        return dict(self._last)

    def annotate(self, frame):
        if cv2 is None:
            return frame
        data = self._last
        color = (74, 90, 198) if data.get("is_fall") else (74, 153, 119) if data.get("pose_detected") else (150, 150, 150)
        cv2.rectangle(frame, (18, 18), (350, 86), (16, 24, 28), -1)
        label = f"VIDEO  {data.get('action', 'unknown').upper()}"
        score = f"fall {float(data.get('fall_score', 0.0)):.2f}  conf {float(data.get('confidence', 0.0)):.2f}"
        cv2.putText(frame, label, (32, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(frame, score, (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 232, 226), 1, cv2.LINE_AA)
        if data.get("bbox"):
            x, y, width, height = data["bbox"]
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 360
            cv2.rectangle(frame, (int(x * scale_x), int(y * scale_y)), (int((x + width) * scale_x), int((y + height) * scale_y)), color, 2)
        if data.get("is_fall"):
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (74, 90, 198), 5)
        return frame

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
