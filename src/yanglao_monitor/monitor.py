"""共享视频流、姿态分析和告警状态。"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .pose_fall import TemporalFallFilter, posture_fall_score


class VideoMonitor:
    """把 OpenCV 视频源转换成浏览器可消费的 MJPEG 流。"""

    def __init__(self, source: str | int = 0, pose: bool = False, weights: str = "yolo11n-pose.pt") -> None:
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.pose_enabled = pose
        self.weights = weights
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._model = None
        self._filter = TemporalFallFilter()
        self._events: deque[dict[str, object]] = deque(maxlen=30)
        self._status: dict[str, object] = {
            "connected": False,
            "source": str(self.source),
            "pose_enabled": pose,
            "detector": "loading" if pose else "off",
            "frames": 0,
            "fps": 0.0,
            "fall_score": 0.0,
            "alerts": 0,
            "last_alert": None,
            "error": None,
        }
        self._jpeg = self._placeholder("等待视频源")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="video-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)

    def events(self) -> list[dict[str, object]]:
        with self._lock:
            return list(self._events)

    def mjpeg_stream(self) -> Iterator[bytes]:
        self.start()
        while not self._stop.is_set():
            with self._lock:
                jpeg = self._jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.05)
            else:
                time.sleep(0.05)

    def _run(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            with self._lock:
                self._jpeg = self._placeholder("视频源不可用")
            self._set_status(connected=False, error=f"无法打开视频源: {self.source}")
            return
        fps_source = capture.get(cv2.CAP_PROP_FPS) or 25.0
        previous = time.monotonic()
        frame_count = 0
        try:
            if self.pose_enabled:
                try:
                    from ultralytics import YOLO

                    self._model = YOLO(self.weights)
                    self._set_status(detector="YOLO Pose")
                except Exception as exc:  # 模型下载/驱动失败时仍保留视频流
                    self._set_status(detector="error", error=f"姿态模型不可用: {exc}")
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    if isinstance(self.source, str):
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    with self._lock:
                        self._jpeg = self._placeholder("视频流已结束")
                    self._set_status(connected=False, error="视频源已结束或读取失败")
                    break
                frame_count += 1
                frame, score, alert = self._analyze(frame)
                encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not encoded:
                    continue
                now = time.monotonic()
                measured_fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                self._set_status(connected=True, frames=frame_count, fps=round(min(measured_fps, fps_source * 1.5), 1), fall_score=round(score, 3))
                with self._lock:
                    self._jpeg = jpeg.tobytes()
                    if alert:
                        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                        self._status["alerts"] = int(self._status["alerts"]) + 1
                        self._status["last_alert"] = stamp
                        self._events.appendleft({"time": stamp, "type": "fall", "score": round(score, 3)})
        finally:
            capture.release()

    def _analyze(self, frame):
        if self._model is None:
            return self._overlay(frame, 0.0, False), 0.0, False
        result = self._model(frame, verbose=False)[0]
        score = 0.0
        best_box = None
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
            for i, (box, cls) in enumerate(zip(boxes, classes)):
                if cls != 0:
                    continue
                current = posture_fall_score(box, keypoints[i] if keypoints is not None else None)
                if current > score:
                    score, best_box = current, box
        alert = self._filter.update(score)
        return self._overlay(frame, score, alert, best_box), score, alert

    @staticmethod
    def _overlay(frame, score: float, alert: bool, box=None):
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if alert else (0, 200, 0), 2)
        cv2.putText(frame, f"fall score {score:.2f}  {'ALERT' if alert else 'monitoring'}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if alert else (255, 255, 255), 2)
        return frame

    @staticmethod
    def _placeholder(message: str) -> bytes:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (16, 22, 29)
        cv2.putText(frame, "CareSense", (42, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (190, 230, 215), 2)
        cv2.putText(frame, message, (44, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 165, 180), 2)
        cv2.putText(frame, "connect a camera or pass --source path/to/video.mp4", (44, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (105, 120, 135), 1)
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()

    def _set_status(self, **updates: object) -> None:
        with self._lock:
            self._status.update(updates)
