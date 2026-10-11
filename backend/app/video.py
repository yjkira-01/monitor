"""视频采集、按原速回放和异步训练模型分析。"""
from collections import deque
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Iterator

import cv2
import numpy as np
from yanglao_monitor.video import ACTIVITY_CLASSES
from .video_model import VideoModelWorker
from .video_tracking import PersonBoxTracker
from .video_people import annotate_people, VideoPoseWorker
from .video_person_state import PersonClips, scene_observation

ROOT = Path(__file__).resolve().parents[2]
cv2.setNumThreads(2)


def preview_frame(frame):
    height, width = frame.shape[:2]
    return cv2.resize(frame, (960, max(1, round(height*960/width))), interpolation=cv2.INTER_AREA) if width > 960 else frame


def _parse_source(value: str | int) -> str | int:
    return int(value.strip()) if isinstance(value, str) and value.strip().isdigit() else value


def open_capture(source):
    if not isinstance(source, int):
        return cv2.VideoCapture(source)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if cap.isOpened():
        if os.name == "nt":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if os.name == "nt" and source == int(os.getenv("VIDEO_CAMERA_INDEX", "1")):
            cap.set(cv2.CAP_PROP_ZOOM, 100)
            # ponytail: -5/gain64 restores ~30 FPS in this room; recalibrate when lighting changes.
            gain = cap.get(cv2.CAP_PROP_GAIN)
            if not (cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) and cap.set(cv2.CAP_PROP_EXPOSURE, -5) and cap.set(cv2.CAP_PROP_GAIN, 64)):
                cap.set(cv2.CAP_PROP_GAIN, gain)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    return cap


def frame_quality(frame, previous=None):
    gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
    brightness = round(float(gray.mean()), 1)
    motion = round(float((cv2.absdiff(gray, previous) > 18).mean()), 4) if previous is not None else 0.0
    quality = {"brightness": brightness, "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
               "motion_ratio": motion, "activity": "明显变化" if motion >= 0.08 else "轻微变化" if motion >= 0.01 else "变化较少",
               "usable": 6 <= brightness <= 245, "note": "缺少有效图像" if brightness < 6 else "过亮" if brightness > 245 else "暗光增强识别" if brightness < 65 else "亮度可用"}
    return quality, gray


class VideoManager:
    def __init__(self, source: str | int | None = None) -> None:
        sample = ROOT / "data/raw/GMDCSA24/Subject 4/Fall/01.mp4"
        default = str(sample) if sample.is_file() else "0"
        self.source = _parse_source(source if source is not None else os.getenv("VIDEO_SOURCE", default))
        self.camera_index = int(os.getenv("VIDEO_CAMERA_INDEX", "1"))
        self._lock = threading.Lock()
        self._source_lock = threading.Lock()
        self._classification_lock = threading.Lock()
        self._classification = None
        self._stop = threading.Event()
        self._closed = False
        self._thread = None
        active_weights = ROOT/'artifacts/care_video_active_inference.pt'
        binary_weights = ROOT/'artifacts/gmdcsa24_r3d18_100ep_inference.pt'
        default_weights = binary_weights if binary_weights.is_file() else active_weights
        weights = Path(os.getenv("VIDEO_MODEL_PATH",str(default_weights)))
        self.worker = VideoModelWorker(weights)
        self.pose_worker = VideoPoseWorker()
        self._generation = 0
        self._jpeg = self._placeholder("WAITING FOR VIDEO")
        self._annotated_jpeg = self._jpeg
        self._status = {"source": self._source_label(), "connected": False, "frames": 0, "fps": 0.0,
                        "width": 0, "height": 0, "error": None, "paused": False, "ended": False,
                        "position_s": 0.0, "duration_s": 0.0, "source_mode": self._source_mode(), "quality": None}

    def _source_label(self):
        if isinstance(self.source, int):
            return "EMEET C60E · USB" if self.source == self.camera_index else f"USB 摄像头 {self.source}"
        return Path(self.source).name if isinstance(self.source, str) and Path(self.source).is_file() else str(self.source)

    def _source_mode(self):
        return "file_replay" if isinstance(self.source, str) and Path(self.source).is_file() else "live"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._status["ended"] or self._closed:
            return
        self._stop.clear()
        self.worker.start()
        self.pose_worker.start()
        self._thread = threading.Thread(target=self._run, name="video-source", daemon=True)
        self._thread.start()

    def _stop_capture(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._thread and self._thread.is_alive():
            raise RuntimeError("上个视频源正在停止，请稍后重试")

    def stop(self) -> None:
        self._closed = True
        self._stop_capture()
        self.worker.stop()
        self.pose_worker.stop()

    def status(self) -> dict:
        self.start()
        with self._lock:
            status = deepcopy(self._status)
        status["analysis"] = self.analysis()
        return status

    def analysis(self) -> dict:
        result = self.worker.snapshot()
        result["people_detection"] = self.pose()
        with self._lock:
            status = dict(self._status)
        age = max(0, int(time.time() * 1000) - result.get("captured_at_ms", int(time.time()*1000)))
        stale = not status["connected"] or (not status["paused"] and not status["ended"] and age > 5000)
        result.update(source="video", device_id="CAM_01", source_mode=status["source_mode"], source_label=status["source"],
                      analysis_backend="trained_r3d18", analysis_enabled=result["model_state"] == "ready", result_age_ms=age,
                      stale=stale, playback_ended=status["ended"], current_quality=status["quality"])
        if stale:
            result.update(is_fall=False, watch=False)
        observation = scene_observation(result['people_detection']['people'], not result['people_detection']['stale'] and result['people_detection']['state'] == 'ready')
        usable = observation['scene_usable'] or result.get('partial_body_trained',False) and observation['scene_video_usable']
        if not usable and result['model_state'] == 'ready':
            result.update(is_fall=False, watch=False, action='unknown', state='observation_limited', assessment=observation['observation_note'])
        result['pose_fall_people'] = observation['pose_fall_people']
        if not stale and observation['pose_fall_people'] and result.get('fall_score', 0) < .5 and result.get('decision_basis') != 'supported_short_window':
            result.update(state='model_disagreement', watch=True, needs_review=True, action='falling',
                          assessment='姿态时序模型出现跌倒特征，RGB模型未识别；请复核画面', decision_basis='trained_pose_tcn')
        return result

    def pose(self):
        result = self.pose_worker.snapshot()
        with self._lock:
            status = dict(self._status)
        age = max(0, int(time.time()*1000)-result.get("captured_at_ms", int(time.time()*1000)))
        result.update(stale=not status["connected"] or (not status["paused"] and not status["ended"] and age > 2000),
                      result_age_ms=age, paused=status["paused"], ended=status["ended"], session_id=self.worker.session_id)
        predictions = self.worker.person_predictions()
        now = int(time.time()*1000)
        for person in result['people']:
            prediction = predictions.get(person.get('track_id'))
            observation = person.get('observation', {})
            if result['stale'] or not observation.get('video_usable', False):
                person['analysis'] = {'state': 'unobservable', 'fall_score': None, 'reason': observation.get('reason', '观测不足')}
            elif prediction and (status['paused'] or status['ended'] or now-prediction['updated_at_ms'] < 5000):
                person['analysis'] = prediction
            else:
                person['analysis'] = {'state': 'warming_up', 'fall_score': None, 'reason': '正在收集该人物的连续片段'}
            pose_prediction = person.get('pose_analysis', {})
            if not result['stale'] and observation.get('video_usable') and pose_prediction.get('state') == 'fall_candidate':
                if (person['analysis'].get('fall_score') is None or person['analysis']['fall_score'] < .5) and person['analysis'].get('short_consecutive_fall_windows', 0) < 2:
                    person['analysis'].update(state='model_disagreement', needs_review=True,
                                              reason='姿态时序模型出现跌倒特征，RGB尚未确认，请复核')
        return result

    def set_file_source(self, path: Path, label: str | None = None) -> None:
        if not path.is_file():
            raise FileNotFoundError("视频文件不存在")
        cap = open_capture(str(path))
        ok, _ = cap.read()
        cap.release()
        if not ok:
            raise ValueError("无法解码此视频，请使用可播放的 MP4/AVI/MOV/WebM")
        self._switch_source(str(path.resolve()), label or path.name)

    def set_camera_source(self):
        if self.source == self.camera_index:
            self._switch_source(self.camera_index, "EMEET C60E · USB")
            return
        cap = open_capture(self.camera_index)
        try:
            ok, _ = cap.read() if cap.isOpened() else (False, None)
        finally:
            cap.release()
        if not ok:
            raise ValueError("无法读取 C60E，请检查 USB 连接、摄像头权限或其他软件占用")
        self._switch_source(self.camera_index, "EMEET C60E · USB")

    def _switch_source(self, source, label):
        with self._source_lock:
            if self._classification_lock.locked():
                raise RuntimeError("正在进行整段分类，请在结束后切换")
            self._stop_capture()
            self.source = source
            self._generation = self.worker.reset()
            self.pose_worker.reset()
            with self._lock:
                self._classification = None
                self._jpeg = self._placeholder("OPENING VIDEO")
                self._annotated_jpeg = self._jpeg
                self._status.update(source=label, connected=False, frames=0, fps=0.0, display_fps=0.0,
                                    width=0, height=0, capture_backend=None, error=None,
                                    paused=False, ended=False, position_s=0.0, duration_s=0.0, source_mode=self._source_mode(), quality=None)
            self._closed = False
            self.start()

    def set_paused(self, paused: bool):
        if self._source_mode() != "file_replay":
            raise ValueError("实时摄像头不能暂停，请选择回放视频")
        self._update(paused=paused)

    def classification(self):
        with self._lock:
            return deepcopy(self._classification)

    def classify_file(self):
        with self._source_lock:
            source = self.source
            if self._source_mode() != "file_replay":
                raise ValueError("请先选择回放视频")
            if not self._classification_lock.acquire(blocking=False):
                raise RuntimeError("已有整段分类在运行")
        try:
            if not self.worker.checkpoint.is_file():
                raise FileNotFoundError("推理权重不存在")
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/predict_video.py"), source,
                                        "--checkpoint", str(self.worker.checkpoint.resolve())],
                                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4"},
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if completed.returncode:
                raise RuntimeError("整段分类失败，请检查 torch/torchvision 与权重")
            result = json.loads(completed.stdout)
            result.pop("video", None)
            if result.get("scope") != "whole_video" or result.get("predicted") not in {"fall", "adl", *ACTIVITY_CLASSES}:
                raise RuntimeError("模型返回无效分类")
            with self._lock:
                self._classification = result
            return dict(result)
        finally:
            self._classification_lock.release()

    def stream(self, annotated: bool = False) -> Iterator[bytes]:
        self.start()
        while not self._closed:
            with self._lock:
                jpeg = self._annotated_jpeg if annotated else self._jpeg
                fps = min(30., max(1., self._status["fps"] or 25.))
                held = self._status['paused'] or self._status['ended']
            if annotated and held:
                # A detector finishing after EOF must still update the held frame's boxes.
                jpeg = self.pose_worker.observation() or jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(1/fps)

    def _run(self):
        cap = open_capture(self.source)
        if not cap.isOpened():
            self._update(connected=False, error="无法打开视频源")
            cap.release()
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if np.isfinite(fps) and 1 <= fps <= 120 else 25.0
        replay = self._source_mode() == "file_replay"
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if replay else 0.0
        self._update(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     capture_backend=cap.getBackendName(), duration_s=round(duration, 2))
        buffer = deque()
        frames = 0
        previous = None
        quality = None
        next_frame = time.monotonic()
        last_sample = last_inference = -1.0
        next_pose = 0.
        generation = self._generation
        start = time.monotonic()
        last_frame = None
        tracker = PersonBoxTracker()
        person_clips = PersonClips()
        display_times = deque(maxlen=30)
        try:
            while not self._stop.is_set():
                with self._lock:
                    paused = self._status["paused"]
                if paused:
                    self._stop.wait(0.08)
                    next_frame = time.monotonic()
                    continue
                read_started = time.monotonic()
                ok, frame = cap.read()
                read_ms = (time.monotonic()-read_started)*1000
                if not ok:
                    if replay and buffer:
                        # EOF 保留最后结果供复核；不让循环回放冲掉短片的推理。
                        if last_frame is not None and buffer[-1][0] < position:
                            buffer.append((position, cv2.resize(last_frame, (171, 128), interpolation=cv2.INTER_AREA)))
                        self.worker.submit([item[1] for item in buffer], buffer[0][0], position, generation, quality, int(time.time()*1000), person_clips.ready(), [item[0] for item in buffer])
                        self.pose_worker.submit(last_frame, position, generation, int(time.time()*1000))
                        self._update(ended=True, paused=True)
                    else:
                        self._update(connected=False, error="视频读取失败")
                    break
                frames += 1
                processing_started = time.monotonic()
                position = (frames - 1) / fps if replay else time.monotonic() - start
                last_frame = frame
                # A fixed cadence avoids halving updates for sources slightly above 15 FPS.
                if position + .0001 >= next_pose:
                    self.pose_worker.submit(frame, position, generation, int(time.time()*1000))
                    next_pose = (int((position + .0001)*15)+1)/15
                buffer.append((position, cv2.resize(frame, (171, 128), interpolation=cv2.INTER_AREA)))
                while buffer and position - buffer[0][0] > 5:
                    buffer.popleft()
                if position - last_sample >= 0.12:
                    quality, previous = frame_quality(frame, previous)
                    detection = self.pose_worker.snapshot()
                    fresh = detection.get('generation') == generation and position-detection.get('frame_s', -10) < .6 and detection['state'] == 'ready'
                    people = detection['people'] if fresh else []
                    quality.update(scene_observation(people, fresh))
                    person_clips.sample(frame, people, position)
                    ready_people = person_clips.ready()
                    quality['short_window_eligible'] = len(people) == 1 and any(p['quality'].get('short_window_eligible') for p in ready_people)
                    last_sample = position
                    if position >= 1 and position - last_inference >= 0.8:
                        self.worker.submit([item[1] for item in buffer], buffer[0][0], position, generation, quality, int(time.time()*1000), ready_people, [item[0] for item in buffer])
                        last_inference = position
                preview = preview_frame(frame)
                encoded, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if encoded:
                    tracked = tracker.update(frame, self.pose_worker.tracking_detection(), position)
                    annotated_jpeg = annotate_people(preview, tracked) if tracked else jpeg.tobytes()
                    display_times.append(time.monotonic())
                    display_fps = (len(display_times)-1)/max(display_times[-1]-display_times[0], 1e-6) if len(display_times)>1 else 0.
                    with self._lock:
                        self._jpeg = jpeg.tobytes()
                        self._annotated_jpeg = annotated_jpeg
                        self._status.update(connected=True, frames=frames, fps=round(fps, 1), display_fps=round(display_fps,1),
                                            display_width=preview.shape[1], display_height=preview.shape[0],
                                            position_s=round(position, 2), error=None, quality=quality,
                                            read_ms=round(read_ms,1), processing_ms=round((time.monotonic()-processing_started)*1000,1))
                if replay:
                    next_frame += 1 / fps
                    self._stop.wait(max(0, next_frame - time.monotonic()))
        finally:
            cap.release()

    def _update(self, **values):
        with self._lock:
            self._status.update(values)

    @staticmethod
    def _placeholder(message):
        frame = np.full((360, 640, 3), (30, 45, 37), dtype=np.uint8)
        cv2.putText(frame, "CareSense", (42, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (195, 230, 210), 2)
        cv2.putText(frame, message, (44, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 210, 202), 1)
        return cv2.imencode(".jpg", frame)[1].tobytes()


video_manager = VideoManager()
