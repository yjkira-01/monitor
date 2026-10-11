"""SR112 串口点云与专用固件生命体征；合成相位仅作明确标注的演示。"""
from __future__ import annotations

from collections import deque
import os
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .mmvital_estimator import synthesize_chest_signal
from .oob_tlv import consume_frames
from .dual_vitals import estimate_dual
from ..models import VitalSource

try:
    import serial
except ImportError:  # 串口采集环境需要 pyserial，离线测试不需要。
    serial = None


class PhaseProvider:
    def next_window(self) -> list[float] | None:
        raise NotImplementedError


class SyntheticPhaseProvider(PhaseProvider):
    def __init__(self, fs: float = 20.0, breath_hz: float = 0.25, heart_hz: float = 1.1, seed: int = 7):
        self.fs, self.breath_hz, self.heart_hz, self.seed = fs, breath_hz, heart_hz, seed

    def next_window(self) -> list[float]:
        return synthesize_chest_signal(self.breath_hz, self.heart_hz, self.fs, 5.0, noise=0.05, seed=self.seed)


@dataclass
class _RadarConfig:
    port: str | None
    baud: int


class RadarPhaseProvider(PhaseProvider):
    """名字保留供既有调用使用；点云不是相位，体征只读固件 TLV 1040。"""

    def __init__(self, fs: float = 20.0, window_size: int = 100):
        self.fs = fs
        self._cfg = _RadarConfig(os.getenv("RADAR_PORT"), int(os.getenv("RADAR_BAUD", "921600")))
        self._serial = None
        self._buffer = b""
        self._pending_vitals = deque(maxlen=32)
        self._latest_points: Any = None
        self._latest_points_ms = 0
        self._telemetry: dict = {}
        self.session_id = uuid4().hex
        self.error: str | None = None
        self.serial_available = serial is not None
        self._enabled = bool(self._cfg.port) and serial is not None
        if self._enabled:
            try:
                self._serial = serial.Serial(port=None, baudrate=self._cfg.baud, timeout=0.2)
                # Disable modem control before opening the DATA UART to avoid resetting the board.
                self._serial.dtr = self._serial.rts = False
                self._serial.port = self._cfg.port
                self._serial.open()
            except Exception as exc:
                self._enabled = False
                self.error = f"数据串口打开失败（{type(exc).__name__}），请检查端口与占用"

    def reset_cache(self):
        self._buffer = b""
        self._pending_vitals.clear()
        self._latest_points, self._latest_points_ms = None, 0
        self._telemetry = {}
        self.session_id = uuid4().hex

    @staticmethod
    def _vital_data(vital, track, captured, frame_number):
        # ponytail: 0.1 m/s is a track-motion heuristic; calibrate before using for alerts.
        moving = sum(v*v for v in track['velocity']) >= 0.1**2
        weak_breath = vital['breathing_deviation'] < 0.02
        return {**vital, "timestamp_ms": captured, "device_id": "RADAR_01",
                "heart_rate": vital['heart_rate'] or None,
                "breath_rate": vital['breath_rate'] if not weak_breath and vital['breath_rate'] > 0 else None,
                "chest_displacement_mm": None, "motion_flag": moving, "ahi_index": None,
                "source": VitalSource.real.value, "signal_type": "ti_vitals_tlv1040", "firmware_frame_number": frame_number,
                "quality": None, "vital_agreement": None, "needs_review": True,
                "watch_reason": "呼吸波形偏弱，需复核；不是呼吸暂停诊断" if weak_breath else "TI固件体征估计，尚未做实机对照标定"}

    def _poll_frames(self):
        if not self._enabled or self._serial is None:
            return []
        try:
            # Poll available bytes; waiting for 4KB blocks the API event loop on sparse radar frames.
            chunk = self._serial.read(min(65536, getattr(self._serial, 'in_waiting', 0)))
        except Exception as exc:
            self._buffer = b""
            self._pending_vitals.clear()
            self._telemetry = {}
            self._latest_points, self._latest_points_ms = None, 0
            self.error = f"串口读取失败（{type(exc).__name__}），请检查设备连接"
            return []
        frames, self._buffer = consume_frames(self._buffer+chunk)
        for frame in frames:
            self.error = None
            captured = int(time.time()*1000)
            self._latest_points, self._latest_points_ms = frame.points, captured
            active_ids = {t['target_id'] for t in frame.tracks}
            latest = {v['target_id']: v for v in self._telemetry.get('vitals', []) if v['target_id'] in active_ids}
            self._pending_vitals = deque((item for item in self._pending_vitals if item[0]['target_id'] in active_ids), maxlen=32)
            for vital in frame.vitals:
                track = next((t for t in frame.tracks if t['target_id'] == vital['target_id']), None)
                if track is not None and vital['breathing_deviation'] > 0:
                    self._pending_vitals.append((vital, track, captured, frame.frame_number))
                    latest[vital['target_id']] = self._vital_data(vital, track, captured, frame.frame_number)
                else:
                    latest.pop(vital['target_id'], None)
                    self._pending_vitals = deque((item for item in self._pending_vitals if item[0]['target_id'] != vital['target_id']), maxlen=32)
            # Replace atomically so read-only API requests see one coherent packet.
            self._telemetry = {"timestamp_ms": captured, "frame_number": frame.frame_number,
                               "tlv_types": list(frame.tlv_types), "tracks": frame.tracks[:256],
                               "points": frame.points[:2048].tolist(), "vitals": list(latest.values())[:256]}
        return frames

    def next_frame(self):
        frames = self._poll_frames()
        return frames[-1].points if frames else None

    def next_window(self) -> None:
        self._poll_frames()
        return None

    def next_vital(self) -> dict | None:
        self._poll_frames()
        while self._pending_vitals:
            vital, track, captured, frame_number = self._pending_vitals.popleft()
            if int(time.time()*1000)-captured > 3000:
                continue
            return self._vital_data(vital, track, captured, frame_number)
        return None


class MmVitalSource:
    def __init__(self, device_id: str = "RADAR_01", fs: float = 20.0, provider: PhaseProvider | None = None):
        self.device_id, self.fs = device_id, fs
        self.provider = provider if provider is not None else RadarPhaseProvider(fs)

    def next_vital(self) -> dict | None:
        if isinstance(self.provider, RadarPhaseProvider):
            result = self.provider.next_vital()
            if result is not None:
                result['device_id'] = self.device_id
            return result
        window = self.provider.next_window()
        if window is None:
            return None
        dual = estimate_dual(window, self.fs)
        return {"timestamp_ms": int(time.time()*1000), "device_id": self.device_id,
                "breath_rate": dual.breath_rate, "heart_rate": dual.heart_rate,
                "chest_displacement_mm": None, "motion_flag": dual.motion_flag, "quality": dual.quality,
                "vital_agreement": dual.agreement, "needs_review": dual.needs_review, "watch_reason": dual.watch_reason,
                "ahi_index": None, "source": VitalSource.mock.value if isinstance(self.provider, SyntheticPhaseProvider) else VitalSource.real.value}
