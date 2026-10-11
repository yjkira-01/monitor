"""毫米波时序的可解释特征提取。

输入可以是锁定距离门后的复数相位/位移序列。真实设备的 ADC 到相位、距离和角度
转换依赖雷达型号，故这里只处理设备无关的时序层，不假装替代厂商 SDK。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def _as_1d(values: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < 8 or not np.isfinite(x).any():
        raise ValueError("雷达时序太短或全为无效值")
    x = np.nan_to_num(x, nan=float(np.nanmedian(x[np.isfinite(x)])))
    return x - np.median(x)


def _bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    if not 0 < low < high < fs / 2:
        raise ValueError(f"频带必须位于 (0, fs/2)，收到 {low}-{high} Hz, fs={fs}")
    sos = signal.butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
    # sosfiltfilt 对极短窗口不稳定，退化为因果滤波而不是直接崩溃。
    return signal.sosfiltfilt(sos, x) if len(x) > 24 else signal.sosfilt(sos, x)


def _dominant_rate(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[float, float]:
    filtered = _bandpass(x, fs, *band)
    frequencies, power = signal.periodogram(filtered, fs=fs, scaling="spectrum")
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    if not np.any(mask):
        return float("nan"), 0.0
    band_power = power[mask]
    index = int(np.argmax(band_power))
    peak_power = float(band_power[index])
    total = float(power[(frequencies > 0) & np.isfinite(power)].sum())
    confidence = peak_power / (total + 1e-12)
    return float(frequencies[mask][index] * 60.0), confidence


def estimate_vital_signs(values: np.ndarray | list[float], fs: float) -> dict[str, float]:
    """估计呼吸率、心率和信号质量，单位为次/分钟。"""
    x = _as_1d(values)
    if fs <= 4:
        raise ValueError("采样率过低，至少需要覆盖心跳频带")
    resp, resp_conf = _dominant_rate(x, fs, (0.10, 0.50))
    heart, heart_conf = _dominant_rate(x, fs, (0.80, 2.00))
    motion_rms = float(np.sqrt(np.mean(np.diff(x) ** 2)))
    return {
        "resp_rate_bpm": resp,
        "heart_rate_bpm": heart,
        "resp_confidence": resp_conf,
        "heart_confidence": heart_conf,
        "motion_rms": motion_rms,
    }


def radar_feature_vector(values: np.ndarray | list[float], fs: float) -> np.ndarray:
    """输出给融合模型的固定长度雷达向量。"""
    x = _as_1d(values)
    vital = estimate_vital_signs(x, fs)
    features = np.array(
        [
            vital["resp_rate_bpm"],
            vital["heart_rate_bpm"],
            vital["resp_confidence"],
            vital["heart_confidence"],
            vital["motion_rms"],
            float(np.std(x)),
            float(np.percentile(x, 95) - np.percentile(x, 5)),
            float(np.mean(np.abs(x))),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_sleep_windows(windows: list[dict[str, float]]) -> dict[str, Any]:
    """将连续窗口特征汇总为可解释的睡眠报告，不等同于临床睡眠分期。"""
    if not windows:
        raise ValueError("至少需要一个睡眠窗口")
    motion = np.array([w.get("motion_rms", 0.0) for w in windows], dtype=float)
    resp = np.array([w.get("resp_rate_bpm", np.nan) for w in windows], dtype=float)
    active = motion > np.nanmedian(motion) * 1.8 if np.nanmedian(motion) > 0 else motion > 0
    return {
        "window_count": len(windows),
        "estimated_sleep_windows": int((~active).sum()),
        "movement_windows": int(active.sum()),
        "movement_ratio": float(active.mean()),
        "mean_resp_rate_bpm": float(np.nanmean(resp)),
        "resp_rate_std_bpm": float(np.nanstd(resp)),
    }
