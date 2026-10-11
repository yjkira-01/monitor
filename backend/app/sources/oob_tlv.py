"""SR112 / xWR6843 的 SDK3 OOB 与人体跟踪/生命体征 TLV 解析。"""
from __future__ import annotations

from dataclasses import dataclass, field
import struct
from typing import Iterable

import numpy as np

MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])
TLV_DETECTED_POINTS = 1
HDR_SIZE = 40
MAX_FRAME_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ParsedOobFrame:
    points: np.ndarray
    total_len: int
    frame_number: int = 0
    subframe_number: int = 0
    tlv_types: tuple[int, ...] = ()
    tracks: list[dict] = field(default_factory=list)
    vitals: list[dict] = field(default_factory=list)


def parse_frame(buf: bytes, pos: int = 0) -> ParsedOobFrame | None:
    """解析一帧 OOB TLV，返回点云与帧长度。

    返回 None 表示当前缓冲区中缺少完整帧或帧头损坏。
    """
    if pos < 0 or pos + HDR_SIZE > len(buf):
        return None
    if buf[pos : pos + len(MAGIC)] != MAGIC:
        return None

    _, _, total_len, _, frame_number, _, num_objects, num_tlv, subframe = struct.unpack_from("<8s8I", buf, pos)
    if not HDR_SIZE <= total_len <= MAX_FRAME_SIZE or pos + total_len > len(buf) or num_tlv > (total_len-HDR_SIZE)//8:
        return None

    body = buf[pos + HDR_SIZE : pos + total_len]
    off = 0
    points = np.zeros((0, 3), dtype=np.float32)
    tracks, vitals, types = [], [], []
    for _ in range(num_tlv):
        if off + 8 > len(body):
            return None
        tlv_type, tlv_len = struct.unpack_from("<II", body, off)
        off += 8
        if tlv_len > len(body)-off:
            return None
        payload = body[off : off + tlv_len]
        off += tlv_len
        types.append(tlv_type)
        if tlv_type == TLV_DETECTED_POINTS:
            if tlv_len != num_objects * 16:
                return None
            values = np.frombuffer(payload, dtype="<f4").reshape(-1, 4)
            if not np.isfinite(values).all():
                return None
            points = values[:, :3].copy()
        elif tlv_type == 1020:
            if tlv_len < 20 or (tlv_len-20) != num_objects * 8:
                return None
            units = np.array(struct.unpack_from("<5f", payload))
            if not np.isfinite(units).all() or np.any(units <= 0):
                return None
            values = np.array(list(struct.iter_unpack("<bbhHH", payload[20:])), dtype=np.float32).reshape(-1, 5)
            elevation, azimuth, distance = values[:, 0]*units[0], values[:, 1]*units[1], values[:, 3]*units[3]
            points = np.column_stack((distance*np.cos(elevation)*np.sin(azimuth), distance*np.cos(elevation)*np.cos(azimuth), distance*np.sin(elevation))).astype(np.float32)
        elif tlv_type == 1010:
            if tlv_len % 112:
                return None
            for values in struct.iter_unpack("<I27f", payload):
                if not np.isfinite(values[1:]).all():
                    return None
                tracks.append({"target_id": values[0], "position": list(values[1:4]), "velocity": list(values[4:7]),
                               "acceleration": list(values[7:10]), "confidence": values[27]})
        elif tlv_type == 1040:
            if tlv_len != 136:
                return None
            values = struct.unpack("<HH33f", payload)
            if not np.isfinite(values[2:]).all() or min(values[2:5]) < 0:
                return None
            vitals.append({"target_id": values[0], "range_bin": values[1], "breathing_deviation": values[2],
                           "heart_rate": values[3], "breath_rate": values[4], "heart_waveform": list(values[5:20]),
                           "breath_waveform": list(values[20:35])})
    if not np.isfinite(points).all():
        return None
    return ParsedOobFrame(points, total_len, frame_number, subframe, tuple(types), tracks, vitals)


def consume_frames(buf: bytes) -> tuple[list[ParsedOobFrame], bytes]:
    """消费完整帧，保留跨串口 read 的半帧与魔数尾部。"""
    frames = []
    pos = 0
    while True:
        magic_at = buf.find(MAGIC, pos)
        if magic_at < 0:
            return frames, buf[pos:][-7:]
        if len(buf)-magic_at < HDR_SIZE:
            return frames, buf[magic_at:]
        total_len = struct.unpack_from("<I", buf, magic_at+12)[0]
        if not HDR_SIZE <= total_len <= MAX_FRAME_SIZE:
            pos = magic_at+1
            continue
        if len(buf)-magic_at < total_len:
            return frames, buf[magic_at:]
        parsed = parse_frame(buf, magic_at)
        if parsed is None:
            pos = magic_at+1
            continue
        frames.append(parsed)
        pos = magic_at+parsed.total_len


def iter_frames(buf: bytes) -> list[np.ndarray]:
    """从连续字节流中提取所有完整点云帧。"""
    return [frame.points for frame in consume_frames(buf)[0]]


def build_detected_points_tlv(points: np.ndarray) -> bytes:
    """构造 TLV type=1 的检测点载荷（测试夹具用）。"""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points 必须是 (N, 3) 数组")
    doppler = np.zeros((pts.shape[0], 1), dtype=np.float32)
    data = np.concatenate([pts, doppler], axis=1).astype("<f4", copy=False)
    payload = data.tobytes()
    return struct.pack("<II", TLV_DETECTED_POINTS, len(payload)) + payload


def build_oob_frame(points: np.ndarray, frame_num: int = 1, timestamp_ms: int = 0) -> bytes:
    """构造单帧 OOB TLV 字节流（测试夹具用）。"""
    tlv = build_detected_points_tlv(points)
    total_len = (HDR_SIZE + len(tlv) + 31)//32 * 32
    header = struct.pack(
        "<8sIIIIIIII",
        MAGIC,
        0x01020304,
        total_len,
        0xA6843,
        frame_num,
        timestamp_ms,
        int(np.asarray(points).shape[0]),
        1,
        0,
    )
    return (header + tlv).ljust(total_len, b"\0")


def build_oob_stream(frames: Iterable[np.ndarray]) -> bytes:
    """把多帧拼成连续 OOB 字节流。"""
    chunks = []
    for idx, pts in enumerate(frames, start=1):
        chunks.append(build_oob_frame(pts, frame_num=idx, timestamp_ms=idx * 50))
    return b"".join(chunks)
