"""IWR6843AOP out-of-box 点云采集脚本（真实数据管线入口）。

连 DATA UART -> 解析 mmWave OOB demo 的 TLV 帧 -> 每帧点云存成 (N,3) float64 .bin
（项目 load_mmwave_frames 可直接读；仅保存 xyz，不保存体征、轨迹、速度/SNR 或原始 I/Q）。

与后端共用 40 字节头的 SDK3/跟踪 TLV 解析器；必须先运行匹配固件并发送配置。
其他 SDK/固件格式需对照其输出协议验证。此脚本不会发送配置或烧录固件。

采集机依赖：pyserial, numpy（不进后端 venv，单独装即可）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.sources.oob_tlv import consume_frames


def main() -> None:
    ap = argparse.ArgumentParser(description="IWR6843AOP OOB 点云采集")
    ap.add_argument("--port", required=True, help="DATA UART 口，如 COM5 / /dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out", required=True, help="输出目录，如 data/radar/E01/S01/A01/mmwave")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0=不限制")
    args = ap.parse_args()
    try:
        import serial
    except ImportError:
        sys.exit("请先装 pyserial: pip install pyserial")

    os.makedirs(args.out, exist_ok=True)
    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    print(f"[采集] 打开 {args.port}@{args.baud}，输出 {args.out}")

    buf = b""
    frame_idx = 0
    start = time.time()
    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                buf += chunk
            frames, buf = consume_frames(buf)
            for frame in frames:
                points = frame.points
                if len(points) > 0:
                    path = os.path.join(args.out, f"frame{frame_idx:04d}.bin")
                    points.astype(np.float64).tofile(path)
                    frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            if time.time() - start > args.seconds:
                break
    finally:
        ser.close()
    print(f"[采集] 完成，共 {frame_idx} 帧 -> {args.out}")


if __name__ == "__main__":
    main()
