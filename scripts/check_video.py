"""检查摄像头或视频文件是否能被 OpenCV 连续读取。"""

from __future__ import annotations

import argparse
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="0", help="摄像头编号或视频路径")
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"video source unavailable: {args.source}")
    started = time.monotonic(); count = 0; shape = None
    while count < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1; shape = tuple(frame.shape)
    elapsed = max(time.monotonic() - started, 1e-6)
    cap.release()
    if count == 0:
        raise SystemExit("video opened but returned no frames")
    print(f"video ok: frames={count}, shape={shape}, read_fps={count/elapsed:.1f}")


if __name__ == "__main__":
    main()
