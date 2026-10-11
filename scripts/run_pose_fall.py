"""用 Ultralytics YOLO Pose 在视频上做可解释的实时跌倒基线。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yanglao_monitor.pose_fall import TemporalFallFilter, posture_fall_score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="视频文件或摄像头编号")
    parser.add_argument("--output", default="artifacts/pose_fall.mp4")
    parser.add_argument("--weights", default="yolo11n-pose.pt")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频源: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = YOLO(args.weights)
    fall_filter = TemporalFallFilter(args.threshold, args.patience)
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model(frame, verbose=False)[0]
        best_score = 0.0
        best_box = None
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
            for i, (box, cls) in enumerate(zip(boxes, classes)):
                if cls != 0:  # COCO person
                    continue
                score = posture_fall_score(box, keypoints[i] if keypoints is not None else None)
                if score > best_score:
                    best_score, best_box = score, box
        alert = fall_filter.update(best_score)
        if best_box is not None:
            x1, y1, x2, y2 = best_box.astype(int)
            color = (0, 0, 255) if alert else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"fall_score={best_score:.2f}  {'ALERT' if alert else 'normal'}"
        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255) if alert else (255, 255, 255), 2)
        writer.write(frame)
        frame_id += 1
    capture.release(); writer.release()
    print(f"processed {frame_id} frames -> {out_path}")


if __name__ == "__main__":
    main()
