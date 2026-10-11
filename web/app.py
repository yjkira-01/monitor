"""本地智能康养监测台：视频流优先，毫米波设备到位后再接入。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yanglao_monitor.monitor import VideoMonitor  # noqa: E402


def create_app(monitor: VideoMonitor) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/video_feed")
    def video_feed():
        return Response(monitor.mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def status():
        return jsonify(monitor.status())

    @app.get("/api/events")
    def events():
        return jsonify(monitor.events())

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "video": monitor.status()})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="摄像头编号或视频路径")
    parser.add_argument("--pose", action="store_true", help="启用 YOLO Pose 跌倒基线")
    parser.add_argument("--weights", default="yolo11n-pose.pt")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址；局域网访问需使用 0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    monitor = VideoMonitor(args.source, pose=args.pose, weights=args.weights)
    monitor.start()
    create_app(monitor).run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
