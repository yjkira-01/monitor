# 实施路线（以真实数据为准）

计划书只提供场景背景；当前正式入口是 FastAPI + React，先把视频流、雷达回放和数据契约跑通，再决定模型。

## 当前阶段

- `backend/app/video.py`：摄像头/视频文件转 MJPEG，无法打开时返回占位帧。
- `frontend/src/components/VideoPreview.tsx`：总览页显示视频、分辨率、FPS 和连接状态。
- `backend/app/sources/`：mock、MMFi、replay、IWR6843AOP OOB 数据源。
- `edge/vision_node.py`：本地 MediaPipe Pose 与跌倒几何分数，结果通过 `/api/edge/pose` 上送。

## 下一步

1. 用 `VIDEO_SOURCE` 接真实摄像头或回放视频，记录不同光照、遮挡和多人场景的误报。
2. 设备到位后验证串口、OOB TLV 帧完整性和时间戳，再校准生命体征双估计器。
3. 用按受试者划分的同步视频/雷达数据训练行为模型，并保留 `mock`、`mmfi`、`replay` 做回归测试。
4. 在融合层加入视频姿态、雷达动作和生命体征的时间窗对齐，再做边缘部署和告警迟滞。
5. 睡眠先报告体动、呼吸率趋势和疑似异常窗口；没有 PSG 标签前不宣称临床睡眠分期或 OSAHS 诊断。
