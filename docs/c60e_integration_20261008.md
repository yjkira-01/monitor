# C60E 低分辨率实时接入记录

后续更新：用户要求扩大拍摄范围，当前输出已改为完整 4:3 的 **640×480，约 30 FPS**。背景匹配估算上下可见场景相对原模式增加约 39%，左右约 4%。下文 640×360 的数据保留为原始实验记录，最新证据见 `artifacts/c60e_fov_probe.json` 和 [最新接入记录](sr112_connection_20261008.md)。

2026-10-08，Windows 实机 EMEET SmartCam C60E 4K，USB 连接。当前项目后端和前端分别运行于 8000 / 5173，最终视频来源保留为 C60E 实时采集。

## 实机配置与原因

DirectShow 枚举：0 为 Integrated Camera，1 为 C60E，2 为 OBS Virtual Camera。C60E 的 USB 标识为 VID 328F / PID 00F3。

本次配置为 DirectShow、MJPEG、640×360、请求 30 FPS。摄像头自动曝光时，实际读取一帧需要 50–70 ms，处理画面只需要 5–10 ms，项目采集约 14 FPS；仅降低分辨率不能解决曝光造成的帧率限制。手动曝光 -5 配合增益 64，独立短测达到 30.09 FPS，缩略图平均亮度 95.9。只降低曝光会使画面过暗，因此同时调整增益。

Windows DirectShow 用 `CAP_PROP_AUTO_EXPOSURE=0` 切到手动曝光，`1` 恢复自动曝光。本机读取此属性返回 -1，不可用返回值恢复。设置控制失败时恢复原增益并切回自动曝光。其他编号的摄像头只应用低分辨率和帧率设置，文件回放保持原配置。

这组曝光和增益只验证了当前室内光照；不同照明需重新校准，自动曝光回退也可能降低帧率。640×360 减少了远处小人物的细节，需要结合实际安装距离验证识别。

## 已接入的功能

- `backend/app/video.py` 共用 `open_capture`，实摄通过驱动控制采集节奏，文件回放继续按源 FPS 软件计时。
- 登录后 `POST /api/video/camera` 可以切换或重新连接配置的 C60E。文件切到摄像头前先检查有效帧；已选摄像头重连时先释放设备，避免预检查占用同一个设备。
- 前端视频来源下拉加入 C60E，实时模式显示 LIVE、重连按钮、实际分辨率和采集 FPS。示例、上传、暂停、继续和重播保留在文件回放模式。
- 实时帧进入既有常驻 R3D 滑窗分类、人物检测、独立姿态队列及三维/点云显示。原视频只画人物框。
- 后端模型实例生成 `session_id`，模型和姿态共享该标识。前端在同一会话内比较代次与结果时间；后台重启后刷新视频地址、清除旧姿态和平滑历史，避免旧页面永久拒绝归零后的代次。
- `scripts/start_backend.ps1` 默认 C60E 编号 1，保留 `VIDEO_CAMERA_INDEX` 和 `VIDEO_SOURCE` 环境变量覆盖。

## 真实测量与验收

最后一次标框 MJPEG 测量在后台会话修复、重启和浏览器验收之后进行：

| 项目 | 结果 |
| --- | --- |
| 采集与实际图像尺寸 | 640×360，DSHOW，源请求 30 FPS |
| 标框视频流不同帧 | 114 帧 / 3.976 秒 |
| 不同帧接收 FPS | 28.42 |
| 后台采集 FPS | 30.3 |
| 最新单帧读取 / 处理耗时 | 29.9 / 1.3 ms |
| 浏览器观测时新增模型窗口 | 6 个 / 6.027 秒 |
| 浏览器观测时姿态处理频率 | 9.62 Hz，末次滑动统计 10.7 Hz |
| 最新分类推理耗时 | 255.1 ms |
| 前端可观察的不同姿态时间 | 30 个 / 6.027 秒 |
| 390px 布局 | 页面宽 390，内容宽 390，无横向溢出 |

视频接收 FPS、模型窗口频率和姿态处理频率是三个不同指标。MJPEG 实测为 localhost 短测，不代表手机、公网或整天连续使用的性能。

隔离的真实 Chrome 验收通过：正常活动样例切换到 C60E、C60E 重连、实时/回放控制切换、640×360 图像解码、模型窗口与姿态持续更新、两个 WebGL 画布和蒙皮人物加载、手机布局。浏览器临时模拟旧后台的高代次与未来结果时间，恢复真实会话后，视频地址和姿态均自动恢复，未要求刷新页面。

最终 API 核验：真实 live、模型 ready、42 个窗口、姿态 ready、约 10.4 Hz、亮度可用；两个人物检测结果中，一个有 33 个相对三维关节，另一个没有可用姿态。这是当前取景观测，不能视为多人姿态可靠率。取景应尽量包含全身；遮挡或只有局部身体会限制三维展示。

测试记录：摄像头实现完成时全项目 66 pytest 通过；后台会话修复后相关 15 项测试通过，包含会话跨切源保持、跨实例变化及姿态/模型会话一致性；TypeScript 检查和 Vite 构建通过。Vite 仍提示大于 500 kB 的 chunk，本轮未另做资源拆包。

本轮使用数值、帧哈希计数和页面元数据验证，未保存摄像头图片或录像。结果文件：

- `artifacts/c60e_low_resolution_benchmark.json`
- `artifacts/c60e_exposure_gain_probe.json`
- `artifacts/c60e_stream_fps.json`：最后一次 28.42 FPS 测量
- `artifacts/c60e_browser_validation.json`
- `artifacts/c60e_final_status.json`

`artifacts/c60e_project_status.json` 是此前自动曝光约 14 FPS 的旧记录，不是最终配置结果。

## 复现

```powershell
cd D:\dachaung
.\scripts\start_backend.ps1
# 前端使用现有 Vite 启动方式，登录 admin / admin123。
```

测试脚本运行前保证服务已启动、摄像头未被其他程序占用；它们会切换视频源，最终保留为 C60E 实时模式。

```powershell
$env:PYTHONPATH = 'D:\dachaung\.deps;D:\dachaung\backend;D:\dachaung\src'
python scripts/validate_camera_integration.py
python scripts/measure_video_stream.py --camera
```

尚未检查长时 USB 稳定性、低照度、远距离小人物、4K 或真实手机性能。本轮未更换跌倒权重，也未修改云端训练任务；毫米波仍是独立采集通道。
