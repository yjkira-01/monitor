# 三维人体与点云优化：实现与验证

2026-10-09 更新：本记录保留早期单人/60 FPS 实验。当前已改为多人独立骨骼、每人历史姿态补全和参考结构，两画布上限 30 FPS；最新双人稳定性验证与观测限制见 [最新记录](video_partial_multistate_20261009.md)。

## 原因与实现

旧版人物姿态与 R3D 分类在同一个线程内执行，约每 0.8 秒才能产生一个新姿态；前端每次收到状态会清空人物 Mesh，并销毁/重建点云几何缓冲区。旧版没有连续动画或姿态插值。球体/圆柱几何和材质原本已经共享，本轮没有证实 GPU 内存泄漏。

本轮将姿态拆为 `VideoPoseWorker`：采集按最多 15 Hz 提交最新帧，队列只保留一帧，MediaPipe Full 使用 VIDEO 跟踪模式；SSDLite 每 0.5 秒检测人物，间隔内复用已有光流跟踪。源切换清空结果、拒绝旧代次结果，并重建 VIDEO 跟踪图以允许时间戳从零开始。姿态错误独立于 R3D 分类。暂停后不继续推算人物运动，EOF 保留最后有效观测。

`GET /api/video/pose` 是受保护的精简接口；三维区域独立轮询，不等待 750 ms 的整页状态轮询，也不把跌倒模型的结果年龄当作姿态年龄。缺失/过期姿态隐藏人物。原始视频仍仅叠加人物框。

人物由本地 Quaternius 连续蒙皮网格替换胶囊部件。只加载一次，按已标定的绑定姿态把关节方向转换为骨骼旋转，复用皮肤权重与几何。左右髋与肩确定躯干朝向，低置信度肢体保留最近有效方向。体型、脸部和手指是资产预设，不代表视频人物的身份或测得的身体细节。

显示端使用 One Euro Filter（min cutoff 1.5 Hz、beta 1.5、derivative cutoff 1 Hz）与约 45 ms 的指数追随，按视频时间计算观测间隔。超过 0.6 秒的缺口、切源或切人物丢弃平滑历史。渲染最多 60 FPS；移出视口/切后台停止渲染。暂停时只完成有限的追随，不外推新姿态。去除实时阴影，像素比最高 1.5。

模拟点云直接从平滑后的关节写入一个固定 2048 点 `DynamicDrawUsage` 缓冲区，最多 704 个示意点；不重建几何。真实雷达模式只更新测量坐标，不按数组下标插值不存在对应关系的点。雷达轮询只在雷达视图启用，模拟数据始终标为非实测。

## 开源项目与论文的取舍

| 参考 | 核验来源 | 本轮用途 |
|---|---|---|
| Quaternius Universal Base Characters | [作者页面，CC0](https://quaternius.com/packs/universalbasecharacters.html) | 实际采用 Superhero Male 绑定网格；原几何/骨骼/蒙皮，材质改为医疗绿 |
| Quaternius GLB 分发副本 | [asset 说明](https://github.com/programasweights/avatar/blob/main/ASSETS.md) | 使用单文件打包版本，不含外部动画；本地资源 741,320 字节 |
| MediaPipe / BlazePose GHUM | [VIDEO API](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)、[Grishchenko et al., 2022](https://arxiv.org/abs/2206.11678) | 实际采用已有 Full 权重的视频跟踪，独立于 R3D 分类 |
| 1€ Filter | [作者实现与 CHI 2012 论文入口](https://gery.casiez.net/1euro/) | 按速度自适应平滑，抑制静止抖动；使用本项目的小型实现 |
| Kalidokit | [开源仓库](https://github.com/yeemachine/kalidokit) | 参考姿态到骨骼旋转的路线；本项目按固定资产标定，不增加此依赖 |
| three-vrm | [MIT 仓库](https://github.com/pixiv/three-vrm) | 考察 VRM 路线；当前不需要角色平台/表情/弹簧骨，因此直接使用现有 Three.js SkinnedMesh |
| Three.js | [SkinnedMesh 与 BufferAttribute 文档](https://threejs.org/docs/) | 实际使用已有依赖的 GPU 蒙皮、持久点云缓冲区与生命周期清理 |

资产 SHA256：`d6f3b64cab629f6ddca9ba063811aeffd3663a8a4e49ade47cd9505dc2bbd70f`。原许可和资源出处分别保留在 `frontend/public/models/QUATERNIUS-LICENSE.txt` 与 `provenance.json`。代码参考与模型资产的许可独立；没有分发 Mixamo 或许可未明确的 SMPL 模型。

## 验证方法与结果

使用本机 RTX 4060 Laptop / ANGLE Direct3D11 的真实无头 Chrome，**没有禁用 GPU**，同一 ADL 示例与双画布，各采样约 8 秒。只统计主画布的颜色缓冲清屏，排除离屏阴影处理；它表示实际 WebGL 画布输出调用，并不是显示器实际呈现或用户手机的 FPS。旧版重复绘制同一姿态会使渲染调用频率高于姿态更新频率。

| 指标 | 改动前 | 最终复测（60 FPS 上限） |
|---|---:|---:|
| 人体主画布渲染 | 2.47 FPS | 51.00 FPS 平均，末段 60 FPS |
| 模拟点云主画布渲染 | 4.45 FPS | 51.00 FPS 平均，末段 60 FPS |
| 新姿态观测 | 1.24 Hz | 实际接受 12.84 Hz；末段后台滑动统计 15.2 Hz |
| 原视频采集 | 29.87 FPS | 29.83 FPS |
| 人物几何/绘制调用 | 未统计 | 4 / 4 |
| 点云几何/绘制调用 | 未统计 | 2 / 2 |

独立 MediaPipe 视频模式探测：首帧约 123 ms，随后连续帧约 30–35 ms。完整后台并发时，本次观测的姿态推理耗时均值约 77 ms；这些耗时不含浏览器网络、轮询与平滑延迟。轮询实际看到的新姿态约 9.10 Hz，不能等同于后台处理频率。初版未限制渲染，在本机高刷新率环境跑到 185 FPS 平均/237 FPS 末段，随后主动限制到 60，避免无用开销。

原始报告：`artifacts/spatial_baseline_metrics.json`、`artifacts/spatial_optimized_metrics.json`。最终复测确认同一视频源代次保持不变；以 JSON 为最终数值依据。截图为 `artifacts/spatial_optimized.png`。

回归：全项目 62 项 pytest 通过；TypeScript/build 与空间自检通过。空间自检覆盖源/人物切换、跟踪缺口、有限矩阵、手臂方向与持续复用几何。真实 Chrome 已验证暂停位置/姿态保持、旋转/复位、模拟/雷达离线切换、390 px 无横向溢出。工作台的 Fall 6 窗与 ADL 16 窗、上传、暂停继续、摘要上下文也已通过。

复现：

```powershell
$env:PYTHONPATH="$PWD\.deps;$PWD\backend;$PWD\src"
python scripts/validate_spatial_views.py
python scripts/validate_video_workbench.py
cd frontend
node scripts/check_spatial.mjs
```

尚未进行真实手机/低端 GPU 与毫米波硬件帧率验证。当前三维仍是单目估计，受遮挡、远景和多人关联影响；更漂亮的网格与高渲染帧率并不改善跌倒分类准确率。云端长训练与既有页面跌倒权重未被本轮替换。
