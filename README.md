# 青禾康养 · 视频与毫米波监测

视频跌倒识别、人员状态跟踪、SR112 雷达点云与生命体征、事件处理及监测报告。提供 Windows C# 桌面应用和 React + FastAPI 网页版。项目仓库：[yjkira-01/monitor](https://github.com/yjkira-01/monitor)。

## 直接使用桌面应用

从 [v0.3.2 发布页](https://github.com/yjkira-01/monitor/releases/tag/v0.3.2) 下载 `CareMonitor-0.3.2-embedded-win-x64.zip`，解压后双击唯一的 `CareMonitor.exe`。支持 Windows 10/11 x64，无需安装 Python、Node.js 或 .NET。

完整模型、图标和运行库已嵌入 EXE。首次运行会释放到 **EXE 所在文件夹的 Runtime 目录**，以后校验并复用。请把程序放在可写文件夹；档案、设置和标定保存在用户数据目录。打开应用后可选择本地视频测试，再连接摄像头或雷达。

## 使用网页版

需要 Python 3.10 或更新版本、Node.js 20 或更新版本。先下载仓库源码或运行：

```powershell
git clone https://github.com/yjkira-01/monitor.git
cd monitor
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install_web.ps1
```

安装脚本会安装依赖，从本项目 Release 下载校验过的模型与两段公开测试视频，并构建页面。完成后双击根目录的 **启动网页版.cmd**，浏览器打开 [127.0.0.1:5173](http://127.0.0.1:5173)。首次账号为 `admin / admin123`。初始使用公开跌倒回放，雷达来源初始为明确标注的记录回放，不自动占用摄像头或雷达。

已准备好依赖的本机可直接双击启动入口。模型资源手动安装：从发布页下载 `web-resources.zip`，按包内目录结构解压到项目根目录。上传自己的视频可以完全不使用摄像头。修改网页后在 `frontend` 下执行 `npm run build` 更新页面。

启动真实设备时可分别运行下列脚本，串口与视频来源也可用环境变量覆盖：

```powershell
.\scripts\start_backend.ps1
# 另一个终端
.\scripts\start_frontend.ps1
```

前端地址为 `http://127.0.0.1:5173`，API 文档为 `http://127.0.0.1:8000/docs`。网页默认早期 GMD 二分类权重；设置 `VIDEO_MODEL_PATH` 可选择已有 Stage C 对照。服务初次加载模型需要等待，画面和推理分别更新。演示账号和当前认证配置用于本地项目使用；公网部署需替换 JWT 密钥、账号及网络访问配置。

## 当前功能

| 功能 | 桌面端 | 网页端 |
| --- | --- | --- |
| 本地视频、摄像头、播放控制 | 支持，另有网络视频 | 支持本地上传与摄像头 |
| 人体框和跟踪状态 | RTMO-m，按人员显示证据 | SSDLite + MediaPipe，按人员显示证据 |
| 跌倒判定 | R3D 主窗口 + 有动作支持的短窗口 | R3D 主窗口 + 姿态 TCN 与短窗口 |
| SR112 点云、轨迹及心率/呼吸 | 实际串口与匹配固件 TLV | 实际串口、记录回放及显式演示 |
| 区域、停留与静止提醒 | 支持，可修改、删除 | 显示视频分析与事件记录 |
| 双摄标定和三维定位 | 支持真实棋盘标定，需现场操作 | 当前未接入桌面双目流程 |
| 住户档案 | 本机真实档案管理 | 尚未接入档案服务，不显示虚构住户 |
| 智能分析接口与监测报告 | 配置、测试、原生排版及导出 | 设置入口、本地摘要与兼容大模型接口 |

雷达独立采集，不进入当前视频主模型训练。三维人物动画已移出监测页面；雷达点云仍在独立工作区展示。各路人员编号属于跟踪编号，不能自动对应住户身份。原视频不上云，大模型接口仅接收结构化观测摘要。

## 0.3.2 检测改进与限制

保留最近5秒的原始模型结果，增加最近2秒复核。短窗口必须得到前后动作与图像位置支持并连续确认，避免把正常躺床直接当作跌倒；长短分值分别显示在人员详情、报告和导出中。本次没有重新训练权重。

桌面真实回放的14段已知公开视频中，7段跌倒检出 **6段**，上一版为4段；7段正常无误报。扩大到Subject4全部20段正常活动后无跌倒误报，但与前一组有6段正常片段重叠。这些是已查看的开发回归，**不是独立测试准确率**，也不能直接套用到网页。仍漏检Fall09，ADL05/07/10仍出现多余人体框；真实多人、暗光和遮挡场景需要继续验证。静止与躺姿不能直接作为睡眠分期，体征也未完成医学精度验证。

- [本轮模型优化与验证](docs/model_optimization_20261011.md)
- [双目标定操作说明](docs/stereo_calibration_guide.md)
- [桌面使用说明](desktop/README.md)
- [数据集与模型许可](MODEL_NOTICES.md)
- [训练、数据与划分记录](docs/architecture_training_v2_20261008.md)
- [项目清理记录](docs/project_cleanup_20261011.md)

## 源码与验证

`frontend` 为当前网页，`backend/app` 为API与推理，`desktop/CareMonitor` 为C# WPF应用，`src/yanglao_monitor` 为模型与训练工具。较早的 `web` 目录只保留独立融合接口示例，正式网页入口为 `frontend`。

```powershell
python -m pytest backend/tests tests/test_project_archive.py tests/test_fetch_web_resources.py
cd frontend
npm ci
npx tsc --noEmit
npm run build
```

桌面编译需要.NET 8 SDK。模型与原生运行资源不进入Git历史：可先下载并运行正式EXE释放资源，将Runtime版本目录里的 `Assets` 和项目需要的运行库复制到开发工程对应位置；也可使用 `scripts/prepare_open_source_models.py`、`scripts/export_binary_fall.py`、`scripts/export_desktop_models.py` 和 `scripts/prepare_desktop_runtime.py` 从已记录来源准备。随后 `dotnet build desktop/CareMonitor/CareMonitor.csproj -c Release`，或执行 `desktop/publish.ps1` 生成新的内嵌包。已有本机离线NuGet源时会复用，其他环境使用标准NuGet恢复。

代码沿用本仓库Apache-2.0许可；第三方代码、模型、数据与图标的独立许可保留在通知文件中。Git仓库不含训练原始数据、私人视频、用户数据库、密钥、云端凭据、历史安装包或本地实验记忆。运行资源通过版本化Release交付。
