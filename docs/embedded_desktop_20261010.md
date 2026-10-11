# 单文件桌面发布与首次资源释放

当前0.3.1 使用原生 C# 启动器。后文0.2.1的实测数据保留为历史验证。完整 WPF 应用、模型、图标、许可、说明、.NET 运行时、ONNX Runtime / DirectML / OpenCV 和 Visual C++ 运行库先发布为自含目录，再压缩并作为 `EmbeddedResource` 编译进启动器。启动器本身也发布为自含单文件。交付目录仅包含 `CareMonitor.exe`，可以复制到空目录运行，无需另外携带 Assets、DLL 或 ZIP。

首次运行把内嵌资源流式释放到 **EXE 所在文件夹下的 `Runtime\0.3.1-<内嵌包SHA前16位>`**。默认使用 `AppContext.BaseDirectory`，以可执行文件的位置为准，与快捷方式的“起始位置”或命令行工作目录无关。移动整个程序文件夹后仍能复用资源；只移动 EXE 时在新位置重新释放。该文件夹需要写入权限，不回退到用户缓存目录。程序不依赖在线下载。启动准备超过半秒时显示进度，可以取消。

每次启动逐文件验证长度和 SHA256，完整缓存直接复用；文件缺失、损坏或出现清单外的 DLL / EXE / ONNX 时从程序内嵌资源重建。安装在临时目录完成，验证后才切换到正式目录。相同资源使用命名互斥锁，避免多个启动进程重复释放；已运行版本持有资源租约，修复前需关闭该版本。中断的临时目录在下次启动清理，不以不完整目录启动应用。拒绝绝对路径、路径穿越、Windows 保留文件名和目录链接。

缓存按版本和内容哈希隔离。用户档案、设置、事件与报告仍保存在 `%LOCALAPPDATA%\CareMonitor` 原有位置，资源释放不覆盖这些数据。旧版缓存不自动删除。原有识别权重和业务加载方式保持，子程序仍从已释放目录加载模型和原生库。参数采用 `ProcessStartInfo.ArgumentList` 原样转发，启动器返回子程序退出码。

Windows 单文件宿主自身的底层原生库由 .NET 的单文件机制自动释放到临时目录 `.net` 下；完整业务应用、模型、图标、OpenCV、DirectML、VC++ 库和业务所需 .NET 运行时使用上述 EXE 旁的 `Runtime` 目录。两者均来自 EXE 内部，无需外挂文件或预先安装 .NET。首次运行需要额外磁盘空间；启动器的自含运行时和内嵌应用运行时会重复一部分文件，这是保留已验证业务加载方式的取舍。

发布：运行 `desktop/publish.ps1`，默认生成 `desktop/release/CareMonitor-0.3.1-embedded-win-x64/CareMonitor.exe`。已有发布目录会被拒绝，旧发布保持。`.packaging` 是开发构建中间文件，不需要分发。单 EXE 的 SHA256、内嵌清单和大小记录在发布目录旁的 `.build.json`。

诊断参数（正常使用无需设置）：

- `--bootstrap-cache <目录>`：将资源缓存放到指定普通文件夹。
- `--bootstrap-extract-only`：完成释放/校验后退出，不启动主应用。
- `--bootstrap-quiet`：隐藏准备界面，用于自动化测试。
- `--bootstrap-report <JSON文件>`：将本次释放结果与子程序退出码保存到指定路径。默认写到缓存根目录的 `bootstrap-last.json`。

启动器打包机制不改变模型准确率，不恢复云端训练。验证结果见 `artifacts/embedded_desktop_20261010`，仅使用公开回放，不采集私人帧。

## 修正版实际验证

用户明确要求释放到程序所在位置，最终交付为 `CareMonitor-0.2.1-embedded-win-x64-r2`，替代此前默认用户缓存目录的封装版。已完成真实单 EXE 默认位置测试，启动工作目录刻意设在 EXE 文件夹之外，仍释放到 EXE 旁的 `Runtime`。整体移动程序文件夹后，使用新路径并复用已有文件。

- 535 个释放文件，677,294,714 字节，全部与内嵌清单 SHA256 一致；48 个模型及图标等资产与此前已验证 r2 业务发布完全一致。
- 首次资源准备约 6.26 秒，移动后全量 SHA256 校验并复用约 1.10 秒。这是本机资源准备耗时，不包含模型预热，也不代表其他硬件速度。
- 缺失图标、同大小同修改时间的模型字节损坏均自动修复；两个独立启动进程只释放一次。取消、未完成释放、版本隔离与路径安全等 35 项开发检查通过。
- 真正发布 EXE 的 27 项检查通过，包含 CPU/GPU 原生自检、失败退出码转发、中文及带空格路径、默认位置、移动复用和界面回归。
- 公开视频 UI 验证：347 个视频帧、129 次姿态观测、418 次单人/场景分值一致检查，七页导航、详情与图标通过；p95 UI tick 48.64 ms，不作为识别 FPS。
- 便携 ZIP 仅一项 `CareMonitor.exe`，CRC 与文件 SHA256 通过。EXE 为 638,156,835 字节，SHA256 `2caff82b26f84a9948e4e120bd638c30e9be0c1d5a1261dbbc74e08f83a0c804`。

证据：`artifacts/embedded_desktop_20261010/package-local/result.json`、`native-ui-result.json` 与 `integrity-local.json`。

## 0.3.0 设备升级版本

上述 0.2.1 为历史交付记录。当前 `desktop/publish.ps1` 默认发布 `CareMonitor-0.3.0-embedded-win-x64`，资源继续释放在 EXE 旁，档案位置不变。本版新增双目与雷达功能，内嵌 538 文件，EXE 为 638,230,563 字节。新的首次释放、移动复用、损坏修复、并发、CPU/GPU 和八页 UI 已实跑通过；完整性检查包含内外 ZIP CRC 和逐资源 SHA256。证据在 `artifacts/device_upgrade_20261010/package/result.json`、`integrity.json`，设备及识别限制见 `device_upgrade_20261010.md`。
