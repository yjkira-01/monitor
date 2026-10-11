# 视频架构优化与扩大训练（2026-10-08）

2026-10-09 更新：下文为早期 A/B 阶段实验，当前页面已部署 Stage C 暗光/局部画面二类权重，十类 Stage D 从第 68 轮关机断点恢复。当前逐人片段推理和多人三维实现见 [最新记录](video_partial_multistate_20261009.md)；下文的旧权重、首轮进度与单人显示描述不作为当前运行状态。

正式训练在用户授权的云端 RTX 3080 Ti 12GB 上运行。旧网页继续加载 `gmdcsa24_r3d18_100ep_inference.pt`；新候选权重通过验证后再替换。毫米波不进入这一轮主模型训练。

## 数据与固定划分

新增源为 [Zenodo 17170592](https://zenodo.org/records/17170592) 的作者动作片段，记录许可 CC BY 4.0。上游 [CAUCAFall v4](https://data.mendeley.com/datasets/7w7fccy7ky/4) 也标为 CC BY 4.0；[UP-Fall 官方站](https://sites.google.com/up.edu.mx/har-up/) 要求引用原始数据论文。保留原始 ZIP、元数据和 MD5，不混入 ZIP 中未被动作清单引用的重复副本。

| 数据 | 人物划分 | 状态 |
|---|---|---|
| GMDCSA24 | 1–3 训练，4 验证；后者已参与旧模型选模 | 已在云端 |
| CAUCAFall | 1–7 训练，8 验证，9–10 独立测试 | 本机和云端完成下载与 MD5 校验 |
| UP-Fall | 1–11 训练，12–14 验证，15–17 独立测试 | 云端3,177,158,271字节下载、MD5校验与解包完成；B阶段接续 |
| URFD | 已查看结果的外部开发基准 | 不加入本轮训练，不称为未见测试 |

第一阶段 14 个受试者标识、367 个不同的视频文件、1824 个窗口：训练 1398、验证 367、测试 59。同一个父动作可能被作者切成多个视频文件，这些文件不视为统计独立样本；划分按人物/父动作执行。

云端完整索引已生成：31人、2299个不同视频文件、816个父动作、4800个窗口；21人/3354窗训练，5人/903窗验证，5人/543窗独立测试。清单SHA256为 `599d7c97b6a6805ef8a7ad9968835de3a10e2deeacc1a752f92569b3aab4be40`。测试仍未用于模型预测/选择。

第一阶段姿态候选已实际完成：1704个有效姿态窗口，剔除61个缺失覆盖较高窗口；200轮上限训练在第43轮早停，验证最佳第3轮，macro-F1 0.831578、fall recall 0.822785，矩阵 `[[240,30],[14,65]]`。这是覆盖筛选后的349个验证窗口，不能声称优于RGB或旧模型，更不能将它直接上线作个人跌倒判断。

第一阶段RGB也已实际完成：第110轮早停，验证最佳第80轮，367个验证窗口 accuracy 0.961853、macro-F1 0.945036、fall recall/precision均0.914634、ADL recall 0.975439，矩阵 `[[278,7],[7,75]]`。这是参与选模的混合窗口验证，不能与旧4人整段分类直接比较，也不是新独立测试。完整31人阶段的48窗口/两轮GPU功能检查通过，正式3354训练窗口/903验证窗口任务已启动并在缓存片段。

最新复连确认：完整阶段缓存3354/903窗已经完成，实际GPU训练第1轮完成（18.4秒、train_acc 0.923、val_macro-F1 0.907、fall recall 0.782、ADL recall 0.985）。主编排PID417803仍运行，nvidia-smi采样1678MiB/90%利用率。首轮指标只记录进度，不作为最终性能结论。

`scripts/index_multifall.py` 只读取 `mmaction2_annotations.txt` 引用的视频。作者类别 1（fall 转变）作为正例，类别 2（fallen）排除，其余明确动作作为 ADL；不把已经倒地混作安全 ADL。长 ADL 片段最多取三个不超过 5 秒的窗口。源 GMD 时间标签仍是未人工复核的粗标签，已有三条异常父视频保持排除。

受试者、父动作和视频不得跨 train/val/test；UP-Fall 的两个摄像机按人物/父动作一起划分。独立测试受试者不进入训练、选模或姿态缓存。

## 实际模型改动

- RGB 对照：官方 Kinetics 预训练 R3D-18，增加 layer3 微调（主学习率的 0.3 倍），layer4 主学习率，分类头 10 倍；保留固定 BatchNorm 统计量。数据集/类别平衡采样，整窗统一镜像、亮度、缩放、平移和对比度变化。主学习率 1e-4，weight decay 1e-3，batch 8，16 帧。上限 150 epoch，macro-F1 30 轮无改善早停。
- 姿态候选：自行实现的轻量 dilated TCN，197634 参数，不冒充 ST-GCN 论文复现。33 关节身体相对 xy、关节可见度、关节和髋的帧间位移、身体尺度变化和躯干有效掩码，共 169 特征；三层 dilation 1/2/4、mean/max 聚合。batch 64，48 帧，学习率 1e-3，weight decay 5e-3。上限 200 epoch，40 轮无改善早停。
- 姿态预处理：官方 MediaPipe full，CPU 提取，15fps，每视频缓存 NPZ，横坐标按宽高比修正；没有检测到的帧/关节保留缺失，不编造位置。窗口躯干可见覆盖低于 20% 时不纳入姿态训练/验证，完整排除表写入 coverage 报告。因此姿态指标不能直接与 RGB 全量指标等同比较。

时序建模参考 [TCN 序列建模论文](https://arxiv.org/abs/1803.01271)，后续图结构候选参考 [ST-GCN 论文](https://arxiv.org/abs/1801.07455) 和 [MMAction2 骨架配置](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton)。本轮先用小 TCN 实际跑通新人物/动作数据，再按测得的误报与漏检决定是否增加图结构或融合。

姿态提取目前选择单个姿态，不等于经过验证的多人身份跟踪；新模型暂未向页面输出个人跌倒分数。位移特征没有米/秒单位。单目姿态不用于声称测得心率、真实深度或医学睡眠分期。

## 云端验证与任务

两轮 RGB 实际 GPU 检查通过。首版 FP16 loss 有限但 GradScaler 梯度溢出；3080 Ti 支持 BF16，切换后 forward/backward 和保存断点通过，训练使用 BF16。

批量缓存首跑暴露 worker 共享标签保留文件句柄的问题。修为缓存私有张量后，云端在 128 文件句柄上限下缓存 600 条真实 DataLoader 样本通过。正式 1398/367 条视频窗口缓存随后完成，并已进入 GPU epoch。

真实 ADL/Fall 姿态窗口 GPU 检查：输入 2×169×48，loss 0.663086，梯度有限，峰值张量显存 18.31MiB。关闭 TF32 的 CPU/CUDA 对照最大差 4.77e-7。首次功能检查用时 0.66 秒，包含冷启动且与 RGB 任务并发，不能当作独立吞吐量基准。

云端目录 `/root/autodl-fs/yanglao-monitor`，解释器 `/root/miniconda3/bin/python`。非交互 SSH PATH 没有 python，必须用此绝对路径。任务信息不包含 SSH 密码。

- `artifacts/multifall_v2_jobs.json`：真实 PID、命令和日志，覆盖失败后的新 PID。
- `artifacts/rgb_stage_a.log`、`multifall_r3d_v2_stage_a.json`：第一阶段 RGB 实际 epoch/指标。
- `artifacts/pose_cache_stage_a.log`：姿态提取进度。
- `artifacts/curriculum_stage_a.log/json`：缓存完成后，姿态两轮检查和 200 轮上限训练。
- `artifacts/curriculum_stage_b.log/json`：已确认 UP-Fall 完整解包和31人索引，进行 RGB 两轮检查、150 轮上限训练、姿态检查和 200 轮上限训练。第一阶段RGB实测峰值约1GiB，因此完整数据准备可与第一阶段余下训练重叠；分别保留输出。任一子步骤失败即停止并写 failed，不能把排队视为完成。
- `artifacts/pose_tcn_gpu_check.json`：真实姿态 GPU 检查。
- `artifacts/multifall_stage_a_source_retry/` 和 source SHA256：实际代码快照；首个失败版本另行保留。

每个模型仍保留最佳 macro-F1、高召回、last、optimizer/RNG、manifest SHA256 和 JSON 历史。上限不是保证跑满；根据验证早停，独立测试保留到最终固定候选后。当前 pooled 验证中 GMD 窗口较多，后续验收必须分别统计每个数据集的误报与漏检。

## 网页帧率与 App

原页面默认将 0.8 秒更新一次的单帧检测截图作为主画面，原 MJPEG 又固定为 12.5FPS。现在主画面使用连续 MJPEG，输出速率随视频源（上限 30FPS），人物检测间隔之间以短时 LK 光流跟随真实像素运动。低纹理、跟踪失败或检测超过 1.5 秒时丢弃框，等待下一次检测；原视频无骨架线。

本机实测带框接口连续 4 秒收到 110 个不同画面，约 27.26FPS，源 29.8FPS、采集 29.7FPS；结果 `artifacts/video_stream_fps.json`。识别/姿态/3D 更新仍按模型结果频率，不把视频播放 FPS 写成模型推理 FPS。真实 Chrome 跌倒/ADL、播放控制、三维和 390px 布局通过。公网网络和手机实机帧率未测。

后续 App 先采用可安装 PWA，复用鉴权、视频源、人物视图、风险记录和大模型配置；静态缓存不得缓存鉴权视频/报告。需要系统后台、蓝牙/串口或原生通知时再决定原生封装，本轮没有宣称已发布 App。
