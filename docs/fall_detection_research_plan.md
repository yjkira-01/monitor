# 跌倒检测开源参考与后续训练计划

更新时间：2026-10-08

本轮已新增 [固定权重泛化审计与全场景优化计划](video_generalization_audit_20261008.md)。URFD 70 段真实外部视频准确率 80%，正常活动误报 13/40，说明需要补充新场景与 hard negatives；不能把原验证 94.6% 当实际部署性能。已准备粗时间窗并修复训练/服务预处理，尚未开始下一轮优化训练。

本文件记录外部检索结果和下一阶段实验方案。视频侧已完成 TinyVideoNet 50 epoch 和 R3D-18 100 epoch 迁移学习，详见 [训练实验报告](training_run_20261008.md)；姿态时序模型和毫米波融合仍需独立验证，不把尚未验证的权重直接接入值班告警。

## 开源项目初筛

| 项目 | 主要方法 | 可以借鉴 | 当前判断 |
| --- | --- | --- | --- |
| [cwlroda/falldetection_openpifpaf](https://github.com/cwlroda/falldetection_openpifpaf) | OpenPifPaf 姿态 + 人体跟踪，多人视频/RTSP/摄像头 | 多人跟踪、姿态时序和事件确认；README 报告 UR Fall 上 Precision 100%、Recall 83.33%，但没有完整负样本评估 | 最适合做结构参考；依赖较旧，不能直接复制到当前环境 |
| [Bigtuo/YOLOv7-Pose-Bytetrack-STGCN](https://github.com/Bigtuo/YOLOv7-Pose-Bytetrack-STGCN) | YOLOv7-Pose + ByteTrack + ST-GCN | 关键点、ID 跟踪、时序图卷积的完整组合 | 最接近我们的目标架构；先复现接口，不直接拉旧依赖 |
| [Bigtuo/YOLO-POSE-Bytetrack-STGCN](https://github.com/Bigtuo/YOLO-POSE-Bytetrack-STGCN) | YOLO-Pose + ByteTrack + ST-GCN | 与上项目类似的姿态到行为识别流水线 | 用作第二份实现对照 |
| [mgei/fall-detection](https://github.com/mgei/fall-detection) | 轻量姿态估计，头部 Y 轴运动和规则判定 | Raspberry Pi / NCS2 端侧部署、低算力兜底 | 可借鉴边缘降级逻辑；项目较旧，不能作为主模型 |
| [RichardChen20/PoseFall](https://github.com/RichardChen20/PoseFall) | 视频跌倒检测和 ROS 系统 | 单图/视频事件接口、机器人端部署思路 | 用于接口和部署参考 |
| [AllenCaoo/Fall-Detect-AlphaPose-ST_GCN](https://github.com/AllenCaoo/Fall-Detect-AlphaPose-ST_GCN) | AlphaPose + ST-GCN | 关键点骨架到时序分类的最小闭环 | 用作 ST-GCN 实验对照 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO 检测/姿态、多目标跟踪 | 当前生态成熟、容易替换 HOG 通道 | AGPL-3.0，若用于发布或商业部署必须先做许可证评估 |
| [torchvision R3D-18](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.video.r3d_18.html) | 3D ResNet，Kinetics-400 预训练 | RGB 时空迁移学习，不依赖额外姿态或人体框标注 | 当前实际训练候选；使用已安装兼容版本，数据少时冻结前层和 BN，仍须做外部场景验证 |

开源仓库的许可证、权重和数据集授权不能只按 README 推断；真正复用代码前要逐仓库核对 LICENSE、模型权重条款和训练数据来源。

## 论文与综述

- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)，Tran 等，CVPR 2018。torchvision R3D-18 官方 API 引用的架构参考；本轮采用其 3D ResNet 基线迁移学习，不宣称已复现论文全部模型或结果。
- [A survey on fall detection: Principles and approaches](https://doi.org/10.1016/j.neucom.2011.09.037)，Mubashir 等，Neurocomputing，2013。用于建立传感器、视觉、阈值和机器学习方法的分类基线。
- [A survey on vision-based fall detection](https://doi.org/10.1145/2769493.2769540)，ACM Computing Surveys，2015。重点关注视角、遮挡、背景变化和误报问题。
- [Elderly Fall Detection Systems: A Literature Survey](https://doi.org/10.3389/frobt.2020.00071)，Frontiers in Robotics and AI，2020。适合整理养老场景中的部署约束和评价指标。
- [Radar-Based Fall Detection: A Survey](https://doi.org/10.1109/mra.2024.3352851)，IEEE Robotics & Automation Magazine，2024。支持毫米波先做独立特征和质量评估，再做晚期融合的路线。
- [Survey on data fusion approaches for fall-detection](https://doi.org/10.1016/j.inffus.2024.102696)，Information Fusion，2025。为视觉与毫米波的 late fusion、置信度校准和缺失模态处理提供依据。
- [Unsupervised Keypoints for Real-Time Fall Detection](https://arxiv.org/abs/2607.15400)，2026。可关注无监督关键点和带宽自适应，但需要在公开数据和本项目数据上独立复现后再采用。
- [Anticipatory Fall Detection in Humans with Hybrid Directed Graph Neural Networks and Long Short-Term Memory](https://arxiv.org/abs/2509.05337)，2025。可作为“跌倒前风险”方向的候选，不纳入第一版告警模型。
- [PifPaf: Composite Fields for Human Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.html)，CVPR 2019。对应 OpenPifPaf 项目的姿态基础。

当前检索结论是：第一版应该优先解决“事件级误报”和“跨受试者泛化”，而不是直接追求更大的 backbone 或大模型解释。

## 我们的优化路线

### 阶段 A：固定可比较基线

1. 保留当前 `TinyVideoNet` 作为训练闭环基线，记录 Accuracy 之外的 macro-F1、fall recall、ADL precision、每小时误报数和端到端延迟。
2. 继续按 Subject 划分，禁止同一受试者的视频同时出现在训练和验证；当前 GMDCSA24 是 Subject 1–3 训练、Subject 4 验证。
3. 增加事件级确认：连续窗口投票、进入/退出滞回、冷却时间，避免同一次跌倒产生多条告警。

### 阶段 B：视觉主模型候选

1. 用 YOLO Pose 或 RTMPose 输出人体框、17 点关键点和置信度；ByteTrack 为每个人保留 ID。
2. 对每个 ID 生成归一化骨架序列：相对髋部坐标、身高归一化、关键点置信度和缺失掩码。
3. 第一候选使用轻量 TCN/GRU，第二候选使用 ST-GCN；输入 32–48 帧、滑窗步长 4–8 帧。
4. 特征加入躯干倾角、髋部下降速度、人体框宽高比、关键点速度/加速度和地面接触后的静止时长。
5. 用坐、躺、弯腰、起身、快速坐下等 hard negative 做重点验证，避免把正常卧床或坐下报成跌倒。

### 阶段 C：训练策略

长 epoch 训练前先固定数据和配置，建议顺序：

```text
候选模型：YOLO/RTMPose + ByteTrack + TCN（先）→ ST-GCN（对照）
训练轮数：50–100，Early Stopping patience 10–15
优化器：AdamW，warmup + cosine decay
损失：交叉熵起步；类别不平衡明显时再试 focal loss
增强：亮度/裁剪/缩放/模糊/遮挡/视角扰动，避免改变跌倒语义
选择指标：subject-held-out fall recall、macro-F1、误报/小时、延迟
保存：最佳验证 F1、最佳 fall recall、最后一轮各保存一份
```

实际训练采用 GMDCSA24、Subject 1–3 的 123 个视频训练/Subject 4 的 37 个视频验证。Tiny 50 epoch 完成，验证准确率 0.649、macro-F1 0.621，ADL 仅识别 7/20，有 13 次误报；这说明简单延长随机初始化小模型不足以解决泛化。

随后完成 R3D-18 100 epoch：官方 Kinetics-400 预训练，固定 stem/layer1–3 和全部 BatchNorm 统计量，更新 layer4/fc；batch 4、16 帧、官方 128x171 缩放/112 crop、AMP、AdamW 1e-4（分类头 1e-3）、5 轮 warmup + cosine。训练用整片一致的横翻和亮度增强，缓存 uint8 片段约 160MiB；FFmpeg 单线程，避免与 worker 叠加。配置和源码 SHA256、清单 SHA256、完整断点以及逐视频预测均保存。最佳验证 accuracy 0.946、macro-F1 0.945，混淆矩阵 [[20,0],[2,15]]；第 4 epoch 达到最佳，其后稳定，保存该轮权重。

主权重按 macro-F1 选择，高召回权重按 recall 后再按 macro-F1 选择，避免全报 fall 的早期模型一直占据“最佳召回”；last 含优化器、调度器、AMP scaler、随机状态。当前按用户长训要求固定跑 100 轮，未启用 early stopping；Subject 4 已用于选择 epoch，不能报告为独立测试集。事件级每小时误报数尚无长时间负样本视频，当前只统计整段视频分类错误。

姿态/跟踪接口和 hard negative 清单完成后，再比较 TCN/ST-GCN 路线，不把本轮 RGB 模型称为已经完成的骨架时序方案。

### 阶段 D：毫米波和双流

毫米波暂时只做 OOB 帧解析、点云/相位/呼吸频带特征、数据质量和独立报告。设备到位并完成时间同步后，再比较：

```text
视觉概率 + 雷达概率 → 置信度校准 → late fusion
缺失一个模态 → 使用另一模态并降低置信度，不补造数据
```

毫米波不进入本轮视觉长 epoch 训练，避免用模拟或未校准数据污染主模型。

## 当前工程状态

2026-10-08 用户已要求开始架构优化与扩大训练，之前“只计划、不启动”的限制已撤销。新一轮 RGB 与姿态候选已实现并在云端执行/排队，最新数据、划分、命令和验证见 [架构优化与扩大训练](architecture_training_v2_20261008.md)。旧的4人整段分类指标不代表新窗口任务指标。

- GMDCSA24 已建立清单，TinyVideoNet 10 轮基线最佳验证准确率 0.541，权重在 `artifacts/gmdcsa24_tiny_10ep.pt`。
- 实时页面现已使用训练导出的 R3D-18 权重；HOG 旧通道不再作为正式页面的主要判定。
- 云端 Tiny 50 epoch、R3D-18 100 epoch 均已结束，已核对完整日志、last 的 epoch=100、最佳权重和 37 条验证预测；R3D 日志为 `artifacts/gmdcsa24_r3d18_100ep.log`。
- `scripts/predict_video.py` 读取训练导出的 `*_inference.pt`，无需摄像头即可分类视频文件；模型分数未校准，整段标签不等同于事件时间定位。实时页面使用持久异步 R3D 滑窗，事件级精度仍待专门评估。

## 2026-10-08 页面持续推理接入

正式页面现已常驻运行训练导出的 R3D-18；最多5秒窗口、16帧、约0.8秒间隔，展示模型分类、窗口曲线、CAM响应、特征变化和质量，并将结构化证据送入值班摘要。回放与现场告警隔离。两段样例只用于接入验收，不能替代下述事件级/外部验证；详细证据见 [工作台验收](video_workbench_validation_20261008.md)。后续优化优先增加事件起止标签、长时正常活动、床上后仰与主动躺下的成对样本，再验证阈值、召回/误报和跨场景泛化。
