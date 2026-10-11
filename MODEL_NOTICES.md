# 模型与样例说明

应用源码沿用目标仓库的 Apache-2.0 许可证；模型、数据和第三方代码各自的许可独立保留，不因源码许可证而改变。

- 跌倒视频权重：项目训练的 R3D-18 二分类 GMD 和 Stage C 对照；架构来自 torchvision（BSD-3-Clause）。GMDCSA24 来源为 [Ekram Alam 等人的数据集](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos)，DOI [10.5281/zenodo.12921216](https://doi.org/10.5281/zenodo.12921216)，其上游 MIT 许可随样例保存。
- 姿态时序权重：项目训练的 Pose TCN；数据来源、划分与限制见 [训练与数据记录](docs/architecture_training_v2_20261008.md) 和 [数据清单](docs/datasets.md)。不同训练数据的使用条件分别适用。
- `pose_landmarker_full.task`：Google MediaPipe 官方模型；[模型说明](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)，MediaPipe 项目 Apache-2.0。
- SSDLite COCO 人体检测权重：torchvision 官方发布；[torchvision](https://github.com/pytorch/vision)，BSD-3-Clause。COCO 数据的原始许可条件分别适用。
- 桌面 RTMO、ByteTrack 和图标等来源与许可随 EXE 释放到 `Assets/Notices`，源码同目录也保留通知文件。

资源包只含当前运行需要的模型及两段已署名的公开视频样例，不含用户上传的视频、个人档案、API 密钥、云端登录凭据或训练原始数据集。

当前模型用于科研演示和辅助复核，未完成独立现场准确率或医学有效性验证。开发片段上的改进不能代表真实夜间、遮挡或多人场景性能，也不能把躺姿、静止当作睡眠分期。
