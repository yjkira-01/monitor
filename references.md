# 开源实现核验记录

截至 2026-10-07，以下仓库可公开访问，且其 README/代码与本项目方向直接相关：

- [AdrianNunez/transformer-based-fall-detection](https://github.com/AdrianNunez/transformer-based-fall-detection)：按 UP-Fall 的 Subject/Activity/Trial/Camera 帧目录训练 Transformer/Video 模型，适合作为视频监督训练参考。
- [EikeSan/video-fall-detection](https://github.com/EikeSan/video-fall-detection)：OpenCV 轮廓宽高比与连续帧计数的传统基线，适合做无模型对照，不作为最终模型。
- [radar-lab/mmfall](https://github.com/radar-lab/mmfall)：4D mmWave 点云 + HVRAE；只用正常日常活动训练，通过异常分数尖峰和人体重心高度下降联合判定跌倒。我们采用这个判据作为毫米波分支的设计参考。
- [JJC-afk/mmWave-Radar-CNN-Fall-Detection](https://github.com/JJC-afk/mmWave-Radar-CNN-Fall-Detection)：TI mmWave 数据预处理、CNN/MLP/RNN 训练示例，适合核对设备字段和数据形状。
- [DarkSZChao/MMWave-radar-human-tracking-and-fall-detection](https://github.com/DarkSZChao/MMWave-radar-human-tracking-and-fall-detection)：TI 雷达多目标跟踪与跌倒工程框架，包含硬件串口配置；适合作为后续设备接入参考。

采用方式：Ultralytics YOLO Pose 负责公开预训练姿态提取，`pose_fall.py` 负责最小可解释时序规则；`radar.py` 负责设备无关的生命体征频带特征；`FusionNet` 负责统一多模态输入。没有把外部仓库整包复制进来，避免许可证、硬件耦合和数据格式混乱。
