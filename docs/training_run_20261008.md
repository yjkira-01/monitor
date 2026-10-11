# 视频长训实验报告（2026-10-08）

已经在云端 RTX 3080 Ti 完成 TinyVideoNet 50 epoch 和 R3D-18 100 epoch，并核对完整日志、最后断点与验证预测。R3D-18 的最佳验证准确率为 94.6%，相同划分下 Tiny 为 64.9%。这是 37 个受试者留出视频的验证结果，Subject 4 已参与模型选择，不能作为独立测试或真实养老环境效果。

## 数据和环境

- GMDCSA24：160 个 MP4，共 1,109,791,938 字节。
- Train：Subject 1–3，123 视频（61 ADL / 62 Fall）。
- Validation：Subject 4，37 视频（20 ADL / 17 Fall）。
- 本地与云端 160 个文件的大小和 SHA-256 全匹配；全帧解码 34,172 帧，无失败。训练代码同时检查受试者与视频路径不交叉。
- GPU：NVIDIA GeForce RTX 3080 Ti，12GB；Python 3.12.3，torch 2.5.1+cu124，torchvision 0.20.1+cu124，CUDA 12.4，OpenCV 5.0.0，NumPy 2.1.3，SciPy 1.18.1。
- 原始数据、训练代码、源码快照、日志和完整断点保存于云端 `/root/autodl-fs/yanglao-monitor`。
- 毫米波数据没有进入训练；生命体征、睡眠和设备校准属于后续独立数据采集。

## 实际训练配置

TinyVideoNet：50 epochs，batch 8，8 帧，96×96，AdamW，AMP，余弦退火。最初逐帧 seek 导致 CPU 解码瓶颈；保留慢速试验产物，再改为顺序采样完成 50 轮。

R3D-18：官方 Kinetics-400 预训练，固定 stem/layer1–3 和全部 BatchNorm 统计量，训练 layer4/fc。100 epochs，batch 4，16 帧，官方预处理 resize 128×171 / crop 112×112，AdamW，layer4 学习率 1e-4、分类头 1e-3，weight decay 1e-4，5 epoch warmup + cosine，AMP。横翻和亮度增强在同一片段所有帧保持一致。使用 uint8 片段缓存（约 160MiB），FFmpeg 单线程、4 个加载 worker，OMP/MKL 线程均 4。

4 个真实片段的 GPU 前向/反向检查：loss 0.751343，参数梯度有限，张量峰值显存约 545MiB。检查修复了 torchvision 0.20 的非连续批量张量 view 问题。正式运行时 GPU 驱动报告总占用约 1.1GB，计算每轮约 0.7 秒，其余时间用于准备数据及写入完整断点。

主模型按验证 macro-F1 选择；高召回模型按 fall recall、macro-F1、accuracy 依次选择。指标并列时保留较早 epoch。最后断点保存 optimizer、scheduler、AMP scaler、Python/Torch/CUDA 随机状态和清单 SHA-256，使用临时文件替换保证原子保存。恢复仅接受相同训练配置和数据清单；已有输出不会被新任务无意覆盖。

## 验证结果

| 模型/选择方式 | Accuracy | Macro-F1 | Fall F1 | Fall precision | Fall recall | ADL recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny，50 轮最佳 | 0.6486 | 0.6210 | 0.7234 | 0.5667 | 1.0000 | 0.3500 |
| R3D-18，100 轮中的 macro-F1 最佳（epoch 4） | 0.9459 | 0.9449 | 0.9375 | 1.0000 | 0.8824 | 1.0000 |
| R3D-18，高召回选择（epoch 7） | 0.8108 | 0.8086 | 0.8293 | 0.7083 | 1.0000 | 0.6500 |

混淆矩阵行表示真值、列表示预测，类别顺序为 ADL / Fall：

| 模型 | 真 ADL → ADL | 真 ADL → Fall（误报） | 真 Fall → ADL（漏检） | 真 Fall → Fall |
| --- | ---: | ---: | ---: | ---: |
| Tiny 最佳 | 7 | 13 | 0 | 17 |
| R3D macro-F1 最佳 | 20 | 0 | 2 | 15 |
| R3D 高召回 | 13 | 7 | 0 | 17 |

最佳验证值首次出现在第 4 epoch，后续稳定，因此保留该轮模型，完整 100 轮日志仍全部保存。训练集准确率很早达到 100%，说明当前小数据已经容易被拟合；进一步优化应优先扩展场景和受试者。

## 漏检分析和无摄像头测试

最佳 R3D 模型漏检：

- `Subject 4/Fall/08.mp4`：fall 模型分数 0.337756。
- `Subject 4/Fall/09.mp4`：fall 模型分数 0.025020。

两段的官方描述均为 “Falling backward on the bed from sitting on the bed”。下一阶段应增加坐姿向后跌倒、正常主动躺下、卧床翻身等成对样本，检查事件时间标签与关键点下降速度；不能直接把所有卧床状态标为跌倒，也不能用此次验证集反复调整阈值后声称外部泛化提升。

云端离线推理测试：`Subject 4/Fall/01.mp4` 返回 Fall（分数 0.584257），`Subject 4/ADL/01.mp4` 返回 ADL（fall 分数 0.000136）。这两段属于同一验证集，只检查推理权重和训练预处理的一致性。

推理入口 `scripts/predict_video.py` 对整段视频均匀采样并分类，结果声明 `whole_video`；模型分数未校准，暂不定位发生时间，不生成实时告警。页面已接入同一 R3D 权重的异步滑窗推理，视频和判断持续更新，具体见 [视频工作台验收](video_workbench_validation_20261008.md)。该适配没有新增事件级标注或修改整段验证指标。

## 产物和复现

本地精简推理权重为 `artifacts/gmdcsa24_r3d18_100ep_inference.pt`；完整最佳、高召回和 last 断点（各约 317MiB）保存在云端同一目录。回传包 `artifacts/cloud_training_results_20261008.tar.gz` 包含推理权重、Tiny 最佳权重、完整日志/指标、逐视频预测、数据校验、运行环境、源码快照和 SHA-256。

关键文件：

- `artifacts/gmdcsa24_r3d18_100ep.json`：100 个 epoch 的完整指标。
- `artifacts/gmdcsa24_r3d18_100ep_evaluation.json`：37 段验证视频的预测及标签。
- `artifacts/gmdcsa24_upload_validation.json`：160 文件的哈希/全帧解码结果。
- `artifacts/training_comparison.json`：三个模型选择方式的指标。
- `artifacts/training_environment.json`：实际训练环境。
- `artifacts/r3d18_long100_source.sha256`：训练源码摘要。
- `artifacts/gmdcsa24_r3d18_100ep_inference.sha256`：推理权重摘要。

```powershell
cd D:\dachaung
python scripts/predict_video.py "data/raw/GMDCSA24/Subject 4/Fall/01.mp4" --checkpoint artifacts/gmdcsa24_r3d18_100ep_inference.pt
```

云端训练命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/root/autodl-fs/yanglao-monitor/artifacts/torch_cache python scripts/train_video.py --manifest data/manifests/gmdcsa24.csv --model r3d18 --epochs 100 --batch-size 4 --clip-len 16 --size 112 --lr 0.0001 --weight-decay 0.0001 --workers 4 --cache-clips --amp --output artifacts/gmdcsa24_r3d18_100ep.pt
```

若在完成前中断，追加 `--resume artifacts/gmdcsa24_r3d18_100ep_last.pt` 恢复同一任务；已完成 100 轮时不追加训练。两次 SSH 断开后复连确认后台继续运行，最终进程正常结束。

本地全部测试：35 passed，覆盖跌倒指标退化、短视频重复采样/RGB、整片增强、旧 torchvision 连续张量约束、真实中断恢复、推理权重加载以及既有后端功能。

尚未完成事件级每小时误报测试、跨数据集外部测试、姿态 TCN/ST-GCN 对照和真实毫米波校准。四位受试者的小样本、两例床上后仰漏检和整段视频标签，限制了当前结果的部署范围。

