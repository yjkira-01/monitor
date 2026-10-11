# 中等规模模型升级与独立训练

用户最新选择是中等规模，取消十亿参数路线。本轮没有下载十亿参数权重，没有更改当前发布的 C# 应用、网页来源、摄像头连接或雷达状态。

## 实际选型

| 链路 | 候选 | 实测参数量 | 当前用途 |
| --- | --- | ---: | --- |
| 视频动作与跌倒 | InternVideo2 蒸馏 Base，K710 → K400 作者权重 | 89,448,202 | 八源十类数据上全参数微调 |
| 人体框选 | RF-DETR Medium，COCO 作者权重 | 33,687,458 | 公共视频验证、独立 ONNX 候选 |

两个模型合计约 1.23 亿参数，分别承担框选和视频动作识别。通用 Kinetics/COCO 指标不能作为本项目跌倒、遮挡或真实暗光精度。

视频源码固定到 OpenGVLab/InternVideo 提交 `3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3`，保留 Apache 2.0 许可。视频模型仓库版本 `449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac`；官方文件 SHA256 为 `aa6ded7709ec41639577aec183f1baeddccda57fb3a7fdf6e76e5ebea2d8996f`。原权重仓库没有单独模型卡，后续对外分发前还需确认权重许可说明。

检测模型版本 `1b5b672408f86dd38e05dd3cf3f2e0834e545a59`，权重 SHA256 为 `e52098adc46969794fbdd16e0548a62b81ba0c0f4b14392676edba50be9a69f6`，模型卡声明 Apache 2.0。检测导出使用工作区隔离目录中的 Transformers 5.19.0，不安装到当前应用环境。

## 视频训练实际配置

复用已审计的八源 v3 清单，复制为独立 `data/manifests/moderate_stage_f_20261009.csv`；SHA256 `d6e99ba317e9762e750cf4f0ed9707d22f3bd43a6cedd1dc604f8457b6171a1f`。5371 个训练窗口来自 3842 个训练视频，验证 862 窗口，测试 871 窗口。总量 7104 是所有划分的窗口数，不是训练视频数。训练和选模不读取测试划分。

类别为 `fall / fallen / lie_down / lying / other / sit_down / sitting / stand_up / standing / walk`。正常卧姿不是睡眠诊断；`fall + fallen` 合并用于风险监督。

- 原生 8 帧、224×224 RGB，完整画面按比例缩放补边，再做 ImageNet 归一化；保持作者预训练的时空位置编码尺寸。
- 全部 8945 万参数参与学习，较早层使用较小学习率；原生 PyTorch SDPA、非重入梯度检查点、BF16 运算与 FP32 参数/优化器。
- 微批次 2，累积 8 次，有效批次 16；不把八个微批次当八个模型。
- AdamW、基础学习率 2e-5、权重衰减 0.05、warmup/cosine、梯度裁剪；最多 120 epoch，25 轮未改善早停。
- 保留光照、噪声、画面截断、遮挡、低分辨率、模糊与正常卧姿训练。变化在整段视频上保持一致。
- 数据集/类别权重采用平方根修正，任何训练窗口的采样概率最多 5/N；这是期望重复次数的上限，不是一次随机采样的硬次数上限，也不等于每个演员/父片段最多重复五次。
- 每轮验证正常画面与七种合成困难条件，逐类别、逐来源记录指标；合成暗光不冒充真实夜间。

参照是同一清单上冻结的 R3D 开发结果，启动时选择其第 61 轮作为参照。新模型不能以随机初始化分类头作为防退化基线。沿用逐来源与困难条件的回归门槛，并要求合并风险召回至少 0.95、精确率至少 0.90、正常 `lying` 召回至少 0.80。这些是开发门槛，最终仍需独立测试与现场验证，稀有动作样本也不能据此宣称可靠。

## 已验证的运行证据

云端 RTX 3080 Ti 12GB，独立目录：

```text
/root/autodl-fs/yanglao-monitor/artifacts/activity_stage_f_internvideo2_base_20261009
```

原 v3 任务及其冻结快照不变。新任务 driver PID `111846`、正式训练 PID `111948`，通过 nohup 在后台运行。首次打包漏带包初始化所需的模块，未进入 GPU 检查；错误日志保留，重试完整快照为 `source_retry_setup`。当前正式任务使用这份固定源码，不在运行中替换。

真实训练视频的一次检查使用一段跌倒和一段正常卧姿，全部参数梯度有限且发生实际权重更新。峰值张量显存 **1427.45 MiB**，缓存峰值 **1586 MiB**；两段视频一批的前向耗时 **101.68 / 92.89 / 97.61 ms**，反向及优化一步 **992.83 ms**。这些测量与原 v3 任务并发，不能直接换算成桌面 App 的实时 FPS。检查使用随机新分类头，不是准确率评估。

正式训练已开始，第 1 轮已观察到 `batch=1900/2686`，日志显示实际前向、反向与优化；尚无完整 epoch 的验证精度。每轮保存模型、优化器、调度器、随机数状态、最佳候选与 last 断点。初始 epoch 中途断电仍会损失该轮未落盘的进度。

2026-10-09 18:25 更新：已完成第 1 轮并保存 best / last 断点，正式训练进入第 2 轮，最近记录 `batch=700/2686`。首轮验证准确率 0.4443155、macro-F1 0.3157193、风险召回 0.0042553、风险精确率 1.0，开发验收门槛未通过；这是初始完整轮次结果，不是最终成绩，不能据此宣称新模型泛化改善。实时进度条、十类召回与困难条件指标见 [云端监控说明](cloud_training_monitor.md)。

证据文件：`artifacts/activity_stage_f_internvideo2_base_20261009/gpu_check.json`、同目录 `status.json`、`r3d_reference.json`；源码哈希记录在 `source_retry_setup/artifacts/moderate_source_sha256.json`。本机根目录测试 81 项、后端 64 项，共 **145 项通过**。

## 接入边界

检测候选的独立 ONNX 已实际导出并通过人体框一致性检查：`artifacts/pretrained/people_rf_detr_medium.onnx`，122,138,444 字节，SHA256 `8e33f2d32592918b2274a12209a80adcdebaab6b21fbc7f9eca168390dc6a72b`。单个 GMD Subject4 Fall01 公开片段取三个时间点，每个分别测试原图、合成暗光及上部裁剪，九例都检出一人。没有人工框标注，不把这九例换算成检测准确率，也没有完成真实多人评测。

另用公开原图在 C# / ONNX Runtime DirectML 1.20.1 中完成 CPU 与 GPU 数值回归，实际执行了 DmlExecutionProvider 内核。两次预热后十次固定输入：CPU 平均 296.54ms、p95 310.89ms；DirectML 默认适配器平均 31.40ms、p95 37.52ms，加载 1596.84ms。最终工具复跑同样 exit 0，CPU 平均 298.69ms、p95 308.47ms，DirectML 平均 29.56ms、p95 32.99ms，完整记录在 `native_result_final.json`。GPU 与参考人体框误差最大 0.0000153（320 坐标尺度），分数误差约 5.96e-8。没有确认适配器具体型号；这是单帧推理基准，不是整机摄像头或 UI FPS。CPU 上该候选较重，需要保留现有快速通道。

验证工具为 `desktop/ModelBench`；报告 `artifacts/detector_benchmark/native_result.json`。`public_input.bin` 只含公开数据集输入，未保存摄像头或私人视频。第一次检测导出因 bicubic 抗锯齿失败，第二次 TopK 形状失败，日志保留；失败 TopK 文件单独命名 `.failed_topk.onnx`，不用于客户端。

新视频模型的输入是 8×224×224，当前 C# 模型是 16×128×171，特征和解释图接口也不同。不能改文件名后直接替换旧 ONNX。训练导出权重可由 `scripts/predict_video.py` 做离线回归；正式客户端接入需同步采集窗口、预处理、十类输出与解释接口，完成数值一致性和性能验证。

RF-DETR 候选输入固定为 1×3×576×576 RGB 0..1，归一化放在 ONNX 内。输出保留 `boxes / scores / labels`，person 类别为 1，框坐标使用原生客户端已有的 320 坐标尺度；客户端采用该候选时仍需调整输入大小。固定原生输入的位置编码不需插值，可避免导出器不支持抗锯齿 bicubic 操作，不能据此宣称支持动态输入尺寸。

新候选不会自动替换发布版本。正式训练完成后比较独立测试、已知漏检视频、正常卧姿、单人重复框、多人遮挡和真实暗光，再决定是否部署。雷达仍只用于独立数据采集与功能，未加入视频主模型训练。

参考：[InternVideo2 官方模型表](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/single_modality/MODEL_ZOO.md)、[作者蒸馏权重](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models)、[RF-DETR Medium](https://huggingface.co/Roboflow/rf-detr-medium)。
