# 复现说明

## 0. 环境和目录

推荐Python3.13、NVIDIA CUDA环境。原实验Torch2.11.0+cu128、NumPy2.3.4、RTX4060 Laptop 8GB；精确已安装核心版本见ENVIRONMENT.json。创建虚拟环境后安装requirements-research.txt，PyTorch从https://download.pytorch.org/whl/cu128安装对应版本。Windows建议在可写目录解压并从paper_handoff_20261016根目录运行。

不要执行旧本机流水线launcher来“续跑”：本快照实验均已complete，launcher要求新目录。已有保存权重可直接重审计、复算报告。下面审计可能重写审计receipt/报告；如需保留交接快照，先复制解压目录。

## 1. 不重训的CPU指标核查

    python tools/verify_results.py

它从主预测NPZ独立计算欧氏误差，与交接主表逐方法逐seed核对。它不替代原始图像推理、权重前向重建或主体分离审计。

## 2. 原始数据与像素审计

从Release下载dataset_release.zip放根目录。若只读表与已缓存特征训练，不需要原视频。要重跑完整像素审计，从DATA_SOURCES.json官方源恢复blurred_videos.zip，确认SHA256；运行tools/restore_raw_videos.py只提取开发16人color0，再执行：

    python research/audit_mri_action_windows.py artifacts/mri_action_windows_resumed_20261016

RTMO固定ONNX已经包含在desktop/CareMonitor/Assets/Realtime/rtmo.onnx，来源与许可随包。不要用不同模型的检测结果替换缓存后继续引用旧数字。

## 3. 保存权重完整GPU重建

    python research/audit_action_window_research.py artifacts/action_window_main_20261016 --cache artifacts/mri_action_windows_resumed_20261016
    python research/report_action_window_research.py artifacts/action_window_main_20261016
    python research/audit_ray_oof_action_windows_20261016.py
    python research/report_ray_action_window_research.py
    python research/audit_report_ray_raw_action_window_control_20261016.py

几何审计会检查OOF训练/留出主体、训练内重新拟合距离、专家观测重建、CUDA及NumPy头部重建。上述脚本预设目录必须保持原名称。原始数据ZIP、所有research源码、主基线/几何/raw载荷均来自主复现包。

## 4. 从头训练（会耗时）

    python research/run_action_window_formal_20261016.py --cache artifacts/mri_action_windows_resumed_20261016 --output artifacts/action_window_retrained --folds 0 --seeds 7 17 27 --epochs 80 --temporal-epochs 80

新输出目录的训练可以独立完成；几何原冻结脚本默认读取原action_window_main_20261016且要求输出目录不存在。因此整流程重训应在新的解压副本中将旧结果目录移出、保持同名目录再按原流水线执行，或明确生成另一个runner并保存修改说明/新SHA。不要把新运行的数据填回旧结果冒充完全一致。

旧1536帧实验的入口是run_ray_oof_research.py、run_ray_oof_strict_controls.py、run_ray_raw_observation_control.py、run_ray_raw_width32_control_fixed_20261016.py；独立审计脚本同目录。X-Fi强输入核心：run_xfi_strong_inputs.py及resume源码，官方X-Fi固定代码在artifacts/paper_baseline_sources_20261013/xfi_checkout。历史native X-Fi复现需另外解压history包，包含tokens_v3及完整权重。

## 5. 最终保留集

3/10/12/18没有模型预测或评价。公开原始ZIP本来含20人，但所有主缓存和训练仅16开发人。交接包没有安排最终测试任务。方法和协议冻结后，由接手研究者决定是否运行最终测试，并单独记录。
