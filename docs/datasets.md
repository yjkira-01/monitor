# 数据集清单

## 已下载到本机

### GMDCSA24

- 来源：[GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos)
- DOI：[10.5281/zenodo.12921216](https://doi.org/10.5281/zenodo.12921216)
- 许可证：上游仓库 `MIT`
- 本地目录：`data/raw/GMDCSA24`
- 规模：4 个受试者、160 个 MP4（79 个 Fall、81 个 ADL），约 1.1 GB
- 标签：每个 Subject 下的 `Fall.csv` 和 `ADL.csv` 含视频时长、动作描述和片段标签；ADL 里包含 walking、sitting、sleeping、reading 等日常状态。
- 当前划分：Subject 1–3 作为训练，Subject 4 作为验证，按人划分以避免同一人的视频泄漏到验证集。

这个数据集适合先验证视频读取、姿态特征和跌倒二分类。它只有 4 名受试者，不能把验证结果当成泛化结论。

### UP-Fall 3D skeleton 子集

- 来源：[Zenodo 12773013](https://doi.org/10.5281/zenodo.12773013)
- 本地目录：`data/raw/UP-Fall-3D-skeletons`
- 规模：5 个受试者、82 个 CSV 序列；当前统计为 2,462 个非冲击帧和 5,438 个冲击帧。
- 标签：每一帧有 33 个关节的三维坐标和 `LABEL`（0=非冲击、1=冲击）。

它不是原始视频，适合先做轻量的骨架时序基线和冲击点检测，之后再把同样的时序接口接到 MediaPipe/YOLO Pose 输出。

两个本地清单由以下命令生成：

```bash
python scripts/index_datasets.py
```

输出在 `data/manifests/gmdcsa24.csv` 和 `data/manifests/upfall_3d_skeletons.csv`。

### URFD 官方 RGB 外部审计集（2026-10-08）

- 来源：[UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)
- 许可：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，非商业学术使用，衍生内容需遵守署名与相同方式共享要求。
- 已下载 camera 0 的 70 个原始 RGB ZIP：30 Fall、40 ADL，11,936 帧，约 4.49 GB；未猜测序列 FPS。
- 路径：`data/raw/URFD`；清单：`data/manifests/urfd_external.csv`，记录 URL、SHA256、类别和镜头，受试者记为 unknown。
- 本轮仅用于固定权重和 0.5 阈值的外部审计，未混入训练，完整结果 `artifacts/urfd_external_audit.json`。
- 引用：Kwolek B., Kepski M., *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014, 489–501。

查看外部结果后继续优化时，URFD 只能作为已查看的开发基准；最终报告另用未查看的独立测试集。

### GMDCSA24 时序窗口准备

`python scripts/index_video_windows.py` 生成 1,561 个粗标签窗口，来自 157 个视频、4 名受试者；3 个视频因格式/时间段问题排除等待复核。数量增加不代表场景扩充。来源记录为 `author_coarse_interval_unreviewed`，本轮尚未据此训练；详情见 [泛化审计](video_generalization_audit_20261008.md)。

## 适合毫米波+视频双流的外部数据

### MM-Fi

- 项目与工具箱：[ybhbingo/MMFi_dataset](https://github.com/ybhbingo/MMFi_dataset)
- 数据下载：项目 README 提供 [Google Drive](https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz?usp=sharing)、[百度网盘](https://pan.baidu.com/s/1IU9okQzdeCIaF7xCr1X_pw?pwd=t316) 和匿名 RGB [阿里云](https://mmfi-dataset.oss-ap-southeast-1.aliyuncs.com/anonymized_rgb_images/all_images.zip)。
- 规模：40 人、4 个环境、27 类日常/康复动作、超过 32 万同步帧；包括 mmWave、WiFi CSI、深度、激光雷达、姿态和匿名 RGB 图像。
- 用法：适合做毫米波与视觉的时间同步、跨模态表征和动作预训练；它没有专门的跌倒类别，不能直接当作跌倒标签集。

### mmFall

- 代码：[radar-lab/mmfall](https://github.com/radar-lab/mmfall)
- 方法：只用 ADL 训练 HVRAE，再用异常分数尖峰和人体重心高度下降确认跌倒。
- 用法：适合作为毫米波分支的异常检测基线。仓库中的点云文件走 Git LFS，当前没有把它当作本地训练集依赖。

### 同步毫米波跌倒数据（受限）

- [Zenodo 17512851](https://doi.org/10.5281/zenodo.17512851)：40 次模拟跌倒（绊倒/晕厥，快/慢），毫米波与 markerless motion capture 同步；记录当前为 restricted，需要申请访问。

## 训练顺序

最新扩大训练使用 GMDCSA24 + CAUCAFall + 下载中的 UP-Fall 作者视频动作片段，新增源及固定受试者 train/val/test 划分见 [本轮架构与数据记录](architecture_training_v2_20261008.md)。不将 UP-Fall CSV 的冲击标签直接当作 RGB 跌倒标签，也不混入已查看的 URFD 开发基准。

1. 用 GMDCSA24 跑视频流和跌倒二分类，按 Subject 做 train/validation。
2. 用 UP-Fall 骨架 CSV 验证轻量时序模型和冲击点标签。
3. 设备到位后，用实际 IWR6843AOP 数据替换毫米波输入；MM-Fi 只作为跨模态预训练/接口参考，不把不同传感器的标签强行拼接。
