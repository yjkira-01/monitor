# 融合实验与相关论文的结果差距核查

日期：2026-10-13。回答“现在对比相关论文的结果差多少”。以下跨论文差值只作数值参照，未运行官方复现，也不是统一协议排名。

## 当前实验参照

采用已经完成并通过独立重载的可见关节中心表示普通拼接作为稳定参照：正常画面 MPJPE **65.860253 mm**，五条件主体宏平均分数 S **90.216644 mm**。S 包含正常、噪声暗光、遮挡、视频缺失、雷达缺失，不能用它代替论文的普通场景 MPJPE。

本项目为 mRI、17 关节、12 名训练主体、4 名开发验证主体，每人 96 画面；一个主体划分、三个初始化，验证共 384 画面。最终保留主体 [3,10,12,18] 未预测、未评价。当前主误差针对髋中心目标，但评价没有再次强制预测髋归零。RGB 输入来自冻结 RTMO 预测，保留检测失败画面；因此不能将当前开发验证成绩称为正式测试成绩。

## 论文原表与纯数值差值

下表统一换算为毫米；差值定义为“本项目正常画面 65.860253 − 论文报告值”。正值表示本项目误差更高，负值表示本项目误差更低。不同数据、场景、关节及输入处理仍未统一。

| 论文 / 结果行 | 数据与条件 | 论文关节误差 mm | 纯数值差值 mm |
| --- | --- | ---: | ---: |
| ImmFusion，ICRA 2023，Table I Lab1 | mmBody，实验室 1 | 41.00 | +24.86 |
| ImmFusion，Table I Lab2 | mmBody，实验室 2 | 37.00 | +28.86 |
| ImmFusion，Table I Average | mmBody，七场景平均 | 59.00 | +6.86 |
| X-Fi，ICLR 2025，Table 1 I+R | MM-Fi，RGB＋mmWave | 88.80 | −22.94 |
| mRI 原论文，NeurIPS 2022，Table 3 RGB S2 P1 | mRI，跨主体、全部 12 动作；单 RGB | 120.10 ± 3.70 | −54.24（对均值） |

ImmFusion Table I 单位为 cm，每场景第一列 joint error，第二列 vertex error；上表只用 joint mean，不混入顶点误差。其正文另有 5.8/7.6 cm 叙述，与 Table I 的 5.9/7.4 cm 不同，本记录明确以 Table I 为准。正常画面相对两个实验室场景高约 25–29 mm；相对七场景均值高约 6.86 mm，但后者与本项目正常画面场景构成不同。

X-Fi I+R 的 PA-MPJPE 为 57.3 mm，与普通 MPJPE 88.8 mm 是不同指标；不能拿 57.3 mm 与本项目普通误差比较。mRI Table 3 是单模态姿态估计表，不能称 RGB＋雷达融合基线，也不能混入动作检测 Table 4。

## 同任务近邻 R-DMRF-HPE 的来源限制

R-DMRF-HPE（Measurement 2026）采用 mRI 的 RGB＋毫米波雷达，是当前较直接的同任务近邻。已读作者实验室摘要原文为 `MPJPE of 91.82  41.81 mm`，关系符号缺失；mmBody 对应原文为 `62.47  22.39 mm`。未取得论文原始结果表，因此不得自行补成“±”，也不得把 41.81 mm 当作另一方法或改进后误差。

**如果 91.82 mm 确实是所报告的 mRI 均值**，则与本项目 65.860253 mm 的纯数值差为 −25.96 mm，即低 28.27%。此条件性差值不是已确认的统一协议胜出结果，后续必须先核对原表、划分、关节及评价对齐。

RadarConf 2026 雷达＋IMU 交叉注意力近邻的已读官方摘要没有给出 MPJPE 数字，当前无法计算差距；其输入组合也不同。

## 相对各自基线的机制增益

这些百分比对应各论文内部不同基线、数据和评价条件，只用于判断当前方法贡献的证据强度，不构成跨方法统一排行榜。

| 比较 | 原基线 → 方法，mm | 误差下降 |
| --- | --- | ---: |
| ImmFusion vs TokenFusion，Table I 七场景平均 joint | 74.00 → 59.00 | 20.27% |
| X-Fi I+R vs 输出平均融合，Table 1 | 101.00 → 88.80 | 12.08% |
| X-Fi I+R vs 特征融合，Table 1 | 140.60 → 88.80 | 36.84% |
| 本项目效用辅助门控 vs 同表示拼接，五条件 S | 90.216644 → 90.193134 | 0.02606% |

本项目效用辅助相对拼接仅改善 **0.023510 mm**，且比没有该辅助的残差门控 90.188169 mm 略差 **0.004966 mm**；没有稳定贡献证据。可见关节中心表示相对原髋表示的拼接 S 从 91.2683 降至 90.2166 mm，约改善 **1.15%**，仍未达到预定扩大实验门槛。

已有局部信号是：髋缺失但仍有预测关节的 74 张遮挡画面，拼接平均误差改善 21.9134 mm；该子组贡献占整体遮挡改善约 91.83%，但主体 20 在该组恶化 3.2723 mm。最后两组严格置信度/几何控制的整体方向依架构，子组配对仍待收尾，因此不能把局部主表示收益概括成已证实的新融合算法。

## 公平比较仍缺的条件

ImmFusion 为 mmBody 的 22 个 SMPL-X 身体关节与 mesh 任务，官方数据加载使用 GT 关节裁图、GT 三维框裁点及 GT pelvis 点云中心化，和当前完整预测输入链不同。X-Fi 数据是 MM-Fi。mRI 原论文 S2 为 16 训练 / 4 测试主体、三个随机主体划分，评价再次对齐预测和目标根关节，RGB 只评价预训练输出与标注的交集关节。以上差异都能改变报告误差。

正式比较应在同一主体划分、共同关节、相同预测输入和失败处理、统一 MPJPE 对齐规则下重训或适配基线；在开发集冻结方法后再评价保留测试主体。当前既不能声称已超过相关论文，也不能仅根据 ImmFusion 的跨数据差值断定本方法泛化落后相同幅度。

## 来源定位

- ImmFusion：[原文](https://arxiv.org/abs/2210.01346)，[正式 DOI](https://doi.org/10.1109/ICRA48891.2023.10161428)。本地提取 `artifacts/fusion_method_research_20261013_literature/immfusion_pdf_fulltext.txt`，Table I 在 PDF 第 5 页，文本 458–642 行均值部分；官方输入审计见 `research/immfusion_official_feasibility_audit_20261013.md`。
- X-Fi：[ICLR 2025 正式 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/f25602918e8a0d0c86e3c752ecfbbaa1-Paper-Conference.pdf)。本地 `artifacts/research_track_20261012/xfi_final_pdf_fulltext.txt`，Table 1 第 7 页，I+R 行 614–624；基线定义 892–903 行。
- mRI：[NeurIPS 2022 正式页面](https://proceedings.neurips.cc/paper_files/paper/2022/hash/af9c9c6d2da701da5a0acf91ec217815-Abstract-Datasets_and_Benchmarks.html)。本地 `artifacts/research_track_20261012/mri_paper_fulltext.txt`，Table 3 第 9 页，RGB S2 P1 行 735；划分与对齐 669–703 行。
- R-DMRF-HPE：[作者实验室摘要](https://erglab.bjtu.edu.cn/archives/4952)，[DOI](https://doi.org/10.1016/j.measurement.2026.120687)。本地 `artifacts/fusion_method_research_20261013_literature/rdmrf_lab_fulltext.txt` 第 94 行。仅题录和作者摘要已核对，原结果表未读。
- 本项目精确结果：`artifacts/fusion_method_research_20261013_expanded_centroid/run_summary.json` 的 `aggregate.models`；当前实验总记录见 `research/fusion_research_progress_20261013.md`。

本次只核对与记录现有结果，没有新增训练、保留测试预测或生产权重改动。
