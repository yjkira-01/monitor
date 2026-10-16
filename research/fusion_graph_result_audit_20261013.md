# 骨架邻接控制结果与独立绘图审计

数据来源：[graph run summary](../artifacts/fusion_method_research_20261013_graph/run_summary.json)。这是完成的 3 方法 × 3 初始化种子探索实验；训练主体和验证主体固定，训练 768 帧、验证 256 帧，每主体 64 帧，CPU，80 epoch 的 last checkpoint。三个模型均为 107,770 参数，只替换同一个图残差架构的固定邻接。最终测试主体未使用。

| 固定邻接 | 五条件宏平均 S / mm | 初始化 seed SD / mm | 相对 self 配对收益 / mm | 新派生探索性主体 bootstrap 95% 区间 / mm |
|---|---:|---:|---:|---:|
| 自连接 graph_self | 79.8103 | 0.5786 | 0 | [0, 0] |
| 解剖 graph_anatomical | 80.0818 | 0.5682 | −0.2715 | [−0.8358, 0.3499] |
| 固定置换 graph_permuted | 79.7783 | 0.4470 | +0.0320 | [−0.2489, 0.2551] |

S 的条件为 clean、dark、video_missing、radar_missing、radar_sparse；这些是旧 64 帧协议的五条件，不能与新版 noisy_dark、occlusion 主分数直接比较。解剖邻接在三个初始化下均未胜过自连接，固定置换也未显示足够稳定的提升。本局部实验不支持“骨架邻接的解剖关系带来收益”的假设；它没有检验所有图模型，也不构成对一般图融合方法的否定。

比较参考是 **graph_self**，并非旧 concat 或旧 joint_gate。三个图头比旧 joint_gate 头更深且多 4,160 参数；不同架构阶段的误差差不能归因为邻接结构。这里保留原阶段边界，不合并新 96 帧 GPU 实验。

原 summary 的 `comparisons_to_concat` 是空字典。绘图没有把空字典当作现成统计，而是从保存的每主体、每条件、跨种子均值派生 self-minus-method 收益。先等权平均五条件和三个初始化，再对验证主体 5、6、19、20 以放回方式重采样，NumPy `default_rng(20261013)`，10,000 次，取 2.5/97.5 百分位。种子与条件均不重采样。四名验证主体、一个固定划分的区间只作探索展示，不作显著性证据。派生统计记录于独立 CSV 和 receipt，原 summary 的 SHA 未变化。

绘图程序 [plot_fusion_results.py](plot_fusion_results.py) 已支持 `--reference-method`；graph 自动用 graph_self，scaled 自动用 scaled_joint_gate，普通融合默认 concat。普通 concat 已有 bootstrap 时复制并验证保存区间；非 concat 参考的新派生方法和随机种子写入 receipt。逐条件总体 mean/SD 和逐主体 seed mean 均与保存 run 指标核对。

产物位于 [graph figures](../artifacts/fusion_method_research_20261013_graph/figures)：两个独立图，各有 PNG（300 dpi）、PDF、可编辑 SVG；四份来源 CSV、图注、绘图 SHA receipt 和 [视觉核对记录](../artifacts/fusion_method_research_20261013_graph/figures/fig_graph_control_visual_verification.json)。两张 PNG 已实际查看，图例、条件标签、纵轴与注释可读，无截断重叠；PNG 的实际 DPI 为 299.9994，SVG 保留文本元素。旧 round1 的 concat 提取与既有区间也已复核，未重算模型预测。

下一步保留该负结果，等待相同 GPU 预算的 hip、visible_centroid 和局部 confidence_only／centroid_fallback 分支完成，再分开解释表示、可用性和融合机制。任何局部几何控制图都应使用真实 metadata 的标签，不能套成纯 centroid 比较。
