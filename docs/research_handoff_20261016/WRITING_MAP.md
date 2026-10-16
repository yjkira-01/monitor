# 论文写作材料定位

| 论文部分 | 主要材料 | 使用边界 |
|---|---|---|
| 引言/相关工作 | research/paper_baseline_selection_20261013.md、fusion_literature_audit_20261013.md、knap_2024_geometric_baseline_card.md | 核验程度逐篇标记；没有新颖性完备证明 |
| 方法 | handoff/论文交接说明、research/geometric_residual_method_20261016.md、ray_gate_models.py、run_ray_oof_action_windows_20261016.py | 以冻结源码为精确定义；旧说明中的小数据结论按当前完整报告更新 |
| 数据/实验设置 | mri_action_windows_resumed_20261016/protocol.json、cache_audit.json、action_window_main_20261016/protocol.json | 一个划分/3初始化；保留集未评价 |
| 主结果/消融 | large_protocol_all_methods.csv、3份正式reporting/results.json | 不混合1536与12288人口，不省略同观测/GRU负结果 |
| 图 | handoff/figures/及旧strict_controls/reporting/figures/ | 大小数据图分别标明人口；误差线是初始化SD |
| 附录/复现 | REPRODUCE.md、MANIFEST.json、逐帧prediction NPZ、audit.json、source_snapshot | CPU指标复算不等于原始推理重建 |
| 限制/后续 | 交接说明第6节、EXCLUSIONS.json | 不声称已完成盲测、第二数据集或临床部署验证 |

小数据与大数据发生结论变化，适合作为进一步研究问题；不应通过改名回避负结果。LaTeX模板是待完善工作稿，作者/单位/目标期刊尚未提供。
