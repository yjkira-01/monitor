# 固定单位门控：新96帧CUDA控制的独立审查

日期：2026-10-13。只读源码审查及CPU核对，不修改模型/runner，不训练或占用GPU。9个正式scaled模型已由root在centroid后按共同预算串行执行，独立核验证据补充在文末。

## 控制是否足够

当前 `ScaledGateFusionModel` 先按语义method调用完整base构造，再用没有随机参数初始化的wrapper包装原gate。固定scale是buffer，不增加trainable参数；顺序为原12维gate证据的姿态6维×10、绝对分歧3维×25、confidence与有效点比例各×1、骨长差×25。它把米尺度信号移到decimetre/40mm单位附近，所有缩放值在这轮前固定，没有新验证grid搜索。

各参数初值可以逐位匹配，但scaled evidence改变了初始gate函数/logits。它是相同容量的单位/优化参数化控制，不是函数级相同初始化。正对角缩放可由第一层相反权重缩放吸收，不增加观测信息或MLP函数族；相同数值初始权重、固定学习率/AdamW和辅助loss系数会改变有效初始化、梯度及正则化路径。不能把后续收益单说成“只是数值更稳定”，也不能说新增融合信息。

三个同scale模型的公共/candidate/gate参数和初始函数相同。scaled utility对scaled joint_gate及scaled random是风险监督最关键的同组控制；跨scale比较只解释该参数化干预。每个scaled方法在GPU runner内先去scaled前缀，继续调用原 `fusion_loss`，因此candidate/direct分支aux、utility BCE及有效域random-label语义均保持。

新9套必须用保持不变的main GPU runner、hip表示、相同1536帧cache、1152/384训练/验证、split7、init7/17/27、四actualpixelconditions、原hip RadarDrop reference、batch64、80epoch最后checkpoint、CUDA IEEE FP32和原optimizer/loss。直接对照为完成的expanded_hip三个同语义模型。旧64帧CPU组不能用于归因该缩放效果。

## 已完成的真实CPU预核查

`artifacts/fusion_scale_preflight_20261013` 中9个语义初始化pair已通过：每对映射后34个state数组逐位相同，video/radar/candidate和valid masks在实际384个clean验证输入上逐位相同，参数数相同，scale固定buffer核对通过。初始gate最大差在.001013至.004720之间，实证说明初gate函数不同。

`audit_fusion_scale_controls.py` 同时核查同scale三个模型的state与video/radar/candidate/gate/fused逐位一致。追加完整同scale核查保存在上述目录的 `verified_groups/`；原预核查源码/receipt保留。

这些只验证初始化与forward逻辑，不能计入9次80epoch训练或性能证据。实际训练后候选和分支轨迹会因fused梯度不同而不同，不要求最终预测逐位一致。

## 正式结果的独立配对分析

正式scaled训练完成后先运行通用独立重载/fallback/补充指标，再运行：

```powershell
E:/miniconda/python.exe research/audit_fusion_scale_controls.py --base artifacts/fusion_method_research_20261013_expanded_hip --scaled artifacts/fusion_method_research_20261013_scaled --output artifacts/fusion_method_research_20261013_scaled/analysis/paired_scale
```

实际目录名由root训练产物确定。分析要求两组已有完整重载receipt；核对source/runner/cache/backend/training/representation、参数数、全部normalization和保存frame/subject/target逐项匹配，再从原保存预测复算主metric。保留每seed/condition及各主体的base−scaled signed gain、初始化差值sample SD和原五条件主体macro score。另直接输出同scale utility对joint/random的主score和条件差。

正gain偏向scaled，负gain偏向base，两者均保留。主metric、checkpoint和历史结果不改。三初始化仍为同一主体划分，四验证人是实际独立单位；不把相关帧、joint或初始化数视作新主体。此追加是在开发反馈下的探索控制；风险辅助若不胜同scale无辅助/随机监督，应停止把它作为机制贡献扩大预算。

## 正式scaled9套与严格配对实际完成

目录 `artifacts/fusion_method_research_20261013_scaled`。9套80epoch完整模型独立CPU重载，293数组最大CPU/CUDA保存预测差.0006103515625mm，819指标复算差0；exact fallback9模型通过（clean382可用/2双缺失），补充髋重中心核对441项原主metric差0。

`analysis/paired_scale/scale_control_audit.json` 核对16个共同contract字段，额外representation/归一来源/factory/params、完整normalization、缓存与source快照、frame/subject/target逐项配对。再复算882项两边原主指标，差0。9个语义初始pair与同scale初始函数检查也通过，保存审计源码copy。

| 语义模型 | 未缩放主S mm | scaled主S mm | base−scaled主S gain mm |
| --- | ---: | ---: | ---: |
| joint_gate | 91.190273 | 90.943151 | +.247122 |
| utility_gate | 91.261750 | 91.418313 | −.156563 |
| utility_random | 91.010937 | 91.528606 | −.517668 |

同scale utility比joint差.475162mm，三个初始化全部更差，七条件均值也全部更差。相对随机监督仅改善.110293mm，2/3seed方向为正；该轻微差不能补偿无辅助控制失败。clean有效gate/AUC utility=.530457/.525798，random=.532059/.523860，仍缺乏可靠个人效用判别。

这轮固定单位优化没有恢复风险辅助的机制证据。joint小幅参数化改善可作为控制结果保留，不能称创新或通过既定投入门槛；不得再按unit选择最有利配置来包装风险贡献。
