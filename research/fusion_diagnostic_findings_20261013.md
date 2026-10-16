# 第一轮融合开发实验：独立重载与逐关节诊断记忆

日期：2026-10-13。此记录便于后续优先续接最近的实际实验与方法判断，不替代最终测试。训练由主任务负责；本记录对应独立分析程序 `analyze_fusion_trials.py` 的已完成运行。

## 已完成证据

实验目录：`artifacts/fusion_method_research_20261013_round1`。13方法×初始化7/17/27共39套模型，80epoch；旧64帧/主体缓存，训练768帧、验证256帧。主体划分仍为12人训练、4人开发验证，三个初始化不是三种划分。第一轮实际条件是clean、像素×0.18 dark、video_missing、radar_missing、radar_sparse；此轮没有新的noisy_dark/occlusion条件。

独立分析安全NPZ strict reload，检查权重keys/shape/dtype/finite、来源SHA、缓存SHA及训练内normalization精确重算。重建全部condition输入后，851项数组检查与2445项指标检查**全部一致，最大差值均为0**。各model的fused/video/radar/gate保存数组及MPJPE、P95、逐关节和逐主体数值已核对；通过全部检查后才计算gate可靠性与GT Oracle。

- 核对凭据：`artifacts/fusion_method_research_20261013_round1/analysis/verification.json`
- 完整诊断：`artifacts/fusion_method_research_20261013_round1/analysis/diagnostics.json`
- 独立聚焦检查：`test_fusion_diagnostics.py`，9项通过；涵盖partial Oracle、零残差/端点限制、inactive回退、AUC并列/退化类别、safe strict NPZ、错误prediction阻止诊断、固定graph factory/adjacency声明核对。

源核心代码有匹配原run SHA的source_snapshot；原trial没有固定所有import依赖的SHA，分析receipt记录当前依赖版本，并以所有预测行为精确匹配补充核查。这项限制仍应披露。

## 关键实际结果

下表均为三个初始化均值。clean MPJPE用全部256帧；gain/AUC/harm等细粒度诊断只在both可用帧joint上计算。clean有255帧、dark有253帧同时可用，失败帧保留在整体指标。S为第一轮五条件主体宏平均，不可和第二轮不同条件S直接比较。

| 方法 | clean MPJPE mm | 第一轮S mm | 有效gate均值 | fullcandidate beneficial AUC | 实际有害joint比例 | 连续segment Oracle相对实际可恢复mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| concat | 65.875 | 79.212 | — | — | 44.12% | — |
| joint_gate，无效用BCE | 67.224 | 79.691 | 0.970 | 0.530 | 47.18% | 3.771 |
| utility_gate | 68.013 | 80.399 | 0.530 | 0.523 | 46.83% | 4.427 |
| utility_random，有效域内随机标签 | 67.772 | 80.141 | 0.529 | 0.529 | 45.83% | 4.295 |
| utility_global | 67.915 | 80.267 | 0.530 | 0.526 | 47.26% | 4.350 |
| utility_nobone | 67.940 | 80.371 | 0.529 | 0.517 | 45.89% | 4.401 |
| joint_attention，隐式g=1 | 67.570 | 80.191 | 1.000 | 0.500 | 47.16% | 3.601 |
| quality_gate，直接单路姿态逐关节门控 | 69.838 | 81.797 | 0.055 | 0.591 | 35.50% | 13.363 |
| precision，每关节各向同性辅助方差 | 68.773 | 81.375 | 0.167 | 0.605 | 38.07% | 13.232 |

utility_gate在g>0.5位置的fullcandidate实际改善precision约0.500。其未门控完整candidate相对video平均收益仅0.298mm，实际gate后收益0.805mm；这些相对它自身的video分支，不是相对独立RGB模型。

当前utility监督没有优于同结构joint_gate或随机标签对照，不能按“风险预测”包装算法收益。BCE组把平均gate从无辅助的约0.97拉回约0.53，但此轮AUC只在约0.52附近；这些只是4名开发主体、相关frame/joint上的描述性判别诊断，没有显著性推断或校准概率声明。

## 方法上的重要诊断

Oracle为每joint固定video→candidate残差线段上的GT投影：`g*=clip(dot(y-v,delta)/||delta||²,0,1)`；零残差令g*=0，所有inactive/missing帧保持真实fallback，不参与gate可靠性。它只是一条固定残差线段内的最优点，不是全姿态空间最优预测，也不是可部署模型。global gate模型在该Oracle下放宽为每joint独立g，因此还包含解除scalar约束的上限。

utility_gate约6.82%的有效joint中，完整candidate不改善video，但部分残差的Oracle能够改善。quality_gate/precision直接video→radar线段对应比例约45.29%/47.57%。因此，用完整candidate的binary beneficial sign来判定整个残差方向有无价值会遗漏部分校正；低AUC或fullcandidate有害不等于所有partial corrections无益。

可恢复4.43mm上限不说明可观测输入足以预测这些GT权重。若以后试连续Oracle权重作训练辅助，需注意它与既有融合MSE对g的监督目标高度重合，可能仅改变梯度权重；不能因换连续标签就自动宣称新增信息或原创性。仍需同结构/同损失预算控制及开发泛化证据。本次没有启动OOF teacher或新训练。

## 续接操作

新实验报告必须status complete后运行：

```powershell
python research/analyze_fusion_trials.py artifacts/fusion_method_research_20261013_round2
```

默认输出对应实验的`analysis/`，已有目录拒绝覆盖；如需另版本可显式`--output`新目录。脚本支持`model_factory=fusion_graph_models.GraphFusionModel`，要求graph源文件在report的source_sha中固定，加载匹配快照，并核对保存固定adjacency与声明control一致。graph_self/anatomical/permuted是同结构控制；不能只与旧少一层的joint_gate比较来归因骨架连接收益。

后续应优先读取本记录及analysis凭据，然后看新96帧/noisy_dark/occlusion轮与graph三控制的实际完成状态，不重复启动或覆盖旧训练。最终保留主体未使用；仍需扩大开发证据、冻结方案后独立最终测试。

## 后续graph同结构控制及补充指标（实际完成）

`fusion_method_research_20261013_graph/analysis` 的9套安全重载、221数组/585指标全部匹配；exact fallback也全过（clean255可用、1双缺失）。self/anatomical/permuted的clean MPJPE为67.253/67.804/67.212mm，五条件S79.810/80.082/79.778mm；正常邻接未胜self或固定置换。clean active gate均值.9676/.9693/.9676，AUC.5266/.5266/.5201，实际有害joint比例49.17%/49.68%/47.76%。门基本常开，不能把该结构解释为可靠关节选择或骨架连接贡献。

独立 `audit_fusion_hip_recenter.py` 已基于完整重载receipt处理round1的39套及graph的9套保存预测，并精确复核原主指标1365/315项，差0。各 `analysis/prediction_hip_recentered_metrics.json` 存储每seed/condition的MPJPE/P95/逐关节/逐主体、预测髋偏移与signed变化。它是补充指标，没有重写原prediction/summary/选择规则。

round1 clean重中心后的变化（recentered−primary）：concat −.008688mm，joint_gate +.226805mm，utility_gate +.172559mm，random +.252914mm；graph self/anatomical/permuted为+.264305/+.140398/+.399756mm。重中心可能变差，正负数均保留；首轮效用监督及graph机制的负结论不变。

## expanded hip正式阶段独立验证（实际完成）

`fusion_method_research_20261013_expanded_hip/analysis` 已独立重载33套×80epoch。935数组检查最大CPU重建CUDA保存预测差.0006103515625mm（事前FP32跨backend界限.01mm），2877保存数组指标精确复算差0；所有exact fallback通过，clean382视频可用/2双缺失。补充髋重中心也已核对1617项原metric差0。最终测试主体未评价。

相同96帧缓存与五条件主S（clean/noisy_dark/occlusion/video_missing/radar_missing）的concat、joint_gate、utility、random分别为clean65.840748/65.989402/65.847422/65.710431mm与S91.268277/91.190273/91.261750/91.010937mm。utility尚未胜同结构无辅助或随机对照，不能按极小concat差包装有效风险贡献。

utility clean/noisy_dark/occlusion的有效gate均值.530583/.534019/.551225；描述性fullcandidate AUC.519358/.509268/.494327；实际有害joint比例45.08%/46.68%/47.81%。连续GTsegment相对实际全部帧可恢复4.675155/5.091917/.437894mm。遮挡中只有21/384帧both有效，该整体Oracle被可用人口稀释；它没有测试另74个原hip缺失但存在raw局部关节帧的几何价值，那是后续partial两模式严格比较的任务。

## expanded visible-centroid正式独立验证（实际完成）

`fusion_method_research_20261013_expanded_centroid/analysis` 已独立重载33套×80epoch。935数组最大CPU重建CUDA预测差.000732421875mm，2877保存指标复算差0；exact fallback33模型通过，clean382视频可用/2双缺失。补充髋重中心1617原主metric核对差0，没有改写主预测或选择规则。

concat/joint_gate/utility/random的clean为65.860253/65.755650/65.868066/66.587031mm，五条件主S90.216644/90.188169/90.193134/90.467962mm。utility虽比random改善约.275mm，但未胜无辅助joint，且相对concat仅.0235mm，均未达到预先投入门槛。表示收益要与融合监督机制分开解释。

utility的clean/noisy/occ activegate均值.540271/.543593/.573863，fullcandidate AUC.515959/.517674/.521120，实际有害joint比例45.16%/45.94%/51.64%；occ中g>.5的beneficial precision45.89%，actual相对自身video分支条件收益仅.136540mm。正常门控joint在同条件相对自己的video为−.417020mm，random为+.216801mm。这些内部video分支本身不同，不能把条件收益差直接当跨方法精度差。

GTsegment整体可恢复clean5.096824/noisy5.437111/occ1.671908mm。centroid的occ both人口为95帧，hip为21帧，conditional统计不能直接跨表示比较；新增信息、锚点、normalization、监督人口和耦合训练变化应由后续paired与partial两控制进一步分离。
