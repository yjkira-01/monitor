# 新96帧融合对照：公平基线与下一假设审计

日期：2026-10-13。只读审查第二轮协议、现有模型/GPU runner、第一轮独立诊断和近邻文献。本报告不训练、不改现有模型或runner，不把准备中的新结果称为已完成。

## 1. 当前11方法足以回答什么

第二轮在同一96帧/主体像素缓存、同一GPU后端和初始化7/17/27上重训11方法，解决了旧64帧CPU与新96帧之间不能直接公平比较的问题。共同主体划分、真实像素条件、同批次与增强日程、相同80轮末轮报告，支持**当前架构与训练配置的整体性能比较**。utility_gate、joint_gate和utility_random的同结构、同直接分支/candidate损失及同初始化对照，尤其适合检验辅助标签是否有用。

共同pose MSE＋0.25 direct losses不是“公平失效”。不同输出组合与附加candidate/NLL使跨方法比较包含目标差异，因此只有更窄的“纯结构、纯注意力、纯不确定性机制”归因需要补消融。不能把这一限制扩大成所有架构比较均无效。

参数量只读实际核算：RGB46,259；容量RGB111,077；Radar25,395；concat111,129；late average71,654；quality gate72,103；precision81,032；joint attention103,161；joint gate/utility gate/random各103,610。容量RGB接近concat，**它没有匹配所有方法的参数和计算成本**；表中记录各自开销即可，不必强迫所有结构完全相同。

## 2. 最需要补齐的公平基线要素

| 缺口 | 影响判断 | 最小补充及当前边界 |
| --- | --- | --- |
| 当前late_average只有0.5/0.5，且使用本模型共同训练的内部单路 | 第一轮雷达明显弱，等权易差；“胜过平均”不能证明自适应必要性。它也不等同冻结独立单路的后融合 | 后续加训练主体内留出/OOF确定的固定全局权重，以及冻结独立RGB/Radar后融合；本轮不训练或用验证真值挑部署权重 |
| attention家族多0.25 candidate MSE，concat没有同样第三预测目标 | attention-vs-concat差距同时含结构与多头训练；方法整体比较仍有效 | 用同实际完整candidate的g=1、常量g、仅pose gate、同candidate auxiliary预算；纯机制消融必须保持其他目标一致 |
| joint_attention没有gate，建模顺序导致attention层初值与joint_gate不同 | 它是结构近似基线，不能称“同一个固定候选只把g置1” | 从同完整模型冻结actual候选，改变同一路径的g；或者模块独立初始化。当前不改模型，不重训 |
| quality_gate实际为逐关节video/radar姿态＋置信、点数、骨长分歧gate | 它比仅原始质量的帧门控强，也与残差gate的混合对象不同 | 图文准确称“逐关节质量/分歧门控”；若结果支持残差utility再加同实际candidate上的quality-only gate |
| precision是每关节各向同性辅助variance拟合 | 不能代表完整联合NLL、相关误差Gaussian或MoNIG官方实现 | 无需否定此启发式基线；若论文写不确定性贡献，则补同均值分支的NLL/固定方差与校准/误差相关诊断 |
| 所有视频模型先经冻结RTMO的51维骨架信息瓶颈 | 当前结论只适用于预测关键点＋点集的小网络，不能推导端到端视频融合的优势 | 正式阶段优先同任务ImmFusion或X-Fi适配；读取原图token的模型必须单列输入/预训练/计算变化 |
| 单主体划分、四名开发验证主体、已有多轮结果反馈 | 初始化种子不增加独立主体；96帧仍包含旧64帧 | 按主体/记录配对描述、原负结果保留；方法收敛后冻结协议再做最终测试和第二数据集 |

固定晚权重的标准做法是用训练主体交叉预测的独立单路结果构造$d=r-v$，最小化OOF总平方误差得到全局$alpha=\operatorname{clip}(\sum(y-v)^\top d/\sum\|d\|^2,0,1)$。这是普通stacking/约束回归，不是创新。其训练成本含单路训练与交叉折模型，必须单独记账。当前runner只保存验证预测，因此无法凭已有文件诚实拟合训练OOF权重；任意验证标签优化的权重只能记为oracle/后验诊断，不能作为已部署baseline。

## 3. 共同直接分支损失的精确解释

当RGB-only模型的fused等于video时，当前loss为1.25×video MSE，Radar-only同理。常数倍不改变该模型的理想最优解，但AdamW epsilon、weight decay和gradient clipping使“完全相同优化轨迹”不是自动保证。因此按实际配置比较，不能把常数倍说成额外信息，也不能仅凭它断言单路被不公平削弱。

concat是fused MSE＋0.25×video MSE＋0.25×radar MSE；direct branches有助于维持单路能力，但共同编码器也接受不同来源梯度。late_average的fused损失直接耦合两单路；quality/precision通过权重耦合；残差家族另有0.25 candidate MSE。这些改变训练动力学。**utility_gate与joint_gate/random才隔离标签辅助这一因素**；它们相同的主损失和直接损失不因gate名称改变。

不能用内部video分支误差替代独立RGB性能，不可把缺失时的raw gate均值解释成实际雷达权重。门控对有害融合的评价应只在实际both可用的位置分析，并同时保留所有失败帧的整体结果。详细静态边界见[现有代码审查](fusion_code_review_20261013.md)。

## 4. 单一最有直接证据的下一假设

**假设：当预测髋锚点不可用、其他关键点仍可用时，保留这些局部视觉预测，可以改善同预算姿态估计；雷达能否进一步互补，应在保留相同局部信息的普通融合和独立视频基线之间判断。** 首先验证观察是否被输入表示抹掉，比继续用新名字包装无效risk辅助更有证据。

这属于输入表示与可用性机制假设，中心归一/局部关节mask均是已有工具，不称新的融合算法。第一轮utility的AUC约0.523且未超过joint/random；4.43mm线段oracle余量不保证可观测证据能预测GT权重。因此当前没有把risk辅助升格为唯一主贡献的实验依据。[首轮诊断](fusion_diagnostic_findings_20261013.md)、[文献与oracle幅度反证](fusion_literature_audit_20261013.md)。

像素缓存独立审计由pixel_cache任务负责。本审查在收到停止重复统计指令前完成了仅poses/features的额外交叉计数，没有读取target：遮挡验证384帧中hip特征可用21、centroid规则可用95；74帧原hip表示为零但仍有预测joint，73帧至少2joint；289帧完全没有预测joint。训练相应为70/1152→279/1152。该计数只能说明有可恢复的**预测观测**，不说明这些预测正确、centroid模型已改善或自然遮挡已解决。原始receipt在[独立审计目录](../artifacts/fusion_expanded_design_audit_20261013/)。已完成的[pixel_cache独立审计](fusion_representation_audit_20261013.md)得到相同计数，且明确centroid改变全部有效帧的锚点及辅助损失支持集；表示结论优先引用该报告。

## 5. 可证伪的最小对照与失败解释

采用两表示×同方法的配对实验：hip、visible_centroid各自重训独立RGB/容量RGB、concat及主候选，且相同cache/backend/seed/epoch/增强。主问题优先比较同方法hip→centroid的误差变化，之后再在每个表示内部比较concat相对容量RGB、utility相对joint/random。二者不能混为一个“融合提升”。

按**预测输入**预定义三个分析组：髋可用、髋不可用但仍有其他预测joint、完全无预测joint。不要按目标关节误差或某模型是否改善选择样本。第二组若centroid独立RGB和concat都不改善，假设不获支持；若仅髋本来可用组改善，更可能是归一锚点/训练分布变化，不能归为恢复局部信息。完全无joint组也大幅变化时应检查学到的fallback/先验和训练组成，而不能说直接恢复了不存在的视觉观测。

centroid还在髋可用帧改变锚点，并扩大video/candidate/utility损失的有效样本人口。因此两组相同epochs和参数不等于纯粹“只恢复74帧”的单因素。若扩大实验值得投入，可加“仅hip不可用时centroid fallback、其他帧仍hip归一”的独立表示控制；它是后续可证伪控制，本轮没有修改现有模型/runner。

雷达人工dropout已固定按原hip reference决定，避免两表示可用性变化偷偷改变独立雷达增强日程。独立Radar-only应成为负控制，若同data/seed/backend下输出不同，先检查日程/代码再解释表示收益。融合内部雷达分支可因共同训练改变，不能要求它与另一表示逐位相同。

如果只centroid本身有收益且utility仍不超joint/random，保留表示结果和risk负结论；不自动转向oracle-g/OOF长训。若utility在新表示内部出现跨种子和主体稳定收益，才补同actual残差的quality-only、固定g、oracle-g重加权和主体OOF对照，成本单列。不得只保留有利condition/关节/种子。

## 6. 七条件绘图与两表示比较已准备

[plot_fusion_results.py](plot_fusion_results.py)默认展示summary中全部evaluation_conditions，支持GPU的clean/dark/noisy_dark/occlusion/video_missing/radar_missing/radar_sparse七条件。收益图仍使用已声明的五主condition selection score，dark/sparse不悄悄进入主分数。

两表示使用独立参数：

```powershell
python research/plot_fusion_results.py --experiment artifacts/fusion_method_research_20261013_expanded_hip --stage expanded_hip
python research/plot_fusion_results.py --experiment <centroid实验目录> --stage expanded_centroid --compare-hip artifacts/fusion_method_research_20261013_expanded_hip
```

比较图在centroid的figures/representation_comparison内单独输出PNG300dpi、PDF、可编辑SVG、配对CSV和receipt。脚本先核验两份complete summary、明确representation标签、缓存SHA、主体/帧数、初始化seed、训练设置、环境、源码、参数量；另逐method/seed比较保存NPZ中的frame_ids/subject_ids/target_mm。控制不一致会拒绝表示配对。以hip-minus-centroid的同seed差显示mean±paired-seed SD，不把两组SD独立相加，也不生成假主体CI。

图和caption明确写“独立表示/可用性变化，不是融合机制改善”。没有把旧64帧CPU结果并入新96帧GPU图，没有加载模型或重算预测。七条件绘图已在真实GPU两轮smoke summary运行验证并视觉检查，caption/receipt明确2epochs，不解释为正式精度；正式新summary未complete前不生成正式七条件结果图。两表示配对图尚待两份正式complete summary执行，不能将代码准备称已产生表示结果。官方强baseline可行性已另写[ImmFusion只读审计](immfusion_official_feasibility_audit_20261013.md)，不由这份绘图receipt背书算法创新。
