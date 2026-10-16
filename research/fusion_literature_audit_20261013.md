# 融合方式近邻文献与新颖性反证核查

日期：2026-10-13。范围：视频＋毫米波人体姿态融合的质量门控、逐关节风险监督、选择性残差与缺模态训练。此文是有界的近邻核查，不是穷尽式系统综述；不输出已被证明的新颖性，也未训练或复现下列论文。

## 1. 对当前候选的直接结论

**“逐关节门控＋预测不确定性／质量／分歧＋训练误差辅助监督＋保留强分支的跨模态残差”所用组件均有相近先例。** 最接近的先例来自同一mRI姿态任务的RadarConf 2026、预测真实单路损失并路由残差的TIER-MoE、反事实细粒度门控GAUGE，以及保留固定基准并选择接纳残差的FA–RC–DAF。普通注意力、骨架查询、模态丢弃、输出平均与不确定性加权同样不能作为原创贡献。

当前值得实验检验的较窄问题是：**能否预测某关节“实际跨模态残差相对可部署基准的有符号净改善”，再接受或拒绝该残差，从而比质量代理和单路风险门控更有效地减少有害融合。** 这是本项目的待证假设。下文只区分已核验的方法差别，不据此宣称首次提出。

现有开发结果普通点集融合改善约8.8%，支持互补性研究；它并不说明预测净收益机制成立。当前时偏恢复收益较小，也不能用复杂门控名称替代必要性证据。原数值与限制见[点云开发记录](point_validation_progress_20261013.md)。

## 2. 检索和核验方式

先读现有研究方案、BibTeX和点云报告；补检索2023–2026年的原始论文，并对基础组件向前追溯。检索维度包括joint-wise reliability/error supervision、conditional modality risk、counterfactual gating、selective/gated residual fusion、missing-modality robust pose。网页搜索只用于发现；具体方法结论优先来自会议官方页面、arXiv作者全文、PMLR正式论文、作者项目页与出版社登记题录。

保存搜索响应、精确查询、HTML/PDF、提取正文与公开请求receipt。receipt记录请求URL、时间、HTTP状态、字节数和SHA256。原始材料位于[本轮证据目录](../artifacts/fusion_method_research_20261013_literature/)。本轮单人核查，存在检索遗漏；“未找到完全相同方法”不等于“方法不存在”。

阅读程度分别写为方法原文核读、正式摘要核读、题录核验；“取得完整PDF”不等于逐字通读。没有复制他人精度进本项目结果表，没有跨数据集和划分直接比较毫米误差。

## 3. 最接近候选的已有方法

| 方法及来源 | 本轮核验程度 | 已有设计 | 对候选的直接约束 |
| --- | --- | --- | --- |
| Kim等，Cross-Attention Fusion of mmWave Radar and IMU for Robust 3D Human Pose Estimation，RadarConf 2026 | 官方摘要、学生贡献说明、session与Crossref题录核验；未取得论文全文 | mRI；output-level双向cross-attention、joint-wise reliability、residual refinement；依据每关节单路误差构造soft reliability辅助标签 | 不能声称首次逐关节误差监督门控；RGB替换IMU本身不足以成为机制创新 |
| Chang等，TIER-MoE，2026预印本 | arXiv题录、方法节及公式1–4核读 | 主体OOF真实单路Brier损失监督conditional risk；输入包含不确定性、技术质量、跨模态分歧和可用性；风险与子空间亲和控制专家残差，保留shared interaction path | “GT误差学风险”“OOF风险监督”“风险路由残差”已有；较窄差别是融合相对基准的净改善，而非单路损失 |
| Shi等，GAUGE，2026预印本 | arXiv题录、Methodology与training/inference核读 | 冻结imputer；encoder粒度的证据单元；以预测类别logit的反事实替换效应作Taylor评分，经连续gate注入attention bias | 细粒度反事实证据门控已有；它衡量预测响应影响，不直接监督毫米姿态误差净改善 |
| Zhang等，IDR，2026预印本 | arXiv题录与IV-B核读 | 冻结VLA；mask模态后与factual输出比较；输出残差L2幅度诊断；门控残差修正基础动作 | 反事实输出差＋选择性残差修正已有；依赖程度不能当成有符号任务收益 |
| Niu等，Reliability-Conditioned Heterogeneous Foundation-Model Residual Fusion for Multimodal Disaster Classification，Electronics 2026 | Crossref题录与出版社摘要核验，检索取得出版社3.3节原文片段；整页/PDF未完整获取 | 保留CLIP单位权基准；BERTweet/SigLIP有界残差；预测熵置信和编码器一致性监督router，选择接纳残差 | “强基准＋confidence/agreement＋bounded residual admission”有直接先例；原作者也将可靠性称为proxy而非校准正确率 |

RadarConf官方页面的“Accept as Lecture”及正式DOI已核验，不能把它仅写成不确定投稿。其完整细节仍未知，特别是soft reliability公式、训练划分及是否使用OOF，不能推测其没有某组件。[官方摘要/贡献](https://epapers2.org/radar2026/ESR/paper_details.php?paper_id=3251)、[正式DOI](https://doi.org/10.1109/radarconf2663773.2026.11675669)。

TIER-MoE明确区分单路可靠性、互补性和expert compatibility。它已经采用质量与分歧证据，不可写“以往只看置信度”。本项目需要证明净收益监督提供超出单路risk监督的信息，而不是重新命名risk。[方法原文](https://arxiv.org/html/2607.27289v1)。

GAUGE以固定的当前预测类别为归因目标，使用梯度与参考替换方向的绝对乘积评分。绝对影响大既可能源于有用信息，也可能源于强错误影响；其原文不是用姿态GT构造有符号收益标签。该差别只支持一个可检验假设，不自动保证本项目优越。[方法原文](https://arxiv.org/html/2608.05608v1)。

IDR利用输出差范数门控动作修正，未用姿态GT学习本项目所述残差净收益。FA–RC–DAF则是置信度和一致性代理控制有界残差；其完整实现还未核读，不能宣称其不存在其他风险监督。[IDR原文](https://arxiv.org/html/2607.25516v1)、[FA–RC–DAF出版社](https://www.mdpi.com/2079-9292/15/18/4222)。

## 4. 同任务和成熟动态融合基线

**ImmFusion应进入后续正式强基线优先清单。** ICRA2023版本直接融合RGB与mmWave做人体关节/mesh；第III-C节已有joint/vertex query对局部image/radar tokens的cross-attention，第III-D节随机mask整模态及部分token。已经取得7页作者公开PDF、读相关方法及输入协议，并核验正式DOI和作者代码。原III-A使用从GT mesh joints产生的框裁剪，因此本项目若坚持推理不使用GT框，应统一以冻结检测器裁剪并明确写“适配版”；不能拿原论文数字与本项目直接比较。[作者原文](https://arxiv.org/html/2210.01346v2)、[正式DOI](https://doi.org/10.1109/ICRA48891.2023.10161428)、[作者代码](https://github.com/Chen3110/ImmFusion)。

另一个直接同任务近邻R-DMRF-HPE发表于Measurement 2026，作者实验室摘要描述5维雷达图网络、反射强度加权消息传递，以及基础质量、可学习质量和模态先验结合的动态融合。题录和作者摘要已核验；出版社全文未取得，不能猜测逐关节程度或监督公式。它应约束“首次雷达视觉quality-aware姿态融合”的措辞。[作者摘要](https://erglab.bjtu.edu.cn/archives/4952)、[正式DOI](https://doi.org/10.1016/j.measurement.2026.120687)。

| 可借鉴的基线类别 | 原始方法依据和核读程度 | 公平对照的具体问题 |
| --- | --- | --- |
| Quality-aware dynamic weighting | QMF，ICML2023，完整17页PDF已取得，§4.1–4.2核读 | 原方法用energy uncertainty与历史training-loss排序约束权重；GT误差指导权重并非新思想，分类→姿态应称适配 |
| Predicted confidence/risk fusion | PDF，ICML2024，完整21页PDF已取得，§3.3–4核读 | 作者预测真实类别概率而非仅用最大softmax，结合Mono/Holo confidence并校准；不可把最简单confidence gate当该论文正式复现 |
| 任意缺模态组合的attention fusion | X-Fi，ICLR2025，本机正式全文§3.2–3.3再次核读 | 随机模态存在列表、cross-modal embedding与modality cross-attention已有；输入与训练协议改动必须写清 |
| 基于预测分布的回归融合 | MoNIG，NeurIPS2021，正式摘要核验，尚未补读完整公式 | 不确定性估计、模态自适应回归和corrupted modality鲁棒已有；inverse variance只作简化数学对照，不能冒充MoNIG |

[QMF正式论文](https://proceedings.mlr.press/v202/zhang23ar.html)、[PDF正式论文](https://proceedings.mlr.press/v235/cao24c.html)、[X-Fi正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/f25602918e8a0d0c86e3c752ecfbbaa1-Paper-Conference.pdf)、[MoNIG正式记录](https://papers.nips.cc/paper/2021/hash/371bce7dc83817b7893bcdeed13799b5-Abstract.html)。

常数平均、拼接、普通crossattention、quality gate、单路risk/uncertainty gate、无风险标签的同容量残差、带标签的utility gate，应共享训练/退化数据和缺模态机制。新门控若仅超过正常训练且无缺模态处理的拼接模型，不能归因于净收益机制。

既有Tac-Mamba/RPT-Fusion/SATM/ActionMAE约束仍保留，以原方案核读记录为准。它们分别涉及trust gating、时差质量处理、视频雷达适配和缺模态重建；本轮不将其旧摘要重新包装成新增全文核读。

## 5. 候选机制应怎样避免证据错位

对关节j，先确定可部署基准b_j和实际未门控候选c_j=b_j+r_j，再构造训练标签u_j=||b_j-y_j||_2−||c_j-y_j||_2。u_j>0表示该候选相对基准降低误差。若另用平方误差，应明示目标与MPJPE不同；若只预测符号，0.1mm与100mm收益会获得相同正标签，必须关注拒绝错大收益与接受大损害的代价。

推理只允许置信度、点数/掩码、预测分歧、局部骨架差异及编码特征进入gate，GT只用于训练辅助监督和事后评估。GT-free推理是必要协议，不独立构成新颖性。

必须区分三类目标：

1. **旧teacher的proxy gain**：标签来自旧普通融合，执行的是新crossattention残差。可以作经验辅助教师，但标签不对应实际接受/拒绝操作，不能称actual residual utility。
2. **在线actual candidate gain**：标签由正在变化的残差分支输出detach后计算。它确实对应当前候选，但存在移动目标和同数据优化偏差；只能先作为消融，不把gate值解释成校准的泛化risk。
3. **冻结actual candidate、按训练主体OOF生成gain**：同一held-out训练主体的b、c均由没训过该主体的fold模型预测，再由GT计算差；只用开发训练主体。最后全训练refit可能带来OOF→refit shift，仍需验证校准。OOF本身已有TIER-MoE先例。

冻结teacher可消除标签非平稳，但不会消除teacher曾见过同一训练样本导致的过拟合偏差。训练内辅助误差合法且常见；应准确说其风险解释较弱，而不是无根据地将它判为测试泄漏。test GT不得参与teacher挑选、threshold、温度、过滤、模型选择或gate监督。

缺失雷达时令门控残差严格为0才能证明video fallback与基准一致；同样需要测试空点数与显式availability。soft gate仅控制残差幅度，不保证误差下降，也不保证混合关节后骨架物理一致。

## 6. 实验必须提供的反证

- 相同实际候选：无gate、constant gate、quality-only gate、单路risk gate、无utility监督gate和utility监督gate，控制容量、更新步数、退化及缺模态训练。否则额外分支或增强是替代解释。
- 同一关节、同一帧比较有害融合：相对固定可部署baseline的误差差，报告比例及有害幅度，另列true-best branch/true-best gate oracle。Oracle不进入可部署指标。
- 区分预测依赖与正确收益：output disagreement、counterfactual output norm、attention强度和置信度可很高但方向错误；净收益标签应在训练主体以外的开发主体验证。
- 报告接受/拒绝净损失、收益符号判别、收益大小相关和按门控值的实测净收益。只看gate与quality高度相关不能证明utility。
- 正常/视觉像素退化/雷达点退化/缺失/错配同时列全；验证集threshold和退化参数冻结。真实暗光部署结论仍需要真实数据，mRI合成退化不替代它。
- 除平均MPJPE，列逐关节、高分位误差、逐主体和种子稳定性；只四个验证主体时不给正式泛化显著性背书。
- 若utility gate不优于同实际残差的quality gate，或只增加容量/增强时有利，就收缩候选；不能保留有利种子、关节、动作或扰动后写有效。

## 7. 题录更新与工具来源核验

新增正式题录已保存Crossref原始记录：Kim等RadarConf 2026 DOI10.1109/radarconf2663773.2026.11675669，Hu/Fang/Bao Measurement2026 DOI10.1016/j.measurement.2026.120687，Niu等Electronics2026 DOI10.3390/electronics15184222，Chen等ImmFusion ICRA2023 DOI10.1109/ICRA48891.2023.10161428。只改本报告和独立证据目录，未修改主BibTeX、proposal或MEMORY。

版本纠错：arXiv2409.04851现页标题为**AdaptiveFusion: Adaptive Multi-Modal Multi-View Fusion for 3D Human Body Reconstruction**，搜索旧标题出现“Towards Weather-Robust…”不能据此认定二者为同一正式版本。该12页现版PDF已取得，尚未核读方法和全部journal关系，本报告不据它建立额外技术结论。

本核查应用[literature-review技能](C:/Users/wanyanji/.agents/skills/literature-review/SKILL.md)的可追溯检索与引用核验流程。按照技能要求，实际访问arXiv2609.00065后再记工具来源；已确认论文存在，最新v2于2026-09-02修订，页面无journal-ref字段。正确作者为Timothy Kassis、Vinayak Agarwal、Yuhuan He、Darshil Patel、Aubrey M. Brueckner。它是程序性文献工作流来源，不是融合创新或实验性能证据。

工具来源引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065), [DOI](https://doi.org/10.48550/arXiv.2609.00065)。引用不附版本后缀；原页版本信息与作者已保存于`verified_arxiv_metadata.json`。

网页限制：ScienceDirect R-DMRF全文403，MDPI整页403/网页工具不可用，IEEE全文工具不可用；通过官方会议摘要、作者公开资料和Crossref核验可完成上述有限结论。没有绕过登录、付费或许可；全文不足的工作保持明确未核读状态。

## 8. 补充：oracle插值幅度与OOF regret的窄机制查重

追加日期：2026-10-13。新候选把二元“完整候选是否改善”改成：在真实候选残差方向上，预测适当的部分校正幅度。这个变更避免了完整一步过冲时一律拒绝的标签错误，**但解析最优幅度只是线段投影／约束最小二乘公式；普通门控MSE本来就优化同一方向。**

### 8.1 已有直接先例

| 原始工作 | 已核读的关系 | 不能据此声称的新组件 |
| --- | --- | --- |
| Jeong/Park/Kwon/Cho/Kwak，SRAIN，ECCV2026 | arXiv原文§3.2为每样本搜索近最优插值权重、§3.3冻结编码器后以Eq7 MSE回归该权重；Springer正式章页面及Crossref核验，2026-09-10上线 | oracle interpolation target、sample-wise weight predictor、两阶段冻结避免监督漂移已有；原任务为rank-aware spherical embedding interpolation，不是关节线段投影 |
| Breiman，Stacked Regressions，Machine Learning1996 | 完整16页作者机构公开论文，§1–2与§8核读：CV产生level-one预测；非负最小二乘组合；讨论和为1的interpolating predictor与残差协方差 | OOF预测训练融合器、CV平方损失权重、约束插值已有；原文主要是总体系数，不能说它已实现本项目逐关节条件gate |
| van der Laan/Polley/Hubbard，Super Learner，2007 | 作者机构摘要及正式Crossref题录核验：V-fold CV选择预测器组合函数，CV risk可表述成回归学习问题；oracle性质有明确假设 | OOF风险最小化与oracle比较已有；当前小主体深网不能直接继承其理论保证 |
| Yu/Zeiler/Kolossa，Reliability-Based Large-Vocabulary Audio-Visual Speech Recognition，Sensors2022 | Europe PMC完整XML已获取，§2.1.2–2.1.3核读：可靠性网络预测动态stream权重；test-GT forced alignment下CE最小化权重作为oracle上界 | “最佳融合权重＋可靠性预测＋oracle差距分析”已有；没有把这篇错误写成oracle权重标签回归，正文所述oracle主要用于上界 |
| Friedman，Greedy Function Approximation: A Gradient Boosting Machine，Annals of Statistics2001 | 正式DOI题录核验，公开大学教学镜像取得原始期刊45页扫描PDF；视觉核读印刷1191页Eq5及1194页§4.1/Algorithm2：负梯度／当前残差方向配合line search校正 | 方向＋最佳步长、残差拟合＋损失降低不是新的数学原理；与逐关节传感器gate的任务、统计单位仍不同 |

[SRAIN方法原文](https://arxiv.org/html/2608.22500v1)、[SRAIN正式出版](https://doi.org/10.1007/978-3-032-37232-1_36)、[Breiman原文](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)、[Super Learner作者记录](https://biostats.bepress.com/ucbbiostat/paper222/)、[AVSR正式论文](https://doi.org/10.3390/s22155501)、[Friedman正式DOI](https://doi.org/10.1214/aos/1013203451)。

SRAIN是最直接的新反证：不仅有sample-wise optimal-weight标签，还明确说明joint optimization会造成标签漂移，从而冻结representation再回归oracle权重。故“冻结实际残差后学习oracle g”不能仅凭步骤区别于已有研究。本轮没找到已核验的“mRI逐关节、主体OOF实际残差、线段投影标签”完全同设定原文，这仅是检索范围内的未发现，不能证明空白。

### 8.2 解析投影说明完整候选标签的不足

对一个关节令基准为$b$、实际候选为$c$、残差为$d=c-b$、真值误差方向为$e=y-b$，记$s=e^\top d$、$a=\|d\|_2^2$。在$g\in[0,1]$下，

$$
q(g)=\|y-(b+gd)\|_2^2=\|e\|_2^2-2gs+g^2a.
$$

若$a>0$，微分并投影到可行区间得到

$$
g^*=\operatorname{clip}(s/a,0,1).
$$

这是把$y$投影到线段$[b,c]$的标准解，不能当原创公式。$a=0$时输出与$g$无关，gate不可辨识；实际可固定为0或不参与幅度监督。添加分母epsilon是数值近似，需要报告尺度与近零残差掩码。

完整候选有害的条件是$q(1)>q(0)$，即$a>2s$；部分步有益只需$s>0$。因此$0<s<a/2$时，完整候选有害，但$0<g^*<0.5$的部分校正有益。例如一维$b=0,y=1,c=3$，完整误差由1增至2，$g^*=1/3$却将误差降为0。用“完整$c$是否更好”的标签会把这一方向误当成无效方向。

对固定的**独立单关节**幅度，最小化欧氏误差和其平方的argmin一致，因为平方单调；若所有关节共享一个$g$，平均欧氏误差与总平方误差的最优$g$一般不同。逐关节独立oracle还可能导致骨架约束冲突，它只是在该线段集合内的任务上界。

### 8.3 当前joint_gate的pose MSE已经隐式执行同向优化

本轮只读核对[fusion_models.py](fusion_models.py)：joint_gate使用fused=video+gate*(candidate-video)，shared pose loss为融合输出MSE；candidate另有辅助MSE。给定当前$b,d$，单关节平方误差导数为$q'(g)=2(ag-s)$；代码对坐标和样本取均值只加一个正比例因子，不改变方向与最优值。所以无utility标签的joint_gate也会推动$g$趋向同一约束最优解；网络共享、容量有限、sigmoid饱和与同时训练残差会影响能否达到该解，但不能据oracle标签声称引入了新的校正方向。

定义regret为$q(g)-q(g^*)$。若实际$b,d$被冻结，$q(g^*)$不依赖gate参数，最小化此regret与直接pose MSE完全等价。内点$0<g^*<1$时：

$$
q(g)-q(g^*)=a(g-g^*)^2.
$$

在clipping边界，一般式为

$$
q(g)-q(g^*)=a(g-g^*)^2+2(ag^*-s)(g-g^*),
$$

第二项由可行方向与KKT条件保证非负，不能忽略后声称所有边界也等于加权权重MSE。

因此额外未加权$(g-g^*)^2$监督，主要改变不同残差幅度样本的权重及边界行为。对内点目标，pose MSE天然按$a$加权，添加该项后权重变为$a+\lambda$；它可能加强微小残差样本，收益并非必然。给定推理证据$z$，直接pose风险的条件最优幅度为

$$
g_{\mathrm{risk}}(z)=\operatorname{clip}\left(
\frac{\mathbb{E}[s\mid z]}{\mathbb{E}[a\mid z]},0,1\right),
$$

而未加权拟合oracle幅度主要学$\mathbb{E}[g^*\mid z]$，两者一般不相等。这是新目标需要实验检验的统计区别，不是新几何公式。将逐样本oracle转化为推理可用预测时，条件信息不足和校准偏差不能靠“oracle”一词消除。

### 8.4 下一阶段真正需要区分的因素

有价值的对照应固定同一actual residual与训练预算，比较无额外监督的pose gate、未加权oracle-g MSE、幅度加权oracle-g MSE／直接pose regret，并比较训练内目标和训练主体OOF目标。若仅改变标签权重就得到相同改善，按重加权结果解释；若优势主要来自OOF，按训练误差偏差和泛化证据解释，承认stacking/TIER已有OOF基础。

OOF至少分别生成$b$和$c$，两路都不能见过该held-out训练主体；若只把gate训练样本打散而残差teacher仍见过该主体，不能称主体OOF轨迹。最终refit造成的残差分布变化须另核验。test GT只用于线段oracle/后悔上界分析；正式部署指标不能使用逐关节投影真值。

新增原始证据在oracle_interpolation_*receipt*.json及对应HTML/XML/PDF中。Friedman出版社请求返回iframe而非PDF，作者旧域SSL失败，后取得原始期刊扫描副本并实际视觉读两页，不能把自动提取的空正文写成已文本通读。SRAIN正式页面通过网页工具核验，独立普通请求返回Client Challenge，实际方法依据是作者arXiv原文；没有绕过访问限制。
