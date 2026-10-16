# 同方向论文基线：选择、实现状态与验收

## 2026-10-16 最新已验证状态

同强RTMO输入的X-Fi融合核心适配已完成三初始化×80轮，并独立重载核验：clean73.810535±0.967243mm、五条件S97.555240±0.946417mm。相同1536帧旧普通concat为65.860253/S90.216644mm；因此强输入X-Fi核心仍弱于普通强流程，不把它作为衬托候选的唯一基线。完整证据在`artifacts/xfi_strong_inputs_resumed_20261016/reporting/comparison.md`和`audit.json`。

这里保留作者原X_Fusion的512维/32token/8头/4层/五槽，而128维本地RGB/point编码器、token适配器与缺失辅助头为本地改动。它是核心适配，不是完整官方网络或MM-Fi复现；X-Fi核心lr0.0001、旧concat lr0.001，容量不同，应明确披露。

AWC的Pearson-Shapley、全训练集逐样本MPJPE梯度平方Fisher、epoch起点加权约束已经按原文重建，4项机制测试通过；稳健版两轮运行试验的11模型独立重建差0。尚无全体开发数据的AWC正式结果，不能提前计入主表。使用本地2模态架构，应命名“AWC训练规则重建适配”，不冒充官方MM-Fi网络。

主连续数据计划16人×12动作×64帧=12288，正在提取；不同人口/稳健损失/三条件协议，必须另表重训所有比较方法。最终四人测试仍封存，ImmFusion与R-DMRF-HPE的原阻碍/未取得完整实现状态不因本轮完成X-Fi而自动解决。

日期2026-10-13。用户明确“基线是指同方向论文”。主要任务改为先建立有出处、保留原方法核心的论文基线，再进行创新实验。通用拼接、门控、固定权重和参数匹配单路仅作为补充控制；不以这些本地小模型代替论文基线。

## 论文集合与实施顺序

**2026-10-16执行更新：**X-Fi原生模块的mRI适配现已完成三目标×三初始化共9套100轮开发训练，原5套核SHA原样保留，补齐4套；完整独立CUDA重载、原图/雷达重建、NumPy指标和主体宏分数复算均通过，差值0。MSE正常161.818mm、MPJPE损失149.932mm、髋中心约束157.856mm；MPJPE对MSE的S改善11.978mm且4/4主体、3/3初始化同向，髋中心约束未通过最强后处理控制。原lr0.001三目标的误差仍高于训练均值姿态144.173mm，显示这些设置存在训练问题；这是同方向论文核心模块适配结果，不是MM-Fi原协议复现或超越论文的证据。追加3套低学习率MSE对照现已完成并独立核验，只改lr0.001→0.0001，正常误差124.544±1.955mm、S130.457±1.414mm；对原MSE的S改善31.235mm（19.32%），四主体/三初始化同向，资源门槛通过。这是训练常规优化，不是融合机制创新，仍未超过旧RTMO65.86mm。见[配对结果](../artifacts/xfi_stability_20261016/reporting/paired_accuracy_report.md)。完整原9套结果见[报告](../artifacts/xfi_accuracy_formal_resumed_20261016/reporting/accuracy_report.md)，[独立核验](../artifacts/xfi_accuracy_formal_resumed_20261016/analysis/verification.json)。后文4帧检查为历史准入记录。

| 工作 | 与本研究的关系 | 原文/源码实际状态 | 当前定位 |
| --- | --- | --- | --- |
| **X-Fi，ICLR 2025** | 同为RGB＋毫米波雷达的3D姿态估计，包含I+R实测行；原数据MM-Fi | 正式全文已读；官方HPE源码已固定到commit；官方RGB/雷达骨干权重已取得 | **第一实施基线：原生模块的mRI RGB＋雷达适配** |
| **ImmFusion，ICRA 2023及后续TCSVT扩展** | 直接RGB＋毫米波融合、恶劣环境人体重建；原数据mmBody，22关节/mesh | 原文与官方源码已审计；原协议GT裁图/裁点/root中心、mesh资产、PointNet扩展和预训练链接仍有适配障碍 | 第二个强基线候选；保持局部/全局融合核心，适配前不得称已复现 |
| **AWC，CVPR 2026** | 同方向多模态3D姿态估计及模态竞争；MM-Fi，含RGB/雷达，但主表为四模态 | 11页原文已读；官方仓库快照只有README＋两张图，**无训练代码** | 新近论文对照；若按公式重建，必须写“论文重建适配”，先验证Fisher/Shapley实现 |
| **R-DMRF-HPE，Measurement 2026** | 与本研究最直接：mRI、RGB＋毫米波雷达、动态融合 | 题录与作者/出版社摘要可核对；原始全文结果表和可用源码未取得 | 高优先相关工作与待补基线；不能用摘要猜网络并声称复现 |

本轮有界补检覆盖网页发现、arXiv/Crossref、Crossref/OpenAlex，查询和原始JSON保存。不是穷尽综述。动作分类、目标检测、单雷达多格式融合、雷达＋IMU不列主要同输入姿态基线。2026-10-02预印本Kinematics-Induced Multimodal 3D Human Pose Estimation with Subject-Level Privacy与结构融合相关，但声明代码接受后发布，保留为近邻而非已可运行基线。

## X-Fi原生模块适配已实际完成的工作

官方仓库：https://github.com/NTUMARS/X-Fi 。固定commit `9ba74cc98beb2d75a8eadbe74b1b151dc3d39614`；本地稀疏源码在`artifacts/paper_baseline_sources_20261013/xfi_checkout`，32个HPE/说明/许可/依赖文件的SHA记录在`xfi_source_receipt.json`。没有使用仓库中的作者预测数组。

`research/xfi_paper_baseline.py`直接编译原源码中的33个函数/类定义，保留原定义体与AST哈希；仅绕过不必要的导入和强制载入五种模态的构造函数。适配使用原生RGB ResNet18、毫米波TD Point Transformer、linear_projector与X_Fusion。保留512维token、每模态32个token、8头、4次融合迭代、RGB=0/radar=2的五槽索引和未激活融合分支，共**31,806,675参数**。没有把它缩成现有51维骨架MLP。

RGB输入沿用作者实际代码的480×640、BGR、float 0–255；mRI发布镜像先恢复，必要尺寸变换明确记录，不使用GT框。雷达用发布196×5点集，不用GT根平移或GT三维框。mRI的17关节语义由新回归头监督学习，不能把MM-Fi输出关节语义直接搬入结果。原RGB/雷达单路预训练回归头均不用于mRI最终预测。

作者README链接的公共HPE权重包已通过普通公开下载取得，ZIP约210.45 MiB。只提取RGB_Resnet18.pt与mmwave_all_random_TD.pt，分别45,275,867和5,887,797字节，SHA、ZIP成员与下载来源齐全。使用`weights_only=True`加载张量字典并严格核对模型字段。

RGB原权重的`batch_norm1.running_var`中58/64项为正无穷；所有学习参数有限，雷达全部状态有限。原文件未改动，异常单独记录，不能将它称为已经通过完整数值核验的直接预训练基线。适配允许明确的**训练观测BN统计重估**，不改学习参数、不接收target、不使用开发验证画面校准。

原生CUDA AdaptiveAvgPool2d反向在本机Torch2.11.0/cu128中没有严格确定性实现。因此保留原操作、IEEE FP32、显式确定性cuDNN、warn_only策略，并披露反向非严格确定性；不能与此前小网络的严格确定性训练混称。

## 已完成的真实运行检查及限制

使用训练主体1、2各两张已认证画面，共4张；通过源视频SHA/CRC、精确帧位置、缓存SHA与发布点云/目标索引核对。没有使用开发验证或最终测试主体。

- 随机初始化原生模型：前向/反向两步、缺RGB/缺雷达/两者均缺失、GPU安全权重重载通过；训练峰值约0.692 GiB。这是运行检查，不能引用为姿态精度。
- 作者预训练参数＋4张训练画面BN重估＋冻结骨干：两步仅融合层更新，原骨干学习参数逐项不变，重估后状态有限；原生两路预测与适配调用精确一致，三种缺失路由与原生子集/训练先验一致，保存参数与同一GPU重载预测精确一致。成功产物为`artifacts/xfi_mri_pretrained_probe_v4_20261013`，训练峰值约0.398 GiB。
- 初次严格确定性反向失败、第一次预训练重载因冻结标志/运行设置差异失败，均保留独立失败目录；未覆盖失败证据。
- 完整CPU重建仍未达到既有0.01 mm容差：随机模型最大差约0.1003 mm，预训练检查约0.3404 mm。原生前端RGB/radar tokens均有差别，初始KNN索引有8230项差异；使用完全相同CPU tokens时，原生融合层CPU/GPU最大差仍约0.0658 mm。该诊断不改变原模型，也没有把容差调大后宣称通过。**同GPU重载可复现；跨设备误差限制保留，正式评价应固定同一后端，并在正式协议中解决验收方式。**

4张画面校准、两步优化与上述显存不是完整训练资源估计；没有新论文基线MPJPE、没有正式3初始化长训、没有用这次检查取代已有65.86 mm开发结果。

## 正式训练前必须固定的内容

1. 使用完整12名训练主体的1152张clean画面重新估计BN统计，禁止将本次4帧统计当正式统计；保存全部统计、帧名单、视频/缓存/权重/源码SHA。
2. 对同方向论文明确“原协议复现”或“mRI适配”。X-Fi适配披露数据、关节、两模态、预训练、BN、训练预算与缺失策略差异；ImmFusion不能保留GT输入便利却称与预测输入公平比较。
3. 固定一路RGB＋雷达、共同帧/主体划分和17关节评价，保留全部失败画面。原始图像模型与冻结RTMO模型的比较是整条输入与估计流程比较；研究纯融合机制时，基线与改进共享同一骨干和同一特征/训练条件。
4. 主要误差沿用髋中心目标MPJPE，另列再次预测根对齐辅助误差；不混PA-MPJPE、顶点误差或五条件S。正常MPJPE与鲁棒五条件S分别报告。
5. 记录原代码训练策略：官方HPE配置100epoch/batch16/lr.001，MSE；其optimizer仅更新projector与fusion，骨干BN在`model.train()`下仍可能更新。若选择冻结骨干、固定BN和缓存tokens，必须声明此适配，不能声称完全原样训练。
6. 验收同后端保存重建与原指标复算，跨设备差异单列；不得拿CPU/GPU不同实现的微小差距当机制收益。三初始化仍是一个主体划分，最终[3,10,12,18]暂不预测。

**创新阶段的准入条件是：已实际运行的同方向论文基线、完整来源和公平协议先成立，再针对其失败模式确定一个机制假设。** 现有在线风险辅助仍无贡献证据，不恢复无依据的名称/结构搜索。

## 可核查来源

- X-Fi正式论文：https://proceedings.iclr.cc/paper_files/paper/2025/file/f25602918e8a0d0c86e3c752ecfbbaa1-Paper-Conference.pdf
- ImmFusion原文：https://arxiv.org/abs/2210.01346 ；既有官方审计：`research/immfusion_official_feasibility_audit_20261013.md`。
- AWC正式会议题录：https://openaccess.thecvf.com/content/CVPR2026/html/Qi_Towards_Balanced_Multi-Modal_Learning_in_3D_Human_Pose_Estimation_CVPR_2026_paper.html ；已读稿：https://arxiv.org/abs/2501.05264v5 ；仓库：https://github.com/MICLAB-BUPT/AWC 。正式题录可检索，直接网页请求403已记录，不影响已取得原稿。
- R-DMRF-HPE DOI：https://doi.org/10.1016/j.measurement.2026.120687 ；作者摘要：https://erglab.bjtu.edu.cn/archives/4952 。Crossref/OpenAlex精确标题核查返回无OA PDF，不能据此断言所有渠道均不存在全文。
- 最新结构/隐私近邻：https://arxiv.org/abs/2610.02943 ，标为预印本，非已确认会议论文。

本阶段使用academic-literature-search与experimental-design技能；沿用既有工具来源核验记录，不将工具元数据作为算法创新证据。
