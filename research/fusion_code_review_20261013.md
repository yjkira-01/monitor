# 融合模型与运行脚本的独立代码审查

日期：2026-10-13。审查方式为只读源码、现有协议和测试定义；审查者没有修改模型/运行代码，也没有启动训练或独立重跑测试。因此，本文区分静态确认、测试覆盖和实际实验完成，不能把本文当作新增模型性能验证。

## 1. 审查范围与结论

审查文件：`fusion_models.py`、`run_fusion_experiments.py`、`test_fusion_models.py`，补充读取 `test_fusion_experiments.py`、`fusion_method_protocol_20261013.md`，以及旧数据loader/指标/协议。

静态阅读没有发现当前CPU路径中目标GT进入forward、风险标签反向操纵姿态误差、缺失点NaN污染注意力或最终测试主体进入开发训练的阻断性错误。风险标签使用未门控候选、误差计算被`no_grad`包围；所有门控证据的预测姿态被统一detach；置信度在标准化前提取；骨长证据加回训练目标均值模板，均与协议的核心约束相符。

首轮静态审查提出的随机标签有效域、主评分宏平均和独立单路基线问题，主任务已修正，审查者随后重新读取代码确认。当前剩余问题主要是门控名称/诊断口径、未门控候选保存和依赖来源完整度；没有发现需要停止当前训练的实质错误。新方法仍属于在线共适应监督，未因detach而变成OOF风险预测。

## 2. 可从代码直接确认的机制

### 2.1 GT与标签梯度

`FusionModel.forward(rgb, points, mask, confidence)`没有target参数。`utility_targets`仅在训练loss中调用，用video与**ungated candidate**对target的欧式误差差构造软标签，误差由米转换为毫米后使用margin=1mm、temperature=10mm。整个标签函数在`torch.no_grad()`内，不能沿标签路径回传梯度。

门控的12维证据为3维video、3维candidate、3维绝对差、1维原始置信度、1维有效点比例、1维骨长差。`evidence`对base/candidate统一detach，再计算绝对差和骨长。因此，辅助BCE无法通过证据路径直接更新video head、残差head或其编码器；姿态主损失仍能通过最终加权预测更新这些分支，这是正常训练。

`utility_gate`与`joint_gate`有相同模块构建顺序和主/直接分支/candidate损失。两者的明确差异为0.001风险BCE。这是当前最直接、最有解释力的辅助标签机制对照。

### 2.2 中心化目标和骨长

训练target为逐坐标减训练集均值后的米尺度偏差，评估加回同一个均值。误差标签不必额外加模板，因为prediction与target减去了相同模板，误差相消。

骨长不具有这种相消性质。代码把`target_mean_mm.reshape(17,3)/1000`拷入buffer，并在骨长差前分别加到video/candidate，避免把中心化偏差向量当作完整人体骨长。这项修正方向正确；模板仅来自训练主体，持久化进入state_dict，没有使用验证姿态修正骨架。`utility_nobone`保留输入维度/参数量，只将骨长项置零，是结构匹配的输入消融。

### 2.3 输入、掩码与缺失回退

`normalize_inputs`先读取51维特征的最后17维原始分数，按0.3得到visible，再把34个XY与17个分数的未观察字段在标准化后置零。`np.repeat(visible,2)`与既有`XY.reshape(-1)`的交错XY布局一致，未误用17个连续XYZ/分数三元组。

雷达只有mask有效点参加标准化。逐点encoder用where把无效点实际置零；mean/max池化及attention均应用mask。attention在全空点云时先把全`-inf`行改成有限零分数、softmax后再置无效权重零，避免全空softmax的NaN。

标准化后的整视频dropout同时清零rgb/confidence，整雷达dropout清零points/mask。最终预测在雷达不可用时精确选择video分支，在视频不可用时精确选择radar分支。两者同时不可用时最后一条规则选择无观测radar head的输出，属于学习到的无观测先验；不是可靠测量，应继续单列both_missing数量。

### 2.4 训练及增强公正性

模型初始化seed、batch顺序generator、augmentation generator和随机标签generator分离。每method重新用同一seed初始化三个generator，增强函数每批调用次数相同；当前代码不会因风险随机标签额外randperm而改变其他方法的样本增强。

训练条件逐样本均匀选择实际缓存特征；10%视频missing、10%雷达missing仅在视频可用时发生；20%样本以0.3概率保留每个有效点，并在稀疏化后保证至少一个原有效点。自然无视觉观测时不再人工清雷达，符合避免不必要双缺失的设计。

全部method使用fused MSE和0.25×两个直接单路MSE；attention/residual家族另有0.25×candidate MSE。风险家族再有0.001 BCE，precision另有0.001 variance NLL。代码与协议一致，但跨整个method表同时改变结构与额外candidate损失，不能把concat与attention的差异称为纯结构因果效应。

## 3. 需要修正或准确披露的事项

### 3.1 随机标签有效域：已修正并静态确认

位置：`fusion_models.py`的`fusion_loss`中`utility_random`分支，审查快照约176–180行。

初读版本先对全部batch×17的label做随机置换，再按`both`筛有效训练项，可能把缺失样本的标签混入有效辅助监督。修订代码先构建valid，再对`labels[valid]`置换并回填。现在只破坏有效joint域内的配对，不再混入无效模态标签。

静态检查确认修正方向正确。有效域完全为空时randperm(0)和空索引回填合法，风险weighted_mean归零，未发现新增边界错误。

仍需准确描述置换每batch重新产生，不是固定全训练集乱序；随机标签与有意义标签可学习性不同，仍不能作为唯一正则化解释控制。

### 3.2 主评分宏平均：已修正并静态确认

位置：`run_fusion_experiments.py:summarise`，审查快照约87–104行。

初读版本selection_score取全帧MPJPE平均。修订代码直接枚举condition×subject的`per_validation_subject_MPJPE_mm`构造每seed评分，现为显式主体宏平均，与配对差值和bootstrap口径一致。

clean_change仍使用clean全帧MPJPE。当前每主体等帧且全部保留，与主体宏平均相等；未来不等帧设置需一并改clean门槛或明确采用微平均。各condition表保留全帧指标可作为次要描述。

### 3.3 独立单路与容量对照：已补充并静态确认

位置：`fusion_models.py:METHODS`与`run_fusion_experiments.py`的video/radar预测保存。

修订METHODS新增`rgb_single`、`rgb_matched`和`radar_single`，共13种方法。独立视频模型仅创建rgb encoder/video head，独立雷达模型仅创建point encoder/radar head，未通过内部融合分支充当单路基线。

容量视频width=245，按层维度核算111,077参数；concat为111,129，差约0.047%。总参数几乎匹配，但不能据此称实际计算或优化难度完全一致。单路缺失时保留其无观测head先验，不借另一模态信息，这个单模态边界正确。

内部分支与独立模型的统计仍需区分：runner保存的video/radar字段在未创建该分支的单路模型中是零偏差模板占位，不能解释成实际分支预测；可用计数也体现该模型可利用的分支，而非所有传感器实际状态。新增单路训练是否完成需由实际产物确认，本文仅确认代码已添加。96帧新结果仍不可直接与旧64帧结果当作方法增益比较。

### 3.4 “quality_gate”实际也是逐关节预测分歧门控（P2，命名/解释）

位置：`fusion_models.py`约116–120行。

它把video/radar姿态、置信度、点比例与骨长差送入共享MLP，得到17个权重。它不是仅用可观测原始质量的整帧scalar gate。`utility_global`才把证据跨joint平均后得到相同整帧权重。

建议：结果描述为“单路姿态间逐关节门控”或“逐关节质量/分歧门控”；若需要严格简单帧级门控，补相应方法。现实现是更强的门控baseline，本身不应削弱，但需准确命名，避免把候选的细粒度结构当作区别于该baseline的创新。

### 3.5 保存的门控平均包含不会生效的门控（P2，诊断）

位置：`run_fusion_experiments.py`约219–223行。

`mean_gate_weight`直接平均全部帧的原始gate；整模态missing时实际fused已被torch.where覆盖，gate不会决定输出。缺失条件下的这个平均值不能解释成实际雷达权重，更不能当风险校准。

建议：单列both=true位置的active gate统计（无有效帧则null），并可另外保存经缺失回退后的effective radar/candidate权重。注意attention residual gate控制的是candidate correction，不等于直接雷达姿态融合权重。

当前negative_fusion_vs_available_video_fraction按整帧平均误差判断，且包括雷达missing时相等回退。它是帧级负迁移比例，不能描述成逐关节负迁移率；更细粒度诊断应另算joint错误差，并至少区分both=true与全体视频可用。

### 3.6 未门控候选、风险标签分布和代码依赖证据不足（P2，复现与解释）

位置：`run_fusion_experiments.py`预测保存与report SHA，约166–167、218行。

目前保存fused、video、radar和gate，未直接保存candidate。这使风险组的未门控误差、候选oracle空间和辅助任务诊断不易独立重算。可以重新加载权重运行，但比直接保存候选证据要求更多依赖。

建议保存candidate预测、video/radar/both可用掩码，以及有效域内utility正例/软标签均值、gate分布与负迁移幅度；这些是诊断，GT标签仅计算结果阶段，不加入推理输入。

修订report新增`input_audit_sha256`和`input_audit_conditions`，可锁定缓存审计内容及condition定义，已静态确认。source_sha仍只有runner与fusion_models；实际依赖还包括`train_mri_point_baselines.py`、`run_mri_baselines.py`、`audit_mri.py`、`validate_mri_point_baselines.py`及`pose_fusion_protocol.json`，建议后续一并归档SHA。新增像素生成脚本也应纳入来源记录；实际pixel audit完整性应由缓存审查与读取日志确认，本文没有验证全部像素缓存。

### 3.7 g=1对照不具有完全相同的attention初始化（P3）

`joint_attention`没有gate模块；`joint_gate`先构建gate再构建attention模块。单次torch.manual_seed相同并不使后来attention层初始化逐位相同，因为gate消耗了随机数。视频/雷达公共模块在条件分支前构建，保持一致；`joint_gate`与`utility_gate`模块完全相同则保持整个初始化一致。

这不是常规跨架构比较的错误，但如果称`joint_attention`是“完全相同候选、只把g设为1”的严格对照，应在相同完整模型上强制g=1，或使用模块级独立初始化generator。当前宜称共享候选结构及损失的attention基线，而不是逐位相同候选的单因素实验。

## 4. precision方法的准确含义

代码使用每关节各向同性方差，softplus后加1e-5 m²，直接单路均值有MSE监督。variance NLL中的均值平方误差detach，因此NLL直接拟合variance head；通过共享encoder，它仍可能间接影响均值特征。它不是完整联合Gaussian-NLL训练的官方模型复现，也没有建模模态误差相关性。

NLL采用平均三坐标平方误差和单joint variance，数值归约对应逐坐标平均的各向同性高斯项；未发现三坐标尺度算错。逆方差权重vv/(vv+rv)方向正确：video variance越大时，radar权重越大。不能把这类variance自动解释为校准误差概率或融合后验。

## 5. 测试覆盖与尚需验证的实际证据

初读5项机制测试定义覆盖：融合method精确回退、掩码NaN/点排列不变、utility label detach/方向、缺失场景有限梯度、风险BCE不直接更新姿态参数。另3项数据测试定义覆盖：原始置信度与标准化缺失字段、共同增强复现/可用性、稀疏验证保留最后有效点。修订后主任务报告9项聚焦检查通过；审查者没有独立执行这些测试，实际通过数及测试对象应引用主任务的输出，不能归为本审查者完成。

当前定义尚未直接覆盖：非零target_template下骨长证据正确性、有效域内random标签分布、主评分宏平均不等长主体、全缺失双方输出的标注政策、权重独立重载与预测重算一致。这些验证要按是否修正代码或新出现不确定性选择，不能把全部未覆盖项都当作训练阻断。

新96缓存的完整读取、全部模型80epoch三seed完成、训练过程中标签分布和candidate学习是否正常、最终结果是否优于同结构joint_gate/随机控制，均不在本次静态审查的证明范围。

## 6. 审查快照

初读与主任务修改有短时间重叠；以下SHA及本文第3.1/3.2/3.3/3.6节的修正确认对应随后重新读取的修订版本。后续修改应另记版本。

| 文件 | SHA256 |
| --- | --- |
| fusion_models.py | C4670238EAC02624B293A6A7D0780353DEC05034E80B4DBC7C52FA2B65820B87 |
| run_fusion_experiments.py | 566858E41BD07DAD72C82B980DA28016F7CAFF219674341344A26E4B278D3E66 |
| test_fusion_models.py | 2C8E6B07DA464640701CF4D6A0AC70BCFFF8BF6DA3579DF043FD69D95C9EEFD9 |
| test_fusion_experiments.py | 184F63D4870FD8E29B3A7E13138CBA7F92BD754644CAC11DEA323B88374FAE74 |

本次审查支持继续在已授权开发范围内运行；它不替代统计证据、新颖性核查、独立最终测试或预训练主体重合审计。
