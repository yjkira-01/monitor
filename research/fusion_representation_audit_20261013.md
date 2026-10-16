# hip与可见中心表示：输入和公平性独立审计

日期2026-10-13。独立读取1536个已认证缓存画面的poses/features，未读取GT姿态或启动训练，未重复缓存完整性核验/ZIP哈希。数字来自`artifacts/fusion_representation_audit_20261013/representation_audit.json`，脚本为`research/audit_fusion_representation.py`。

## 实际可用信息

训练每条件1152帧。centroid-valid定义为至少一个raw joint confidence≥0.3，因此与any-raw-visible相等；hip-valid要求两个髋关节均达到阈值。

| 训练条件 | hip valid | any raw visible | centroid valid | 缺根但有局部关节 | raw全空 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 1133 | 1133 | 1133 | 0 | 19 |
| dark | 1136 | 1136 | 1136 | 0 | 16 |
| noisy_dark | 1042 | 1042 | 1042 | 0 | 110 |
| occlusion | 70 | 279 | 279 | 209 | 873 |

验证每条件384帧。

| 验证条件 | hip valid | any raw visible | centroid valid | 缺根但有局部关节 | raw全空 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 382 | 382 | 382 | 0 | 2 |
| dark | 380 | 380 | 380 | 0 | 4 |
| noisy_dark | 365 | 365 | 365 | 0 | 19 |
| occlusion | 21 | 95 | 95 | 74 | 289 |

遮挡训练209局部帧均至少2个可见关节，175帧至少4关节；验证74局部帧中73帧至少2关节、58帧至少4关节，另外1帧只有单关节。单关节中心表示的相对XY为0，只保留关节身份与confidence，不能称保留相对几何。四主体分布如下，所有主体仍保留全部96帧主评估。

| 验证主体 | hip valid | partial | raw全空 | partial中仅1关节 |
| --- | ---: | ---: | ---: | ---: |
| 5 | 5 | 32 | 59 | 1 |
| 6 | 6 | 15 | 75 | 0 |
| 19 | 5 | 16 | 75 | 0 |
| 20 | 5 | 11 | 80 | 0 |

partial pooled结果的32/74帧来自主体5，应同时给逐主体结果；74帧、关节或三初始化种子均不增加独立验证主体数量。

## 源码语义和GT范围

`visible_centroid_features`只接收独立预测poses与公开焦距：以confidence≥0.3选可见关节、以clip(score,0,1)加权中心、将可见XY相对中心按焦距缩放，原始可见confidence作为最后17维。GPU loader从公开cam1_intrinsic取焦距，不用GT框、GT坐标或目标误差选择锚点。未发现GT进入该表示的路径。监督目标与原固定GT筛查仍是既有训练/开发协议的一部分；本审计不把整个监督训练描述成不使用GT。

原始confidence范围在本缓存均为[0,1]以内，anchor的clip在当前数据上不产生额外截断。confidence是检测器分数，不是校准过的3D误差概率。

`normalize_inputs`先复制raw confidence side channel，再对51维RGB作clean-training mean/std归一，随后将不可见XY/conf字段置0。model的video_valid及gate evidence使用该raw side channel，不把zscore当原始score。遮挡验证中319/722个可见confidence的zscore为负，它们仍可用；这种中心化不会使valid判定错误。

当前clean训练没有缺根但有局部信息的帧，hip与centroid的clean confidence逐项相同，因此17维confidence的mean/std也精确相同。两表示的XY mean/std各自由clean训练拟合，未使用验证拟合。

## 公平比较和解释限制

同一表示内所有方法读取同一特征和mask，模型输入仍51维，工厂不接收representation，不因表示更改参数量。GPU runner使用原hip缓存availability作为共同radar-dropout reference，条件/批次/增强generator在每方法和每种子重新固定。这避免centroid新增可用帧改变独立radar训练的输入日程。实际weights/predictions/history一致性仍需两个正式目录complete后核查。

centroid并非只改partial帧：它重定所有可见画面的坐标中心，包括382个clean验证有效帧。clean验证中原hip中心与centroid中心的像素距离中位70.35px、p95 104.55px；同一帧可见关节的两两几何保持，公共平移锚点及其训练XY统计变化。跨表示结果不能单归因为新增局部信息。

centroid还将partial帧的video_valid/both置真，使video branch、candidate和utility辅助损失在更多帧上参与；hip输入在这些帧直接采用radar分支。相同loss公式并不意味着有效监督人口相同。这是表示/可用性处理的联合变化，不能作为单一融合组件增益。

raw全空的289验证帧在两表示均video_valid=False，融合模型按源码精确返回各自已训练的radar分支。此组若有跨训练改善，来源可以是训练梯度耦合造成radar分支变化，不是局部视觉输入被恢复。21个hip-valid帧与74个partial帧应另报结果，主指标保留全部384帧。整体遮挡收益可按21/384、74/384、289/384的组比例分解；关节可见/不可见误差是诊断，不把相关关节视为独立样本。

下一阶段已经冻结`research/fusion_partial_protocol_20261013.md`：hip-preserving centroid fallback对confidence-only fallback，共享原hip-clean mean/std和相同raw confidence/valid/loss support。原始conf-only XY=0在观测字段归一后是固定-mean/std，不能称标准化后零。该对照更准确地检验局部几何；当前函数与7项检查通过，没有本阶段正式性能结果。

边界修复：最初helper在整批无partial时调用centroid空数组，独立复现reshape(0,-1)失败。待root确认confidence-only CPU smoke快照封存后，仅在partial helper增加partial.any() guard，并补全hip-valid及全raw-missing两项边界检查，7项全部exit0；main `fusion_pose_representation.py`保持不变。

已审查源码的SHA保存在JSON，主要位置为`fusion_pose_representation.py`、`run_fusion_experiments_gpu.py`的load_data/normalize_inputs/augmented_batch，以及`fusion_models.py`的valid/fallback/fusion_loss。正式paired分析会再核对两目录的cache SHA、主体/seed/budget/backend，以及独立radar逐项相同，之后给出三组和可见/不可见关节误差。

## partial输入尺度检查

结果保存在`artifacts/fusion_representation_audit_20261013/partial_input_scale_audit.json`，由`research/audit_partial_input_scale.py`执行。只读已完成hip阶段的normalization和开发缓存预测输入，未读GT目标/原始标签，未重做ZIP核验或训练。焦距从clean预测像素偏移与既有hip-relative XY恢复，所有主体重建误差≤1e−10；hip-clean mean/std与训练缓存精确一致。

34维XY的最小clean-training std为0.00385213，最大0.120185，无字段位于1e−8 floor或低于1e−6，clean训练中也没有从未观测过的关节。所有float32归一输入有限。

| 遮挡partial输入 | train maxabs | val maxabs | train p95 / p99 / p99.9 | val p95 / p99 / p99.9 | train / val 值数>10 |
| --- | ---: | ---: | --- | --- | --- |
| centroid | 46.2073 | 49.5929 | 2.5809 / 5.3207 / 24.137 | 2.91135 / 5.48138 / 30.6105 | 57 / 19 |
| confidence-only | 5.95267 | 5.95267 | 4.11189 / 5.94451 / 5.95267 | 4.92275 / 5.94451 / 5.95267 | 0 / 0 |

上述百分位分母是partial帧全部51维（含置零的不可用字段），不是仅观测XY；JSON另给仅观测XY统计。没有绝对值>100输入，不存在1e−8 floor造成的巨大放大。但centroid的少量46–50个标准差尾部可能影响优化，这项诊断本身不证明局部几何有益或有害。保持冻结mean/std、floor与输入不截断，避免根据验证观察调参。confidence-only原始XY=0仍对应观测归一字段的固定−mean/std，不能称归一XY为0。
