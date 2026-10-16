# 保留髋根坐标、分离局部几何：下一阶段固定协议

日期2026-10-13。本计划依据输入可用性审计和已有探索确定，在expanded hip/centroid正式成对结果读取前冻结；仍属同一开发划分上的探索，不预测最终测试主体。当前只落地GT-free表示函数和计算检查，不修改runner/model，不启动本阶段训练。

## 两种表示与共同对照

输入为相同独立RTMO原始预测poses、公开cam1焦距以及原缓存raw hip features。函数没有target、GT框或GT关键点参数。

- `hip_centroid_fallback`：hip-valid帧逐项保留原hip features；仅hip-zero但仍有raw joint confidence≥0.3的局部帧，用可见关节置信加权中心XY和相同原始confidence；完全无可见预测时全zero。
- `hip_confidence_only`：hip-valid帧同样逐项保留原hip features；同一局部帧的原始34维XY置0，保留与前者完全相同的17维raw confidence；完全无可见预测时全zero。

两组feature维数仍51，所有raw confidence及video-valid/auxiliary-valid masks完全相同。在当前1536帧缓存，clean/dark/noisy_dark没有root-missing partial，故两fallback在这些条件均逐项等于原hip。遮挡局部帧训练209、验证74；验证中只有73帧有至少两个可见关节，另1帧仅保留confidence/identity，中心XY为0。

所有实验使用**相同的原hip-clean训练mean/std**，明确从原缓存clean_features计算，不重新对两fallback分别拟合。原缓存features与`reference_available`保持原值，用原hip availability固定radar dropout。同一批次顺序、条件选择、图像丢弃、点稀疏与radar丢弃随机数均共享。

conf-only的XY=0是原始特征定义；观测字段按共同hip-clean统计归一后通常是固定的`-hip_mean/hip_std`，缺失字段才再次置0。不能称模型获得“标准化后全零XY”。两fallback相减时共同mean抵消，局部几何差为centered_XY/hip_std，置信度、mask与归一化参考相同。

## 固定实验预算

- 方法：`rgb_matched`、`concat`、`joint_gate`。
- 两fallback × 3方法 × 初始化种子[7,17,27]，共18训练；各80epoch、最后epoch报告，不用验证选checkpoint。
- 相同1536帧缓存、split seed7、训练主体12/1152帧、验证主体4/384帧；同GPU/backend、IEEE FP32、无TF32/AMP、batch64、AdamW lr0.001/weight_decay0.0001、共同原hip增强及各方法原语义loss。
- 原hip的同预算3方法对照来自正在执行的expanded_hip正式阶段；比较时须确认cache SHA、subjects、seeds、80轮预算、loss、增强和backend相同。

主要机制比较为同架构`hip_centroid_fallback`对`hip_confidence_only`，检验在同confidence/availability/loss support下局部相对几何的增益。原hip对conf-only同时改变局部置信信息和有效监督人口，应按联合表示/可用性变化解释。跨模型只能解释整架构差异，不归因为某一融合组件。

## 报告与边界

保留全部384验证帧主指标与原五条件等权主体宏分数；遮挡另报hip-valid21、partial74、all-raw-missing289组、各主体计数、可见/不可见关节误差。不能把74相邻/相关帧或关节数当独立样本；种子波动是初始化波动。仅1个可见关节的帧不含相对XY几何，应单独标明。

整体全mask组没有新增局部输入；其跨训练收益可能来自分支训练梯度变化。若只有局部组获益，须用组比例核算对全部384帧的实际贡献。原continue-investment规则沿用探索协议，不作为显著性或新颖性证明。

实现：`research/fusion_partial_representation.py`；测试：`research/test_fusion_partial_representation.py`。root在完成expanded hip/centroid两阶段后负责接入独立runner、真实smoke与正式训练，当前没有本阶段性能结果。
