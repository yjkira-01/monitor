# 门控证据单位缩放：优化条件探索

首轮后半程根据早期观察追加，属于已见开发结果后的探索。当前gate证据中的姿态与骨长差是米尺度，而置信度和点比例是约0–1；骨长/残差差异可能只有几毫米。固定单位缩放有可能改变优化条件，不能将其改善称为新增融合机制。

本轮对同参数的joint_gate、utility_gate、utility_random三模型共同应用同一预先固定缩放：姿态乘10，绝对姿态分歧与骨长差乘25，原始置信度和点比例不变。没有按验证误差搜索不同缩放值；尺度不是重新拟合验证统计。数据、80轮预算、3seed、共同增强与各自原语义损失不变。

最重要比较为scaled_utility_gate vs scaled_joint_gate及scaled_utility_random，排除“更合适单位”或额外正则化替代解释；与未缩放对应模型的比较只解释预处理/优化效果。若风险辅助仍无优势，就停止把它作为贡献候选扩训。

实现：[fusion_scaled_gate_models.py](fusion_scaled_gate_models.py)、[run_scaled_gate_experiments.py](run_scaled_gate_experiments.py)。单独保留所有尝试结果。

## 新96帧/GPU执行安排（正式结果读取前固定）

此前只实现/检查，尚未正式训练。现新96帧hip组已完成，三残差模型主分数joint91.190、utility91.262、random91.011mm均未达门槛。可见关节中心组仍在执行。增加一个有限的单位优化控制：在完全相同1536帧缓存、四实际像素条件、主体、初始化[7,17,27]、IEEE FP32 CUDA、80epoch末轮/batch64和原loss预算上训练三个scaled模型，共9套。

使用保持不变的`run_fusion_experiments_gpu.py --family scaled --representation hip`，未缩放对照为刚完成的`expanded_hip`同backend结果，不引用旧CPU64帧来归因。固定scale仍为原先10/25/1，不按新验证误差调整。source快照、权重/全部预测/逐主体与各条件结果保存到独立目录；诊断后解释同方法缩放收益和三scaled方法内部risk标签对比。

该追加是在同一开发结果反馈下确定的探索，不是预注册确认性试验。即使scaled整体提升，也只说明优化条件变化；只有utility稳定胜同scale无辅助及随机控制才支持继续检验其训练辅助目标，仍不能据此证明新颖性或泛化保证。
