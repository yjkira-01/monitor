# 局部几何对照：独立源码及真实CPU smoke 审查

日期：2026-10-13。只审查/重载，不启动或争用训练 GPU；正式训练由root串行负责。

## 表示与共同训练约束

`hip_preserving_fallback_features` 仅接收预测poses、公开focal、原raw hip特征和mode，没有GT参数。两髋预测confidence>=.3的原有效帧逐项copy不变；仅髋缺失但存在其它预测可见关节的帧改变特征。centroid/confidence_only共享原confidence与availability，有效joint/loss masks因此相同；全无可见预测保持zero。

只读对比 `run_fusion_partial_experiments_gpu.py` 与 main GPU runner：主要变化限于 partial loader、表示选项、原hip-clean共同RGB统计及report/source/protocol metadata。batch64、AdamW参数、FP32/noAMP/noTF32、CPU batch/augmentation RNG、same-device risk RNG、原hip availability RadarDrop reference、loss与最后epoch策略均保留，未发现阻断问题。

loader在改写各condition前保存 `row["hip_clean"]=row["clean"].copy()`；每condition的 `reference_available_*` 来自原缓存hip feature置信度。conf-only原XY=0在共同统计下可成为可见joint固定−mean/std，不应写成模型标准化后获得全零XY。单可见关节的centroid相对XY=0，仅提供置信度/关节身份，没有相对几何。

root与pixel_cache后来发现并修复了centroid空partial的边界：空batch不能用原 `visible_centroid_features(...).reshape(0,-1)`。当前helper只在partial.any()时调用centroid，main pose helper不改。该修复没有追溯替换已封存confidence-only CPU smoke的源码。

## 已完成的真实CPU smoke独立重载

目录：`artifacts/fusion_partial_cpu_smoke_20261013`。相同实际1536帧缓存，train1152/validation384；rgb_matched/concat/joint_gate×seed7/17×1epoch，confidence-only表示。只证明运行链路，不是精度结果。

- matching source snapshot加载原helper SHA `84bf6ce57e8007c54502cdadc6d780d16b4f856a4ee3b54bbeb69ca193f42c6e`。
- 安全权重keys/shape/dtype/finite重载6模型，169数组与518指标检查全部一致，最大差0。
- 从原hip_clean训练数据重算全部normalization逐项完全一致，cache/audit/subject/frame/target核对通过，最终保留主体未用于预测或指标。
- saved-array缺模态身份全部精确：video_missing等radar；radar_missing的382可用视频帧等video，另外2双缺失帧等声明radar先验；独立rgb模型只使用自身分支。

凭据在 `analysis/verification.json`、`analysis/diagnostics.json`、`analysis/fallback_verification.json`。分析器12项聚焦检查通过，包括pinned helper、alias恢复、原hip_clean共同统计与错误stats拒绝；补充髋重中心4项检查通过。

后续centroid-fallback GPU smoke须使用修复后新helper SHA，并归档独立源码，旧smoke保留。18个完整训练必须仍共用原hip_clean统计、原RadarDrop reference及正式预算；不因1epoch smoke指标选择架构。

## 修复后geometry真实GPU smoke已完成独立核验

`artifacts/fusion_partial_gpu_smoke_20261013`：同1536实际cache、rgb_matched/concat/joint_gate×seed7/17×2epoch、hip_centroid_fallback。六套安全权重重载，169数组最大CPU重建CUDA预测差.00048828125mm（事前.01mm界限），518保存指标精确复算差0；shared原hip_clean训练normalization逐项exact。六套exact fallback382可用视频/2双缺失先验全部通过。

匹配封存修复helper SHA `5daa677c8a1c8609515b798d564b4180dde171ddc3376e39368c55f4f956ed04`，partial runner SHA `ed54757a92a545316b9c4886c7114c1bacc58fd811108f3d18083188596030fa`。该runner新增protocol快照归档不改变训练操作。审计通过凭据已交root按冻结协议启动18正式；本2epoch smoke不计入精度表或模型选择。
