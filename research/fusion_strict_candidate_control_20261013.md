# 严格同初始化的 g=1 对照与后续来源固定建议

日期：2026-10-13。独立设计审查建议，尚未实现或启动本对照训练。现有正式 hip/visible-centroid 的 runner/model 保持冻结。

## 当前对照的准确含义

`joint_attention` 与 `joint_gate` 共享 candidate 结构和 candidate 辅助 MSE，但前者不构建 gate，后者在 attention 层构造前初始化 gate。相同 `torch.manual_seed` 不能让两者后续 attention/candidate 参数逐位一致。现有比较可解释架构基线差异，不能称完全相同候选初始化下仅改变 g 的单因素对照。

`joint_gate`、`utility_gate`、`utility_random` 则构建完全相同模块；它们是目前辅助效用监督的较严格对照。随机标签只在 both 有效域内部打乱，不借助最终测试。

## 建议的严格 g=1 实施

构造完整 `FusionModel("joint_gate")`，全部模块按原顺序初始化，再冻结不参与输出的 gate 参数。不要在构造前跳过 gate，也不要额外创建消耗随机数的新层。

最小 wrapper 可以调用原 forward，再仅在 both=true 的帧把 fused 换为 candidate。原 forward 已经执行了缺模态 `where`，所以 both=false 保留原 fused，即视频缺失等 radar、雷达缺失等 video、双缺失等声明 radar 零特征先验。保存的有效 gate 为 1；inactive effective gate 若保存为 0，必须明确是回退占位，而不是网络风险预测。

继续按 `fusion_loss(..., method="joint_gate")` 使用 fused MSE、.25×video/radar direct MSE、.25×candidate MSE，不加入 utility BCE。两组均采用相同数据、初始化、batch顺序、像素condition、augmentation原hip reference、optimizer、80epoch最后checkpoint和数值backend。

冻结 gate 是为了避免零梯度张量触发无效 AdamW weight decay；冻结发生在初始化后，不影响公共参数 RNG。gate 的 449 个参数仍计入总参数，但 trainable 参数需单列；wrapper调用原 gate 时计算量仍存在。若以后优化掉该计算，也不能把两者说成实际计算完全一致。

训练前必须核对两组全部 state_dict，包括 candidate/attention/shared branches，初始化数组逐位相等；同一输入的初始 video/radar/candidate 也应相等。训练中 fused 的梯度不同是本控制的干预，不能宣称之后的 candidate 轨迹仍相同。缺模态最终身份检查沿用现有 exact fallback 审计即可。

这项控制验证 learned gating 相对常开候选的作用，不代表 attention 原创或雷达语义对应已得到证明。只有跨主体可重复的增益才值得进一步增加拟合风险教师的预算。

## 后续来源固定

已有原 trial 没有记录所有 import 依赖的当时 SHA，不能现在把当前 SHA 追认成历史版本。独立重载 receipt 已明示这项限制；行为精确重建是补充证据。

之后的新 trial 宜一并固定并归档以下实际依赖：`train_mri_point_baselines.py`、`run_mri_baselines.py`、`audit_mri.py`、`validate_mri_point_baselines.py`、缓存生成脚本、representation helpers、使用的 protocol 文件。它们应在训练前写入 source/dependency SHA manifest，与 dataset/cache/model 来源区分，记录 Python/Torch/Numpy/backend/version。复制的 protocol 也应核对其摘要所列 SHA。

分析器若要真正从这些快照导入依赖，应按 import 顺序设置对应模块 alias，最后恢复原 `sys.modules`/`sys.path`。仅把当时文件复制到目录、运行时仍从当前工作区 import，不等于完整固定重载来源。

## 已落地的严格分析兼容

`analyze_fusion_trials.py` 现已按摘要 `runner_file` 加载 partial 独立 runner，按 matching SHA 加载原 partial helper，并以 `rgb_normalization_key="hip_clean"` 重算共同原 hip-clean 训练统计；partial若缺 pinned helper 或使用其它归一来源将拒绝。每次新分析自动归档 `analyzer_source.py`，历史 analysis 不覆盖。

`audit_fusion_hip_recenter.py` 只在完整独立重载 receipt 后读取原保存 fused prediction，输出补充 prediction-hip-recentered MPJPE/P95/逐关节/逐主体及 signed变化。预测减自身11/12髋中点，GT原本已髋中心化；无 Procrustes、无GT参与该操作、无checkpoint选择或主metric改写。正变化表示更差，负变化表示更好，均保留。四项边界检查验证纯平移移除、可以变差、平移不变以及非髋中心GT拒绝。
