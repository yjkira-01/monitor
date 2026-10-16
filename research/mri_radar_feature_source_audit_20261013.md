# mRI发布雷达featuremap来源：有限输入审计

日期2026-10-13。当前本地loader直接读取作者发布的`features/radar/subject*_featuremap.npy`并作训练集归一，未追加GT框裁点或GT根中心化。仅凭这一事实和XYZ字段名，不能把作者发布数据的全部生成过程声明为已验证GT-free。

核读归档mRI主论文§3.1、根README、现有数据audit、已归档官方源代码ZIP，随后取得官方Supplement并读A.4。主论文描述将去噪后的雷达点云合并三连续帧，64→192点；Supplement A.4（PDF第4–5页）引用其[1]来说明点云生成/软硬件/预处理，引用[2]来说明连续帧融合减轻稀疏，未具体交代GT人体框/根中心化、发布强度归一统计拟合范围或每个feature的camera邻接窗口。README说明特征来自同步数据，维度为(frames,14,14,5)，最后5维是XYZ、Doppler与intensity。归档源码ZIP共60项，只有根README/license和ActionFormer action_localization，未找到雷达feature生成器。主论文的MPJPE根对齐是评估规定，不能据此推断雷达输入也用GT根中心化。作者明确记录连续帧聚合，但本次下一camera点、复制计数和精确中心窗口的数值结论仅适用于所查subject1/camera138。

对固定expanded开发缓存subject1的第一个既选camera frame138作有限查验，未按GT或模型误差选帧。读取该主体发布feature、aligned radar CSV和raw radar CSV，未读pose_labels或最终测试主体，复用先前完整数据包认证，不重hash大包。数值证据在`artifacts/fusion_representation_audit_20261013/mri_radar_feature_source_audit.json`，可复现脚本为`research/audit_mri_radar_feature_source.py`。

- 该feature共196行，55行非零、141行全零padding。
- 55行XYZ/Doppler多重集与aligned camera137、138、139的全部点**精确一致**；与前两帧+当前的camera136、137、138窗口不一致。137/138映射同一raw雷达frame212，各10点；139映射raw frame213，共35点。因此feature实际包含raw212的点重复两次和raw213的35点。
- 55是发布feature非零行数，包含重复的raw212点，不能解释为55个独立物理雷达观测。当前quality gate所用count是非零行数/196，同样不能称独立物理点密度。
- 对应raw XYZ/Doppler最大绝对差4.84375e−5，符合aligned CSV舍入。该帧没有相对这三个camera帧全部雷达点的GT根平移或额外裁点；不能把一帧核查扩大到全部预处理决策。
- 发布intensity与匹配raw intensity满足仿射关系：斜率0.0347224291841595、截距−0.8865417125892164，最大残差1.78e−15。其统计拟合范围和生成机制未确认，不能据此声称全部特征是未预处理raw。
- camera138的发布feature包含下一camera139点。这份离线缓存不能支持严格因果、瞬时单帧或零前视延迟的实时输入声明；此观察不改当前冻结开发阶段。

可支持的表述是“使用作者发布的雷达featuremap；本地loader不引入GT裁点/根中心化，有限既选帧核查未发现这些操作”。发布feature生成全过程是否不依赖GT仍**未确认**，不能以字段名或监督标签另存为理由自动排除输入预处理泄漏。

下一camera索引本身不等于已经定量证明未来物理时刻或前视毫秒数；此处没有据索引推算实际延迟。后续实时协议应核对真实camera/radar时间轴，并单独构建因果raw输入。

文献代理随后完成[A.4及引用独立核读](mri_supplement_preprocessing_audit_20261013.md)，原文与receipt在`artifacts/mri_supplement_independent_audit_20261013/`。A.4所引FUSE在§3.2式(3)以M=1写出前/当前/后一帧中心窗口，支持本次数值匹配的解释；MARS描述固定range/depth区域与保留ghost点、XYZ排序和zero padding、radar原点。引用方法可说明作者的处理背景，不能替代mRI缺失的生成代码或扩充为全部发布帧GT-free已验证；强度拟合统计范围仍未知。

已封存所查帧feature、各非零行对应raw行和三相邻camera的aligned行到`artifacts/fusion_representation_audit_20261013/mri_radar_selected_frame_receipt.npz`，JSON记录NPZ文件SHA、逐数组dtype/shape/内容SHA、审计脚本SHA以及原ZIP成员size/CRC。只封存开发输入，不包含GT标签；来源大ZIP保持既有完整认证。

来源：

- mRI官方归档README：`artifacts/research_track_20261012/mri_readme.md`，原入口[官方仓库](https://github.com/SizheAn/mRI)。
- mRI原论文：`artifacts/research_track_20261012/mri_paper.pdf`及`mri_paper_fulltext.txt`；既有认证SHA256为135fcb8b910b57fa0e51f76a26aae5caea4aa7839350cf5b92895a6f56c8e39f、13页。
- [官方Supplement](https://proceedings.neurips.cc/paper_files/paper/2022/file/af9c9c6d2da701da5a0acf91ec217815-Supplemental-Datasets_and_Benchmarks.pdf)：`artifacts/fusion_representation_audit_20261013/mri_supplemental.pdf`、对应`.txt`及`mri_supplemental_receipt.json`，3056484bytes、8页、SHA256为3ec8b750fcd0a6a6bfc464940b1581b9ee0124e221669a7693eef0650becec91，文件头/实际页数/全文可读已核验。
- 数据入口：[Dryad最新版](https://datadryad.org/dataset/doi:10.5061/dryad.9ghx3ffpp)。数据包认证与CameraFrame映射记录：`artifacts/research_execution_20261012/mri_data_audit.json`。
- 官方项目旧入口`https://sizhean.github.io/mri/`此次只读请求HTTP404，未反复重试；后续有限搜索取得上述NeurIPS官方Supplement，已交文献代理独立核读。
