# mRI 官方补充 A.4 的独立雷达输入预处理审计

本审计复用 pixel_cache 下载的 [NeurIPS 官方补充](https://proceedings.neurips.cc/paper_files/paper/2022/file/af9c9c6d2da701da5a0acf91ec217815-Supplemental-Datasets_and_Benchmarks.pdf)，8 页，SHA256 `3ec8b750fcd0a6a6bfc464940b1581b9ee0124e221669a7693eef0650becec91`。实际读 PDF 第 4–5 页 A.4、第 8 页 references，渲染第 4 页核对 Table B 和正文；原 PDF、下载 receipt 在 `artifacts/fusion_representation_audit_20261013/`。独立页定位、提取文本、参考全文和 SHA receipts 在 `artifacts/mri_supplement_independent_audit_20261013/`。

## A.4 直接说明的范围

A.4 把点云生成的软件／硬件设置、数据预处理归于参考 [1] MARS，把连续帧点云聚合归于参考 [2] Fast and Scalable／FUSE。Table B 的每帧最大点数是 64；Doppler 表示点相对雷达的速度，图中颜色表示反射信号能量强度。图 C 把雷达标在原点附近，图与正文都不是每帧 GT 人体根点中心化的规定。

该节 **没有直接规定** GT 框裁点、GT pelvis 平移、发布 intensity 的标准化公式／拟合主体范围／是否使用验证或测试主体统计，亦没有写明三帧索引或首尾序列边界处理。不能把“A.4 没有提到 GT”改写为“发布数据完整生成过程已经核验不依赖 GT”。

## 对两条实际引用的有限核读

使用普通公开请求取得 [MARS 的 NSF 开放作者稿](https://par.nsf.gov/servlets/purl/10334240)（22 页；DOI `10.1145/3477003`）和 [FUSE 的作者 arXiv v1](https://arxiv.org/pdf/2205.00097v1)（6 页；DAC DOI `10.1145/3489517.3530522`）。下载均 HTTP200，未绕过访问限制，未联系任何人。它们是引用方法的资料，不能替代未公开的 mRI feature 生成脚本。

| 问题 | 引用文献中的明确描述 | mRI 发布 feature 的证据边界 |
|---|---|---|
| 单帧数量／采样 | MARS §3、§4.1.1（PDF 7–8 页）处理首 64 个点，少于 64 则零填充。 | mRI A.4 确认 maximum 64；不能据此假定全部 mRI 边界和抽点代码与 MARS 逐项相同。 |
| 点顺序／矩阵形式 | MARS §4.1.2（PDF 9 页）按 x、y、z 的词典顺序排序，64×5 reshape 为 8×8×5，保留数值。 | mRI 发布形状是 14×14×5，多帧后可有额外 padding；本文没有取得生成器，不追认其全部排序实现。 |
| “区域”是否是 GT 框裁剪 | MARS §4.1.3（PDF 10 页）用固定 x,z∈[−1,1]m、y∈[0,3]m 标记 in/out-of-range；文中明确把 out-of-range ghost 点保留在训练和推理。 | 这是固定雷达坐标范围说明，不是基于每帧 GT 人体框的过滤证据；mRI 一帧全部点匹配结果另见输入审计。 |
| 根中心化 | MARS §4.1.1 将雷达设为原点，预处理章节未定义 GT pelvis 平移。§4.2.1 的 Kinect x 轴反向、传感器对齐与参考标签相关。 | 不能将标签坐标变换和评估的 GT 根对齐混成雷达输入根中心化；也不能仅靠文献缺省证明整个 mRI pipeline。 |
| 三帧聚合 | FUSE §3.2 Eq.(3)（PDF 3 页）定义中心窗口，M=1 时 F[k]={f[k−1],f[k],f[k+1]}。实际渲染核对公式。 | 支持 pixel_cache 在 subject1 既选 camera138 核到 camera137/138/139 全部 XYZ/Doppler 多重集的解释；含下一 aligned camera 帧，不能支持瞬时单帧／零前视的严格因果输入声明。具体物理延迟仍需时间戳，不把该参考的 100ms 周期强加给 mRI。 |
| intensity 拟合范围 | 两引用描述反射强度通道；MARS 的排序／reshape 保值。没有找到发布 intensity 仿射变换的拟合范围说明。 | subject1 单帧 observed affine relation 的统计来源仍未知；不宣称 train-only，也不宣称已证明跨测试泄漏。 |

## 与数值溯源协同的结论

[pixel_cache 的有限输入溯源](mri_radar_feature_source_audit_20261013.md) 已从预先选定的开发帧核查：55 个非零行是 camera137/138/139 的 XYZ/Doppler 多重集，包含相同 raw frame 的重复点；非零行数不等于独立物理反射观测数。其 intensity 与 raw 存在精确仿射关系，但生成统计范围未确认。本审计只补充官方 A.4 和它实际引用的文献证据，没有重做数值匹配、没有读取 GT pose_labels，没有访问最终测试主体。

目前可严谨表述为：**使用作者发布的离线三帧雷达 featuremap；本地 loader 不追加 GT 裁点或 GT 根中心化；有限既选帧与引用方法均未显示每帧 GT 裁点／根平移，但发布 feature 的全部生成流程与 intensity 统计范围尚未完整确认。** 全阶段仍是冻结开发划分的离线探索；这些输入溯源边界应随结果保留，不能以本地训练归一化仅用训练主体为理由覆盖上游未知范围。
